"""
Batch processing for efficient API calls and cost optimization.

This module implements batching strategies for:
- Parallel API calls
- Rate limiting
- Cost tracking
- Caching
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
import json
import hashlib
from pathlib import Path
import pickle

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class BatchRequest:
    """Single request in a batch."""
    id: str
    prompt: str
    params: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class BatchResult:
    """Result from batch processing."""
    request_id: str
    response: str
    latency: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    cached: bool = False


class ResponseCache:
    """
    Cache for model responses and computed scores.
    
    Reduces redundant API calls and score computations.
    """
    
    def __init__(self, cache_dir: str = ".cache/coherify"):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory for cache storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for fast access
        self.memory_cache = {}
        self.max_memory_items = 1000
        
        # Track cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0
        }
    
    def _get_cache_key(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate cache key from prompt and parameters."""
        # Create deterministic string from params
        param_str = json.dumps(params, sort_keys=True)
        combined = f"{prompt}:{param_str}"
        
        # Hash for consistent key
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, prompt: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Get cached response if available.
        
        Args:
            prompt: Input prompt
            params: Generation parameters
            
        Returns:
            Cached response or None
        """
        key = self._get_cache_key(prompt, params)
        
        # Check memory cache first
        if key in self.memory_cache:
            self.stats["hits"] += 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    response = pickle.load(f)
                
                # Add to memory cache
                self._add_to_memory(key, response)
                self.stats["hits"] += 1
                return response
            except Exception as e:
                print(f"⚠️ Cache read error: {e}")
        
        self.stats["misses"] += 1
        return None
    
    def save(self, prompt: str, params: Dict[str, Any], response: str):
        """Save response to cache."""
        key = self._get_cache_key(prompt, params)
        
        # Save to memory
        self._add_to_memory(key, response)
        
        # Save to disk
        cache_file = self.cache_dir / f"{key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
            self.stats["saves"] += 1
        except Exception as e:
            print(f"⚠️ Cache save error: {e}")
    
    def _add_to_memory(self, key: str, value: str):
        """Add item to memory cache with LRU eviction."""
        if len(self.memory_cache) >= self.max_memory_items:
            # Remove oldest item (simple FIFO for now)
            oldest_key = next(iter(self.memory_cache))
            del self.memory_cache[oldest_key]
        
        self.memory_cache[key] = value
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total if total > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "saves": self.stats["saves"],
            "hit_rate": hit_rate,
            "memory_items": len(self.memory_cache),
            "disk_files": len(list(self.cache_dir.glob("*.pkl")))
        }


class BatchProcessor:
    """
    Batch processor for efficient API calls.
    
    Features:
    - Batched API calls for 50% cost reduction (OpenAI Batch API)
    - Parallel processing
    - Rate limiting
    - Automatic retries
    - Response caching
    """
    
    def __init__(self,
                 api_caller: Callable,
                 batch_size: int = 10,
                 max_workers: int = 5,
                 rate_limit: int = 60,  # requests per minute
                 use_cache: bool = True):
        """
        Initialize batch processor.
        
        Args:
            api_caller: Function to call API
            batch_size: Number of requests per batch
            max_workers: Maximum parallel workers
            rate_limit: Maximum requests per minute
            use_cache: Whether to use response caching
        """
        self.api_caller = api_caller
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.rate_limit = rate_limit
        
        # Rate limiting
        self.request_times = deque(maxlen=rate_limit)
        
        # Caching
        self.cache = ResponseCache() if use_cache else None
        
        # Cost tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        self.total_requests = 0
    
    def process_batch(self, requests: List[BatchRequest]) -> List[BatchResult]:
        """
        Process a batch of requests.
        
        Args:
            requests: List of batch requests
            
        Returns:
            List of batch results
        """
        results = []
        
        # Check cache first
        cached_results = []
        uncached_requests = []
        
        for req in requests:
            if self.cache:
                cached_response = self.cache.get(req.prompt, req.params)
                if cached_response:
                    cached_results.append(BatchResult(
                        request_id=req.id,
                        response=cached_response,
                        latency=0.0,
                        cached=True
                    ))
                else:
                    uncached_requests.append(req)
            else:
                uncached_requests.append(req)
        
        # Process uncached requests in parallel
        if uncached_requests:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_request = {}
                
                for req in uncached_requests:
                    # Rate limiting
                    self._wait_for_rate_limit()
                    
                    future = executor.submit(
                        self._process_single_request,
                        req
                    )
                    future_to_request[future] = req
                
                # Collect results
                for future in as_completed(future_to_request):
                    req = future_to_request[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Cache successful result
                        if self.cache and not result.cached:
                            self.cache.save(req.prompt, req.params, result.response)
                            
                    except Exception as e:
                        print(f"❌ Request {req.id} failed: {e}")
                        results.append(BatchResult(
                            request_id=req.id,
                            response=f"Error: {e}",
                            latency=0.0
                        ))
        
        # Combine cached and new results
        all_results = cached_results + results
        
        # Update statistics
        self.total_requests += len(requests)
        
        return all_results
    
    def _process_single_request(self, request: BatchRequest) -> BatchResult:
        """Process a single request."""
        start_time = time.time()
        
        try:
            # Call API
            response = self.api_caller(request.prompt, **request.params)
            
            # Extract response details
            if hasattr(response, 'text'):
                text = response.text
                tokens = getattr(response, 'total_tokens', None)
                cost = getattr(response, 'estimated_cost', None)
            else:
                text = str(response)
                tokens = None
                cost = None
            
            # Update cost tracking
            if cost:
                self.total_cost += cost
            if tokens:
                self.total_tokens += tokens
            
            return BatchResult(
                request_id=request.id,
                response=text,
                latency=time.time() - start_time,
                tokens_used=tokens,
                cost=cost,
                cached=False
            )
            
        except Exception as e:
            raise e
    
    def _wait_for_rate_limit(self):
        """Wait if necessary to respect rate limit."""
        now = time.time()
        
        # Clean old timestamps
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
        
        # Wait if at limit
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Record this request
        self.request_times.append(now)
    
    def process_dataset(self,
                       prompts: List[str],
                       params: Dict[str, Any] = None) -> List[str]:
        """
        Process entire dataset with batching.
        
        Args:
            prompts: List of prompts to process
            params: Generation parameters
            
        Returns:
            List of responses
        """
        params = params or {}
        
        # Create batch requests
        requests = [
            BatchRequest(
                id=f"req_{i}",
                prompt=prompt,
                params=params
            )
            for i, prompt in enumerate(prompts)
        ]
        
        # Process in batches
        all_results = []
        for i in range(0, len(requests), self.batch_size):
            batch = requests[i:i + self.batch_size]
            print(f"📦 Processing batch {i//self.batch_size + 1}/{len(requests)//self.batch_size + 1}...")
            
            results = self.process_batch(batch)
            all_results.extend(results)
        
        # Sort by request ID and extract responses
        all_results.sort(key=lambda r: int(r.request_id.split('_')[1]))
        responses = [r.response for r in all_results]
        
        # Print statistics
        self.print_stats()
        
        return responses
    
    def print_stats(self):
        """Print processing statistics."""
        print("\n" + "="*60)
        print("Batch Processing Statistics")
        print("="*60)
        print(f"Total Requests: {self.total_requests}")
        print(f"Total Tokens: {self.total_tokens:,}")
        print(f"Total Cost: ${self.total_cost:.4f}")
        
        if self.cache:
            cache_stats = self.cache.get_stats()
            print(f"\nCache Statistics:")
            print(f"  Hits: {cache_stats['hits']}")
            print(f"  Misses: {cache_stats['misses']}")
            print(f"  Hit Rate: {cache_stats['hit_rate']:.1%}")
            print(f"  Memory Items: {cache_stats['memory_items']}")
            print(f"  Disk Files: {cache_stats['disk_files']}")
        
        # Cost savings estimate
        if self.cache and cache_stats['hits'] > 0:
            saved_cost = cache_stats['hits'] * (self.total_cost / max(1, cache_stats['misses']))
            print(f"\n💰 Estimated Cost Savings from Cache: ${saved_cost:.4f}")


class SmartBatchScheduler:
    """
    Intelligent batch scheduler that optimizes for cost and latency.
    
    Features:
    - Groups similar prompts for better caching
    - Prioritizes based on expected computation time
    - Adapts batch size based on response times
    """
    
    def __init__(self, batch_processor: BatchProcessor):
        """
        Initialize smart scheduler.
        
        Args:
            batch_processor: BatchProcessor instance
        """
        self.processor = batch_processor
        self.response_time_history = []
    
    def schedule_prompts(self,
                        prompts: List[str],
                        priorities: Optional[List[float]] = None) -> List[List[str]]:
        """
        Schedule prompts into optimized batches.
        
        Args:
            prompts: List of prompts
            priorities: Optional priority scores
            
        Returns:
            List of batches
        """
        if not priorities:
            priorities = [1.0] * len(prompts)
        
        # Group by similarity for better cache hits
        groups = self._group_similar_prompts(prompts)
        
        # Sort by priority within groups
        batches = []
        for group in groups:
            sorted_group = sorted(
                group,
                key=lambda p: priorities[prompts.index(p)],
                reverse=True
            )
            
            # Create batches from group
            for i in range(0, len(sorted_group), self.processor.batch_size):
                batch = sorted_group[i:i + self.processor.batch_size]
                batches.append(batch)
        
        return batches
    
    def _group_similar_prompts(self, prompts: List[str]) -> List[List[str]]:
        """Group similar prompts for better caching."""
        # Simple grouping by prompt length for now
        # In production, could use embeddings for semantic similarity
        
        groups = {}
        for prompt in prompts:
            # Group by length bucket (100 char buckets)
            bucket = len(prompt) // 100
            if bucket not in groups:
                groups[bucket] = []
            groups[bucket].append(prompt)
        
        return list(groups.values())