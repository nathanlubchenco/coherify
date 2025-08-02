"""
Caching utilities for computational efficiency.
Provides embedding and computation caching to speed up coherence evaluation.
"""

import hashlib
import pickle
import os
import time
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
import numpy as np


class EmbeddingCache:
    """
    Cache for text embeddings to avoid recomputation.
    """
    
    def __init__(self, 
                 cache_dir: str = ".coherify_cache",
                 max_size: int = 10000,
                 ttl_seconds: int = 24 * 60 * 60):  # 24 hours
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached embeddings
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._memory_cache = {}
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _get_cache_key(self, text: str, model_name: str = "default") -> str:
        """Generate cache key for text and model."""
        content = f"{model_name}:{text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Get file path for cache key."""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def _load_cache(self):
        """Load cache from disk."""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                cache_key = filename[:-4]  # Remove .pkl
                try:
                    cache_path = self._get_cache_path(cache_key)
                    with open(cache_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    # Check if entry is still valid
                    if time.time() - entry['timestamp'] < self.ttl_seconds:
                        self._memory_cache[cache_key] = entry
                    else:
                        # Remove expired entry
                        os.remove(cache_path)
                
                except Exception:
                    # Remove corrupted cache file
                    try:
                        os.remove(cache_path)
                    except:
                        pass
    
    def get(self, text: str, model_name: str = "default") -> Optional[np.ndarray]:
        """Get cached embedding for text."""
        cache_key = self._get_cache_key(text, model_name)
        
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            # Check TTL
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                return entry['embedding']
            else:
                # Remove expired entry
                self._remove_entry(cache_key)
        
        return None
    
    def put(self, text: str, embedding: np.ndarray, model_name: str = "default"):
        """Cache embedding for text."""
        cache_key = self._get_cache_key(text, model_name)
        
        entry = {
            'text': text,
            'embedding': embedding,
            'model_name': model_name,
            'timestamp': time.time()
        }
        
        # Add to memory cache
        self._memory_cache[cache_key] = entry
        
        # Save to disk
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass  # Continue even if disk cache fails
        
        # Evict old entries if cache is too large
        if len(self._memory_cache) > self.max_size:
            self._evict_old_entries()
    
    def _remove_entry(self, cache_key: str):
        """Remove cache entry."""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        cache_path = self._get_cache_path(cache_key)
        try:
            os.remove(cache_path)
        except:
            pass
    
    def _evict_old_entries(self):
        """Evict oldest entries when cache is full."""
        # Sort by timestamp and remove oldest entries
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        num_to_remove = len(self._memory_cache) - self.max_size + 100  # Remove extra for buffer
        
        for i in range(min(num_to_remove, len(sorted_entries))):
            cache_key = sorted_entries[i][0]
            self._remove_entry(cache_key)
    
    def clear(self):
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except:
            pass
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "memory_entries": len(self._memory_cache),
            "cache_dir": self.cache_dir,
            "max_size": self.max_size,
            "ttl_seconds": self.ttl_seconds
        }


class ComputationCache:
    """
    Generic computation cache for expensive operations.
    """
    
    def __init__(self, 
                 cache_dir: str = ".coherify_cache/computations",
                 max_size: int = 1000,
                 ttl_seconds: int = 60 * 60):  # 1 hour
        """
        Initialize computation cache.
        
        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cached computations
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache_dir = cache_dir
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._memory_cache = {}
        
        os.makedirs(cache_dir, exist_ok=True)
        self._load_cache()
    
    def _get_cache_key(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call."""
        # Create a hashable representation
        key_data = {
            'function': function_name,
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        
        # Convert to string and hash
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk."""
        if not os.path.exists(self.cache_dir):
            return
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                cache_key = filename[:-4]
                try:
                    cache_path = os.path.join(self.cache_dir, filename)
                    with open(cache_path, 'rb') as f:
                        entry = pickle.load(f)
                    
                    if time.time() - entry['timestamp'] < self.ttl_seconds:
                        self._memory_cache[cache_key] = entry
                    else:
                        os.remove(cache_path)
                
                except Exception:
                    try:
                        os.remove(cache_path)
                    except:
                        pass
    
    def get(self, function_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result for function call."""
        cache_key = self._get_cache_key(function_name, args, kwargs)
        
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]
            
            if time.time() - entry['timestamp'] < self.ttl_seconds:
                return entry['result']
            else:
                self._remove_entry(cache_key)
        
        return None
    
    def put(self, function_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache result for function call."""
        cache_key = self._get_cache_key(function_name, args, kwargs)
        
        entry = {
            'function_name': function_name,
            'args': args,
            'kwargs': kwargs,
            'result': result,
            'timestamp': time.time()
        }
        
        self._memory_cache[cache_key] = entry
        
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            with open(cache_path, 'wb') as f:
                pickle.dump(entry, f)
        except Exception:
            pass
        
        if len(self._memory_cache) > self.max_size:
            self._evict_old_entries()
    
    def _remove_entry(self, cache_key: str):
        """Remove cache entry."""
        if cache_key in self._memory_cache:
            del self._memory_cache[cache_key]
        
        try:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pkl")
            os.remove(cache_path)
        except:
            pass
    
    def _evict_old_entries(self):
        """Evict oldest entries."""
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        num_to_remove = len(self._memory_cache) - self.max_size + 50
        
        for i in range(min(num_to_remove, len(sorted_entries))):
            cache_key = sorted_entries[i][0]
            self._remove_entry(cache_key)
    
    def clear(self):
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    os.remove(os.path.join(self.cache_dir, filename))
        except:
            pass


def cached_computation(cache: Optional[ComputationCache] = None, 
                      ttl_seconds: int = 3600):
    """
    Decorator for caching expensive computations.
    
    Args:
        cache: ComputationCache instance. If None, uses default.
        ttl_seconds: Time-to-live for cached results.
    """
    if cache is None:
        cache = ComputationCache(ttl_seconds=ttl_seconds)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get from cache
            result = cache.get(func.__name__, args, kwargs)
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(func.__name__, args, kwargs, result)
            return result
        
        # Add cache control methods to the wrapped function
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_stats = lambda: cache.stats() if hasattr(cache, 'stats') else {}
        
        return wrapper
    
    return decorator


class CachedEncoder:
    """
    Wrapper for text encoders that adds caching.
    """
    
    def __init__(self, encoder, cache: Optional[EmbeddingCache] = None):
        """
        Initialize cached encoder.
        
        Args:
            encoder: Underlying encoder (must have encode method)
            cache: EmbeddingCache instance. If None, uses default.
        """
        self.encoder = encoder
        self.cache = cache or EmbeddingCache()
        
        # Try to get model name for cache key
        self.model_name = getattr(encoder, 'model_name', 
                                 getattr(encoder.__class__, '__name__', 'unknown'))
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Encode texts with caching."""
        single_text = isinstance(texts, str)
        if single_text:
            texts = [texts]
        
        results = []
        texts_to_encode = []
        indices_to_encode = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self.cache.get(text, self.model_name)
            if cached_embedding is not None:
                results.append(cached_embedding)
            else:
                results.append(None)
                texts_to_encode.append(text)
                indices_to_encode.append(i)
        
        # Encode uncached texts
        if texts_to_encode:
            fresh_embeddings = self.encoder.encode(texts_to_encode, **kwargs)
            if isinstance(fresh_embeddings, np.ndarray) and fresh_embeddings.ndim == 1:
                fresh_embeddings = [fresh_embeddings]
            
            # Cache and insert results
            for i, (text, embedding) in enumerate(zip(texts_to_encode, fresh_embeddings)):
                self.cache.put(text, embedding, self.model_name)
                results[indices_to_encode[i]] = embedding
        
        # Convert results to appropriate format
        if single_text:
            return results[0]
        else:
            return np.array(results) if all(isinstance(r, np.ndarray) for r in results) else results
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.cache.clear()


# Global caches for convenience
_default_embedding_cache = None
_default_computation_cache = None


def get_default_embedding_cache() -> EmbeddingCache:
    """Get default embedding cache."""
    global _default_embedding_cache
    if _default_embedding_cache is None:
        _default_embedding_cache = EmbeddingCache()
    return _default_embedding_cache


def get_default_computation_cache() -> ComputationCache:
    """Get default computation cache."""
    global _default_computation_cache
    if _default_computation_cache is None:
        _default_computation_cache = ComputationCache()
    return _default_computation_cache


def clear_all_caches():
    """Clear all default caches."""
    global _default_embedding_cache, _default_computation_cache
    
    if _default_embedding_cache:
        _default_embedding_cache.clear()
    
    if _default_computation_cache:
        _default_computation_cache.clear()