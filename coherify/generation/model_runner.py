"""
Model runner for generating actual responses from LLMs.

This module handles the critical task of actually calling model APIs
to generate responses, which is essential for proper benchmark evaluation.
"""

import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import time

from coherify.providers import ProviderManager, OpenAIProvider, AnthropicProvider


@dataclass 
class GenerationResult:
    """Result from model generation."""
    text: str
    model: str
    provider: str
    temperature: float
    latency: float
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class ModelRunner:
    """
    Handles actual model API calls for benchmark evaluation.
    
    This is essential for proper benchmark evaluation - we need to
    generate real responses, not use mock data or ground truth answers.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize model runner with configuration.
        
        Args:
            model_config: Configuration dict with provider, model, temperature, etc.
        """
        self.model_config = model_config
        self.provider = self._setup_provider(model_config)
        
    def _setup_provider(self, config: Dict[str, Any]):
        """Setup the appropriate provider based on configuration."""
        provider_name = config.get("provider", "mock")
        
        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
                return None
            return OpenAIProvider(
                api_key=api_key,
                model_name=config.get("model", "gpt-4o-mini")
            )
            
        elif provider_name == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("⚠️  No Anthropic API key found. Set ANTHROPIC_API_KEY environment variable.")
                return None
            return AnthropicProvider(
                api_key=api_key,
                model_name=config.get("model", "claude-3-sonnet-20240229")
            )
            
        else:
            print(f"⚠️  Using mock provider (no real generation)")
            return None
            
    def generate_response(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate a single response from the model.
        
        Args:
            prompt: The input prompt/question
            **kwargs: Additional generation parameters
            
        Returns:
            GenerationResult with the generated text and metadata
        """
        start_time = time.time()
        
        if self.provider is None:
            # Mock generation for testing
            return GenerationResult(
                text="This is a mock response for testing.",
                model=self.model_config.get("model", "mock"),
                provider=self.model_config.get("provider", "mock"),
                temperature=self.model_config.get("temperature", 0.7),
                latency=time.time() - start_time
            )
            
        # Real generation
        try:
            # Extract temperature and max_tokens to avoid duplicate arguments
            temperature = kwargs.pop("temperature", self.model_config.get("temperature", 0.7))
            max_tokens = kwargs.pop("max_tokens", self.model_config.get("max_tokens", 1000))
            
            response = self.provider.generate_text(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            return GenerationResult(
                text=response.text,
                model=getattr(response, 'model', self.model_config.get("model")),
                provider=self.model_config.get("provider"),
                temperature=getattr(response, 'temperature', temperature),
                latency=time.time() - start_time,
                tokens_used=getattr(response, 'total_tokens', None),
                cost=getattr(response, 'estimated_cost', None)
            )
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            # Return a fallback response
            return GenerationResult(
                text=f"[Generation failed: {e}]",
                model=self.model_config.get("model", "unknown"),
                provider=self.model_config.get("provider", "unknown"),
                temperature=self.model_config.get("temperature", 0.7),
                latency=time.time() - start_time
            )
            
    def generate_k_responses(self, prompt: str, k: int = 5, **kwargs) -> List[GenerationResult]:
        """
        Generate K responses for the same prompt (for majority voting/coherence selection).
        
        Args:
            prompt: The input prompt/question
            k: Number of responses to generate
            **kwargs: Additional generation parameters
            
        Returns:
            List of K GenerationResults
        """
        responses = []
        
        # Use temperature variation for diversity
        base_temp = self.model_config.get("temperature", 0.7)
        temperatures = kwargs.get("temperatures", [base_temp + 0.1 * i for i in range(k)])
        
        for i in range(k):
            temp = temperatures[i % len(temperatures)]
            response = self.generate_response(prompt, temperature=temp, **kwargs)
            responses.append(response)
            
        return responses
        
    def generate_for_benchmark(self, samples: List[Dict[str, Any]], 
                              question_key: str = "question") -> List[str]:
        """
        Generate responses for a benchmark dataset.
        
        Args:
            samples: List of benchmark samples
            question_key: Key in sample dict containing the question
            
        Returns:
            List of generated responses (predictions)
        """
        predictions = []
        
        for i, sample in enumerate(samples):
            question = sample.get(question_key, "")
            if not question:
                predictions.append("")
                continue
                
            # Generate response
            result = self.generate_response(question)
            predictions.append(result.text)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(samples)} responses...")
                
        return predictions


class KPassGenerator:
    """
    Generator for K-pass evaluation (Stage 2 of research pipeline).
    
    Generates multiple responses per question for majority voting
    or coherence-based selection.
    """
    
    def __init__(self, model_runner: ModelRunner, k: int = 5):
        """
        Initialize K-pass generator.
        
        Args:
            model_runner: ModelRunner instance for generation
            k: Number of responses to generate per question
        """
        self.model_runner = model_runner
        self.k = k
        
    def generate_k_pass_dataset(self, samples: List[Dict[str, Any]], 
                                question_key: str = "question") -> List[List[GenerationResult]]:
        """
        Generate K responses for each sample in the dataset.
        
        Args:
            samples: List of benchmark samples
            question_key: Key in sample dict containing the question
            
        Returns:
            List of lists, where each inner list contains K GenerationResults
        """
        all_responses = []
        
        for i, sample in enumerate(samples):
            question = sample.get(question_key, "")
            if not question:
                all_responses.append([])
                continue
                
            # Generate K responses
            k_responses = self.model_runner.generate_k_responses(question, self.k)
            all_responses.append(k_responses)
            
            # Progress indicator
            print(f"  Generated K={self.k} responses for sample {i + 1}/{len(samples)}")
            
        return all_responses