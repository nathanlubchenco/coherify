"""
Tests for utility modules to improve coverage.
"""

import pytest
import os
import tempfile
import numpy as np
from unittest.mock import Mock, patch

from coherify.utils.caching import EmbeddingCache, ComputationCache, cached_computation
from coherify.utils.clean_output import enable_clean_output, clean_output
from coherify.utils.transformers_utils import create_pipeline_with_suppressed_warnings


class TestEmbeddingCache:
    """Test embedding cache with better coverage."""

    def test_initialization_default(self):
        """Test default initialization."""
        cache = EmbeddingCache()
        assert cache.cache_dir.endswith("coherify_embeddings")
        assert cache.max_size == 1000
        assert cache.ttl_seconds == 3600

    def test_initialization_custom(self):
        """Test custom initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(
                cache_dir=temp_dir,
                max_size=500,
                ttl_seconds=1800
            )
            assert cache.cache_dir == temp_dir
            assert cache.max_size == 500
            assert cache.ttl_seconds == 1800

    def test_cache_key_generation(self):
        """Test cache key generation."""
        cache = EmbeddingCache()
        
        key1 = cache._get_cache_key("test text", "model1")
        key2 = cache._get_cache_key("test text", "model2")
        key3 = cache._get_cache_key("different text", "model1")
        
        # Different models or text should generate different keys
        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_put_and_get(self):
        """Test putting and getting embeddings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            embedding = np.array([1.0, 2.0, 3.0])
            text = "test embedding"
            
            # Put embedding
            cache.put(text, embedding)
            
            # Get embedding
            retrieved = cache.get(text)
            
            assert retrieved is not None
            np.testing.assert_array_equal(retrieved, embedding)

    def test_get_nonexistent(self):
        """Test getting non-existent embedding."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            result = cache.get("nonexistent text")
            assert result is None

    def test_eviction(self):
        """Test cache eviction when max_size is reached."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Small cache for testing eviction
            cache = EmbeddingCache(cache_dir=temp_dir, max_size=2)
            
            # Add embeddings
            cache.put("text1", np.array([1.0]))
            cache.put("text2", np.array([2.0]))
            cache.put("text3", np.array([3.0]))  # Should trigger eviction
            
            # Check that oldest was evicted
            assert len(cache._memory_cache) <= 2

    def test_clear_cache(self):
        """Test clearing cache."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = EmbeddingCache(cache_dir=temp_dir)
            
            # Add some embeddings
            cache.put("text1", np.array([1.0]))
            cache.put("text2", np.array([2.0]))
            
            # Clear cache
            cache.clear()
            
            # Check cache is empty
            assert len(cache._memory_cache) == 0
            assert cache.get("text1") is None


class TestComputationCache:
    """Test computation cache coverage."""

    def test_initialization(self):
        """Test initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ComputationCache(cache_dir=temp_dir, max_size=100)
            assert cache.cache_dir == temp_dir
            assert cache.max_size == 100

    def test_cache_function_result(self):
        """Test caching function results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ComputationCache(cache_dir=temp_dir)
            
            # Cache a result
            result = {"score": 0.8, "details": "test"}
            cache.put("test_function", (1, 2), {"param": "value"}, result)
            
            # Retrieve result
            retrieved = cache.get("test_function", (1, 2), {"param": "value"})
            assert retrieved == result

    def test_cache_key_generation_computation(self):
        """Test computation cache key generation."""
        cache = ComputationCache()
        
        key1 = cache._get_cache_key("func", (1, 2), {"a": 1})
        key2 = cache._get_cache_key("func", (1, 2), {"a": 2})
        key3 = cache._get_cache_key("func", (1, 3), {"a": 1})
        
        # Different args/kwargs should generate different keys
        assert key1 != key2
        assert key1 != key3

    def test_remove_computation(self):
        """Test removing cached computation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = ComputationCache(cache_dir=temp_dir)
            
            # Add and then remove
            cache.put("func", (1,), {}, "result")
            cache.remove("func", (1,), {})
            
            # Should be gone
            assert cache.get("func", (1,), {}) is None


class TestCachedComputationDecorator:
    """Test cached computation decorator."""

    def test_cached_computation_decorator(self):
        """Test the cached computation decorator."""
        call_count = 0
        
        @cached_computation()
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Shouldn't increment
        
        # Different args - should call function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2


class TestCleanOutput:
    """Test clean output utilities."""

    def test_enable_clean_output(self):
        """Test enabling clean output."""
        # Should not raise any exceptions
        enable_clean_output()

    @patch('coherify.utils.clean_output.warnings')
    def test_clean_output_context(self, mock_warnings):
        """Test clean output context manager."""
        with clean_output():
            pass  # Should execute without errors
        
        # Should have called warnings filterwarnings
        assert mock_warnings.filterwarnings.called

    def test_clean_output_nested(self):
        """Test nested clean output contexts."""
        with clean_output():
            with clean_output():
                pass  # Should work without issues


class TestTransformersUtils:
    """Test transformer utilities."""

    @patch('coherify.utils.transformers_utils.pipeline')
    def test_create_pipeline_success(self, mock_pipeline):
        """Test successful pipeline creation."""
        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe
        
        result = create_pipeline_with_suppressed_warnings(
            "text-classification", "model-name"
        )
        
        assert result == mock_pipe
        mock_pipeline.assert_called_once()

    @patch('coherify.utils.transformers_utils.pipeline')
    @patch('coherify.utils.transformers_utils.logging')
    def test_create_pipeline_with_warnings_suppressed(self, mock_logging, mock_pipeline):
        """Test that warnings are properly suppressed."""
        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe
        
        create_pipeline_with_suppressed_warnings(
            "text-classification", "model-name"
        )
        
        # Should have set logging levels
        mock_logging.getLogger.assert_called()

    @patch('coherify.utils.transformers_utils.pipeline')
    def test_create_pipeline_with_kwargs(self, mock_pipeline):
        """Test pipeline creation with additional kwargs."""
        mock_pipe = Mock()
        mock_pipeline.return_value = mock_pipe
        
        create_pipeline_with_suppressed_warnings(
            "text-classification", 
            "model-name",
            return_all_scores=True,
            device=-1
        )
        
        mock_pipeline.assert_called_once_with(
            "text-classification",
            model="model-name",
            return_all_scores=True,
            device=-1
        )