"""Embedding utilities for tool content vectorization."""

import functools

import numpy as np
import structlog
from fastembed import TextEmbedding

logger = structlog.get_logger(__name__)


@functools.lru_cache(maxsize=1000)
def _cached_embedding_function(model_name: str, text: str, model: TextEmbedding) -> np.ndarray:
    """Module-level cached function for generating embeddings."""
    logger.debug(f"Computing embedding for text of length {len(text)}")
    embeddings_list = list(model.embed([text]))
    return embeddings_list[0] if embeddings_list else np.array([])


class EmbeddingManager:
    """Manages embedding generation and operations for tool content.

    IMPORTANT - Embedding Dimension Configuration:
    The embedding dimension is determined by the FastEmbed model used and must match
    the dimension configured in the database schema (currently 384 for BAAI/bge-small-en-v1.5).

    When changing the embedding model:
    1. Verify the new model's embedding dimension
    2. If dimension differs from 384, update the database schema via migration
    3. Re-embed all existing data with the new model
    4. Update any hardcoded dimension values in tests and validation code

    The default model (BAAI/bge-small-en-v1.5) produces 384-dimensional embeddings.
    See database migration file for the configured dimension in vector tables.
    """

    def __init__(
        self,
        model_name: str,
        enable_cache: bool,
        threads: int | None,
        fastembed_cache_path: str | None,
    ) -> None:
        """Initialize with specified embedding model.

        Args:
            model_name: Name of FastEmbed model to use.
                       Default is a lightweight, fast model.
                       WARNING: Changing models may require database migration if
                       the new model has a different embedding dimension.
            enable_cache: Whether to enable embedding caching.
            threads: Number of threads to use for embedding generation.
                    None = use all available CPU cores (default FastEmbed behavior).
                    Set to 1-4 to limit CPU usage in production.
        """
        self.model_name = model_name
        self._model: TextEmbedding | None = None
        self.enable_cache = enable_cache
        self.threads = threads
        self.fastembed_cache_path = fastembed_cache_path

    @property
    def model(self) -> TextEmbedding:
        """Lazy load the embedding model."""
        if self._model is None:
            # Enable local_files_only when cache_dir is set for offline/airgapped deployments
            local_files_only = self.fastembed_cache_path is not None
            self._model = TextEmbedding(
                model_name=self.model_name,
                threads=self.threads,
                cache_dir=self.fastembed_cache_path,
                local_files_only=local_files_only,
            )
        return self._model

    def _generate_single_cached_embedding(self, text: str) -> np.ndarray:
        """Generate and cache a single text embedding."""
        # Use a module-level cache function to avoid memory leaks with instance methods
        return _cached_embedding_function(self.model_name, text, self.model)

    def generate_embedding(self, sentences: list[str]) -> np.ndarray:
        """Generate embedding for tool details or user queries with caching support.

        Args:
            sentences: List of sentences to embed

        Returns:
            numpy array embeddings
        """
        if not sentences:
            # For compatibility with existing tests, still call the model for empty lists
            embeddings_list = list(self.model.embed(sentences))
            return np.array(embeddings_list)

        # Use cached generation for single sentences if caching is enabled
        if len(sentences) == 1 and self.enable_cache:
            return np.array([self._generate_single_cached_embedding(sentences[0])])

        # For batch processing, generate all embeddings at once
        logger.debug(f"Computing embeddings for {len(sentences)} sentences")
        embeddings_list = list(self.model.embed(sentences))
        return np.array(embeddings_list)

    def switch_model(self, new_model_name: str) -> None:
        """Switch to a different embedding model.

        Args:
            new_model_name: Name of new FastEmbed model to use

        Warning:
            Switching models may require a database migration if the new model has a
            different embedding dimension than the current configuration (384).
            Verify the new model's dimension before switching in production.
        """
        if new_model_name != self.model_name:
            self.model_name = new_model_name
            self._model = None  # Reset to trigger lazy loading with new model
            # Clear cache since embeddings are model-specific
            if self.enable_cache:
                _cached_embedding_function.cache_clear()

    def get_cache_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary with cache size and hit/miss statistics
        """
        if not self.enable_cache:
            return {"enabled": False}

        cache_info = _cached_embedding_function.cache_info()
        return {
            "enabled": True,
            "hits": cache_info.hits,
            "misses": cache_info.misses,
            "maxsize": cache_info.maxsize,
            "currsize": cache_info.currsize,
        }

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        if self.enable_cache:
            _cached_embedding_function.cache_clear()
            logger.info("Embedding cache cleared")
