"""Unit tests for embedding utilities."""

from unittest.mock import Mock, patch

import numpy as np

from mcp_optimizer.embeddings import EmbeddingManager


class TestEmbeddingManager:
    """Test cases for EmbeddingManager class."""

    def test_initialization_with_default_model(self):
        """Test EmbeddingManager initialization with default model."""
        manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)
        assert manager.model_name == "BAAI/bge-small-en-v1.5"
        assert manager._model is None

    def test_initialization_with_custom_model(self):
        """Test EmbeddingManager initialization with custom model."""
        custom_model = "sentence-transformers/all-MiniLM-L6-v2"
        manager = EmbeddingManager(model_name=custom_model, enable_cache=True)
        assert manager.model_name == custom_model
        assert manager._model is None

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_lazy_model_loading(self, mock_text_embedding):
        """Test that the model is lazily loaded on first access."""
        mock_model_instance = Mock()
        mock_text_embedding.return_value = mock_model_instance

        manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)
        assert manager._model is None

        # First access should trigger loading
        model = manager.model
        assert model == mock_model_instance
        assert manager._model == mock_model_instance
        mock_text_embedding.assert_called_once_with(model_name="BAAI/bge-small-en-v1.5")

        # Second access should reuse existing model
        model2 = manager.model
        assert model2 == mock_model_instance
        assert mock_text_embedding.call_count == 1

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_generate_embedding_single_sentence(self, mock_text_embedding):
        """Test generating embedding for a single sentence."""
        mock_model = Mock()
        mock_embedding = np.array([[0.1, 0.2, 0.3, 0.4]])
        mock_model.embed.return_value = iter([mock_embedding[0]])
        mock_text_embedding.return_value = mock_model

        manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)
        sentences = ["This is a test sentence."]

        result = manager.generate_embedding(sentences)

        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 4)
        np.testing.assert_array_equal(result, mock_embedding)
        mock_model.embed.assert_called_once_with(sentences)

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_generate_embedding_multiple_sentences(self, mock_text_embedding):
        """Test generating embeddings for multiple sentences (batch processing)."""
        mock_model = Mock()
        mock_embeddings = [
            np.array([0.1, 0.2, 0.3, 0.4]),
            np.array([0.5, 0.6, 0.7, 0.8]),
            np.array([0.9, 1.0, 1.1, 1.2]),
        ]
        mock_model.embed.return_value = iter(mock_embeddings)
        mock_text_embedding.return_value = mock_model

        manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)
        sentences = ["First test sentence.", "Second test sentence.", "Third test sentence."]

        result = manager.generate_embedding(sentences)

        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 4)
        expected = np.array(mock_embeddings)
        np.testing.assert_array_equal(result, expected)
        mock_model.embed.assert_called_once_with(sentences)

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_generate_embedding_empty_list(self, mock_text_embedding):
        """Test generating embeddings for empty list."""
        mock_model = Mock()
        mock_model.embed.return_value = iter([])
        mock_text_embedding.return_value = mock_model

        manager = EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)
        sentences = []

        result = manager.generate_embedding(sentences)

        assert isinstance(result, np.ndarray)
        assert result.shape == (0,)
        mock_model.embed.assert_called_once_with(sentences)

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_switch_model_different_model(self, mock_text_embedding):
        """Test switching to a different model."""
        mock_model1 = Mock()
        mock_model2 = Mock()
        mock_text_embedding.side_effect = [mock_model1, mock_model2]

        manager = EmbeddingManager(model_name="model1", enable_cache=True)

        # Load first model
        model1 = manager.model
        assert model1 == mock_model1
        assert manager.model_name == "model1"

        # Switch to different model
        manager.switch_model("model2")
        assert manager.model_name == "model2"
        assert manager._model is None  # Should reset cached model

        # Load second model
        model2 = manager.model
        assert model2 == mock_model2
        assert mock_text_embedding.call_count == 2

    @patch("mcp_optimizer.embeddings.TextEmbedding")
    def test_switch_model_same_model(self, mock_text_embedding):
        """Test switching to the same model should not reset."""
        mock_model = Mock()
        mock_text_embedding.return_value = mock_model

        manager = EmbeddingManager(model_name="model1", enable_cache=True)

        # Load model
        model1 = manager.model
        assert model1 == mock_model

        # Switch to same model
        manager.switch_model("model1")
        assert manager.model_name == "model1"
        assert manager._model == mock_model  # Should not reset

        # Access model again
        model2 = manager.model
        assert model2 == mock_model
        assert mock_text_embedding.call_count == 1  # Should not create new model
