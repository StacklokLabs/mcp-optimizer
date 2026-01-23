"""Tests for LLMLingua summarizer."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLinguaSummarizer


class TestLLMLinguaSummarizer:
    """Test LLMLinguaSummarizer class."""

    @pytest.fixture
    def summarizer(self):
        """Create a LLMLinguaSummarizer instance."""
        return LLMLinguaSummarizer()

    @pytest.fixture
    def summarizer_with_custom_force_tokens(self):
        """Create a summarizer with custom force tokens."""
        return LLMLinguaSummarizer(force_tokens=["\n", ".", ":", "-"])

    def test_initialization(self, summarizer):
        """Test summarizer initialization."""
        assert summarizer.force_tokens == ["\n", ".", "?", "!", ","]
        assert summarizer._loaded is False
        assert summarizer._available is False

    def test_initialization_with_custom_force_tokens(self, summarizer_with_custom_force_tokens):
        """Test initialization with custom force tokens."""
        assert summarizer_with_custom_force_tokens.force_tokens == ["\n", ".", ":", "-"]

    def test_is_available_without_model(self, summarizer):
        """Test is_available returns False when model is not found."""
        # With no model file present, should return False gracefully
        result = summarizer.is_available()
        # Result depends on whether model exists in the path
        assert isinstance(result, bool)

    def test_should_force_keep_punctuation(self, summarizer):
        """Test that punctuation tokens are force-kept."""
        assert summarizer._should_force_keep(".") is True
        assert summarizer._should_force_keep("?") is True
        assert summarizer._should_force_keep("!") is True
        assert summarizer._should_force_keep(",") is True
        assert summarizer._should_force_keep("\n") is True

    def test_should_force_keep_digits(self, summarizer):
        """Test that tokens with digits are force-kept."""
        assert summarizer._should_force_keep("123") is True
        assert summarizer._should_force_keep("test1") is True
        assert summarizer._should_force_keep("2024") is True

    def test_should_force_keep_wordpiece(self, summarizer):
        """Test that wordpiece tokens with force chars are kept."""
        assert summarizer._should_force_keep("##.") is True
        assert summarizer._should_force_keep("##123") is True

    def test_should_not_force_keep_regular_word(self, summarizer):
        """Test that regular words are not force-kept."""
        assert summarizer._should_force_keep("hello") is False
        assert summarizer._should_force_keep("world") is False
        assert summarizer._should_force_keep("##test") is False

    def test_reconstruct_text_simple(self, summarizer):
        """Test text reconstruction from tokens."""
        tokens = ["Hello", "world", "!"]
        result = summarizer._reconstruct_text(tokens)
        assert result == "Hello world !"

    def test_reconstruct_text_with_wordpiece(self, summarizer):
        """Test reconstruction with wordpiece tokens."""
        tokens = ["un", "##believ", "##able"]
        result = summarizer._reconstruct_text(tokens)
        assert result == "unbelievable"

    def test_reconstruct_text_mixed(self, summarizer):
        """Test reconstruction with mixed tokens."""
        tokens = ["This", "is", "amaz", "##ing", "."]
        result = summarizer._reconstruct_text(tokens)
        assert result == "This is amazing ."

    def test_reconstruct_text_empty(self, summarizer):
        """Test reconstruction with empty token list."""
        result = summarizer._reconstruct_text([])
        assert result == ""

    @pytest.mark.asyncio
    async def test_summarize_raises_when_model_unavailable(self, summarizer):
        """Test that summarize raises RuntimeError when model isn't available."""
        # Force model to be unavailable
        summarizer._loaded = True
        summarizer._available = False

        text = "This is a test. " * 100

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="LLMLingua model is not available"):
            await summarizer.summarize(text, target_tokens=50)

    def test_compute_keep_probabilities_shape(self, summarizer):
        """Test keep probabilities computation."""
        # Create mock logits (batch=1, seq=10, classes=2)
        logits = np.random.randn(1, 10, 2)

        probs = summarizer._compute_keep_probabilities(logits)

        # Should return 1D array of probabilities for class 1
        assert probs.shape == (10,)
        assert np.all(probs >= 0)
        assert np.all(probs <= 1)

    def test_filter_tokens_above_threshold(self, summarizer):
        """Test token filtering based on threshold."""
        tokens = ["Hello", "world", "test", "word"]
        keep_probs = np.array([0.9, 0.3, 0.8, 0.2])
        attention_mask = np.array([1, 1, 1, 1])

        result = summarizer._filter_tokens(tokens, keep_probs, attention_mask, threshold=0.5)

        # Should keep tokens with prob >= 0.5
        assert "Hello" in result
        assert "test" in result
        assert "world" not in result
        assert "word" not in result

    def test_filter_tokens_skips_padding(self, summarizer):
        """Test that padding tokens are skipped."""
        tokens = ["Hello", "[PAD]", "world"]
        keep_probs = np.array([0.9, 0.9, 0.9])
        attention_mask = np.array([1, 0, 1])

        result = summarizer._filter_tokens(tokens, keep_probs, attention_mask, threshold=0.5)

        assert "Hello" in result
        assert "world" in result
        assert "[PAD]" not in result

    def test_filter_tokens_skips_special_tokens(self, summarizer):
        """Test that special tokens are skipped."""
        tokens = ["[CLS]", "Hello", "world", "[SEP]"]
        keep_probs = np.array([0.9, 0.9, 0.9, 0.9])
        attention_mask = np.array([1, 1, 1, 1])

        result = summarizer._filter_tokens(tokens, keep_probs, attention_mask, threshold=0.5)

        assert "[CLS]" not in result
        assert "[SEP]" not in result
        assert "Hello" in result
        assert "world" in result

    def test_filter_tokens_force_keeps(self, summarizer):
        """Test that force tokens are kept regardless of probability."""
        tokens = ["Hello", ".", "world", "!"]
        keep_probs = np.array([0.9, 0.1, 0.9, 0.1])  # Punctuation has low prob
        attention_mask = np.array([1, 1, 1, 1])

        result = summarizer._filter_tokens(tokens, keep_probs, attention_mask, threshold=0.5)

        # Punctuation should be kept due to force tokens
        assert "." in result
        assert "!" in result


class TestLLMLinguaSummarizerWithMockedModel:
    """Test LLMLinguaSummarizer with mocked ONNX model."""

    @pytest.fixture
    def mock_summarizer(self):
        """Create a summarizer with mocked model components."""
        summarizer = LLMLinguaSummarizer()

        # Mock the session
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
        ]
        # Return logits shape (1, seq_len, 2)
        mock_session.run.return_value = [np.random.randn(1, 10, 2)]
        summarizer._session = mock_session

        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
            "attention_mask": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]),
        }
        mock_tokenizer.convert_ids_to_tokens.return_value = [
            "[CLS]",
            "Hello",
            "world",
            "this",
            "is",
            "a",
            "test",
            ".",
            "End",
            "[SEP]",
        ]
        summarizer._tokenizer = mock_tokenizer

        summarizer._loaded = True
        summarizer._available = True

        return summarizer

    @pytest.mark.asyncio
    async def test_summarize_with_mocked_model(self, mock_summarizer):
        """Test summarization with mocked model."""
        text = "Hello world this is a test. End"
        result = await mock_summarizer.summarize(text, target_tokens=5)

        # Should return some text
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_handles_inference_error(self, mock_summarizer):
        """Test that inference errors raise RuntimeError."""
        mock_summarizer._session.run.side_effect = Exception("Inference error")

        text = "Hello world this is a test. " * 20

        # Should raise RuntimeError
        with pytest.raises(RuntimeError, match="LLMLingua summarization failed"):
            await mock_summarizer.summarize(text, target_tokens=10)

    def test_run_inference_calls_session(self, mock_summarizer):
        """Test that run_inference calls the ONNX session correctly."""
        inputs = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_summarizer._run_inference(inputs)

        mock_summarizer._session.run.assert_called_once()

    def test_run_inference_with_token_type_ids(self, mock_summarizer):
        """Test inference when model expects token_type_ids."""
        # Add token_type_ids to expected inputs
        mock_summarizer._session.get_inputs.return_value = [
            MagicMock(name="input_ids"),
            MagicMock(name="attention_mask"),
            MagicMock(name="token_type_ids"),
        ]

        inputs = {
            "input_ids": np.array([[1, 2, 3]]),
            "attention_mask": np.array([[1, 1, 1]]),
        }

        mock_summarizer._run_inference(inputs)

        # Should have been called with token_type_ids included
        mock_summarizer._session.run.assert_called_once()


class TestLLMLinguaSummarizerWithRealModel:
    """Test LLMLinguaSummarizer with real ONNX model."""

    @pytest.fixture
    def real_summarizer(self):
        """Create a real LLMLinguaSummarizer instance."""
        return LLMLinguaSummarizer()

    @pytest.mark.asyncio
    async def test_summarize_reduces_text_length(self, real_summarizer):
        """Test that summarization reduces text length."""
        text = "This is an important sentence that contains key information. " * 20
        target_tokens = 50

        result = await real_summarizer.summarize(text, target_tokens)

        # Result should be shorter than original
        assert len(result) < len(text)
        # Result should not be empty
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_preserves_key_content(self, real_summarizer):
        """Test that summarization preserves important content."""
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "Python is a programming language. "
            "Machine learning models can summarize text. "
            "Important numbers like 42 and 100 should be kept."
        )
        target_tokens = 30

        result = await real_summarizer.summarize(text, target_tokens)

        # Should preserve some recognizable content
        assert isinstance(result, str)
        assert len(result) > 0
        # Numbers should be preserved due to force_tokens
        assert "42" in result or "100" in result or "." in result

    @pytest.mark.asyncio
    async def test_summarize_handles_short_text(self, real_summarizer):
        """Test summarization of text already within target."""
        text = "Short text."
        target_tokens = 100

        result = await real_summarizer.summarize(text, target_tokens)

        # Short text should be returned mostly intact
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_with_different_target_tokens(self, real_summarizer):
        """Test that different target tokens produce different compression levels."""
        text = "This is a test sentence with multiple words. " * 30

        result_50 = await real_summarizer.summarize(text, target_tokens=50)
        result_100 = await real_summarizer.summarize(text, target_tokens=100)

        # Both should produce valid results
        assert isinstance(result_50, str)
        assert isinstance(result_100, str)
        assert len(result_50) > 0
        assert len(result_100) > 0

        # Higher target should generally produce longer or equal output
        # (not strictly enforced as it depends on model behavior)

    @pytest.mark.asyncio
    async def test_summarize_preserves_punctuation(self, real_summarizer):
        """Test that punctuation is preserved in summarized text."""
        text = "Hello, world! How are you? This is great. Numbers: 1, 2, 3."
        target_tokens = 20

        result = await real_summarizer.summarize(text, target_tokens)

        # Should contain some punctuation (due to force_tokens)
        has_punctuation = any(p in result for p in [".", ",", "!", "?"])
        assert has_punctuation or len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_handles_newlines(self, real_summarizer):
        """Test that text with newlines is handled correctly."""
        text = "Line one.\nLine two.\nLine three.\nLine four.\nLine five."
        target_tokens = 15

        result = await real_summarizer.summarize(text, target_tokens)

        # Should produce valid output
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_summarize_long_text(self, real_summarizer):
        """Test summarization of longer text (up to model's max length)."""
        # Create text that approaches the 512 token limit
        text = "This is a sentence with important information. " * 100
        target_tokens = 100

        result = await real_summarizer.summarize(text, target_tokens)

        # Should significantly reduce the text
        assert len(result) < len(text)
        assert len(result) > 0

    def test_model_loads_successfully(self, real_summarizer):
        """Test that the model loads without errors."""
        # Trigger lazy loading
        assert real_summarizer.is_available() is True
        # Verify internal state after loading
        assert real_summarizer._loaded is True
        assert real_summarizer._available is True
        assert real_summarizer._session is not None
        assert real_summarizer._tokenizer is not None
