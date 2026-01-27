"""Tests for TruncationSummarizer."""

import pytest

from mcp_optimizer.response_optimizer.summarizers.truncation import TruncationSummarizer


class TestTruncationSummarizer:
    """Test TruncationSummarizer class."""

    @pytest.fixture
    def summarizer(self):
        """Create a TruncationSummarizer instance."""
        return TruncationSummarizer()

    def test_is_always_available(self, summarizer):
        """Test that truncation summarizer is always available."""
        assert summarizer.is_available() is True

    @pytest.mark.asyncio
    async def test_short_text_not_truncated(self, summarizer):
        """Test that short text within budget is not truncated."""
        text = "This is a short text."
        target_tokens = 100  # Well above the text length

        result = await summarizer.summarize(text, target_tokens)

        assert result == text
        assert "TRUNCATED" not in result

    @pytest.mark.asyncio
    async def test_long_text_truncated(self, summarizer):
        """Test that long text is truncated with marker."""
        text = "A" * 1000  # ~250 tokens
        target_tokens = 50  # Much smaller budget

        result = await summarizer.summarize(text, target_tokens)

        assert len(result) < len(text)
        assert "TRUNCATED" in result

    @pytest.mark.asyncio
    async def test_truncation_marker_present(self, summarizer):
        """Test that truncation marker is added when content is truncated."""
        text = "This is a long text " * 100
        target_tokens = 20

        result = await summarizer.summarize(text, target_tokens)

        assert "[...TRUNCATED...]" in result

    @pytest.mark.asyncio
    async def test_truncation_preserves_beginning(self, summarizer):
        """Test that truncation preserves the beginning of the text."""
        text = "START " + "middle " * 100 + "END"
        target_tokens = 20

        result = await summarizer.summarize(text, target_tokens)

        assert result.startswith("START")
        assert "END" not in result  # End should be truncated

    @pytest.mark.asyncio
    async def test_empty_text(self, summarizer):
        """Test handling of empty text."""
        result = await summarizer.summarize("", 100)
        assert result == ""

    @pytest.mark.asyncio
    async def test_very_small_target(self, summarizer):
        """Test handling of very small target token count."""
        text = "Some text that will be truncated."
        target_tokens = 1  # Very small

        result = await summarizer.summarize(text, target_tokens)

        # Should return just the truncation marker or minimal content
        assert "TRUNCATED" in result or len(result) <= len("[...TRUNCATED...]")

    @pytest.mark.asyncio
    async def test_newline_breaking(self, summarizer):
        """Test that truncation tries to break at newlines."""
        text = "Line 1\nLine 2\nLine 3\n" + "A" * 1000
        target_tokens = 30

        result = await summarizer.summarize(text, target_tokens)

        # Should truncate at or after a newline if possible
        # The result should contain the truncation marker
        assert "TRUNCATED" in result

    @pytest.mark.asyncio
    async def test_exact_boundary(self, summarizer):
        """Test text that's exactly at the boundary."""
        # Create text that's exactly at the 4 chars per token estimate
        target_tokens = 25
        text = "A" * (target_tokens * 4)  # Exactly 100 chars = 25 tokens

        result = await summarizer.summarize(text, target_tokens)

        # Should not be truncated since it's at the limit
        assert result == text
        assert "TRUNCATED" not in result

    @pytest.mark.asyncio
    async def test_multiline_text(self, summarizer):
        """Test truncation of multiline text."""
        lines = [f"Line {i}: Some content here" for i in range(100)]
        text = "\n".join(lines)
        target_tokens = 50

        result = await summarizer.summarize(text, target_tokens)

        assert "TRUNCATED" in result
        assert result.startswith("Line 0:")

    def test_custom_chars_per_token(self):
        """Test summarizer with custom chars_per_token."""
        summarizer = TruncationSummarizer(chars_per_token=3)
        assert summarizer.chars_per_token == 3


class TestTruncationSummarizerIntegration:
    """Integration tests for TruncationSummarizer with ResponseOptimizer."""

    @pytest.mark.asyncio
    async def test_used_as_fallback_when_llmlingua_unavailable(self):
        """Test that truncation is used when llmlingua is unavailable."""
        from mcp_optimizer.response_optimizer.optimizer import ResponseOptimizer

        # Create optimizer with truncation method
        optimizer = ResponseOptimizer(
            token_threshold=100,
            summarizer_method="truncation",
        )

        # Verify truncation summarizer is used
        assert isinstance(optimizer._summarizer, TruncationSummarizer)
        assert optimizer.is_summarizer_available()

    @pytest.mark.asyncio
    async def test_optimizer_with_truncation_method(self):
        """Test ResponseOptimizer explicitly configured with truncation."""
        from mcp_optimizer.response_optimizer.optimizer import ResponseOptimizer

        optimizer = ResponseOptimizer(
            token_threshold=50,
            summarizer_method="truncation",
        )

        # Large content that needs optimization
        content = "Test content " * 100
        result = await optimizer.optimize(content, "test_tool")

        # Should be optimized since content exceeds threshold
        assert result.was_optimized
        assert result.token_metrics.tokens_saved > 0
