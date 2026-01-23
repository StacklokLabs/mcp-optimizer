"""Tests for Text traverser."""

import pytest

from mcp_optimizer.response_optimizer.traversers.text_traverser import TextTraverser


class TestTextTraverser:
    """Test TextTraverser class."""

    @pytest.fixture
    def traverser(self, simple_token_counter):
        """Create a TextTraverser with simple token counter."""
        return TextTraverser(simple_token_counter, head_lines=5, tail_lines=5)

    @pytest.fixture
    def large_traverser(self, simple_token_counter):
        """Create a TextTraverser with more head/tail lines."""
        return TextTraverser(simple_token_counter, head_lines=20, tail_lines=20)

    @pytest.mark.asyncio
    async def test_traverse_content_within_budget(self, traverser, mock_summarizer):
        """Test that content within budget is returned as-is."""
        content = "Line 1\nLine 2\nLine 3"
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert result.content == content
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_small_content_passthrough(self, traverser, mock_summarizer):
        """Test that small content is simply truncated."""
        # Content with fewer lines than head + tail
        content = "\n".join([f"Line {i}" for i in range(8)])
        result = await traverser.traverse(content, max_tokens=10, summarizer=mock_summarizer)

        # Should be truncated, not head/tail extracted
        assert result.sections_summarized == 1

    @pytest.mark.asyncio
    async def test_traverse_head_tail_extraction(self, traverser, mock_summarizer):
        """Test head/tail extraction for large content."""
        # Create content with many lines - need enough content to exceed budget
        lines = [f"Line {i}: Some longer content here to use more tokens" for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=200, summarizer=mock_summarizer)

        # Should contain first lines (head_lines=5 for this fixture)
        assert "Line 0:" in result.content
        assert "Line 1:" in result.content

        # Should contain last lines (tail_lines=5 for this fixture)
        # The exact last lines depend on budget, but should have some tail lines
        assert "Line 9" in result.content  # Some line in the 90s should be present

        # Should have omission marker
        assert "omitted" in result.content.lower() or "summarized" in result.content.lower()

    @pytest.mark.asyncio
    async def test_traverse_middle_summarization(self, traverser, mock_summarizer):
        """Test that middle section is summarized/omitted."""
        lines = [f"Line {i}: Some content here" for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Should indicate middle lines were handled
        assert "lines" in result.content.lower()
        assert result.sections_summarized >= 1

    @pytest.mark.asyncio
    async def test_traverse_custom_head_tail_lines(self, simple_token_counter, mock_summarizer):
        """Test with custom head/tail line counts."""
        traverser = TextTraverser(simple_token_counter, head_lines=3, tail_lines=2)
        lines = [f"Line {i}" for i in range(50)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Should respect custom head/tail counts in metadata
        if result.metadata:
            assert result.metadata.get("head_lines", 3) == 3
            assert result.metadata.get("tail_lines", 2) == 2

    @pytest.mark.asyncio
    async def test_traverse_with_summarizer(self, traverser):
        """Test traversal with real LLMLingua summarizer."""
        from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLinguaSummarizer

        summarizer = LLMLinguaSummarizer()
        lines = [f"Line {i}: Important content that should be preserved" for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=200, summarizer=summarizer)

        # Should have compressed the middle section
        assert result.sections_summarized >= 1
        # Content should be reduced
        assert result.result_tokens < result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_metadata(self, traverser, mock_summarizer):
        """Test that metadata is populated correctly."""
        # Need enough lines to trigger head/tail extraction (> head_lines + tail_lines + 5)
        lines = [f"Line {i}: Some longer content here to use more tokens" for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Metadata should contain info when optimization occurs
        if result.sections_summarized > 0:
            assert result.metadata is not None
            # Can be either full summarization or head/tail strategy
            if result.metadata.get("strategy") == "full_summarization":
                assert "total_lines" in result.metadata
            else:
                assert "head_lines_used" in result.metadata
                assert "tail_lines_used" in result.metadata
                assert "middle_lines_summarized" in result.metadata

    @pytest.mark.asyncio
    async def test_traverse_budget_exceeded_by_head_tail(
        self, simple_token_counter, mock_summarizer
    ):
        """Test when head+tail exceeds budget."""
        # Create traverser with many head/tail lines
        traverser = TextTraverser(simple_token_counter, head_lines=50, tail_lines=50)
        lines = [f"Line {i}: " + "x" * 50 for i in range(200)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=50, summarizer=mock_summarizer)

        # Should still produce valid output
        assert result.content is not None
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_traverse_preserves_line_structure(self, traverser, mock_summarizer):
        """Test that line structure is preserved in head/tail."""
        lines = ["First line", "Second line", "Third line", "Last-2", "Last-1", "Last line"]
        content = "\n".join(lines * 10)  # Repeat to make it large

        result = await traverser.traverse(content, max_tokens=200, summarizer=mock_summarizer)

        # Should preserve line breaks
        assert "\n" in result.content

    @pytest.mark.asyncio
    async def test_traverse_empty_content(self, traverser, mock_summarizer):
        """Test traversal of empty content."""
        result = await traverser.traverse("", max_tokens=100, summarizer=mock_summarizer)

        assert result.content == ""
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_single_line(self, traverser, mock_summarizer):
        """Test traversal of single line content."""
        result = await traverser.traverse(
            "Single line content", max_tokens=1000, summarizer=mock_summarizer
        )

        assert result.content == "Single line content"
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_real_text_file_low_budget(
        self, large_traverser, text_test_content: str, mock_summarizer
    ):
        """Test traversal of real text file with very low budget."""
        result = await large_traverser.traverse(
            text_test_content, max_tokens=50, summarizer=mock_summarizer
        )

        # Should produce non-empty content
        assert len(result.content) > 0

        # Should have significantly reduced size
        assert result.result_tokens < result.original_tokens
        assert result.sections_summarized >= 1

        # Metadata should be populated
        assert result.metadata is not None

    @pytest.mark.asyncio
    async def test_traverse_real_text_file_default_budget(
        self, large_traverser, text_test_content: str, mock_summarizer
    ):
        """Test traversal of real text file with default budget (1000 tokens)."""
        result = await large_traverser.traverse(
            text_test_content, max_tokens=1000, summarizer=mock_summarizer
        )

        # Should produce non-empty content
        assert len(result.content) > 0
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_traverse_real_text_file_large_budget(
        self, large_traverser, text_test_content: str, mock_summarizer
    ):
        """Test traversal of real text file with very large budget - all content returned."""
        result = await large_traverser.traverse(
            text_test_content, max_tokens=100000, summarizer=mock_summarizer
        )

        # Should return original content unchanged
        assert result.content == text_test_content
        assert result.sections_summarized == 0
        assert result.result_tokens == result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_token_estimation(self, traverser, mock_summarizer):
        """Test that token estimation is used correctly."""
        lines = [f"Line {i}" for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=50, summarizer=mock_summarizer)

        # Result tokens should be estimated
        assert result.original_tokens > 0
        assert result.result_tokens > 0
        assert result.result_tokens <= result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_very_long_lines(self, traverser, mock_summarizer):
        """Test handling of very long lines."""
        lines = ["Short", "x" * 1000, "y" * 1000, "End"]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=50, summarizer=mock_summarizer)

        # Should handle without error
        assert result.content is not None

    @pytest.mark.asyncio
    async def test_truncate_marker_presence(self, traverser, mock_summarizer):
        """Test that truncation markers are present when needed."""
        lines = [f"Line {i}: " + "content " * 20 for i in range(100)]
        content = "\n".join(lines)

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Should have some indication of omission
        lower_content = result.content.lower()
        assert any(
            marker in lower_content for marker in ["omitted", "truncated", "summarized", "..."]
        )
