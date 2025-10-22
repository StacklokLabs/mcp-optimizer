"""Tests for token limiting functionality."""

import json

from mcp.types import CallToolResult, TextContent

from mcp_optimizer.token_limiter import (
    count_content_tokens,
    estimate_tokens,
    limit_tool_response,
)


class TestEstimateTokens:
    """Test token estimation."""

    def test_estimate_tokens_empty_string(self):
        """Test estimating tokens for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short_text(self):
        """Test estimating tokens for short text."""
        # "Hello world" is 11 chars, should be ~2-3 tokens
        tokens = estimate_tokens("Hello world")
        assert tokens == 2  # 11 // 4 = 2

    def test_estimate_tokens_long_text(self):
        """Test estimating tokens for longer text."""
        text = "This is a longer piece of text " * 10
        tokens = estimate_tokens(text)
        # Should be roughly 1/4 of character count
        assert tokens == len(text) // 4


class TestCountContentTokens:
    """Test content token counting."""

    def test_count_content_tokens_empty(self):
        """Test counting tokens in empty content."""
        assert count_content_tokens([]) == 0

    def test_count_content_tokens_text_only(self):
        """Test counting tokens in text content."""
        content = [
            TextContent(type="text", text="Hello world"),
            TextContent(type="text", text="More text here"),
        ]
        tokens = count_content_tokens(content)
        # Should be roughly (11 + 14) / 4 = 6
        assert tokens > 0
        assert tokens == estimate_tokens("Hello world") + estimate_tokens("More text here")


class TestLimitToolResponse:
    """Test complete tool response limiting."""

    def test_limit_tool_response_under_limit(self):
        """Test response under token limit is not modified."""
        result = CallToolResult(
            content=[TextContent(type="text", text="Short response")], isError=False
        )

        limited = limit_tool_response(result, max_tokens=1000)

        assert not limited.was_truncated
        assert limited.result.content == result.content
        assert limited.truncation_message is None

    def test_limit_tool_response_text_truncation(self):
        """Test that long text responses cause content items to be omitted."""
        long_text = "A" * 10000
        result = CallToolResult(content=[TextContent(type="text", text=long_text)], isError=False)

        limited = limit_tool_response(result, max_tokens=100)

        assert limited.was_truncated
        assert limited.original_tokens > 100
        assert limited.final_tokens <= 100
        assert limited.truncation_message is not None
        assert "truncated" in limited.truncation_message.lower()

        # The large text item should be omitted entirely
        assert len(limited.result.content) == 0

    def test_limit_tool_response_json_list(self):
        """Test that large JSON responses are omitted when they exceed limit."""
        data = [{"id": i, "value": f"item_{i}" * 10} for i in range(200)]
        json_text = json.dumps(data, indent=2)

        result = CallToolResult(content=[TextContent(type="text", text=json_text)], isError=False)

        limited = limit_tool_response(result, max_tokens=500)

        assert limited.was_truncated
        assert limited.final_tokens <= 500

        # The large JSON should be omitted
        assert len(limited.result.content) == 0

    def test_limit_tool_response_multiple_content_items(self):
        """Test limiting response with multiple content items."""
        result = CallToolResult(
            content=[
                TextContent(type="text", text="First item " * 10),  # ~30 tokens
                TextContent(type="text", text="Second item " * 10),  # ~30 tokens
                TextContent(type="text", text="Third item " * 100),  # ~300 tokens
            ],
            isError=False,
        )

        limited = limit_tool_response(result, max_tokens=100)

        assert limited.was_truncated
        assert limited.final_tokens <= 100
        # Should keep first two items, omit the third
        assert len(limited.result.content) == 2

    def test_limit_tool_response_preserves_error_flag(self):
        """Test that error flag is preserved."""
        result = CallToolResult(
            content=[TextContent(type="text", text="Error message " * 1000)], isError=True
        )

        limited = limit_tool_response(result, max_tokens=100)

        assert limited.result.isError is True

    def test_limit_tool_response_very_small_limit(self):
        """Test behavior with very small token limit."""
        result = CallToolResult(
            content=[TextContent(type="text", text="Some text here")], isError=False
        )

        limited = limit_tool_response(result, max_tokens=1)

        # With a very small limit, no items will fit
        assert len(limited.result.content) == 0
        assert limited.was_truncated

    def test_limit_tool_response_exact_limit(self):
        """Test response exactly at token limit."""
        text = "word " * 100  # Roughly 100 tokens
        result = CallToolResult(content=[TextContent(type="text", text=text)], isError=False)

        tokens = estimate_tokens(text)
        limited = limit_tool_response(result, max_tokens=tokens)

        # Should not be truncated if exactly at limit
        assert not limited.was_truncated


class TestTokenLimitingEdgeCases:
    """Test edge cases in token limiting."""

    def test_empty_content_list(self):
        """Test handling empty content list."""
        result = CallToolResult(content=[], isError=False)
        limited = limit_tool_response(result, max_tokens=100)

        assert not limited.was_truncated
        assert limited.result.content == []

    def test_large_single_item(self):
        """Test single large item is omitted when it exceeds limit."""
        text = "[This is not JSON] " * 200
        result = CallToolResult(content=[TextContent(type="text", text=text)], isError=False)

        limited = limit_tool_response(result, max_tokens=100)

        assert limited.was_truncated
        # Item should be omitted entirely
        assert len(limited.result.content) == 0

    def test_json_object_too_large(self):
        """Test large JSON object is omitted."""
        data = {"key": "value " * 500}
        json_text = json.dumps(data, indent=2)

        result = CallToolResult(content=[TextContent(type="text", text=json_text)], isError=False)

        limited = limit_tool_response(result, max_tokens=50)

        assert limited.was_truncated
        # Item should be omitted
        assert len(limited.result.content) == 0
