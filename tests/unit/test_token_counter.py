"""Tests for TokenCounter utility."""

from mcp.types import Tool as McpTool

from mcp_optimizer.response_optimizer.token_counter import TokenCounter, estimate_tokens


class TestEstimateTokens:
    """Test estimate_tokens function."""

    def test_estimate_tokens_empty(self):
        """Test estimating tokens for empty string."""
        assert estimate_tokens("") == 0

    def test_estimate_tokens_short(self):
        """Test estimating tokens for short text."""
        # 12 characters / 4 = 3 tokens
        assert estimate_tokens("Hello world!") == 3

    def test_estimate_tokens_long(self):
        """Test estimating tokens for longer text."""
        text = "A" * 100
        assert estimate_tokens(text) == 25  # 100 / 4


class TestTokenCounter:
    """Test TokenCounter class."""

    def test_initialization(self):
        """Test TokenCounter initializes with default encoding."""
        counter = TokenCounter(encoding_name="cl100k_base")
        # Lazy loading - encoding not loaded until first use
        assert counter.encoding_name == "cl100k_base"
        assert not counter._loaded

    def test_initialization_with_custom_encoding(self):
        """Test TokenCounter initializes with custom encoding."""
        counter = TokenCounter(encoding_name="p50k_base")
        assert counter.encoding_name == "p50k_base"

    def test_count_tokens_simple_text(self):
        """Test counting tokens for simple text."""
        counter = TokenCounter(encoding_name="cl100k_base")
        token_count = counter.count_tokens("Hello, world!")
        assert token_count > 0
        assert isinstance(token_count, int)
        # "Hello, world!" with cl100k_base should be around 4 tokens
        assert 3 <= token_count <= 5

    def test_count_tokens_empty_string(self):
        """Test counting tokens for empty string."""
        counter = TokenCounter(encoding_name="cl100k_base")
        token_count = counter.count_tokens("")
        assert token_count == 0

    def test_count_tokens_long_text(self):
        """Test counting tokens for longer text."""
        counter = TokenCounter(encoding_name="cl100k_base")
        text = "This is a longer piece of text with multiple words and sentences."
        token_count = counter.count_tokens(text)
        assert token_count > 10
        assert isinstance(token_count, int)

    def test_count_tool_tokens_minimal_tool(self):
        """Test counting tokens for minimal MCP tool."""
        counter = TokenCounter(encoding_name="cl100k_base")
        tool = McpTool(name="test_tool", inputSchema={})
        token_count = counter.count_tool_tokens(tool)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_count_tool_tokens_full_tool(self):
        """Test counting tokens for complete MCP tool."""
        counter = TokenCounter(encoding_name="cl100k_base")
        tool = McpTool(
            name="test_tool",
            description="A test tool for counting tokens",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "integer", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        )
        token_count = counter.count_tool_tokens(tool)
        assert token_count > 0
        assert isinstance(token_count, int)
        # Full tool should have more tokens than minimal tool
        assert token_count > 10

    def test_count_tool_tokens_consistency(self):
        """Test that counting the same tool multiple times gives consistent results."""
        counter = TokenCounter(encoding_name="cl100k_base")
        tool = McpTool(name="consistent_tool", description="Test consistency", inputSchema={})
        count1 = counter.count_tool_tokens(tool)
        count2 = counter.count_tool_tokens(tool)
        assert count1 == count2

    def test_fallback_to_estimation_on_invalid_encoding(self):
        """Test that counter falls back to estimation with invalid encoding."""
        counter = TokenCounter(encoding_name="invalid_encoding_that_does_not_exist")
        # Should not raise, just use estimation fallback
        token_count = counter.count_tokens("Hello world")
        assert token_count > 0
        assert counter.is_using_estimation()

    def test_is_using_estimation_false_with_valid_encoding(self):
        """Test that is_using_estimation returns False with valid encoding."""
        counter = TokenCounter(encoding_name="cl100k_base")
        # Trigger loading
        counter.count_tokens("test")
        assert not counter.is_using_estimation()

    def test_lazy_loading(self):
        """Test that encoding is lazily loaded."""
        counter = TokenCounter(encoding_name="cl100k_base")
        # Before first use
        assert not counter._loaded
        assert counter._encoding is None

        # After first use
        counter.count_tokens("test")
        assert counter._loaded
        assert counter._encoding is not None
