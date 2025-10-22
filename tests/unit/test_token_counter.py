"""Tests for TokenCounter utility."""

from mcp.types import Tool as McpTool

from mcp_optimizer.token_counter import TokenCounter


class TestTokenCounter:
    """Test TokenCounter class."""

    def test_initialization(self):
        """Test TokenCounter initializes with default encoding."""
        counter = TokenCounter(encoding_name="cl100k_base")
        assert counter.encoding is not None
        assert counter.encoding.name == "cl100k_base"

    def test_initialization_with_custom_encoding(self):
        """Test TokenCounter initializes with custom encoding."""
        counter = TokenCounter(encoding_name="p50k_base")
        assert counter.encoding is not None
        assert counter.encoding.name == "p50k_base"

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
