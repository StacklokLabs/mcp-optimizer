"""Token counting utility using tiktoken for LLM-compatible tokenization."""

import tiktoken
from mcp.types import Tool as McpTool


class TokenCounter:
    """Token counting utility using tiktoken.

    This class provides methods to count tokens in text and serialized MCP tools
    using tiktoken, which matches the tokenization used by OpenAI's LLM models.
    """

    def __init__(self, encoding_name: str):
        """
        Initialize token counter with specified encoding.

        Args:
            encoding_name: tiktoken encoding to use
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in given text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def count_tool_tokens(self, tool: McpTool) -> int:
        """
        Count tokens in serialized MCP tool.

        Args:
            tool: MCP Tool to count tokens for

        Returns:
            Number of tokens in JSON serialized tool
        """
        tool_json = tool.model_dump_json()
        return self.count_tokens(tool_json)
