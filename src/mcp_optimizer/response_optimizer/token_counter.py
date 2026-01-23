"""Token counting utility using tiktoken for LLM-compatible tokenization."""

import structlog
import tiktoken
from mcp.types import Tool as McpTool

logger = structlog.get_logger(__name__)


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a simple character-based estimation: roughly 4 characters per token.
    This is an approximation that works well for English text.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


class TokenCounter:
    """Token counting utility using tiktoken.

    This class provides methods to count tokens in text and serialized MCP tools
    using tiktoken, which matches the tokenization used by OpenAI's LLM models.

    If the specified encoding is not available, falls back to character-based
    estimation (approximately 4 characters per token).
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter with specified encoding.

        Args:
            encoding_name: tiktoken encoding to use (default: cl100k_base)
        """
        self.encoding_name = encoding_name
        self._encoding: tiktoken.Encoding | None = None
        self._use_estimation = False
        self._loaded = False

    def _load_encoding(self) -> None:
        """Load the tiktoken encoding, falling back to estimation if unavailable."""
        if self._loaded:
            return

        try:
            self._encoding = tiktoken.get_encoding(self.encoding_name)
            self._use_estimation = False
            logger.debug(
                "Loaded tiktoken encoding",
                encoding_name=self.encoding_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to load tiktoken encoding, using estimation fallback",
                encoding_name=self.encoding_name,
                error=str(e),
            )
            self._encoding = None
            self._use_estimation = True

        self._loaded = True

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in given text.

        Uses tiktoken encoding if available, otherwise falls back to
        character-based estimation (approximately 4 characters per token).

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens (or estimated count if tiktoken unavailable)
        """
        self._load_encoding()

        if self._use_estimation or self._encoding is None:
            return estimate_tokens(text)

        return len(self._encoding.encode(text))

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

    def is_using_estimation(self) -> bool:
        """
        Check if the counter is using estimation fallback.

        Returns:
            True if using character-based estimation, False if using tiktoken
        """
        self._load_encoding()
        return self._use_estimation
