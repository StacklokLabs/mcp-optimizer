"""Simple truncation-based summarizer for token limiting."""

from mcp_optimizer.response_optimizer.summarizers.base import BaseSummarizer
from mcp_optimizer.response_optimizer.token_counter import estimate_tokens


class TruncationSummarizer(BaseSummarizer):
    """
    Simple summarizer that truncates text to fit within token budget.

    This summarizer preserves the beginning of the text and truncates
    the rest when content exceeds the target token count. It adds a
    truncation marker to indicate content was removed.

    Use this as a fallback when more sophisticated summarization
    (like LLMLingua) is not available.
    """

    def __init__(self, chars_per_token: int = 4):
        """
        Initialize the truncation summarizer.

        Args:
            chars_per_token: Approximate characters per token for estimation
        """
        self.chars_per_token = chars_per_token

    async def summarize(self, text: str, target_tokens: int) -> str:
        """
        Truncate text to approximately fit within target token count.

        Preserves the beginning of the text and adds a truncation marker
        when content is removed.

        Args:
            text: The text to truncate
            target_tokens: Target maximum token count for the result

        Returns:
            Truncated text with marker if truncation occurred
        """
        current_tokens = estimate_tokens(text)

        # If already within budget, return as-is
        if current_tokens <= target_tokens:
            return text

        # Calculate approximate character limit
        # Reserve space for truncation marker
        marker = "\n\n[...TRUNCATED...]"
        marker_chars = len(marker)
        max_chars = (target_tokens * self.chars_per_token) - marker_chars

        if max_chars <= 0:
            # Edge case: target is so small we can only show the marker
            return marker.strip()

        # Truncate at character boundary, try to end at newline for cleaner output
        truncated = text[:max_chars]

        # Try to find a good breaking point (newline)
        last_newline = truncated.rfind("\n")
        if last_newline > max_chars // 2:
            # Found a newline in the second half, use it
            truncated = truncated[:last_newline]

        return truncated + marker

    def is_available(self) -> bool:
        """
        Check if the summarizer is available.

        The truncation summarizer is always available as it has no
        external dependencies.

        Returns:
            True (always available)
        """
        return True
