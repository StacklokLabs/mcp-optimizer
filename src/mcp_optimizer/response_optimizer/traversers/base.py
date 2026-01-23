"""Base traverser interface for structure-aware content compression."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Protocol

from mcp_optimizer.response_optimizer.models import TraversalResult


class Summarizer(Protocol):
    """Protocol for summarizers used during traversal."""

    async def summarize(self, text: str, target_tokens: int) -> str:
        """Summarize text to target token count."""
        ...


class BaseTraverser(ABC):
    """
    Base class for content traversers.

    Traversers implement structure-aware compression using breadth-first traversal.
    They preserve the structural context (keys, headers, etc.) while summarizing
    nested content that exceeds the token budget.
    """

    def __init__(self, token_estimator: Callable[[str], int]):
        """
        Initialize the traverser.

        Args:
            token_estimator: Function that estimates token count for text
        """
        self.estimate_tokens = token_estimator

    @abstractmethod
    async def traverse(
        self,
        content: str,
        max_tokens: int,
        summarizer: Summarizer,
    ) -> TraversalResult:
        """
        Traverse and compress content to fit within token budget.

        Args:
            content: The content to traverse
            max_tokens: Maximum tokens for the result
            summarizer: Summarizer for compressing sections

        Returns:
            TraversalResult with compressed content and metadata
        """
        pass

    def _create_summary_placeholder(self, description: str) -> str:
        """Create a placeholder for summarized content."""
        return f"[SUMMARIZED: {description}]"

    def _create_array_placeholder(self, count: int, sample: str | None = None) -> str:
        """Create a placeholder for summarized array content."""
        if sample:
            return f"[...{count} more items similar to: {sample}]"
        return f"[...{count} more items]"
