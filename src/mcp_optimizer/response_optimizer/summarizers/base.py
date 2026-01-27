"""Base summarizer interface."""

from abc import ABC, abstractmethod


class BaseSummarizer(ABC):
    """
    Base class for text summarizers.

    Summarizers compress text while preserving key information.
    They are used during traversal to compress sections that exceed
    the token budget.
    """

    @abstractmethod
    async def summarize(self, text: str, target_tokens: int) -> str:
        """
        Summarize text to approximately fit within target token count.

        Args:
            text: The text to summarize
            target_tokens: Target maximum token count for the result

        Returns:
            Summarized text
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the summarizer is available and ready to use.

        Returns:
            True if the summarizer can be used
        """
        pass
