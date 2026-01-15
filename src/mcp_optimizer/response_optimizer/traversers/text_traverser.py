"""Text traverser for unstructured content using head/tail extraction."""

from collections.abc import Callable

from mcp_optimizer.response_optimizer.models import TraversalResult
from mcp_optimizer.response_optimizer.traversers.base import BaseTraverser, Summarizer


class TextTraverser(BaseTraverser):
    """
    Text traverser using head/tail extraction.

    Algorithm:
    1. Extract first N lines (default: 20)
    2. Extract last M lines (default: 20)
    3. Summarize middle section to fit remaining budget
    4. Return: [head] + [SUMMARIZED: middle] + [tail]

    Rationale:
    - Beginning often contains: command output headers, initial status, setup info
    - End often contains: final results, error messages, exit codes, summaries
    - Middle typically contains: verbose logs, repeated patterns, incremental progress
    """

    def __init__(
        self,
        token_estimator: Callable[[str], int],
        head_lines: int = 20,
        tail_lines: int = 20,
    ):
        """
        Initialize the text traverser.

        Args:
            token_estimator: Function that estimates token count for text
            head_lines: Number of lines to preserve from start
            tail_lines: Number of lines to preserve from end
        """
        super().__init__(token_estimator)
        self.head_lines = head_lines
        self.tail_lines = tail_lines

    async def traverse(
        self,
        content: str,
        max_tokens: int,
        summarizer: Summarizer | None = None,
    ) -> TraversalResult:
        """Traverse unstructured text using head/tail extraction."""
        original_tokens = self.estimate_tokens(content)

        # If already within budget, return as-is
        if original_tokens <= max_tokens:
            return TraversalResult(
                content=content,
                original_tokens=original_tokens,
                result_tokens=original_tokens,
                sections_summarized=0,
            )

        lines = content.split("\n")
        total_lines = len(lines)

        # If content is small enough, just truncate
        if total_lines <= self.head_lines + self.tail_lines + 5:
            # Not enough lines to do head/tail extraction
            truncated = self._simple_truncate(content, max_tokens)
            return TraversalResult(
                content=truncated,
                original_tokens=original_tokens,
                result_tokens=self.estimate_tokens(truncated),
                sections_summarized=1,
            )

        # Extract head and tail
        head = "\n".join(lines[: self.head_lines])
        tail = "\n".join(lines[-self.tail_lines :])
        middle = "\n".join(lines[self.head_lines : -self.tail_lines])

        head_tokens = self.estimate_tokens(head)
        tail_tokens = self.estimate_tokens(tail)
        middle_tokens = self.estimate_tokens(middle)
        middle_lines = total_lines - self.head_lines - self.tail_lines

        # Calculate budget for middle summary
        overhead_tokens = 50  # For markers and formatting
        remaining_budget = max_tokens - head_tokens - tail_tokens - overhead_tokens

        sections_summarized = 0

        if remaining_budget <= 0:
            # Head + tail already exceeds budget, need to trim them
            half_budget = (max_tokens - overhead_tokens) // 2
            head = self._truncate_to_tokens(head, half_budget)
            tail = self._truncate_to_tokens(tail, half_budget)
            middle_summary = f"[...{middle_lines} lines omitted...]"
            sections_summarized = 1
        elif summarizer and remaining_budget >= 50:
            # Have budget for summary
            middle_summary = await summarizer.summarize(middle, remaining_budget)
            middle_summary = f"[...{middle_lines} lines summarized:]\n{middle_summary}"
            sections_summarized = 1
        else:
            # No summarizer or not enough budget, just indicate omission
            middle_summary = f"[...{middle_lines} lines omitted ({middle_tokens} tokens)...]"
            sections_summarized = 1

        # Build result
        result = f"{head}\n\n{middle_summary}\n\n{tail}"
        result_tokens = self.estimate_tokens(result)

        return TraversalResult(
            content=result,
            original_tokens=original_tokens,
            result_tokens=result_tokens,
            sections_summarized=sections_summarized,
            metadata={
                "head_lines": self.head_lines,
                "tail_lines": self.tail_lines,
                "middle_lines_omitted": middle_lines,
            },
        )

    def _simple_truncate(self, content: str, max_tokens: int) -> str:
        """Simple truncation for small content."""
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content

        # Keep beginning with truncation marker
        truncated = content[: max_chars - 30]
        return truncated + "\n\n[...TRUNCATED...]"

    def _truncate_to_tokens(self, content: str, max_tokens: int) -> str:
        """Truncate content to fit within token budget."""
        current_tokens = self.estimate_tokens(content)
        if current_tokens <= max_tokens:
            return content

        # Binary search for the right length
        lines = content.split("\n")
        low, high = 0, len(lines)

        while low < high:
            mid = (low + high + 1) // 2
            test_content = "\n".join(lines[:mid])
            if self.estimate_tokens(test_content) <= max_tokens:
                low = mid
            else:
                high = mid - 1

        return "\n".join(lines[:low])
