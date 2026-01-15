"""JSON traverser with breadth-first structure-aware compression."""

import json
from typing import Any

from mcp_optimizer.response_optimizer.models import TraversalResult
from mcp_optimizer.response_optimizer.traversers.base import BaseTraverser, Summarizer


class JsonTraverser(BaseTraverser):
    """
    JSON traverser using breadth-first expansion.

    Algorithm:
    1. Start with skeleton: all top-level keys with type indicators
    2. Expand arrays/objects level by level while budget permits
    3. For arrays: include first N elements, summarize rest
    4. For nested objects: preserve keys, summarize values exceeding budget
    """

    async def traverse(
        self,
        content: str,
        max_tokens: int,
        summarizer: Summarizer | None = None,
    ) -> TraversalResult:
        """Traverse JSON content using breadth-first expansion."""
        original_tokens = self.estimate_tokens(content)

        # If already within budget, return as-is
        if original_tokens <= max_tokens:
            return TraversalResult(
                content=content,
                original_tokens=original_tokens,
                result_tokens=original_tokens,
                sections_summarized=0,
            )

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Not valid JSON, return with placeholder
            return TraversalResult(
                content=self._create_summary_placeholder("Invalid JSON content"),
                original_tokens=original_tokens,
                result_tokens=self.estimate_tokens("[SUMMARIZED: Invalid JSON content]"),
                sections_summarized=1,
            )

        # Traverse and compress
        result, sections_summarized = await self._traverse_value(
            data, max_tokens, summarizer, depth=0
        )

        result_content = json.dumps(result, indent=2, ensure_ascii=False)
        result_tokens = self.estimate_tokens(result_content)

        return TraversalResult(
            content=result_content,
            original_tokens=original_tokens,
            result_tokens=result_tokens,
            sections_summarized=sections_summarized,
        )

    async def _traverse_value(
        self,
        value: Any,
        budget: int,
        summarizer: Summarizer | None,
        depth: int,
    ) -> tuple[Any, int]:
        """
        Recursively traverse a JSON value with budget constraints.

        Returns:
            Tuple of (processed_value, sections_summarized)
        """
        if isinstance(value, dict):
            return await self._traverse_dict(value, budget, summarizer, depth)
        elif isinstance(value, list):
            return await self._traverse_list(value, budget, summarizer, depth)
        elif isinstance(value, str):
            # Check if string is too long
            value_tokens = self.estimate_tokens(value)
            if value_tokens > budget and summarizer:
                summarized = await summarizer.summarize(value, budget)
                return summarized, 1
            elif value_tokens > budget:
                # Truncate string
                return self._truncate_string(value, budget), 1
            return value, 0
        else:
            # Primitive types (int, float, bool, None)
            return value, 0

    async def _traverse_dict(
        self,
        obj: dict,
        budget: int,
        summarizer: Summarizer | None,
        depth: int,
    ) -> tuple[dict, int]:
        """Traverse a dictionary with breadth-first expansion."""
        sections_summarized = 0

        # First pass: create skeleton with type indicators
        skeleton: dict[str, Any] = {}
        for key, value in obj.items():
            skeleton[key] = self._get_type_indicator(value)

        skeleton_json = json.dumps(skeleton, indent=2)
        skeleton_tokens = self.estimate_tokens(skeleton_json)

        if skeleton_tokens >= budget:
            # Even skeleton doesn't fit, need to summarize keys
            summary = self._summarize_dict_structure(obj)
            return {self._create_summary_placeholder(summary): None}, 1

        # Budget for expanding values
        remaining_budget = budget - skeleton_tokens
        result: dict[str, Any] = {}

        for key, value in obj.items():
            # Estimate budget per key
            key_budget = remaining_budget // max(len(obj), 1)

            processed, summarized = await self._traverse_value(
                value, key_budget, summarizer, depth + 1
            )
            result[key] = processed
            sections_summarized += summarized

            # Update remaining budget
            result_json = json.dumps(result, indent=2)
            used_tokens = self.estimate_tokens(result_json)
            remaining_budget = budget - used_tokens

            if remaining_budget <= 0:
                # Out of budget, summarize remaining keys
                remaining_keys = list(obj.keys())[len(result) :]
                if remaining_keys:
                    result[self._create_summary_placeholder(f"{len(remaining_keys)} more keys")] = (
                        None
                    )
                    sections_summarized += 1
                break

        return result, sections_summarized

    async def _traverse_list(
        self,
        arr: list,
        budget: int,
        summarizer: Summarizer | None,
        depth: int,
    ) -> tuple[list, int]:
        """Traverse a list with breadth-first expansion."""
        if not arr:
            return [], 0

        sections_summarized = 0
        result: list[Any] = []

        # Calculate how many items we can include
        item_budget = budget // max(len(arr), 1)
        min_items = min(3, len(arr))  # Always try to include at least 3 items

        for i, item in enumerate(arr):
            processed, summarized = await self._traverse_value(
                item, item_budget, summarizer, depth + 1
            )
            result.append(processed)
            sections_summarized += summarized

            # Check budget
            result_json = json.dumps(result, indent=2)
            used_tokens = self.estimate_tokens(result_json)

            if used_tokens >= budget and i >= min_items - 1:
                # Out of budget, add placeholder for remaining items
                remaining = len(arr) - len(result)
                if remaining > 0:
                    sample = self._get_sample_description(arr[i]) if arr else None
                    result.append(self._create_array_placeholder(remaining, sample))
                    sections_summarized += 1
                break

        return result, sections_summarized

    def _get_type_indicator(self, value: Any) -> Any:
        """Get a type indicator for a value (string, bool, number, or None)."""
        if isinstance(value, dict):
            keys = list(value.keys())[:3]
            key_preview = ", ".join(keys)
            if len(value) > 3:
                key_preview += f", ... ({len(value)} keys total)"
            return f"[Object: {{{key_preview}}}]"
        elif isinstance(value, list):
            return f"[Array({len(value)} items)]"
        elif isinstance(value, str):
            if len(value) > 50:
                return f"[String: {value[:50]}...]"
            return value
        elif isinstance(value, bool):
            return value
        elif isinstance(value, (int, float)):
            return value
        elif value is None:
            return None
        else:
            return f"[{type(value).__name__}]"

    def _summarize_dict_structure(self, obj: dict) -> str:
        """Create a structural summary of a dictionary."""
        keys = list(obj.keys())
        if len(keys) <= 5:
            return f"Object with keys: {', '.join(keys)}"
        return f"Object with {len(keys)} keys: {', '.join(keys[:5])}, ..."

    def _get_sample_description(self, item: Any) -> str | None:
        """Get a sample description of a list item."""
        if isinstance(item, dict):
            keys = list(item.keys())[:3]
            return f"{{{', '.join(keys)}}}"
        elif isinstance(item, str):
            return f'"{item[:30]}..."' if len(item) > 30 else f'"{item}"'
        elif isinstance(item, (int, float, bool)):
            return str(item)
        return None

    def _truncate_string(self, s: str, max_tokens: int) -> str:
        """Truncate a string to fit within token budget."""
        # Rough estimate: 4 chars per token
        max_chars = max_tokens * 4
        if len(s) <= max_chars:
            return s
        return s[: max_chars - 20] + "... [TRUNCATED]"
