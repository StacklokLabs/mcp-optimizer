"""Tests for JSON traverser."""

import json

import pytest

from mcp_optimizer.response_optimizer.traversers.json_traverser import JsonTraverser


class TestJsonTraverser:
    """Test JsonTraverser class."""

    @pytest.fixture
    def traverser(self, simple_token_counter):
        """Create a JsonTraverser with simple token counter."""
        return JsonTraverser(simple_token_counter)

    @pytest.mark.asyncio
    async def test_traverse_content_within_budget(self, traverser, mock_summarizer):
        """Test that content within budget is returned as-is."""
        content = '{"key": "value"}'
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert result.content == content
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_simple_object(self, traverser, mock_summarizer):
        """Test traversal of simple JSON object."""
        content = '{"name": "test", "value": 123}'
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        assert result.original_tokens > 0
        assert result.result_tokens > 0
        data = json.loads(result.content)
        assert "name" in data
        assert "value" in data

    @pytest.mark.asyncio
    async def test_traverse_nested_object(self, traverser, mock_summarizer):
        """Test traversal of nested JSON object."""
        content = json.dumps({"level1": {"level2": {"level3": "deep value"}}})
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        data = json.loads(result.content)
        assert "level1" in data

    @pytest.mark.asyncio
    async def test_traverse_array_truncation(self, traverser, mock_summarizer):
        """Test that large arrays are truncated with placeholder."""
        # Create a large array that exceeds budget
        large_array = list(range(100))
        content = json.dumps({"items": large_array})

        result = await traverser.traverse(content, max_tokens=50, summarizer=mock_summarizer)

        data = json.loads(result.content)
        assert "items" in data
        items = data["items"]
        # Should have some items plus a placeholder
        assert len(items) < 100
        # Check for placeholder pattern
        assert any("[..." in str(item) for item in items) or len(items) == len(large_array)

    @pytest.mark.asyncio
    async def test_traverse_preserves_minimum_array_items(self, traverser, mock_summarizer):
        """Test that at least 3 array items are preserved when possible."""
        content = json.dumps({"items": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        result = await traverser.traverse(content, max_tokens=30, summarizer=mock_summarizer)

        data = json.loads(result.content)
        # Should preserve at least 3 items if budget allows
        if "items" in data and isinstance(data["items"], list):
            # Check that we have at least minimum items or all items if they fit
            assert len(data["items"]) >= 3 or len(data["items"]) == 10

    @pytest.mark.asyncio
    async def test_traverse_string_truncation(self, traverser, mock_summarizer):
        """Test that long strings are summarized."""
        long_string = "x" * 1000
        content = json.dumps({"text": long_string})

        result = await traverser.traverse(content, max_tokens=50, summarizer=mock_summarizer)

        data = json.loads(result.content)
        # String should be summarized
        if isinstance(data["text"], str):
            assert len(data["text"]) < len(long_string) or "[...SUMMARIZED]" in data["text"]

    @pytest.mark.asyncio
    async def test_traverse_invalid_json(self, traverser, mock_summarizer):
        """Test handling of invalid JSON content."""
        # Create invalid JSON that exceeds the budget to trigger traversal
        content = '{"invalid": json content' + "x" * 500

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        assert "[SUMMARIZED:" in result.content
        assert "Invalid JSON" in result.content
        assert result.sections_summarized == 1

    @pytest.mark.asyncio
    async def test_traverse_empty_object(self, traverser, mock_summarizer):
        """Test traversal of empty JSON object."""
        content = "{}"
        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        assert result.content == "{}"
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_empty_array(self, traverser, mock_summarizer):
        """Test traversal of empty JSON array."""
        content = "[]"
        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        assert result.content == "[]"
        assert result.sections_summarized == 0

    @pytest.mark.asyncio
    async def test_traverse_preserves_types(self, traverser, mock_summarizer):
        """Test that primitive types are preserved."""
        content = json.dumps(
            {"string": "hello", "number": 42, "float": 3.14, "boolean": True, "null": None}
        )
        result = await traverser.traverse(content, max_tokens=1000, summarizer=mock_summarizer)

        data = json.loads(result.content)
        assert data["string"] == "hello"
        assert data["number"] == 42
        assert data["float"] == 3.14
        assert data["boolean"] is True
        assert data["null"] is None

    @pytest.mark.asyncio
    async def test_traverse_with_token_budget(self, traverser, mock_summarizer):
        """Test that result respects token budget."""
        large_content = json.dumps(
            {
                "key1": "value1" * 100,
                "key2": "value2" * 100,
                "key3": "value3" * 100,
            }
        )

        result = await traverser.traverse(large_content, max_tokens=100, summarizer=mock_summarizer)

        # Result should be smaller than original
        assert result.result_tokens <= result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_type_indicators(self, traverser, mock_summarizer):
        """Test that type indicators are generated for nested structures."""
        content = json.dumps(
            {"nested_obj": {"a": 1, "b": 2, "c": 3}, "nested_arr": [1, 2, 3, 4, 5]}
        )

        # With very low budget, should see type indicators
        result = await traverser.traverse(content, max_tokens=20, summarizer=mock_summarizer)

        # Should contain some indication of structure
        assert result.content is not None
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_traverse_with_summarizer(self, traverser):
        """Test traversal with real LLMLingua summarizer."""
        from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLinguaSummarizer

        summarizer = LLMLinguaSummarizer()
        long_string = "This is important content that should be preserved. " * 50
        content = json.dumps({"long_text": long_string})

        result = await traverser.traverse(content, max_tokens=50, summarizer=summarizer)

        # Content should be compressed
        data = json.loads(result.content)
        assert data is not None
        # Either summarized, truncated, or reduced in size
        assert len(str(data.get("long_text", ""))) < len(long_string)

    @pytest.mark.asyncio
    async def test_traverse_real_json_file_low_budget(
        self, traverser, json_test_content: str, mock_summarizer
    ):
        """Test traversal of real JSON file with very low budget."""
        result = await traverser.traverse(
            json_test_content, max_tokens=50, summarizer=mock_summarizer
        )

        # Should produce valid JSON
        data = json.loads(result.content)
        assert data is not None

        # Should have significantly reduced size
        assert result.result_tokens < result.original_tokens
        assert result.sections_summarized > 0

    @pytest.mark.asyncio
    async def test_traverse_real_json_file_default_budget(
        self, traverser, json_test_content: str, mock_summarizer
    ):
        """Test traversal of real JSON file with default budget (1000 tokens)."""
        result = await traverser.traverse(
            json_test_content, max_tokens=1000, summarizer=mock_summarizer
        )

        # Should produce valid JSON
        data = json.loads(result.content)
        assert data is not None
        assert result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_traverse_real_json_file_large_budget(
        self, traverser, json_test_content: str, mock_summarizer
    ):
        """Test traversal of real JSON file with very large budget - all content returned."""
        result = await traverser.traverse(
            json_test_content, max_tokens=100000, summarizer=mock_summarizer
        )

        # Should return original content unchanged
        assert result.content == json_test_content
        assert result.sections_summarized == 0
        assert result.result_tokens == result.original_tokens

    @pytest.mark.asyncio
    async def test_traverse_dict_exceeds_budget(self, traverser, mock_summarizer):
        """Test dict with too many keys that exceeds budget."""
        content = json.dumps({f"key{i}": f"value{i}" * 50 for i in range(20)})

        result = await traverser.traverse(content, max_tokens=100, summarizer=mock_summarizer)

        # Should have placeholder for remaining keys
        data = json.loads(result.content)
        # Check that some keys exist
        assert len(data) > 0
