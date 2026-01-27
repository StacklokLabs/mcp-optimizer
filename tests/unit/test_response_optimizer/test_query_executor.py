"""Tests for query executor."""

import shutil
from unittest.mock import patch

import pytest

from mcp_optimizer.response_optimizer.models import ContentType
from mcp_optimizer.response_optimizer.query_executor import (
    QueryExecutionError,
    execute_jq_query,
    execute_query,
    execute_text_query,
    extract_markdown_section,
)


class TestExecuteJqQuery:
    """Test execute_jq_query function."""

    @pytest.fixture
    def json_content(self):
        """Sample JSON content for testing."""
        return '{"name": "test", "value": 42, "items": [1, 2, 3]}'

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_simple_key(self, json_content):
        """Test simple key extraction."""
        result = execute_jq_query(json_content, ".name")
        assert result == '"test"'

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_number_value(self, json_content):
        """Test number value extraction."""
        result = execute_jq_query(json_content, ".value")
        assert result == "42"

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_array_index(self, json_content):
        """Test array index access."""
        result = execute_jq_query(json_content, ".items[0]")
        assert result == "1"

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_array_length(self, json_content):
        """Test array length query."""
        result = execute_jq_query(json_content, ".items | length")
        assert result == "3"

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_nested(self):
        """Test nested object query."""
        content = '{"data": {"nested": {"value": "deep"}}}'
        result = execute_jq_query(content, ".data.nested.value")
        assert result == '"deep"'

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_keys(self, json_content):
        """Test keys extraction."""
        result = execute_jq_query(json_content, "keys")
        assert "name" in result
        assert "value" in result
        assert "items" in result

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_jq_query_invalid(self, json_content):
        """Test invalid jq query raises error."""
        with pytest.raises(QueryExecutionError) as exc_info:
            execute_jq_query(json_content, ".invalid[")
        assert "jq query failed" in str(exc_info.value)

    def test_jq_not_installed(self, json_content):
        """Test error when jq is not found."""
        with patch("shutil.which", return_value=None):
            with pytest.raises(QueryExecutionError) as exc_info:
                execute_jq_query(json_content, ".name")
            assert "jq command not found" in str(exc_info.value)


class TestExtractMarkdownSection:
    """Test extract_markdown_section function."""

    @pytest.fixture
    def markdown_content(self):
        """Sample Markdown content for testing."""
        return """# Introduction

This is the introduction section.

## Getting Started

Here's how to get started.

### Prerequisites

You need these things.

## Usage

How to use the tool.

## Conclusion

Final thoughts.
"""

    def test_extract_section_by_exact_header(self, markdown_content):
        """Test extraction with exact header level."""
        result = extract_markdown_section(markdown_content, "## Getting Started")

        assert "Getting Started" in result
        assert "how to get started" in result.lower()
        # Should not include next same-level section
        assert "## Usage" not in result

    def test_extract_section_any_level(self, markdown_content):
        """Test extraction without specifying header level."""
        result = extract_markdown_section(markdown_content, "Introduction")

        assert "Introduction" in result
        assert "introduction section" in result.lower()

    def test_extract_section_with_subsections(self, markdown_content):
        """Test that subsections are included."""
        result = extract_markdown_section(markdown_content, "## Getting Started")

        # Should include the H3 subsection
        assert "Prerequisites" in result
        assert "need these things" in result.lower()

    def test_extract_section_case_insensitive(self, markdown_content):
        """Test case-insensitive matching."""
        result = extract_markdown_section(markdown_content, "GETTING STARTED")

        assert "Getting Started" in result

    def test_extract_section_partial_match(self, markdown_content):
        """Test partial title matching."""
        result = extract_markdown_section(markdown_content, "Started")

        assert "Getting Started" in result

    def test_extract_section_not_found(self, markdown_content):
        """Test error when section not found."""
        with pytest.raises(QueryExecutionError) as exc_info:
            extract_markdown_section(markdown_content, "## Nonexistent Section")
        assert "not found" in str(exc_info.value)

    def test_extract_last_section(self, markdown_content):
        """Test extraction of last section (no following header)."""
        result = extract_markdown_section(markdown_content, "## Conclusion")

        assert "Conclusion" in result
        assert "Final thoughts" in result

    def test_extract_h1_section(self, markdown_content):
        """Test extraction of H1 section."""
        result = extract_markdown_section(markdown_content, "# Introduction")

        assert "Introduction" in result


class TestExecuteTextQuery:
    """Test execute_text_query function."""

    @pytest.fixture
    def text_content(self):
        """Sample text content for testing."""
        return "\n".join([f"Line {i}: Some content here" for i in range(1, 51)])

    def test_head_default(self, text_content):
        """Test head command with default count."""
        result = execute_text_query(text_content, "head")

        lines = result.split("\n")
        assert len(lines) == 10
        assert "Line 1:" in result
        assert "Line 10:" in result

    def test_head_with_count(self, text_content):
        """Test head command with specified count."""
        result = execute_text_query(text_content, "head -n 5")

        lines = result.split("\n")
        assert len(lines) == 5
        assert "Line 1:" in result
        assert "Line 5:" in result
        assert "Line 6:" not in result

    def test_head_alternate_syntax(self, text_content):
        """Test head command with alternate syntax."""
        result = execute_text_query(text_content, "head 5")

        lines = result.split("\n")
        assert len(lines) == 5

    def test_tail_default(self, text_content):
        """Test tail command with default count."""
        result = execute_text_query(text_content, "tail")

        lines = result.split("\n")
        assert len(lines) == 10
        assert "Line 50:" in result
        assert "Line 41:" in result

    def test_tail_with_count(self, text_content):
        """Test tail command with specified count."""
        result = execute_text_query(text_content, "tail -n 3")

        lines = result.split("\n")
        assert len(lines) == 3
        assert "Line 50:" in result
        assert "Line 48:" in result

    def test_lines_range(self, text_content):
        """Test lines range command."""
        result = execute_text_query(text_content, "lines 5-10")

        lines = result.split("\n")
        assert len(lines) == 6  # 5, 6, 7, 8, 9, 10
        assert "Line 5:" in result
        assert "Line 10:" in result
        assert "Line 4:" not in result
        assert "Line 11:" not in result

    def test_lines_range_alternate_syntax(self, text_content):
        """Test lines range with 'line' singular."""
        result = execute_text_query(text_content, "line 1-3")

        lines = result.split("\n")
        assert len(lines) == 3

    def test_grep_pattern(self, text_content):
        """Test grep command."""
        result = execute_text_query(text_content, "grep 'Line 1:'")

        # Should match Line 1:, Line 10:, Line 11:, etc.
        assert "Line 1:" in result

    def test_grep_case_insensitive(self, text_content):
        """Test grep with case insensitive flag."""
        result = execute_text_query(text_content, "grep -i 'LINE 1:'")

        assert "Line 1:" in result

    def test_grep_regex_pattern(self, text_content):
        """Test grep with regex pattern."""
        result = execute_text_query(text_content, "grep 'Line [12]:'")

        assert "Line 1:" in result
        assert "Line 2:" in result

    def test_grep_invalid_regex_fallback(self):
        """Test that invalid regex falls back to literal match."""
        content = "Line with [bracket] here\nAnother line"
        result = execute_text_query(content, "grep '[bracket]'")

        # Should fall back to literal match
        assert "[bracket]" in result

    def test_grep_no_matches(self, text_content):
        """Test grep with no matching lines."""
        result = execute_text_query(text_content, "grep 'nonexistent'")

        assert "No lines matching" in result

    def test_unsupported_query(self, text_content):
        """Test unsupported query command."""
        with pytest.raises(QueryExecutionError) as exc_info:
            execute_text_query(text_content, "cat")
        assert "Unsupported text query" in str(exc_info.value)


class TestExecuteQuery:
    """Test unified execute_query function."""

    @pytest.mark.skipif(shutil.which("jq") is None, reason="jq not installed")
    def test_routes_json_to_jq(self):
        """Test that JSON content routes to jq."""
        content = '{"key": "value"}'
        result = execute_query(content, ContentType.JSON, ".key")

        assert result == '"value"'

    def test_routes_markdown_to_section_extraction(self):
        """Test that Markdown content routes to section extraction."""
        content = "# Title\n\nContent here."
        result = execute_query(content, ContentType.MARKDOWN, "Title")

        assert "Title" in result
        assert "Content here" in result

    def test_routes_unstructured_to_text_query(self):
        """Test that unstructured content routes to text query."""
        content = "Line 1\nLine 2\nLine 3"
        result = execute_query(content, ContentType.UNSTRUCTURED, "head -n 2")

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" not in result


class TestQueryExecutionError:
    """Test QueryExecutionError exception."""

    def test_error_message_format(self):
        """Test error message formatting."""
        error = QueryExecutionError(query=".test", reason="test failed")

        assert ".test" in str(error)
        assert "test failed" in str(error)

    def test_error_attributes(self):
        """Test error attributes are set."""
        error = QueryExecutionError(query=".query", reason="reason text")

        assert error.query == ".query"
        assert error.reason == "reason text"
