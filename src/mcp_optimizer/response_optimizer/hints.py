"""Query hint generator for retrieving original content."""

import json
import re

from mcp_optimizer.response_optimizer.models import ContentType, QueryHint


def generate_query_hints(
    content: str,
    content_type: ContentType,
    response_id: str,
) -> QueryHint:
    """
    Generate query hints based on content type.

    Provides retrieval instructions so the LLM can request specific parts
    of the original content that was stored in the KV store.

    Args:
        content: The original content
        content_type: The detected content type
        response_id: The ID of the stored response

    Returns:
        QueryHint with tool and example queries
    """
    if content_type == ContentType.JSON:
        return _generate_json_hints(content, response_id)
    elif content_type == ContentType.MARKDOWN:
        return _generate_markdown_hints(content, response_id)
    else:
        return _generate_text_hints(content, response_id)


def _generate_json_hints(content: str, response_id: str) -> QueryHint:
    """Generate hints for JSON content."""
    examples = []

    try:
        data = json.loads(content)

        # Generate examples based on structure
        if isinstance(data, dict):
            keys = list(data.keys())[:5]
            for key in keys:
                examples.append(f".{key}")

            # Check for arrays
            for key, value in data.items():
                if isinstance(value, list) and len(value) > 0:
                    examples.append(f".{key}[0]")
                    examples.append(f".{key} | length")
                    if isinstance(value[0], dict):
                        examples.append(f".{key}[] | keys")
                    break

        elif isinstance(data, list):
            examples.append(".[0]")
            examples.append(". | length")
            if len(data) > 0 and isinstance(data[0], dict):
                examples.append(".[] | keys")

    except json.JSONDecodeError:
        examples = [".keys", ".[0]", ". | length"]

    return QueryHint(
        tool="jq",
        examples=examples[:5],
        description=(
            f"Use jq to query the original JSON response (ID: {response_id}). "
            "Request specific fields or array elements as needed."
        ),
    )


def _generate_markdown_hints(content: str, response_id: str) -> QueryHint:
    """Generate hints for Markdown content."""
    examples = []

    # Find headers
    header_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    headers = header_pattern.findall(content)

    for level, title in headers[:5]:
        header_marker = "#" * len(level)
        examples.append(f'Section: "{header_marker} {title}"')

    if not examples:
        examples = [
            'Section: "## Getting Started"',
            'Section: "# Introduction"',
            "Lines: 1-50",
        ]

    return QueryHint(
        tool="section",
        examples=examples[:5],
        description=(
            f"Request specific sections from the original Markdown (ID: {response_id}). "
            "Specify section headers or line ranges."
        ),
    )


def _generate_text_hints(content: str, response_id: str) -> QueryHint:
    """Generate hints for unstructured text."""
    lines = content.split("\n")
    total_lines = len(lines)

    examples = [
        f"head -n 50 (first 50 of {total_lines} lines)",
        f"tail -n 50 (last 50 of {total_lines} lines)",
        "grep 'error'",
        "grep 'warning'",
        f"lines 100-200 (of {total_lines} total)",
    ]

    # Look for common patterns to suggest
    if any("error" in line.lower() for line in lines):
        examples.insert(0, "grep -i 'error'")
    if any("exception" in line.lower() for line in lines):
        examples.insert(1, "grep -i 'exception'")

    return QueryHint(
        tool="text",
        examples=examples[:5],
        description=(
            f"Use text tools to query the original content (ID: {response_id}). "
            f"Total: {total_lines} lines. Use head/tail/grep/line ranges."
        ),
    )
