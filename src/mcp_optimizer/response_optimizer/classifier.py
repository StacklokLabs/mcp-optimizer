"""Content type classifier for tool responses."""

import json
import re

from mcp_optimizer.response_optimizer.models import ContentType


def classify_content(content: str) -> ContentType:
    """
    Classify content type using heuristic pattern matching.

    Detection order:
    1. JSON: Starts with { or [, valid JSON parse
    2. Markdown: Contains headers (#), code blocks, tables, or other MD syntax
    3. Unstructured: Default fallback

    Args:
        content: The content to classify

    Returns:
        The detected ContentType
    """
    content = content.strip()

    if not content:
        return ContentType.UNSTRUCTURED

    # Check for JSON first
    if _is_json(content):
        return ContentType.JSON

    # Check for Markdown
    if _is_markdown(content):
        return ContentType.MARKDOWN

    # Default to unstructured
    return ContentType.UNSTRUCTURED


def _is_json(content: str) -> bool:
    """Check if content is valid JSON."""
    # Must start with { or [
    if not content.startswith(("{", "[")):
        return False

    try:
        json.loads(content)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _is_markdown(content: str) -> bool:
    """
    Check if content appears to be Markdown.

    Looks for common Markdown patterns:
    - Headers (# Header)
    - Code blocks (``` or ~~~)
    - Tables (|---|)
    - Lists (* item, - item, 1. item)
    - Links [text](url)
    - Bold/italic (**text**, *text*, __text__, _text_)
    """
    # Header pattern: # at start of line followed by space
    header_pattern = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
    if header_pattern.search(content):
        return True

    # Code block pattern: ``` or ~~~
    code_block_pattern = re.compile(r"^(`{3,}|~{3,})", re.MULTILINE)
    if code_block_pattern.search(content):
        return True

    # Table pattern: | followed by content and |
    table_pattern = re.compile(r"^\|.+\|$", re.MULTILINE)
    table_separator = re.compile(r"^\|[\s\-:]+\|$", re.MULTILINE)
    if table_pattern.search(content) and table_separator.search(content):
        return True

    # Link pattern: [text](url) - but not just brackets
    link_pattern = re.compile(r"\[.+?\]\(.+?\)")
    if link_pattern.search(content):
        return True

    # Count markdown indicators
    indicators = 0

    # Bullet lists: * or - at start of line followed by space
    bullet_pattern = re.compile(r"^[\*\-]\s+\S", re.MULTILINE)
    if bullet_pattern.search(content):
        indicators += 1

    # Numbered lists: digit. at start of line
    numbered_pattern = re.compile(r"^\d+\.\s+\S", re.MULTILINE)
    if numbered_pattern.search(content):
        indicators += 1

    # Bold/italic: **text** or *text* or __text__ or _text_
    emphasis_pattern = re.compile(r"(\*\*|__).+?(\*\*|__)|(\*|_)[^*_\s].+?(\*|_)")
    if emphasis_pattern.search(content):
        indicators += 1

    # Blockquote: > at start of line
    blockquote_pattern = re.compile(r"^>\s+", re.MULTILINE)
    if blockquote_pattern.search(content):
        indicators += 1

    # Horizontal rule: --- or *** or ___ on its own line
    hr_pattern = re.compile(r"^(---|\*\*\*|___)$", re.MULTILINE)
    if hr_pattern.search(content):
        indicators += 1

    # If we have multiple markdown indicators, it's likely markdown
    return indicators >= 2
