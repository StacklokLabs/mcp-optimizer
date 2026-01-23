"""Query executor for searching and filtering stored tool responses."""

import re
import shutil
import subprocess  # nosec B404 - subprocess used for trusted jq tool only

from mcp_optimizer.response_optimizer.models import ContentType


class QueryExecutionError(Exception):
    """Exception raised when query execution fails."""

    def __init__(self, query: str, reason: str):
        self.query = query
        self.reason = reason
        super().__init__(f"Query '{query}' failed: {reason}")


def execute_jq_query(content: str, query: str) -> str:
    """Execute a JQ query on JSON content using the jq command-line tool.

    Args:
        content: The JSON content to query
        query: The JQ query expression (e.g., ".results", ".[0].name")

    Returns:
        The query result as a string

    Raises:
        QueryExecutionError: If jq is not installed or the query fails
    """
    jq_path = shutil.which("jq")
    if jq_path is None:
        raise QueryExecutionError(
            query=query,
            reason="jq command not found. Please install jq to query JSON responses.",
        )

    try:
        result = subprocess.run(  # noqa: S603 # nosec B603 - jq is a trusted tool, path validated
            [jq_path, query],
            input=content,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown jq error"
            raise QueryExecutionError(query=query, reason=f"jq query failed: {error_msg}")

        return result.stdout.strip()
    except subprocess.TimeoutExpired as e:
        raise QueryExecutionError(
            query=query,
            reason="jq query timed out after 10 seconds",
        ) from e


def _parse_section_query(section_query: str) -> tuple[int | None, str]:
    """Parse a markdown section query into target level and title."""
    header_match = re.match(r"^(#{1,6})\s*(.+)$", section_query.strip())
    if header_match:
        return len(header_match.group(1)), header_match.group(2).strip().lower()
    return None, section_query.strip().lower()


def _find_section_bounds(
    lines: list[str], target_level: int | None, target_title: str
) -> tuple[int | None, int | None, int | None]:
    """Find section start, end, and actual level in markdown lines."""
    section_start = None
    section_end = None
    actual_level = target_level

    for i, line in enumerate(lines):
        header_pattern = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not header_pattern:
            continue

        level = len(header_pattern.group(1))
        title = header_pattern.group(2).strip().lower()

        if section_start is None:
            # Check if this header matches our target
            level_matches = target_level is None or level == target_level
            if level_matches and target_title in title:
                section_start = i
                actual_level = level
        elif level <= actual_level:
            # Found end of section
            section_end = i
            break

    return section_start, section_end, actual_level


def extract_markdown_section(content: str, section_query: str) -> str:
    """Extract a section from markdown content based on header matching.

    Args:
        content: The markdown content to search
        section_query: Section to extract. Can be:
            - "## Section Name" - Match exact header level
            - "Section Name" - Match any header containing this text

    Returns:
        The extracted section content including the header

    Raises:
        QueryExecutionError: If the section is not found
    """
    lines = content.split("\n")
    target_level, target_title = _parse_section_query(section_query)
    section_start, section_end, _ = _find_section_bounds(lines, target_level, target_title)

    if section_start is None:
        raise QueryExecutionError(
            query=section_query,
            reason=f"Section '{section_query}' not found in markdown content",
        )

    if section_end is None:
        section_end = len(lines)

    return "\n".join(lines[section_start:section_end]).strip()


def execute_text_query(content: str, query: str) -> str:
    """Execute a text query (grep, head, tail, lines) on unstructured content.

    Args:
        content: The text content to query
        query: The query command. Supported:
            - "head [-n N]" - First N lines (default 10)
            - "tail [-n N]" - Last N lines (default 10)
            - "lines X-Y" - Lines X through Y (1-indexed)
            - "grep [-i] pattern" - Lines matching pattern

    Returns:
        The matching lines

    Raises:
        QueryExecutionError: If the query format is not supported
    """
    query = query.strip()

    # Handle 'head' command
    head_match = re.match(r"head\s*(?:-n\s*)?(\d+)?", query, re.IGNORECASE)
    if head_match:
        n = int(head_match.group(1) or 10)
        lines = content.split("\n")
        return "\n".join(lines[:n])

    # Handle 'tail' command
    tail_match = re.match(r"tail\s*(?:-n\s*)?(\d+)?", query, re.IGNORECASE)
    if tail_match:
        n = int(tail_match.group(1) or 10)
        lines = content.split("\n")
        return "\n".join(lines[-n:])

    # Handle 'lines X-Y' command
    lines_match = re.match(r"lines?\s+(\d+)\s*-\s*(\d+)", query, re.IGNORECASE)
    if lines_match:
        start = int(lines_match.group(1)) - 1  # Convert to 0-indexed
        end = int(lines_match.group(2))
        lines = content.split("\n")
        start = max(0, start)
        end = min(len(lines), end)
        return "\n".join(lines[start:end])

    # Handle 'grep' command
    grep_match = re.match(r"grep\s+(?:-i\s+)?['\"]?(.+?)['\"]?\s*$", query, re.IGNORECASE)
    if grep_match:
        pattern = grep_match.group(1)
        case_insensitive = "-i" in query.lower()
        lines = content.split("\n")
        flags = re.IGNORECASE if case_insensitive else 0

        try:
            matching_lines = [line for line in lines if re.search(pattern, line, flags)]
        except re.error:
            # If regex fails, try literal string match
            if case_insensitive:
                matching_lines = [line for line in lines if pattern.lower() in line.lower()]
            else:
                matching_lines = [line for line in lines if pattern in line]

        if not matching_lines:
            return f"No lines matching '{pattern}' found"
        return "\n".join(matching_lines)

    raise QueryExecutionError(
        query=query,
        reason=(
            "Unsupported text query. Supported commands: "
            "'head [-n N]', 'tail [-n N]', 'lines X-Y', 'grep [-i] pattern'"
        ),
    )


def execute_query(content: str, content_type: ContentType, query: str) -> str:
    """Execute a query on content based on its type.

    Args:
        content: The content to query
        content_type: The type of content (JSON, MARKDOWN, UNSTRUCTURED)
        query: The query string appropriate for the content type

    Returns:
        The query result

    Raises:
        QueryExecutionError: If the query fails
    """
    if content_type == ContentType.JSON:
        return execute_jq_query(content, query)
    elif content_type == ContentType.MARKDOWN:
        return extract_markdown_section(content, query)
    else:  # UNSTRUCTURED
        return execute_text_query(content, query)
