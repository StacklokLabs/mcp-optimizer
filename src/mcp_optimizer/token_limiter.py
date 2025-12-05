"""Token limiting utilities for tool responses."""

import structlog
from mcp.types import (
    AudioContent,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    ResourceLink,
    TextContent,
)
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


class TokenLimitResult(BaseModel):
    """Result of token limiting operation."""

    result: CallToolResult
    was_truncated: bool
    original_tokens: int
    final_tokens: int
    truncation_message: str | None


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.

    Uses a simple character-based estimation: roughly 4 characters per token.
    This is an approximation that works well for English text.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    return len(text) // 4


def count_content_tokens(  # noqa: C901
    content: list[TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource],
) -> int:
    """
    Count tokens in CallToolResult content.

    Args:
        content: List of content items from CallToolResult

    Returns:
        Total estimated token count
    """
    total_tokens = 0

    for item in content:
        if isinstance(item, TextContent):
            total_tokens += estimate_tokens(item.text)
        elif isinstance(item, ImageContent):
            # Images are complex - estimate based on URL/data length
            # In practice, images use many tokens but we'll use a conservative estimate
            total_tokens += 100  # Base cost for image
            if hasattr(item, "data") and item.data:
                total_tokens += estimate_tokens(item.data[:1000])  # Sample of data
        elif isinstance(item, AudioContent):
            # Audio content uses tokens similar to images
            total_tokens += 100  # Base cost for audio
            if hasattr(item, "data") and item.data:
                total_tokens += estimate_tokens(item.data[:1000])  # Sample of data
        elif isinstance(item, ResourceLink):
            # Resource links are typically URIs
            total_tokens += 50  # Base cost for resource link
            if hasattr(item, "uri"):
                total_tokens += estimate_tokens(str(item.uri))
        elif isinstance(item, EmbeddedResource):
            # Resources contain TextResourceContents or BlobResourceContents
            # Estimate based on the resource content
            total_tokens += 50  # Base cost for resource
            if hasattr(item.resource, "uri"):
                total_tokens += estimate_tokens(str(item.resource.uri))
            if hasattr(item.resource, "text"):
                total_tokens += estimate_tokens(str(item.resource.text))

    return total_tokens


def _process_content_items(  # noqa: C901
    content: list[TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource],
    max_tokens: int,
) -> list[TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource]:
    """Process content items in order, stopping when the next item would exceed the limit."""
    limited_content: list[
        TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
    ] = []
    tokens_used = 0

    for item in content:
        # Calculate tokens for this item
        if isinstance(item, TextContent):
            item_tokens = estimate_tokens(item.text)
        elif isinstance(item, ImageContent):
            item_tokens = 100  # Base cost for image
        elif isinstance(item, AudioContent):
            item_tokens = 100  # Base cost for audio
        elif isinstance(item, ResourceLink):
            item_tokens = 50  # Base cost for resource link
            if hasattr(item, "uri"):
                item_tokens += estimate_tokens(str(item.uri))
        elif isinstance(item, EmbeddedResource):
            item_tokens = 50  # Base cost
            if hasattr(item.resource, "uri"):
                item_tokens += estimate_tokens(str(item.resource.uri))
            if hasattr(item.resource, "text"):
                item_tokens += estimate_tokens(str(item.resource.text))
        else:
            item_tokens = 0

        # Check if adding this item would exceed the limit
        if tokens_used + item_tokens > max_tokens:
            break

        # Add the item and update token count
        limited_content.append(item)
        tokens_used += item_tokens

    return limited_content


def _create_truncation_message(
    original_content: list[
        TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
    ],
    limited_content: list[
        TextContent | ImageContent | AudioContent | ResourceLink | EmbeddedResource
    ],
    original_tokens: int,
    max_tokens: int,
) -> str:
    """Create a message describing the truncation."""
    items_removed = len(original_content) - len(limited_content)
    return (
        f"⚠️ Response truncated: {original_tokens} tokens reduced to ~{max_tokens} tokens per "
        f"mcp-optimizer config. Config set by environment variable MAX_TOOL_RESPONSE_TOKENS. "
        f"{items_removed} content item(s) omitted to fit within token limit."
    )


def limit_tool_response(result: CallToolResult, max_tokens: int) -> TokenLimitResult:
    """
    Limit tool response to fit within max_tokens.

    Processes content items in order and stops when adding the next item would
    exceed the token limit.

    Args:
        result: The CallToolResult to limit
        max_tokens: Maximum number of tokens allowed

    Returns:
        TokenLimitResult with limited response and metadata
    """
    original_tokens = count_content_tokens(result.content)

    if original_tokens <= max_tokens:
        return TokenLimitResult(
            result=result,
            was_truncated=False,
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            truncation_message=None,
        )

    # Need to limit the response
    logger.warning(
        "Tool response exceeds token limit",
        original_tokens=original_tokens,
        max_tokens=max_tokens,
    )

    # Process content items to fit within token limit
    limited_content = _process_content_items(result.content, max_tokens)

    # Create truncation message
    truncation_message = _create_truncation_message(
        result.content, limited_content, original_tokens, max_tokens
    )

    final_tokens = count_content_tokens(limited_content)

    limited_result = CallToolResult(
        content=limited_content,
        isError=result.isError,
    )

    return TokenLimitResult(
        result=limited_result,
        was_truncated=True,
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        truncation_message=truncation_message,
    )
