"""Pydantic models for the response optimizer."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from mcp_optimizer.db.models import TokenMetrics


class ContentType(str, Enum):
    """Content type classification for tool responses."""

    JSON = "json"
    MARKDOWN = "markdown"
    UNSTRUCTURED = "unstructured"


class TraversalResult(BaseModel):
    """Result of traversing and structurally compressing content."""

    content: str = Field(description="The traversed/compressed content")
    original_tokens: int = Field(description="Token count of original content")
    result_tokens: int = Field(description="Token count after traversal")
    sections_summarized: int = Field(
        default=0, description="Number of sections that were summarized"
    )
    metadata: dict = Field(default_factory=dict, description="Additional traversal metadata")


class SummaryResult(BaseModel):
    """Result of summarizing text content."""

    content: str = Field(description="The summarized content")
    original_tokens: int = Field(description="Token count before summarization")
    result_tokens: int = Field(description="Token count after summarization")
    compression_ratio: float = Field(description="Compression ratio achieved")


class QueryHint(BaseModel):
    """Query hint for retrieving specific parts of original content."""

    tool: str = Field(description="Tool to use for querying (e.g., 'jq', 'grep')")
    examples: list[str] = Field(description="Example queries for this content type")
    description: str = Field(description="Description of how to use the query tool")


class OptimizedResponse(BaseModel):
    """Result of optimizing a tool response."""

    content: str = Field(description="The optimized content (actual text, not nested JSON)")
    response_id: str = Field(description="UUID for retrieving original content from KV store")
    session_key: str = Field(description="Session key for grouping related responses")
    content_type: ContentType = Field(description="Detected content type")
    was_optimized: bool = Field(description="Whether optimization was applied")
    query_hints: QueryHint | None = Field(
        default=None, description="Hints for querying original content"
    )
    token_metrics: TokenMetrics = Field(description="Token efficiency metrics for this response")


class StoredToolResponse(BaseModel):
    """Model for a tool response stored in the KV store."""

    id: str = Field(description="Unique identifier for the stored response")
    session_key: str = Field(description="Session key for grouping related responses")
    tool_name: str = Field(description="Name of the tool that generated this response")
    original_content: str = Field(description="The original unmodified content")
    content_type: ContentType = Field(description="Detected content type")
    created_at: datetime = Field(description="When the response was stored")
    expires_at: datetime = Field(description="When the response will expire")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")
