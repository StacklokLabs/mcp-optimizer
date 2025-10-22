# Data Model: Token Savings Metrics

**Feature**: Token Savings Metrics for find_tool
**Phase**: Phase 1 - Data Model Definition
**Date**: 2025-10-20

## Overview

This document defines the data entities and their relationships for the token savings metrics feature. All entities use Pydantic for validation and native Python types following project conventions.

## Database Entities

### Tool (Modified)

**Location**: `src/mcp_optimizer/db/models.py`

**Purpose**: Represents a tool from an MCP server with token count for efficiency metrics

**Fields**:
- `id`: str (UUID4) - Primary key, unique identifier for the tool
- `mcpserver_id`: str (UUID4) - Foreign key to mcpserver table
- `details`: McpTool - Complete MCP Tool specification (JSON serialized in DB)
- `details_embedding`: np.ndarray | None - 1D numpy array for semantic search
- `token_count`: int - **NEW** Number of tokens in the serialized tool details
- `last_updated`: datetime - Last modification timestamp
- `created_at`: datetime - Creation timestamp

**Changes**:
```python
class Tool(BaseModel):
    """Result from searching for tools based on a user query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # UUID4
    mcpserver_id: str  # UUID4
    details: McpTool
    details_embedding: np.ndarray | None = Field(default=None, exclude=True)
    token_count: int = Field(default=0, ge=0)  # NEW: Token count, must be non-negative
    last_updated: datetime
    created_at: datetime
```

**Validation Rules**:
- `token_count` must be >= 0 (non-negative integer)
- Default value is 0 to handle edge cases gracefully
- Calculated during tool ingestion using tiktoken

**Database Schema**:
```sql
CREATE TABLE tool (
    id TEXT PRIMARY KEY,
    mcpserver_id TEXT NOT NULL,
    details TEXT NOT NULL,
    details_embedding BLOB NOT NULL,
    token_count INTEGER NOT NULL DEFAULT 0,  -- NEW
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mcpserver_id) REFERENCES mcpserver (id) ON DELETE CASCADE
)
```

### ToolUpdateDetails (Modified)

**Location**: `src/mcp_optimizer/db/models.py`

**Purpose**: Defines fields that can be updated for a Tool entity

**Changes**:
```python
class ToolUpdateDetails(BaseUpdateDetails):
    """Details for updating a Tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mcpserver_id: str | None = None
    details: McpTool | None = None
    details_embedding: np.ndarray | None = None
    token_count: int | None = Field(default=None, ge=0)  # NEW: Allow updating token count
```

**Validation Rules**:
- If `token_count` is provided, it must be >= 0
- When `details` is updated, `token_count` should typically be recalculated

## Response Entities

### TokenMetrics (New)

**Location**: `src/mcp_optimizer/db/models.py` or `src/mcp_optimizer/server.py`

**Purpose**: Represents token savings metrics returned in find_tool response

**Fields**:
- `baseline_tokens`: int - Total tokens for all tools from running servers
- `returned_tokens`: int - Total tokens for filtered/returned tools
- `tokens_saved`: int - Difference between baseline and returned (savings)
- `savings_percentage`: float - Percentage of tokens saved (0-100)

**Definition**:
```python
class TokenMetrics(BaseModel):
    """Token efficiency metrics for tool filtering."""

    baseline_tokens: int = Field(ge=0, description="Total tokens for all running server tools")
    returned_tokens: int = Field(ge=0, description="Total tokens for returned/filtered tools")
    tokens_saved: int = Field(ge=0, description="Number of tokens saved by filtering")
    savings_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of tokens saved")

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        """Validate token metrics consistency."""
        if self.tokens_saved != self.baseline_tokens - self.returned_tokens:
            raise ValueError("tokens_saved must equal baseline_tokens - returned_tokens")

        if self.baseline_tokens > 0:
            expected_pct = (self.tokens_saved / self.baseline_tokens) * 100
            if abs(self.savings_percentage - expected_pct) > 0.01:
                raise ValueError("savings_percentage does not match calculated value")
        else:
            if self.savings_percentage != 0.0:
                raise ValueError("savings_percentage must be 0 when baseline_tokens is 0")

        return self
```

**Validation Rules**:
- All token counts must be non-negative
- `tokens_saved` = `baseline_tokens` - `returned_tokens`
- `savings_percentage` = (`tokens_saved` / `baseline_tokens`) * 100 (or 0 if baseline is 0)
- `savings_percentage` must be between 0 and 100

### FindToolResponse (New)

**Location**: `src/mcp_optimizer/server.py`

**Purpose**: Extended response for find_tool endpoint with token metrics

**Fields**:
- `tools`: list[McpTool] - List of matching tools
- `token_metrics`: TokenMetrics - Token savings information

**Definition**:
```python
class FindToolResponse(BaseModel):
    """Response from find_tool endpoint with token savings metrics."""

    tools: list[McpTool]
    token_metrics: TokenMetrics
```

**Usage**:
```python
async def find_tool(tool_description: str, tool_keywords: str) -> FindToolResponse:
    # ... existing logic to get matching_tools ...

    # Calculate token metrics
    baseline_tokens = await calculate_baseline_tokens(running_server_tools)
    returned_tokens = sum(tool.token_count for tool in matching_tools)
    tokens_saved = baseline_tokens - returned_tokens
    savings_percentage = (tokens_saved / baseline_tokens * 100) if baseline_tokens > 0 else 0.0

    token_metrics = TokenMetrics(
        baseline_tokens=baseline_tokens,
        returned_tokens=returned_tokens,
        tokens_saved=tokens_saved,
        savings_percentage=savings_percentage
    )

    return FindToolResponse(
        tools=[tool.details for tool in matching_tools],
        token_metrics=token_metrics
    )
```

## Service Layer Entities

### TokenCounter (New)

**Location**: `src/mcp_optimizer/token_counter.py`

**Purpose**: Utility class for calculating token counts using tiktoken

**Interface**:
```python
class TokenCounter:
    """Token counting utility using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize with specified encoding (default: cl100k_base for GPT-4)."""
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in given text."""
        return len(self.encoding.encode(text))

    def count_tool_tokens(self, tool: McpTool) -> int:
        """Count tokens in serialized tool JSON."""
        tool_json = tool.model_dump_json()
        return self.count_tokens(tool_json)
```

**Usage in Ingestion**:
```python
# During tool ingestion
token_counter = TokenCounter()
token_count = token_counter.count_tool_tokens(mcp_tool)

# Store in database
await tool_ops.create_tool(
    mcpserver_id=server_id,
    details=mcp_tool,
    details_embedding=embedding,
    token_count=token_count  # NEW parameter
)
```

## Entity Relationships

```
McpServer (1) ----< (N) Tool
                       │
                       └─ Contains: token_count (NEW)

Tool (N) ----< (N) ToolWithMetadata
                   │
                   └─ Used by: find_tool

FindToolResponse
├── tools: list[McpTool]
└── token_metrics: TokenMetrics
    ├── baseline_tokens (from all running server tools)
    ├── returned_tokens (from filtered tools)
    ├── tokens_saved (difference)
    └── savings_percentage (calculated)
```

## State Transitions

### Tool Lifecycle with Token Count

1. **Creation** (during ingestion):
   - Tool details received from MCP server
   - TokenCounter calculates token_count from details.model_dump_json()
   - Tool record created with token_count field populated

2. **Query** (during find_tool):
   - Tools filtered by semantic/BM25 search
   - Token counts summed for baseline (all running server tools)
   - Token counts summed for filtered subset
   - TokenMetrics calculated and returned

3. **Update** (if tool details change):
   - New details provided
   - Token count recalculated
   - Both details and token_count updated atomically

## Data Flow

```
1. Ingestion Flow:
   MCP Server → Tool Details → TokenCounter → token_count → DB (tool table)

2. Query Flow:
   User Query → find_tool → Filter Tools → Sum token_counts → TokenMetrics → Response

3. Token Calculation:
   McpTool → model_dump_json() → tiktoken.encode() → len() → token_count
```

## Validation Summary

| Entity | Field | Validation |
|--------|-------|------------|
| Tool | token_count | >= 0, integer |
| TokenMetrics | baseline_tokens | >= 0, integer |
| TokenMetrics | returned_tokens | >= 0, integer, <= baseline_tokens |
| TokenMetrics | tokens_saved | >= 0, equals baseline - returned |
| TokenMetrics | savings_percentage | 0.0 - 100.0, matches calculated value |

## Migration Notes

- No database migration needed (database is ephemeral)
- Update existing table creation script in `migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py`
- Add `token_count INTEGER NOT NULL DEFAULT 0` column to tool table
- Existing code will need updates to pass token_count parameter
