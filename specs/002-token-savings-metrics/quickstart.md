# Quick Start: Token Savings Metrics Implementation

**Feature**: Token Savings Metrics for find_tool
**Branch**: `002-token-savings-metrics`
**Date**: 2025-10-20

## Overview

This guide walks through implementing token savings metrics for the find_tool endpoint. The implementation calculates token counts during tool ingestion and returns savings metrics at query time.

## Prerequisites

- Python 3.13+
- `uv` package manager
- Existing mcp-optimizer development environment
- Familiarity with SQLAlchemy, Pydantic, and async Python

## Development Setup

### 1. Install Dependencies

```bash
# Add tiktoken dependency
uv add tiktoken

# Verify installation
uv run python -c "import tiktoken; print(tiktoken.__version__)"
```

### 2. Verify Environment

```bash
# Run existing tests to ensure baseline functionality
task test

# Run linting and type checking
task lint
task typecheck
```

## Implementation Steps

### Step 1: Create Token Counter Utility

**File**: `src/mcp_optimizer/token_counter.py`

Create a new module for token counting logic:

```python
"""Token counting utility using tiktoken for LLM-compatible tokenization."""

import tiktoken
from mcp.types import Tool as McpTool


class TokenCounter:
    """Token counting utility using tiktoken."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize token counter with specified encoding.

        Args:
            encoding_name: tiktoken encoding to use (default: cl100k_base for GPT-4)
        """
        self.encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in given text.

        Args:
            text: Text to tokenize

        Returns:
            Number of tokens
        """
        return len(self.encoding.encode(text))

    def count_tool_tokens(self, tool: McpTool) -> int:
        """
        Count tokens in serialized MCP tool.

        Args:
            tool: MCP Tool to count tokens for

        Returns:
            Number of tokens in JSON serialized tool
        """
        tool_json = tool.model_dump_json()
        return self.count_tokens(tool_json)
```

**Test**: `tests/unit/test_token_counter.py`

```python
import pytest
from mcp.types import Tool as McpTool
from mcp_optimizer.token_counter import TokenCounter


def test_count_tokens_simple():
    counter = TokenCounter()
    token_count = counter.count_tokens("Hello, world!")
    assert token_count > 0
    assert isinstance(token_count, int)


def test_count_tool_tokens():
    counter = TokenCounter()
    tool = McpTool(
        name="test_tool",
        description="A test tool",
        inputSchema={"type": "object", "properties": {}}
    )
    token_count = counter.count_tool_tokens(tool)
    assert token_count > 0
```

### Step 2: Update Database Schema

**File**: `migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py`

Add `token_count` column to the tool table:

```python
# Find the CREATE TABLE tool statement and add the token_count column
conn.execute(text("""
    CREATE TABLE tool (
        id TEXT PRIMARY KEY,
        mcpserver_id TEXT NOT NULL,
        details TEXT NOT NULL,
        details_embedding BLOB NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0,  -- ADD THIS LINE
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (mcpserver_id) REFERENCES mcpserver (id) ON DELETE CASCADE
    )
"""))
```

### Step 3: Update Data Models

**File**: `src/mcp_optimizer/db/models.py`

Add token_count field to Tool model:

```python
class Tool(BaseModel):
    """Result from searching for tools based on a user query."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # UUID4
    mcpserver_id: str  # UUID4
    details: McpTool
    details_embedding: np.ndarray | None = Field(default=None, exclude=True)
    token_count: int = Field(default=0, ge=0)  # ADD THIS LINE
    last_updated: datetime
    created_at: datetime
```

Add TokenMetrics model:

```python
class TokenMetrics(BaseModel):
    """Token efficiency metrics for tool filtering."""

    baseline_tokens: int = Field(ge=0)
    returned_tokens: int = Field(ge=0)
    tokens_saved: int = Field(ge=0)
    savings_percentage: float = Field(ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        """Validate token metrics consistency."""
        if self.tokens_saved != self.baseline_tokens - self.returned_tokens:
            raise ValueError("tokens_saved must equal baseline_tokens - returned_tokens")
        return self
```

### Step 4: Update CRUD Operations

**File**: `src/mcp_optimizer/db/crud.py`

Update `create_tool` to accept and store token_count:

```python
async def create_tool(
    self,
    mcpserver_id: str,
    details: McpTool,
    details_embedding: np.ndarray,
    token_count: int,  # ADD THIS PARAMETER
    conn: AsyncConnection | None = None,
) -> Tool:
    """Create a new tool record."""
    new_tool = Tool(
        id=str(uuid.uuid4()),
        mcpserver_id=mcpserver_id,
        details=details,
        details_embedding=details_embedding,
        token_count=token_count,  # ADD THIS LINE
        last_updated=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )

    query = """
    INSERT INTO tool (id, mcpserver_id, details, details_embedding, token_count, last_updated, created_at)
    VALUES (:id, :mcpserver_id, :details, :details_embedding, :token_count, :last_updated, :created_at)
    """

    params = new_tool.model_dump()
    params["details"] = details.model_dump_json()
    params["details_embedding"] = details_embedding.tobytes()

    await self.db.execute_non_query(query, params, conn=conn)
    return new_tool
```

Add method to sum token counts:

```python
async def sum_token_counts_for_running_servers(
    self,
    allowed_groups: list[str] | None = None,
    conn: AsyncConnection | None = None,
) -> int:
    """
    Sum token counts for all tools from running MCP servers.

    Args:
        allowed_groups: Optional list of groups to filter by
        conn: Optional database connection

    Returns:
        Total token count for all running server tools
    """
    query = """
    SELECT COALESCE(SUM(t.token_count), 0) as total_tokens
    FROM tool t
    INNER JOIN mcpserver m ON t.mcpserver_id = m.id
    WHERE m.status = :status
    """

    params = {"status": McpStatus.RUNNING.value}

    if allowed_groups:
        placeholders = ",".join([f":group_{i}" for i in range(len(allowed_groups))])
        query += f" AND m.group IN ({placeholders})"
        for i, group in enumerate(allowed_groups):
            params[f"group_{i}"] = group

    result = await self.db.execute_query(query, params, conn=conn)
    row = result.fetchone()
    return row[0] if row else 0
```

### Step 5: Update Ingestion Logic

**File**: `src/mcp_optimizer/ingestion.py`

Initialize token counter in `__init__`:

```python
def __init__(self, db_config: DatabaseConfig, embedding_manager: EmbeddingManager):
    self.db_config = db_config
    self.embedding_manager = embedding_manager
    self.tool_ops = ToolOps(db_config)
    self.server_ops = McpServerOps(db_config)
    self.token_counter = TokenCounter()  # ADD THIS LINE
```

Update `_sync_tools` to calculate token counts:

```python
# After generating embeddings for tools, calculate token counts
token_counts = [self.token_counter.count_tool_tokens(tool) for tool in tools.tools]

# When creating tools, pass token_count
for i, tool in enumerate(tools.tools):
    await self.tool_ops.create_tool(
        mcpserver_id=server_id,
        details=tool,
        details_embedding=embeddings[i],
        token_count=token_counts[i],  # ADD THIS LINE
        conn=conn,
    )
```

### Step 6: Update find_tool Endpoint

**File**: `src/mcp_optimizer/server.py`

Update find_tool to return token metrics:

```python
async def find_tool(tool_description: str, tool_keywords: str) -> dict:
    """Find and return tools with token savings metrics."""
    if embedding_manager is None or _config is None or tool_ops is None:
        raise RuntimeError("Server components not initialized")

    try:
        # ... existing tool discovery logic ...

        # Calculate baseline tokens (all running server tools)
        baseline_tokens = await tool_ops.sum_token_counts_for_running_servers(
            allowed_groups=_config.allowed_groups
        )

        # Calculate returned tokens (filtered tools)
        returned_tokens = sum(tool.tool.token_count for tool in similar_db_tools)

        # Calculate savings
        tokens_saved = baseline_tokens - returned_tokens
        savings_percentage = (
            (tokens_saved / baseline_tokens * 100) if baseline_tokens > 0 else 0.0
        )

        # Build response with token metrics
        return {
            "tools": matching_tools,
            "token_metrics": {
                "baseline_tokens": baseline_tokens,
                "returned_tokens": returned_tokens,
                "tokens_saved": tokens_saved,
                "savings_percentage": round(savings_percentage, 2),
            },
        }
    except Exception as e:
        logger.exception(f"Error in find_tool: {e}")
        raise
```

## Testing

### Run Unit Tests

```bash
# Test token counter
uv run pytest tests/unit/test_token_counter.py -v

# Test updated models
uv run pytest tests/unit/test_models.py -v

# Test updated CRUD operations
uv run pytest tests/unit/test_crud.py -v
```

### Run Integration Tests

```bash
# Test full ingestion flow
uv run pytest tests/integration/test_ingestion.py -v

# Test find_tool endpoint
uv run pytest tests/integration/test_server.py -v
```

### Manual Testing

```bash
# Start mcp-optimizer server
uv run mcp-optimizer serve

# In another terminal, test find_tool
uv run python -c "
from mcp_optimizer.mcp_client import MCPServerClient
import asyncio

async def test():
    client = MCPServerClient('http://localhost:9900')
    result = await client.find_tool('search the web', 'web search')
    print(result)

asyncio.run(test())
"
```

## Quality Gates

### 1. Format Code

```bash
task format
```

### 2. Run Linter

```bash
task lint
```

### 3. Type Check

```bash
task typecheck
```

### 4. Run Tests

```bash
task test
```

All gates must pass before proceeding to code review.

## Verification Checklist

- [ ] tiktoken dependency added to pyproject.toml
- [ ] TokenCounter class created and tested
- [ ] Database schema updated with token_count column
- [ ] Tool model updated with token_count field
- [ ] TokenMetrics model created with validation
- [ ] create_tool method updated to accept token_count
- [ ] Token counting integrated into ingestion flow
- [ ] find_tool endpoint returns token_metrics
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] All quality gates pass (format, lint, typecheck, test)

## Next Steps

After implementation is complete:
1. Run `/speckit.tasks` to generate implementation tasks
2. Follow tasks in priority order
3. Create PR with commit message following project standards
4. Ensure all CI checks pass

## Troubleshooting

### tiktoken Import Errors

```bash
# Verify tiktoken is installed
uv run python -c "import tiktoken; print('OK')"

# Reinstall if needed
uv remove tiktoken
uv add tiktoken
```

### Database Schema Errors

```bash
# Database is ephemeral, restart server to recreate
# Check migrations file was updated correctly
```

### Test Failures

```bash
# Run specific test with verbose output
uv run pytest tests/unit/test_token_counter.py::test_name -vv

# Check logs
grep ERROR logs/mcp-optimizer.log
```

## References

- [Feature Spec](./spec.md)
- [Research Document](./research.md)
- [Data Model](./data-model.md)
- [API Contracts](./contracts/)
- [tiktoken Documentation](https://github.com/openai/tiktoken)
