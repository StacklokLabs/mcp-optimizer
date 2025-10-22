# Group Filtering

MCP-Optimizer supports filtering tool lookups by ToolHive groups, allowing you to restrict which tools are discoverable based on their group membership.

## Overview

When running multiple MCP servers across different ToolHive groups (e.g., `production`, `staging`, `development`), you may want to limit tool discovery to specific groups. Group filtering enables this by allowing you to specify which groups to search when looking up tools.

## Configuration

Group filtering can be configured using two methods:

### 1. Environment Variable

Set the `ALLOWED_GROUPS` environment variable:

```bash
# Single group
export ALLOWED_GROUPS="production"

# Multiple groups (comma-separated)
export ALLOWED_GROUPS="production,staging"

# Start the server
mcp-optimizer
```

### 2. Default Behavior

If the environment variable is not set, all groups are searched (no filtering).

## Usage Examples

### Example 1: Development Environment

Restrict tool discovery to development and testing groups:

```bash
export ALLOWED_GROUPS="development,testing"
mcp-optimizer
```

### Example 2: Production Environment

Only discover tools from the production group:

```bash
export ALLOWED_GROUPS="production"
mcp-optimizer
```

## How It Works

1. **Server Group Assignment**: When MCP-Optimizer ingests MCP servers from ToolHive, it automatically captures and stores each server's group information.

2. **Tool Discovery**: When searching for tools using `find_tool`, `list_tools`, or `search_registry`, MCP-Optimizer:
   - Checks if group filtering is configured via the `ALLOWED_GROUPS` environment variable
   - Filters the search to only include tools from servers in the specified groups
   - Returns only matching tools

3. **Group Filtering Behavior**:
   - `ALLOWED_GROUPS` not set (default): Search all groups
   - `ALLOWED_GROUPS="group-a"`: Only search servers in `group-a`
   - `ALLOWED_GROUPS="group-a,group-b"`: Search servers in either `group-a` or `group-b`

## Technical Details

### Database Schema

The `mcpserver` table includes a `group` column (nullable) that stores the ToolHive group name. An index (`idx_mcpserver_group`) is created for efficient group-based queries.

### Query Filtering

Group filtering is applied at the database query level, affecting:
- Semantic similarity search (vector embeddings)
- BM25 full-text search
- Tool listing operations

### Backward Compatibility

- Existing servers without group assignments have `group = NULL`
- Servers with `NULL` groups are excluded when group filtering is active
- No breaking changes to existing APIs

## Integration with ToolHive

MCP-Optimizer automatically discovers and respects ToolHive group assignments:

```bash
# Create groups in ToolHive
thv group create production
thv group create staging

# Run servers in different groups
thv run --group production github
thv run --group staging postgres

# Configure MCP-Optimizer to only see production tools
export ALLOWED_GROUPS="production"
mcp-optimizer
```

## See Also

- [ToolHive Groups Documentation](https://docs.stacklok.com/toolhive/)
- [MCP-Optimizer README](../README.md)

