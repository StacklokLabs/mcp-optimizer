# Data Model: Fix Remote Workload Matching

**Feature**: 001-fix-remote-workload-matching
**Date**: 2025-10-20

## Overview

This feature modifies the semantic meaning of existing database columns rather than changing the schema. The key change is how the `package` column in the McpServer table is populated for remote workloads.

## Entity: McpServer (Existing)

**Purpose**: Represents an MCP server (container or remote workload) stored in the database

**Key Attributes** (focusing on affected fields):

| Attribute | Type | Description | Change |
|-----------|------|-------------|--------|
| id | UUID | Primary key | No change |
| name | String | Workload name from ToolHive | No change |
| url | String | URL where server is accessible | No change |
| package | String | Workload identifier | **CHANGED SEMANTICS** |
| remote | Boolean | Whether this is a remote workload | No change |
| from_registry | Boolean | Whether server came from registry | No change |
| status | Enum | Server status (RUNNING, STOPPED, REGISTRY) | No change |
| transport | Enum | Transport type (SSE, STREAMABLE) | No change |

### Package Column Semantics Change

**Before (Current Behavior)**:
- Container workloads: `package` = Docker image name (e.g., "mcp/time")
- Remote workloads: `package` = workload name (e.g., "github-server")

**After (New Behavior)**:
- Container workloads: `package` = Docker image name (e.g., "mcp/time") **[UNCHANGED]**
- Remote workloads: `package` = workload URL (e.g., "https://api.github.com/mcp") **[CHANGED]**

**Rationale**: URLs are stable identifiers for remote workloads. Names can vary, but URLs uniquely identify the remote service endpoint.

### Validation Rules

**For Remote Workloads**:
- `package` MUST contain a valid URL when `remote=True`
- `package` MUST be fetched from workload details endpoint
- If workload details fetch fails, skip that workload (don't create/update)

**For Container Workloads**:
- `package` MUST contain Docker image name when `remote=False`
- Existing validation rules unchanged

### State Transitions

No changes to state transitions. Status flow remains:
- REGISTRY → RUNNING (when workload starts)
- RUNNING → STOPPED (when workload stops)
- STOPPED → REGISTRY (when workload removed and from_registry=True)
- STOPPED → [DELETED] (when workload removed and from_registry=False)

## Entity: Workload (Existing Pydantic Model)

**Purpose**: Represents a workload from ToolHive API

**Key Attributes**:

| Attribute | Type | Description | Usage |
|-----------|------|-------------|-------|
| name | String | Workload name | Identifier for fetching details |
| url | String | Workload URL | **Used for matching** |
| package | String | Package/image name | Used for containers |
| remote | Boolean | Is remote workload | Determines matching strategy |
| tool_type | String | Type of tool ("mcp", "remote") | Filtering |
| status | String | Workload status | Determines if processed |

### URL Field Importance

The `url` field is the critical attribute for remote workload matching:
- Present in workload list response (may be partial)
- **Must be fetched from details endpoint** (`/api/v1beta/workloads/{name}`) for accurate matching
- Used to match against registry entry URLs
- Stored in McpServer.package column for remote workloads

## Entity: RemoteServerMetadata (Existing Registry Model)

**Purpose**: Represents a remote server entry in the registry

**Key Attributes**:

| Attribute | Type | Description | Change |
|-----------|------|-------------|--------|
| name | String | Server name in registry | No change |
| url | String | Canonical server URL | **Used for matching** |
| description | String | Server description | No change |
| tags | List[String] | Server tags | No change |
| tools | List[String] | Available tool names | No change |

### URL Usage

- `url` field contains the canonical URL for the remote server
- Used to match running workloads to registry entries
- Stored in McpServer.package column when ingesting from registry

## Matching Logic

### Remote Workload Matching Flow

```
1. Fetch workload list from ToolHive
2. For each remote workload (tool_type="remote"):
   a. Fetch workload details: GET /api/v1beta/workloads/{name}
   b. Extract URL from response
   c. Query database: find_server_by_package(url, remote=True)
   d. If found: update server status/metadata
   e. If not found: create new server with package=url
3. For each container workload (tool_type="mcp"):
   [Existing logic unchanged - match by package name]
```

### Cleanup Matching Flow

```
1. Collect active workload identifiers:
   - Remotes: (url, True) tuples
   - Containers: (package, False) tuples
2. Query all servers from database
3. For each server not in active identifiers:
   - If from_registry=True: change status to REGISTRY
   - If from_registry=False: delete server
```

## Database Queries

### New/Modified Queries

**Find Remote Server by URL**:
```sql
SELECT * FROM mcpserver
WHERE package = ? AND remote = true
```
This replaces the current query which uses name-based matching.

**Find Container Server by Package** (Unchanged):
```sql
SELECT * FROM mcpserver
WHERE package = ? AND remote = false
```

## Relationships

No changes to relationships between entities:
- McpServer has many Tools (one-to-many)
- McpServer belongs to one Group (many-to-one)
- McpServer has embeddings for semantic search

## Data Migration

**Migration Type**: Semantic update (no schema change)

**Migration Process**:
1. No ALTER TABLE statements required
2. Existing remote workload entries will have package=name
3. On next ingestion cycle:
   - Fetch workload details to get URLs
   - Update package column with URLs via normal upsert logic
4. Natural migration as ingestion runs

**Rollback**: If needed, can revert code changes. Old package values (names) will remain in database but won't cause issues since matching logic changes are in application code, not constraints.

## Validation Summary

| Rule | Entity | Check |
|------|--------|-------|
| URL format | McpServer | package must be valid URL when remote=True |
| Package not null | McpServer | package must be non-empty string |
| Remote flag consistency | McpServer | remote=True implies package contains URL |
| Container package format | McpServer | remote=False implies package contains image name |
| Workload URL presence | Workload | url field must be present for remote workloads |

## Edge Cases

1. **Multiple workloads with same URL**: Allowed. Both stored with same package value, different names.
2. **URL changes**: Old URL entry becomes orphaned, treated as custom deployment.
3. **Fetch details fails**: Workload skipped for this cycle, retried next cycle.
4. **Container workload with URL in package**: Won't happen - remote flag distinguishes.
5. **Empty URL**: Validation failure, workload skipped with error log.
