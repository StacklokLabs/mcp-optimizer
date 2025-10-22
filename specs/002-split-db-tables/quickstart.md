# Quickstart: Database Table Separation Implementation

**Feature**: 002-split-db-tables
**Date**: 2025-10-21

## Overview

This quickstart guide provides a step-by-step walkthrough for implementing the database table separation feature. Follow these steps in order to ensure successful implementation.

## Prerequisites

- Python 3.13+ with uv installed
- Access to mcp-optimizer repository on branch `002-split-db-tables`
- Existing database with old schema (mcpserver, tool tables)
- Familiarity with SQLAlchemy async and Pydantic

## Implementation Phases

### Phase 1: Database Schema and Models (Priority: P1)

**Goal**: Create new table structure and Pydantic models.

#### Step 1.1: Update Pydantic Models

**File**: `src/mcp_optimizer/db/models.py`

1. Create new model classes:
   - `RegistryServer` - Registry MCP server model
   - `WorkloadServer` - Workload MCP server model
   - `RegistryTool` - Registry tool model
   - `WorkloadTool` - Workload tool model (add `available` field)
   - `WorkloadWithRegistry` - Combined model for relationship queries

2. Add validation methods:
   - `validate_identifier()` - Ensure URL (remote) XOR package (container)
   - `validate_description()` - Ensure autonomous workloads have description

3. Keep existing models temporarily:
   - `McpServer` - Mark as deprecated, remove after migration
   - `Tool` - Mark as deprecated, remove after migration

**Test**: Run `task typecheck` to verify model definitions.

#### Step 1.2: Create Migration

**File**: `migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py`

1. Add drop statements at beginning:
   ```python
   op.execute("DROP TABLE IF EXISTS tool")
   op.execute("DROP TABLE IF EXISTS mcpserver")
   op.execute("DROP TABLE IF EXISTS server_vector")
   op.execute("DROP TABLE IF EXISTS tool_vectors")
   op.execute("DROP TABLE IF EXISTS tool_fts")
   ```

2. Create new tables:
   - `mcpservers_registry` with constraints (UNIQUE on URL for remote, UNIQUE on package for container)
   - `mcpservers_workload` with constraints (UNIQUE on workload_name, FK to registry with ON DELETE SET NULL)
   - `tools_registry` with constraints (FK to registry server with ON DELETE CASCADE, matches current tool schema)
   - `tools_workload` with constraints (FK to workload server with ON DELETE CASCADE, matches current tool schema)

3. Create 6 virtual tables:
   - `registry_server_vector` (sqlite-vec)
   - `registry_tool_vectors` (sqlite-vec)
   - `registry_tool_fts` (FTS5)
   - `workload_server_vector` (sqlite-vec)
   - `workload_tool_vectors` (sqlite-vec)
   - `workload_tool_fts` (FTS5)

**Test**: Run migration on test database, verify tables created correctly.

### Phase 2: Operations Classes (Priority: P1)

**Goal**: Implement CRUD operations for each table.

#### Step 2.1: RegistryServerOps

**File**: `src/mcp_optimizer/db/registry_server_ops.py`

1. Implement core methods:
   - `create_server()` - Insert with unique constraint validation
   - `get_server_by_id()` - Simple SELECT
   - `get_server_by_url()` - SELECT WHERE remote=True AND url=?
   - `get_server_by_package()` - SELECT WHERE remote=False AND package=?
   - `find_matching_servers()` - Type-specific matching (returns list for duplicate detection)
   - `update_server()` - UPDATE with name change detection
   - `delete_server()` - DELETE (triggers workload SET NULL via FK)
   - `list_servers()` - SELECT with filters

2. Implement virtual table sync:
   - `sync_server_vectors()` - Update registry_server_vector

**Test**: Write unit tests in `tests/unit/test_registry_server_ops.py` covering all methods.

#### Step 2.2: WorkloadServerOps

**File**: `src/mcp_optimizer/db/workload_server_ops.py`

1. Implement core methods:
   - `create_server()` - Insert with autonomous/linked validation
   - `get_server_by_id()` - Simple SELECT
   - `get_server_by_workload_name()` - SELECT WHERE workload_name=?
   - `get_server_with_registry()` - LEFT JOIN to registry table
   - `list_servers_by_registry()` - SELECT WHERE registry_server_id=?
   - `update_server()` - UPDATE with re-matching on URL/package change
   - `remove_registry_relationship()` - SET NULL + recalculate embeddings
   - `delete_server()` - DELETE (no cascade to registry)
   - `list_servers()` - SELECT with filters

2. Implement virtual table sync:
   - `sync_server_vectors()` - Update workload_server_vector (autonomous only)

**Test**: Write unit tests in `tests/unit/test_workload_server_ops.py` covering all methods.

#### Step 2.3: RegistryToolOps

**File**: `src/mcp_optimizer/db/registry_tool_ops.py`

1. Implement core methods:
   - `create_tool()` - Insert with unique (server_id, name)
   - `get_tool_by_id()` - Simple SELECT
   - `get_tool_by_name()` - SELECT WHERE server_id=? AND name=?
   - `list_tools_by_server()` - SELECT WHERE server_id=?
   - `update_tool()` - UPDATE (name immutable)
   - `delete_tool()` - DELETE
   - `bulk_upsert_tools()` - INSERT OR REPLACE for efficiency

2. Implement virtual table sync:
   - `sync_tool_vectors()` - Update registry_tool_vectors
   - `sync_tool_fts()` - Update registry_tool_fts

**Test**: Write unit tests in `tests/unit/test_registry_tool_ops.py` covering all methods.

#### Step 2.4: WorkloadToolOps

**File**: `src/mcp_optimizer/db/workload_tool_ops.py`

1. Implement core methods:
   - `create_tool()` - Insert with details and details_embedding (matches current Tool schema)
   - `get_tool_by_id()` - Simple SELECT
   - `get_tool_by_name()` - SELECT WHERE mcpserver_id=? AND details.name=?
   - `list_tools_by_server()` - SELECT WHERE mcpserver_id=?
   - `update_tool()` - UPDATE details and details_embedding
   - `delete_tool()` - DELETE
   - `bulk_upsert_tools()` - INSERT OR REPLACE with server name context for embeddings

2. Implement virtual table sync:
   - `sync_tool_vectors()` - Update workload_tool_vectors (filter by RUNNING server status)
   - `sync_tool_fts()` - Update workload_tool_fts (filter by RUNNING server status)

**Note**: Embedding failures should update parent server status (STOPPED/ERROR), not individual tool availability. This matches existing behavior where tools are filtered by `server.status = McpStatus.RUNNING`.

**Test**: Write unit tests in `tests/unit/test_workload_tool_ops.py` covering all methods.

**Checkpoint**: Run `task test` to verify all unit tests pass.

### Phase 3: Ingestion Logic Updates (Priority: P2)

**Goal**: Update ingestion to use new ops classes and implement registry matching.

#### Step 3.1: Update Ingestion Methods

**File**: `src/mcp_optimizer/ingestion.py`

1. Replace `McpServerOps` with type-specific ops:
   - Determine if source is registry or workload
   - Use `RegistryServerOps` + `RegistryToolOps` for registry ingestion
   - Use `WorkloadServerOps` + `WorkloadToolOps` for workload ingestion

2. Implement registry matching for workloads:
   ```python
   # Find matching registry servers
   registry_ops = RegistryServerOps(self.db)
   matches = await registry_ops.find_matching_servers(
       url=server_url,
       package=server_package,
       remote=is_remote
   )

   # Handle duplicate detection
   if len(matches) > 1:
       raise DuplicateRegistryServersError(matches)

   # Establish relationship if single match
   registry_id = matches[0].id if matches else None
   registry_name = matches[0].name if matches else None
   ```

3. Update change detection methods:
   - Replace `_has_server_changed()` with `_has_registry_server_changed()` and `_has_workload_server_changed()`
   - Reuse `_compare_tools()` and `_tools_have_changed()` for both types

4. Update embedding context:
   - For workload tools: use `registry_server_name` if linked, else `workload_server.name`
   - Track registry server name in workload record

**Test**: Write integration tests in `tests/integration/test_registry_workload_matching.py` covering:
- Remote-to-remote matching
- Container-to-container matching
- No match scenario (autonomous workload)
- Duplicate registry detection

#### Step 3.2: Handle Registry Name Changes

**File**: `src/mcp_optimizer/ingestion.py` (or new `src/mcp_optimizer/registry_sync.py`)

1. Detect registry server name changes:
   ```python
   if old_registry.name != new_registry.name:
       # Trigger workload tool re-embedding
       workload_ops = WorkloadServerOps(self.db)
       affected = await workload_ops.list_servers_by_registry(registry_id)

       for workload in affected:
           await re_embed_workload_tools(
               workload_id=workload.id,
               new_registry_name=new_registry.name
           )
   ```

2. Implement re-embedding logic:
   - Fetch all workload tools for server
   - Batch calculate embeddings with new server name context
   - Update tools with new embeddings
   - Sync virtual tables

**Test**: Write integration tests in `tests/integration/test_relationship_management.py` covering:
- Registry name change triggers workload re-embedding
- Registry deletion breaks relationship and calculates autonomous embeddings
- Workload tool changes don't break registry relationship

### Phase 4: Server and CLI Updates (Priority: P3)

**Goal**: Update MCP server query methods and CLI admin commands.

#### Step 4.1: Update Server Query Methods

**File**: `src/mcp_optimizer/server.py`

1. Update tool search queries:
   - Query both `workload_tool_vectors` AND `registry_tool_vectors`
   - Filter workload tools by parent server status (RUNNING)
   - Merge results and rank by similarity

2. Update FTS queries:
   - Query both `workload_tool_fts` AND `registry_tool_fts`
   - Filter workload tools by parent server status (RUNNING)
   - Merge results

**Note**: Virtual tables should already filter by RUNNING status (done in sync methods), matching existing behavior.

**Test**: Test search returns results from both registry and workload tools.

#### Step 4.2: Update CLI Commands

**File**: `src/mcp_optimizer/cli.py`

1. Update list commands:
   - Separate `list-registry-servers` and `list-workload-servers`
   - Or add `--source` parameter to existing command

2. Update admin commands:
   - Delete commands should specify server type
   - Add commands to view relationships

**Test**: Manual testing of CLI commands.

### Phase 5: Migration and Data Verification (Priority: P3)

**Goal**: Run migration and verify data integrity.

#### Step 5.1: Run Migration

1. Backup existing database (if needed)
2. Run Alembic migration: `uv run alembic upgrade head`
3. Verify tables created: Check schema in SQLite

#### Step 5.2: Re-ingest Data

1. Trigger registry ingestion (from registry API)
2. Trigger workload ingestion (from ToolHive API)
3. Verify relationships established correctly
4. Check virtual tables populated

**Test**: Write migration test in `tests/integration/test_migration.py`:
- Verify old tables dropped
- Verify new tables created with correct schema
- Verify virtual tables created

### Phase 6: Cleanup (Priority: P3)

**Goal**: Remove deprecated code and finalize.

#### Step 6.1: Remove Old Code

1. Delete `src/mcp_optimizer/db/crud.py` (old ops file)
2. Remove deprecated models from `models.py` (`McpServer`, `Tool`)
3. Update imports across codebase

#### Step 6.2: Final Quality Gates

1. Run `task format` - Auto-format all code
2. Run `task lint` - Fix any linting issues
3. Run `task typecheck` - Fix any type errors
4. Run `task test` - Ensure all tests pass

**Checkpoint**: All quality gates must pass.

## Testing Strategy

### Unit Tests (per ops class)

Test each CRUD method in isolation:
- Happy path (successful create, read, update, delete)
- Error cases (not found, duplicate, foreign key violation)
- Edge cases (NULL handling, empty results)
- Virtual table sync operations

### Integration Tests

Test cross-component behavior:
- **Registry-Workload Matching**: Type-specific matching works correctly
- **Relationship Lifecycle**: Establish, maintain, break relationships
- **Registry Changes**: Name changes trigger workload re-embedding
- **Duplicate Detection**: Multiple registry matches rejected with error
- **Embedding Failure**: Tools marked unavailable, retry logic works

### Manual Testing

Test CLI and server operations:
- List servers (registry and workload separately)
- Search tools (returns results from both sources)
- Delete servers (verify no cascade to unrelated data)
- View relationships

## Common Pitfalls

1. **Forgetting virtual table sync**: Always sync after embedding changes
2. **Not handling NULL registry_server_id**: Autonomous workloads must have own embeddings
3. **Incorrect type matching**: Remote must match remote, container must match container
4. **Batch operations**: Always use `bulk_upsert_tools()` not iterative inserts
5. **Transaction scope**: Use single transaction for related operations (server + tools)
6. **Tool schema mismatch**: Ensure `details` field contains McpTool object, `details_embedding` stores vector (matches current Tool model)
7. **Status-based filtering**: Filter tools by parent server status (RUNNING), not individual tool availability flags

## Rollback Plan

If issues arise:
1. Keep old `crud.py` file until confident in new ops classes
2. Migration can be reverted by manually dropping new tables and recreating old
3. Data is re-ingestible from source systems (registry API + ToolHive)

## Success Criteria

- [ ] All 4 ops classes implemented with full test coverage
- [ ] Migration runs successfully without errors
- [ ] Data re-ingestion completes and relationships established
- [ ] All quality gates pass (format, lint, typecheck, test)
- [ ] MCP server search returns results from both registry and workload
- [ ] CLI commands work correctly for both server types
- [ ] Virtual tables populated and search works correctly
- [ ] No old code remains (crud.py, deprecated models removed)

## Next Steps

After completing implementation:
1. Run `/speckit.tasks` to generate actionable task list
2. Follow task list to implement changes
3. Commit changes with clear message describing refactoring
4. Create PR with link to this spec

## References

- **Specification**: `specs/002-split-db-tables/spec.md`
- **Data Model**: `specs/002-split-db-tables/data-model.md`
- **Research**: `specs/002-split-db-tables/research.md`
- **Contracts**: `specs/002-split-db-tables/contracts/ops-interfaces.md`

