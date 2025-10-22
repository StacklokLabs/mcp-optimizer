# Research: Database Table Separation

**Feature**: 002-split-db-tables
**Date**: 2025-10-21
**Status**: Complete

## Overview

This document consolidates research findings for splitting the unified MCP server storage into separate registry and workload tables with automatic relationship management and type-specific matching logic.

## Key Decisions

### 1. Table Structure Strategy

**Decision**: Create 4 separate tables (mcpservers_registry, mcpservers_workload, tools_registry, tools_workload) with optional foreign key relationship from workload to registry servers.

**Rationale**:
- Clean separation of concerns: registry servers represent catalog entries (stable metadata), workload servers represent running instances (dynamic state)
- Simplifies lifecycle management: deleting registry servers doesn't cascade to workload servers
- Enables multiple workload servers to reference same registry server without data duplication
- Supports independent evolution: registry and workload schemas can diverge if needed

**Alternatives Considered**:
- Single table with `source_type` discriminator: Rejected because it couples lifecycle management and makes relationship queries complex
- Inheritance/polymorphic tables: Rejected because it violates database agnosticism principle (SQLAlchemy ORM feature)

### 2. Relationship Management Strategy

**Decision**: Store optional `registry_server_id` (UUID) in workload table. When relationship exists, workload server reads description/embeddings from registry server. When relationship is NULL, workload server has own description/embeddings calculated from tool mean pooling.

**Rationale**:
- Simple schema: single nullable foreign key column
- Clear semantics: NULL means autonomous, non-NULL means inherited
- Efficient queries: can fetch registry metadata with single JOIN when needed
- No cascading deletes: when registry server deleted, SET NULL on workload side, triggering recalculation

**Alternatives Considered**:
- Junction table (many-to-many): Rejected because relationship is strictly one-to-many (workload → registry)
- Duplicate data in workload table: Rejected because it creates synchronization complexity and data staleness

### 3. Type-Specific Matching Logic

**Decision**: Match by URL for remote servers (workload.url = registry.url WHERE workload.remote=true AND registry.remote=true), match by package for container servers (workload.package = registry.package WHERE workload.remote=false AND registry.remote=false).

**Rationale**:
- Type safety: prevents meaningless matches (e.g., remote workload matching container registry)
- Aligns with real-world identity: remote servers are identified by URL, containers by package name
- Simplifies duplicate detection: each type has single unique constraint

**Alternatives Considered**:
- Match by name only: Rejected because names are not globally unique and can be arbitrary
- Match by both URL and package: Rejected because it's ambiguous when only one matches
- Complex scoring system: Rejected for simplicity; exact match on type-specific identifier is sufficient

### 4. Operations Class Organization

**Decision**: Create 4 separate ops classes (RegistryServerOps, WorkloadServerOps, RegistryToolOps, WorkloadToolOps) with consistent interface patterns. Each ops class manages its own table plus corresponding virtual tables (vector, FTS).

**Rationale**:
- Code clarity: each class has single responsibility (one table)
- Maintainability: changes to registry logic don't risk breaking workload logic
- Testability: can test each ops class in isolation with focused fixtures
- Follows existing pattern: current codebase has McpServerOps and ToolOps, we're extending the pattern

**Alternatives Considered**:
- Single unified ops class with method prefixes: Rejected because it creates a god class (800+ lines)
- Ops class per entity (2 classes: ServerOps with source parameter, ToolOps with source parameter): Rejected because conditional logic branches throughout methods reduce clarity

### 5. Virtual Table Sync Strategy

**Decision**: Each ops class maintains its own virtual tables. Sync operations triggered only by changes to source table:
- RegistryServerOps manages registry_server_vector
- RegistryToolOps manages registry_tool_vectors + registry_tool_fts
- WorkloadServerOps manages workload_server_vector
- WorkloadToolOps manages workload_tool_vectors + workload_tool_fts

**Rationale**:
- Consistency: follows existing pattern where virtual tables sync on source changes
- Performance: avoids unnecessary syncs (e.g., registry tool changes don't affect workload virtual tables)
- Isolation: failures in one virtual table don't cascade to others

**Alternatives Considered**:
- Central sync manager: Rejected because it creates coupling and single point of failure
- Manual sync calls from ingestion: Rejected because it's error-prone and violates encapsulation

### 6. Embedding Context for Tools

**Decision**: Workload tools use registry server name (if linked) or workload server name (if autonomous) as context when calculating embeddings. Store `registry_server_name` in workload server table for this purpose. When registry server name changes, trigger re-embedding of all linked workload tools.

**Rationale**:
- Semantic consistency: tools from same logical server should have consistent context regardless of workload instance
- Cache-friendly: registry server name changes are rare, so re-embedding overhead is acceptable
- Simplicity: single context source (server name) rather than complex composite context

**Alternatives Considered**:
- Always use workload name: Rejected because it breaks semantic consistency across workload instances
- Store full registry server metadata in workload table: Rejected because it duplicates data and creates synchronization complexity
- No context differentiation: Rejected because server name provides important semantic context for tool embeddings

### 7. Migration Strategy

**Decision**: Update existing migration file to drop old tables (mcpserver, tool) and create new tables (mcpservers_registry, mcpservers_workload, tools_registry, tools_workload). Since migration system is currently unused and data is refreshed continuously, no data preservation logic is needed.

**Rationale**:
- Simplicity: clean slate approach avoids complex data migration logic
- Current practice: data is refreshed from source systems (registry + ToolHive), not preserved across schema changes
- Migration file exists: can modify existing file rather than create new revision

**Alternatives Considered**:
- Preserve existing data: Rejected because data is ephemeral and re-ingested from source
- Create new migration revision: Rejected because it adds complexity (migration chain) when single file update suffices

### 8. Duplicate Registry Handling

**Decision**: When multiple registry servers match workload server (duplicate URLs or packages in registry), reject workload ingestion with clear error message indicating which duplicates exist and requiring resolution at registry source.

**Rationale**:
- Data integrity: prevents ambiguous relationships and silent arbitrary choices
- Fail-fast: surfaces data quality issues immediately rather than hiding them
- Clear responsibility: registry is source of truth and must maintain uniqueness

**Alternatives Considered**:
- Match first result: Rejected because it's non-deterministic (query order dependency)
- Match most recent: Rejected because it requires registry timestamps and may not reflect intent
- Create workload without relationship: Rejected because it silently loses potentially valuable metadata inheritance

## Technical Patterns

### Ops Class Interface Pattern

Each ops class will follow this consistent interface:

```python
class {Entity}Ops:
    def __init__(self, db: DatabaseConfig):
        self.db = db

    async def create_{entity}(...) -> {Entity}Model
    async def get_{entity}_by_id(id: str) -> {Entity}Model | None
    async def update_{entity}(...) -> {Entity}Model
    async def delete_{entity}(id: str) -> None
    async def list_{entities}(...) -> list[{Entity}Model]

    # Virtual table management
    async def sync_{entity}_vectors(...) -> None
    async def sync_{entity}_fts(...) -> None  # For tools only
```

### Relationship Query Pattern

When workload server needs registry metadata:

```sql
SELECT
    w.*,
    r.name as registry_name,
    r.description as registry_description,
    r.server_embedding as registry_embedding
FROM mcpservers_workload w
LEFT JOIN mcpservers_registry r ON w.registry_server_id = r.id
WHERE w.id = ?
```

If `registry_server_id` is NULL, use workload's own description/embedding.

### Type-Specific Matching Pattern

For remote servers:
```sql
SELECT r.id, r.name
FROM mcpservers_registry r
WHERE r.remote = true
  AND r.url = ?
```

For container servers:
```sql
SELECT r.id, r.name
FROM mcpservers_registry r
WHERE r.remote = false
  AND r.package = ?
```

Enforce at most 1 result; if multiple found, raise error with duplicate details.

## Implementation Considerations

### Change Detection

Current code has `_has_server_changed()` method. This will become:
- `_has_registry_server_changed()`: compares incoming registry data to existing registry record
- `_has_workload_server_changed()`: compares incoming workload data to existing workload record

Tool comparison methods `_compare_tools()` and `_tools_have_changed()` can remain unchanged and be reused for both registry and workload tools.

### Embedding Text Generation

Current `_create_server_text_to_embed()` can remain with slight modification:
- For registry servers: use registry server's own description
- For workload servers with registry link: include registry description + registry server name
- For autonomous workload servers: use tool mean pooling

### Error Handling

New error scenarios:
- `DuplicateRegistryServersError`: raised when multiple registry servers match workload
- `EmbeddingCalculationFailedError`: raised when embedding fails, allows workload creation with unavailable tools flag
- `RegistryServerNameChangedEvent`: triggers re-embedding of linked workload tools

### Testing Strategy

**Unit Tests** (per ops class):
- CRUD operations (create, read, update, delete)
- Virtual table sync operations
- Edge cases (NULL handling, empty results, constraint violations)

**Integration Tests**:
- Type-specific matching (remote-to-remote, container-to-container)
- Relationship lifecycle (establish, maintain through changes, break on delete)
- Registry name change triggering workload re-embedding
- Duplicate registry detection and error handling
- Embedding failure handling with tool unavailability marking

## Dependencies

No new external dependencies required. Using existing stack:
- SQLAlchemy (async + text())
- Pydantic (validation)
- FastEmbed (batch embeddings)
- sqlite-vec (vector operations)
- structlog (logging)

## Migration Path

1. Update migration file to drop old tables: `DROP TABLE IF EXISTS mcpserver;` `DROP TABLE IF EXISTS tool;`
2. Create new tables with appropriate constraints and indexes
3. Create 6 virtual tables for vector/FTS operations
4. Run migration to apply schema changes
5. Re-ingest data from source systems (registry API + ToolHive API)

## References

- Existing patterns: `src/mcp_optimizer/db/crud.py` (McpServerOps, ToolOps)
- Ingestion flow: `src/mcp_optimizer/ingestion.py`
- Embedding generation: `src/mcp_optimizer/embeddings.py`
- Database config: `src/mcp_optimizer/db/config.py`
- Models: `src/mcp_optimizer/db/models.py`

