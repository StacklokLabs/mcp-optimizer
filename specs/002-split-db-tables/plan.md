# Implementation Plan: Database Table Separation for Registry and Workload MCP Servers

**Branch**: `002-split-db-tables` | **Date**: 2025-10-21 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-split-db-tables/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Refactor the single unified MCP server storage structure into separate tables for registry servers (from catalog) and workload servers (running instances). This separation enables proper lifecycle management, simplifies cleanup operations, and supports optional relationships where workload servers can inherit metadata from matching registry servers. The system will maintain 4 core tables (mcpservers_registry, mcpservers_workload, tools_registry, tools_workload) with type-specific matching (remote-to-remote by URL, container-to-container by package) and automatic relationship management.

## Technical Context

**Language/Version**: Python 3.13+ (using uv for package management)
**Primary Dependencies**:
- SQLAlchemy (async, raw SQL with text())
- Pydantic (data validation and serialization)
- FastEmbed (embedding generation with batch operations)
- sqlite-vec (vector similarity search)
- structlog (structured logging)

**Storage**: SQLite with sqlite-vec extension for vector operations
- 4 main tables: mcpservers_registry, mcpservers_workload, tools_registry, tools_workload
- 6 virtual tables: registry_server_vector, registry_tool_vectors, registry_tool_fts, workload_server_vector, workload_tool_vectors, workload_tool_fts

**Testing**: pytest with pytest-asyncio and pytest-cov for coverage tracking

**Target Platform**: Linux server (Docker containers)

**Project Type**: Single Python project with CLI and MCP server components

**Performance Goals**:
- Server deletion: <100ms per operation
- Workload ingestion matching: <300ms including relationship establishment
- Embedding calculation: <200ms for fallback, <500ms for batch re-embedding
- Support 1000+ registry and workload servers without degradation

**Constraints**:
- Database agnostic (no SQLite-specific features except sqlite-vec for vectors)
- Zero data loss during migration
- Maintain backward compatibility with existing MCP server API
- Batch embedding operations (never single-item iteration)

**Scale/Scope**:
- Expected: 1000+ registry servers, 1000+ workload servers
- Multiple workload servers can reference same registry server
- Each server has multiple tools (10-100 typical)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### ✅ I. Tooling Consistency
- Using uv for package management (`uv add`, `uv run`)
- Using taskfile for automation (`task format`, `task lint`, `task typecheck`, `task test`)
- All Python execution via `uv run python`

### ✅ II. Centralized Configuration
- pyproject.toml is single source of truth
- Ruff, ty, pytest, alembic all configured in pyproject.toml
- No standalone config files for linters/formatters

### ✅ III. Modern Python Standards
- Native types (`list`, `dict` not `List`, `Dict`)
- Pydantic models for all data structures
- Type hints for all function signatures
- Python 3.13+ features

### ✅ IV. Database Agnosticism
- SQLAlchemy text() for all queries
- NO ORM features (no declarative base, no relationship() mappings)
- UUID primary keys via uuid.uuid4()
- Direct Connection.execute() usage
- No database-specific SQL dialects

### ✅ V. Quality Gates (NON-NEGOTIABLE)
- Sequential execution: format → lint → typecheck → test
- All gates must pass before commit
- task format auto-fixes what it can

### ✅ VI. Testing Coverage
- pytest framework
- Tests in tests/ directory
- test_*.py naming convention
- pytest-asyncio for async tests
- Coverage tracking enabled

### ✅ VII. CLI Design Patterns
- Commands group functionality with parameters
- Use logger (structlog) not click.echo
- Follow patterns in src/mcp_optimizer/cli.py

### ✅ VIII. Batch Operations
- FastEmbed processes lists of texts in single calls
- Database operations batched where possible
- No iterative single-item embedding calls

**Result**: ✅ ALL GATES PASS - No violations requiring justification

## Project Structure

### Documentation (this feature)

```
specs/002-split-db-tables/
├── spec.md              # Feature specification
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── ops-interfaces.md  # Database operations interface contracts
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/mcp_optimizer/
├── db/
│   ├── config.py                    # Database connection management (UNCHANGED)
│   ├── models.py                    # Pydantic models (MODIFY: split server/tool models)
│   ├── crud.py                      # Legacy ops (REMOVE after migration)
│   ├── registry_server_ops.py       # NEW: Registry server CRUD operations
│   ├── workload_server_ops.py       # NEW: Workload server CRUD operations
│   ├── registry_tool_ops.py         # NEW: Registry tool CRUD operations
│   └── workload_tool_ops.py         # NEW: Workload tool CRUD operations
├── ingestion.py                     # Ingestion logic (MODIFY: use new ops classes)
├── embeddings.py                    # Embedding generation (UNCHANGED - batch operations)
├── server.py                        # MCP server (MODIFY: update query methods)
└── cli.py                           # CLI commands (MODIFY: update admin commands)

migrations/versions/
└── 2025_08_18_0743-d2977d4c8c53_create_initial_tables.py  # MODIFY: drop old tables, create new

tests/
├── unit/
│   ├── test_registry_server_ops.py  # NEW: Registry server ops tests
│   ├── test_workload_server_ops.py  # NEW: Workload server ops tests
│   ├── test_registry_tool_ops.py    # NEW: Registry tool ops tests
│   ├── test_workload_tool_ops.py    # NEW: Workload tool ops tests
│   └── test_ingestion.py            # MODIFY: Update for new table structure
└── integration/
    ├── test_registry_workload_matching.py  # NEW: Test type-specific matching
    ├── test_relationship_management.py     # NEW: Test relationship lifecycle
    └── test_migration.py                   # NEW: Test data migration
```

**Structure Decision**: Single Python project with modular database operations. The refactoring creates 4 separate ops classes (one per table) to improve code clarity and maintainability while keeping the existing DatabaseConfig and ingestion flow largely intact. Virtual table management is distributed across the ops classes, with each responsible for syncing its corresponding vector/FTS tables.

## Complexity Tracking

*No constitutional violations - this section is empty*

