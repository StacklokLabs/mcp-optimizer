# Implementation Plan: Fix Remote Workload Matching

**Branch**: `001-fix-remote-workload-matching` | **Date**: 2025-10-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-fix-remote-workload-matching/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Fix the ingestion logic to match remote MCP workloads to registry entries by URL comparison instead of name matching. Currently, remote workloads with custom names are incorrectly ingested then immediately deleted because the name-based matching fails. The solution fetches detailed workload information from ToolHive API (`/api/v1beta/workloads/{name}`) to obtain the workload URL, then matches it against registry entry URLs. The `package` column in the McpServer database table will be updated to store URLs as the stable identifier for remote workloads.

## Technical Context

**Language/Version**: Python 3.13+
**Primary Dependencies**:
- SQLAlchemy (database interactions with text() for raw SQL)
- aiosqlite (async SQLite database)
- httpx (HTTP client for ToolHive API calls)
- structlog (structured logging)
- Pydantic (data validation)
- pytest, pytest-asyncio (testing framework)

**Storage**: SQLite database with McpServer and Tool tables, existing schema
**Testing**: pytest with async support (pytest-asyncio), pytest-cov for coverage tracking
**Target Platform**: Linux/macOS server, Docker containers, Kubernetes
**Project Type**: Single project (existing codebase modification)
**Performance Goals**: No degradation from current ingestion cycle performance (~same time bounds)
**Constraints**: Must maintain backward compatibility with container workload matching, must handle ToolHive API failures gracefully, must work with both Docker and Kubernetes runtime modes
**Scale/Scope**: Modifying existing ingestion service in `src/mcp_optimizer/ingestion.py`, adding ToolHive API client method, updating database operations, adding comprehensive tests

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### I. Tooling Consistency ✅ PASS
- Using uv for package management (no new dependencies needed)
- Task automation via taskfile (task format, lint, typecheck, test)
- Python execution via uv

### II. Centralized Configuration ✅ PASS
- pyproject.toml remains single source of truth
- No new tool configuration needed

### III. Modern Python Standards ✅ PASS
- Using native Python types (list, dict)
- Python 3.13+ compatible
- Pydantic models for Workload data validation (already exists)
- Type hints for new/modified functions

### IV. Database Agnosticism ✅ PASS
- Continue using SQLAlchemy text() for raw SQL
- No ORM features
- UUID primary keys maintained
- Connection.execute() with text() statements

### V. Quality Gates (NON-NEGOTIABLE) ✅ PASS
- All changes will go through: format → lint → typecheck → test
- Tests required before implementation

### VI. Testing Coverage ✅ PASS
- pytest with pytest-asyncio for async tests
- Coverage tracking with pytest-cov
- Tests in tests/ directory following test_*.py naming
- Tests explicitly requested by user for new functionality and regression prevention

### VII. CLI Design Patterns ✅ PASS
- No CLI changes required
- Internal ingestion service modification only

### VIII. Batch Operations ✅ PASS
- Existing batch processing for workloads maintained
- Single API call per remote workload to fetch details (acceptable overhead)

**GATE STATUS**: ✅ ALL GATES PASS - Proceed to Phase 0

## Project Structure

### Documentation (this feature)

```
specs/[###-feature]/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/mcp_optimizer/
├── ingestion.py                    # Main modification: IngestionService class
├── toolhive/
│   ├── toolhive_client.py         # Add get_workload_details() method
│   ├── k8s_client.py              # May need workload detail fetching for K8s
│   └── api_models/
│       ├── core.py                # Workload model (already exists)
│       └── v1.py                  # API response models
├── db/
│   ├── crud.py                    # Update find_server_by_package logic for URLs
│   └── models.py                  # McpServer model (package column semantics change)
└── config.py                      # Configuration (no changes expected)

tests/
├── test_ingestion.py              # Add tests for URL matching logic
├── test_toolhive_client.py        # Add tests for get_workload_details
└── test_*.py                      # Update existing tests if affected
```

**Structure Decision**: Single project structure. This is a modification to the existing ingestion service, not a new feature requiring separate modules. Key changes are localized to:
1. `src/mcp_optimizer/ingestion.py` - Core matching logic
2. `src/mcp_optimizer/toolhive/toolhive_client.py` - API client for workload details
3. Database CRUD operations for URL-based lookups
4. Comprehensive tests for new and existing functionality

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

No constitutional violations. All principles followed.

