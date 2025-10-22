# Implementation Plan: Token Savings Metrics for find_tool

**Branch**: `002-token-savings-metrics` | **Date**: 2025-10-20 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/002-token-savings-metrics/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Add token savings metrics to the find_tool endpoint to demonstrate efficiency gains from intelligent tool filtering. Token counts will be calculated during tool ingestion using tiktoken on the tool details field and persisted in the database. At query time, find_tool will sum the pre-calculated token counts for all running server tools (baseline) and the filtered subset (returned tools), then include the savings (difference) in the response.

## Technical Context

**Language/Version**: Python 3.13+
**Primary Dependencies**: tiktoken (for token counting), FastAPI/FastMCP (existing server), Pydantic (existing data models), SQLAlchemy (existing database layer)
**Storage**: SQLite with sqlite-vec extension (ephemeral, recreated on each server start)
**Testing**: pytest with pytest-asyncio and pytest-cov
**Target Platform**: Linux/macOS server (where mcp-optimizer runs)
**Project Type**: Single project (server application)
**Performance Goals**: Token calculation during ingestion should add minimal overhead to startup; query-time aggregation should add <10ms to find_tool response time
**Constraints**: Database is ephemeral (recreated each server start); must use tiktoken for LLM-compatible token counting; changes must not break existing find_tool API clients
**Scale/Scope**: Typical deployment has 5-20 MCP servers with 50-200 tools total; find_tool handles ~10-100 queries per session

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

### Initial Check (Pre-Phase 0)

| Principle | Compliant | Notes |
|-----------|-----------|-------|
| I. Tooling Consistency | ✅ | Will use `uv add tiktoken` for dependency, `task format/lint/typecheck/test` for quality gates |
| II. Centralized Configuration | ✅ | pyproject.toml will be updated with tiktoken dependency |
| III. Modern Python Standards | ✅ | Using native types (`list`, `dict`), Pydantic models for data validation, type hints |
| IV. Database Agnosticism | ✅ | Using SQLAlchemy `text()` for raw SQL, no ORM features, UUID primary keys |
| V. Quality Gates | ✅ | All changes will pass through format → lint → typecheck → test sequence |
| VI. Testing Coverage | ✅ | Will add tests for token calculation logic, ingestion changes, find_tool response |
| VII. CLI Design Patterns | ✅ | No CLI changes required for this feature |
| VIII. Batch Operations | ✅ | Token calculation during ingestion happens once per tool; query-time is simple aggregation |

**Status**: ✅ **PASSED** - No constitutional violations. Feature aligns with all principles.

**Rationale**: This feature extends existing ingestion and query logic without introducing new architectural patterns or violating established conventions. Token counting is a straightforward addition to the existing tool ingestion flow.

## Project Structure

### Documentation (this feature)

```
specs/002-token-savings-metrics/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
│   └── find_tool_response.json  # Extended response schema with token metrics
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```
src/mcp_optimizer/
├── server.py                    # MODIFY: Update find_tool to include token savings
├── db/
│   ├── models.py               # MODIFY: Add token_count field to Tool model
│   ├── crud.py                 # MODIFY: Update create_tool to accept token_count
│   └── config.py               # No changes
├── ingestion.py                # MODIFY: Calculate token count during tool ingestion
├── token_counter.py            # NEW: Token counting utility using tiktoken
└── embeddings.py               # No changes

migrations/versions/
└── 2025_08_18_0743-d2977d4c8c53_create_initial_tables.py  # MODIFY: Add token_count column to tool table schema

tests/
├── unit/
│   ├── test_token_counter.py   # NEW: Tests for token counting logic
│   ├── test_ingestion.py       # MODIFY: Add tests for token count calculation
│   ├── test_crud.py            # MODIFY: Add tests for token_count field
│   └── test_server.py          # MODIFY: Add tests for find_tool token savings
└── integration/
    └── test_end_to_end.py      # MODIFY: Add integration test for full flow
```

**Structure Decision**: Single project structure. This is a server application with all code under `src/mcp_optimizer/`. The database is ephemeral and recreated on each server start, so no migration is needed. Instead, the existing table creation script in `migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py` will be modified to include the `token_count` column. The feature requires:
1. New token counting module (`token_counter.py`)
2. Modification to existing table creation script to add `token_count` column to `tool` table
3. Modifications to existing ingestion, CRUD, and server modules
4. Comprehensive test coverage across unit and integration levels

## Complexity Tracking

*Fill ONLY if Constitution Check has violations that must be justified*

**Status**: N/A - No constitutional violations

## Constitution Check (Post-Phase 1)

*Re-evaluation after design and planning completion*

| Principle | Compliant | Design Notes |
|-----------|-----------|--------------|
| I. Tooling Consistency | ✅ | tiktoken added via `uv add tiktoken`, all quality gates apply |
| II. Centralized Configuration | ✅ | tiktoken dependency added to pyproject.toml |
| III. Modern Python Standards | ✅ | Native types used, Pydantic models for all entities (TokenMetrics, FindToolResponse) |
| IV. Database Agnosticism | ✅ | Using SQLAlchemy `text()` for queries, INTEGER column is database-agnostic |
| V. Quality Gates | ✅ | All changes will pass format → lint → typecheck → test |
| VI. Testing Coverage | ✅ | Comprehensive tests planned: test_token_counter.py, test_ingestion.py, test_crud.py, test_server.py |
| VII. CLI Design Patterns | ✅ | No CLI changes |
| VIII. Batch Operations | ✅ | Token calculation happens once per tool during ingestion |

**Status**: ✅ **PASSED** - Design maintains constitutional compliance

**Rationale**: The detailed design confirms no new architectural patterns or violations. The feature extends existing components following established patterns (new utility module, Pydantic models, SQLAlchemy queries, standard ingestion flow).

