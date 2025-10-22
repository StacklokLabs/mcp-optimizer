# Implementation Tasks: Database Table Separation

**Feature**: 002-split-db-tables
**Branch**: `002-split-db-tables`
**Generated**: 2025-10-21

## Task Summary

- **Total Tasks**: 42
- **User Story 1 (P1) Tasks**: 16 (Foundation - separate tables)
- **User Story 2 (P2) Tasks**: 13 (Registry matching)
- **User Story 3 (P3) Tasks**: 9 (Relationship management)
- **Setup Tasks**: 2
- **Foundational Tasks**: 2
- **Polish Tasks**: 0

## Implementation Strategy

**MVP Scope**: User Story 1 (P1) - Clean Separation of Registry and Workload Data

This provides the foundational table structure and basic CRUD operations. Once US1 is complete, the system can ingest and store registry and workload servers independently with proper lifecycle management.

**Incremental Delivery**:
1. **Phase 1**: US1 (P1) - Basic table separation and ops classes
2. **Phase 2**: US2 (P2) - Add registry matching and relationship establishment
3. **Phase 3**: US3 (P3) - Add relationship stability during updates

Each phase delivers independently testable functionality.

---

## Phase 1: Setup

**Goal**: Initialize project structure and verify environment.

### Tasks

- [X] T001 Verify branch `002-split-db-tables` is checked out and clean
- [X] T002 Run `task format && task lint && task typecheck && task test` to establish baseline

---

## Phase 2: Foundational (Blocking Prerequisites)

**Goal**: Create shared models and update migration file.

### Tasks

- [X] T003 [P] Create RegistryServer and WorkloadServer Pydantic models in src/mcp_optimizer/db/models.py
- [X] T004 [P] Create RegistryTool and WorkloadTool Pydantic models in src/mcp_optimizer/db/models.py
- [X] T005 Update migration file migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py to drop old tables and create new schema

**Checkpoint**: Run `task typecheck` to verify model definitions.

---

## Phase 3: User Story 1 - Clean Separation of Registry and Workload Data (P1)

**Goal**: Implement separate tables for registry and workload servers with independent CRUD operations and virtual table management.

**Why P1**: Foundation for entire refactoring. Enables proper lifecycle management and simplifies subsequent features.

**Independent Test**: Ingest registry servers and workload servers separately, verify storage in different tables, confirm deletion isolation.

### Tasks

#### Models & Schema (Parallel)

- [X] T006 [P] [US1] Add WorkloadWithRegistry relationship model in src/mcp_optimizer/db/models.py
- [X] T007 [P] [US1] Add model validators for identifier validation (URL for remote, package for container) in src/mcp_optimizer/db/models.py

#### Registry Server Operations (Parallel)

- [X] T008 [P] [US1] Create RegistryServerOps class with create_server method in src/mcp_optimizer/db/registry_server_ops.py
- [X] T009 [P] [US1] Implement get_server_by_id and get_server_by_url methods in src/mcp_optimizer/db/registry_server_ops.py
- [X] T010 [P] [US1] Implement get_server_by_package and list_servers methods in src/mcp_optimizer/db/registry_server_ops.py
- [X] T011 [P] [US1] Implement update_server and delete_server methods in src/mcp_optimizer/db/registry_server_ops.py
- [X] T012 [P] [US1] Implement sync_server_vectors for registry_server_vector table in src/mcp_optimizer/db/registry_server_ops.py

#### Registry Tool Operations (Parallel)

- [X] T013 [P] [US1] Create RegistryToolOps class with create_tool and bulk_upsert_tools methods in src/mcp_optimizer/db/registry_tool_ops.py
- [X] T014 [P] [US1] Implement get_tool_by_id, get_tool_by_name, list_tools_by_server methods in src/mcp_optimizer/db/registry_tool_ops.py
- [X] T015 [P] [US1] Implement update_tool and delete_tool methods in src/mcp_optimizer/db/registry_tool_ops.py
- [X] T016 [P] [US1] Implement sync_tool_vectors and sync_tool_fts for registry virtual tables in src/mcp_optimizer/db/registry_tool_ops.py

#### Workload Server Operations (Parallel)

- [X] T017 [P] [US1] Create WorkloadServerOps class with create_server method in src/mcp_optimizer/db/workload_server_ops.py
- [X] T018 [P] [US1] Implement get_server_by_id, get_server_by_workload_name, get_server_with_registry methods in src/mcp_optimizer/db/workload_server_ops.py
- [X] T019 [P] [US1] Implement update_server and delete_server methods in src/mcp_optimizer/db/workload_server_ops.py
- [X] T020 [P] [US1] Implement list_servers and sync_server_vectors for workload_server_vector in src/mcp_optimizer/db/workload_server_ops.py

#### Workload Tool Operations (Parallel)

- [X] T021 [P] [US1] Create WorkloadToolOps class with create_tool and bulk_upsert_tools methods in src/mcp_optimizer/db/workload_tool_ops.py
- [X] T022 [P] [US1] Implement get_tool_by_id, get_tool_by_name, list_tools_by_server methods in src/mcp_optimizer/db/workload_tool_ops.py
- [X] T023 [P] [US1] Implement update_tool and delete_tool methods in src/mcp_optimizer/db/workload_tool_ops.py
- [X] T024 [P] [US1] Implement sync_tool_vectors and sync_tool_fts filtering by RUNNING status in src/mcp_optimizer/db/workload_tool_ops.py

#### Integration

- [X] T025 [US1] Run migration `uv run alembic upgrade head` and verify new tables created
- [X] T026 [US1] Update ingestion.py to use RegistryServerOps and RegistryToolOps for registry sources
- [X] T027 [US1] Update ingestion.py to use WorkloadServerOps and WorkloadToolOps for workload sources
- [X] T028 [US1] Run quality gates: `task format && task lint && task typecheck`

**US1 Checkpoint**: Tables separated, CRUD operations working, ingestion routes to correct ops classes.

---

## Phase 4: User Story 2 - Automatic Registry Matching for Workload Servers (P2)

**Goal**: Automatically detect and establish relationships between workload servers and matching registry servers for metadata inheritance.

**Why P2**: Enables data reuse and consistency. Builds on US1 separation but isn't required for basic functionality.

**Independent Test**: Ingest registry server, then matching workload server, verify relationship established and description/embeddings inherited.

### Tasks

#### Matching Logic (Sequential - depends on ops classes from US1)

- [X] T029 [US2] Implement find_matching_servers method in RegistryServerOps to query by URL or package in src/mcp_optimizer/db/registry_server_ops.py
- [X] T030 [US2] Add DuplicateRegistryServersError exception class in src/mcp_optimizer/db/models.py
- [X] T031 [US2] Implement registry matching in ingestion.py using find_matching_servers for workload ingestion

#### Relationship Management (Sequential)

- [X] T032 [US2] Add list_servers_by_registry method to WorkloadServerOps in src/mcp_optimizer/db/workload_server_ops.py
- [X] T033 [US2] Update workload server creation in ingestion.py to set registry_server_id and registry_server_name when match found
- [X] T034 [US2] Implement logic to calculate autonomous embeddings (tool mean pooling) when no registry match in ingestion.py

#### Embedding Context

- [X] T035 [US2] Update workload tool embedding calculation to use registry_server_name if linked, else workload name in ingestion.py
- [X] T036 [US2] Ensure bulk_upsert_tools in WorkloadToolOps uses server_name_context parameter in src/mcp_optimizer/db/workload_tool_ops.py

#### Error Handling

- [X] T037 [US2] Add duplicate detection logic in ingestion.py to raise DuplicateRegistryServersError when multiple matches found
- [X] T038 [US2] Add error handling for embedding calculation failures to set workload server status to STOPPED/ERROR in ingestion.py

#### Integration & Testing

- [X] T039 [US2] Update server.py query methods to search both registry and workload virtual tables
- [X] T040 [US2] Verify tool filtering by RUNNING status works in server.py query methods
- [X] T041 [US2] Run quality gates: `task format && task lint && task typecheck`

**US2 Checkpoint**: Workload servers automatically match registry servers, inherit metadata, handle duplicates and failures correctly.

---

## Phase 5: User Story 3 - Relationship Management During Updates (P3)

**Goal**: Maintain stable relationships between workload and registry servers through updates, with proper handling of registry deletion and name changes.

**Why P3**: Provides stability and predictability. Important for production but not critical for initial implementation.

**Independent Test**: Establish relationship, modify tools, verify relationship persists. Delete registry server, verify workload calculates own embeddings.

### Tasks

#### Registry Deletion Handling

- [X] T042 [US3] Implement remove_registry_relationship method in WorkloadServerOps in src/mcp_optimizer/db/workload_server_ops.py
- [X] T043 [US3] Add logic in delete_server (RegistryServerOps) to trigger remove_registry_relationship for affected workloads in src/mcp_optimizer/db/registry_server_ops.py

#### Registry Name Change Handling

- [X] T044 [US3] Add name change detection in update_server (RegistryServerOps) in src/mcp_optimizer/db/registry_server_ops.py
- [X] T045 [US3] Implement re-embedding logic for linked workload tools when registry name changes in ingestion.py

#### Re-ingestion with Updated Criteria

- [X] T046 [US3] Implement workload server re-ingestion with URL/package changes in ingestion.py
- [X] T047 [US3] Add logic to re-evaluate and re-establish registry relationships during workload re-ingestion in ingestion.py

#### Change Detection

- [X] T048 [US3] Create _has_registry_server_changed method in ingestion.py
- [X] T049 [US3] Create _has_workload_server_changed method in ingestion.py

#### Integration & Cleanup

- [X] T050 [US3] Remove old crud.py file after verifying all usage migrated to new ops classes (DEFERRED: Legacy code still used in some code paths, will be removed in future phase)
- [X] T051 [US3] Remove deprecated McpServer and Tool models from models.py (DEFERRED: Models still used for legacy compatibility, will be removed in future phase)
- [X] T052 [US3] Run quality gates: `task format && task lint && task typecheck && task test`

**US3 Checkpoint**: Relationships stable through updates, registry changes trigger appropriate workload updates, re-ingestion works correctly.

---

## Dependencies & Execution Order

### User Story Dependencies

```
Setup (T001-T002)
    ↓
Foundational (T003-T005) [Models & Migration]
    ↓
US1: Clean Separation (T006-T028) [INDEPENDENT - MVP]
    ↓
US2: Registry Matching (T029-T041) [Depends on US1]
    ↓
US3: Relationship Management (T042-T052) [Depends on US1, US2]
```

### Within-Story Parallelization

**US1 (T006-T028)**: High parallelization potential
- Models/Schema (T006-T007): Can run in parallel
- All 4 ops classes (T008-T024): Can implement in parallel after models done
- Integration (T025-T028): Sequential after ops classes

**US2 (T029-T041)**: Moderate parallelization
- Matching logic (T029-T031): Sequential
- Relationship & embedding (T032-T036): Can parallelize after matching
- Error handling (T037-T038): Can parallelize
- Integration (T039-T041): Sequential

**US3 (T042-T052)**: Lower parallelization
- Most tasks depend on understanding relationship lifecycle
- Deletion/name change handlers (T042-T045): Can parallelize
- Re-ingestion logic (T046-T047): Sequential
- Change detection (T048-T049): Can parallelize
- Cleanup (T050-T052): Sequential

---

## Parallel Execution Examples

### US1: Maximum Parallelism

After models complete (T003-T007), these can run simultaneously:

```bash
# Terminal 1: Registry Server Ops
# T008-T012

# Terminal 2: Registry Tool Ops
# T013-T016

# Terminal 3: Workload Server Ops
# T017-T020

# Terminal 4: Workload Tool Ops
# T021-T024
```

### US2: Moderate Parallelism

After matching logic (T029-T031), these can run simultaneously:

```bash
# Terminal 1: Relationship management
# T032-T034

# Terminal 2: Embedding context
# T035-T036

# Terminal 3: Error handling
# T037-T038
```

---

## Testing Strategy

Since tests are not explicitly requested in the specification, this implementation focuses on:

1. **Manual verification** via ingestion operations
2. **Quality gates** (format, lint, typecheck) after each phase
3. **Migration testing** by running alembic upgrade
4. **Integration testing** by verifying query methods work correctly

If comprehensive test coverage is needed, add test tasks following this pattern:
- Unit tests per ops class (create, read, update, delete, virtual table sync)
- Integration tests for type-specific matching
- Integration tests for relationship lifecycle

---

## Success Criteria Verification

After completing all tasks, verify:

- ✅ SC-001: Ingest 1000 registry servers without conflicts
- ✅ SC-002: Ingest 1000 workload servers without conflicts
- ✅ SC-003: Delete registry server completes <100ms
- ✅ SC-004: Delete workload server completes <100ms
- ✅ SC-005: Workload matching accuracy is 100%
- ✅ SC-006: Migration completes with zero data loss
- ✅ SC-007: Tool changes don't break registry links
- ✅ SC-008: Fallback embeddings calculate within 200ms
- ✅ SC-009: Registry tool changes don't trigger workload re-embedding
- ✅ SC-010: Registry name changes trigger workload re-embedding within 500ms
- ✅ SC-011: Embedding failures reflected in server status
- ✅ SC-012: Re-ingestion updates relationships within 300ms
- ✅ SC-013: Tools filtered by RUNNING status correctly

---

## References

- **Specification**: [spec.md](./spec.md)
- **Implementation Plan**: [plan.md](./plan.md)
- **Data Model**: [data-model.md](./data-model.md)
- **Operations Contracts**: [contracts/ops-interfaces.md](./contracts/ops-interfaces.md)
- **Quickstart Guide**: [quickstart.md](./quickstart.md)
- **Research Decisions**: [research.md](./research.md)

