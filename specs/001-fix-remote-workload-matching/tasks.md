# Tasks: Fix Remote Workload Matching

**Input**: Design documents from `/specs/001-fix-remote-workload-matching/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Tests explicitly requested by user for new functionality and regression prevention.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2)
- Include exact file paths in descriptions

## Path Conventions
- **Single project**: `src/`, `tests/` at repository root
- Paths shown below use single project structure per plan.md

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and verification - no new infrastructure needed

- [X] T001 Verify existing test infrastructure is set up (pytest, pytest-asyncio, pytest-cov in pyproject.toml)

**Checkpoint**: Setup verified - ready for foundational work

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core API client capability that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [X] T002 Add get_workload_details method to ToolhiveClient in src/mcp_optimizer/toolhive/toolhive_client.py
- [X] T003 [P] Add unit tests for get_workload_details method in tests/test_toolhive_client.py

**Checkpoint**: Foundation ready - user story implementation can now begin

---

## Phase 3: User Story 1 - Stable Remote Workload Ingestion (Priority: P1) 🎯 MVP

**Goal**: Fix the core bug where remote workloads with custom names are matched by URL instead of name, preventing false deletions

**Independent Test**: Deploy a remote MCP workload with custom name (different from registry name), run ingestion, verify workload persists and is not deleted

### Tests for User Story 1 ⚠️

**NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T004 [P] [US1] Add unit test for remote workload URL fetching in tests/test_ingestion.py
- [X] T005 [P] [US1] Add unit test for URL-based matching logic in tests/test_ingestion.py
- [X] T006 [P] [US1] Add unit test for error handling when workload details fetch fails in tests/test_ingestion.py
- [X] T007 [P] [US1] Add regression test for container workload matching (unchanged) in tests/test_ingestion.py
- [X] T008 [P] [US1] Add integration test for end-to-end remote workload ingestion with custom name in tests/test_ingestion.py

### Implementation for User Story 1

- [ ] T009 [US1] Update IngestionService._upsert_server to fetch workload details for remote workloads in src/mcp_optimizer/ingestion.py
- [ ] T010 [US1] Modify package column population logic for remote workloads (use URL) in src/mcp_optimizer/ingestion.py
- [ ] T011 [US1] Update workload identifier collection in ingest_workloads to use URLs for remotes in src/mcp_optimizer/ingestion.py
- [ ] T012 [US1] Update cleanup logic in _cleanup_removed_servers to match remote workloads by URL in src/mcp_optimizer/ingestion.py
- [ ] T013 [US1] Add structured logging for workload detail fetch operations in src/mcp_optimizer/ingestion.py
- [ ] T014 [US1] Add error handling for workload detail fetch failures (skip and continue) in src/mcp_optimizer/ingestion.py
- [ ] T015 [US1] Run task format to auto-format code
- [ ] T016 [US1] Run task lint to check for linting issues
- [ ] T017 [US1] Run task typecheck to verify type correctness
- [ ] T018 [US1] Run task test to ensure all tests pass

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently. Remote workloads with custom names should persist after ingestion.

---

## Phase 4: User Story 2 - Registry Server URL Updates (Priority: P2)

**Goal**: Ensure registry metadata updates (URL changes) are handled gracefully without disrupting running workloads

**Independent Test**: Update a registry entry's URL, run ingestion with remote workload using old URL, verify workload treated as custom deployment

### Tests for User Story 2 ⚠️

- [ ] T019 [P] [US2] Add unit test for registry URL mismatch scenario in tests/test_ingestion.py
- [ ] T020 [P] [US2] Add unit test for registry URL match scenario in tests/test_ingestion.py
- [ ] T021 [P] [US2] Add integration test for registry metadata update handling in tests/test_ingestion.py

### Implementation for User Story 2

- [ ] T022 [US2] Update _upsert_registry_server to use URLs for remote registry entries in src/mcp_optimizer/ingestion.py
- [ ] T023 [US2] Extract URL from RemoteServerMetadata when ingesting registry servers in src/mcp_optimizer/ingestion.py
- [ ] T024 [US2] Update registry server package column to store URLs for remotes in src/mcp_optimizer/ingestion.py
- [ ] T025 [US2] Run task format to auto-format code
- [ ] T026 [US2] Run task typecheck to verify type correctness
- [ ] T027 [US2] Run task test to ensure all tests pass

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently. Registry updates handled gracefully.

---

## Phase 5: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories and final validation

- [ ] T028 [P] Verify test coverage meets requirements (run pytest with coverage report)
- [ ] T029 [P] Update quickstart.md validation by running test scenarios from quickstart.md
- [ ] T030 [P] Review and update logging messages for clarity across ingestion service
- [ ] T031 Run final quality gate sequence (task format && task lint && task typecheck && task test)
- [ ] T032 Verify no regressions in container workload handling (run existing container tests)
- [ ] T033 Verify performance: measure ingestion cycle time (should match baseline)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User Story 1 can proceed after Foundational
  - User Story 2 can proceed after Foundational (can run in parallel with US1 if staffed)
- **Polish (Phase 5)**: Depends on all user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - Independent of US1 (different code paths)

### Within Each User Story

- Tests MUST be written and FAIL before implementation
- Implementation tasks follow order: fetch details → update matching → update cleanup → quality gates
- Quality gates (format, lint, typecheck, test) run after implementation tasks

### Parallel Opportunities

- **Foundational Phase**: T002 and T003 can run in parallel (different concerns - implementation vs tests)
- **User Story 1 Tests**: T004, T005, T006, T007, T008 can all be written in parallel (different test cases)
- **User Story 2 Tests**: T019, T020, T021 can all be written in parallel
- **Polish Phase**: T028, T029, T030 can run in parallel (independent validation tasks)
- **User Stories**: US1 and US2 can be implemented in parallel by different developers (different code sections)

---

## Parallel Example: User Story 1 Tests

```bash
# Launch all tests for User Story 1 together:
Task: "Add unit test for remote workload URL fetching in tests/test_ingestion.py"
Task: "Add unit test for URL-based matching logic in tests/test_ingestion.py"
Task: "Add unit test for error handling when workload details fetch fails in tests/test_ingestion.py"
Task: "Add regression test for container workload matching in tests/test_ingestion.py"
Task: "Add integration test for end-to-end remote workload ingestion in tests/test_ingestion.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (get_workload_details method)
3. Complete Phase 3: User Story 1 (core URL-based matching)
4. **STOP and VALIDATE**: Test User Story 1 independently
   - Deploy remote workload with custom name
   - Run ingestion
   - Verify workload persists
   - Verify no false deletions
5. If validation passes, MVP is ready

### Incremental Delivery

1. Complete Setup + Foundational → API client ready
2. Add User Story 1 → Test independently → Core fix working (MVP!)
3. Add User Story 2 → Test independently → Registry updates handled
4. Polish phase → Final validation and optimization
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1 (core URL matching)
   - Developer B: User Story 2 (registry updates) - can start in parallel
3. Both stories complete and integrate independently
4. Team collaborates on Polish phase

---

## Notes

- [P] tasks = different files or independent concerns, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Tests must fail before implementing (TDD approach)
- Quality gates run after each user story implementation
- No new dependencies needed (all tools already in place)
- Package column semantics change is transparent (no schema migration)
- Focus: Minimal changes for maximum impact (bug fix, not feature)

## Task Count Summary

- **Total Tasks**: 33
- **Setup Phase**: 1 task
- **Foundational Phase**: 2 tasks
- **User Story 1 (P1)**: 15 tasks (5 tests + 10 implementation/validation)
- **User Story 2 (P2)**: 9 tasks (3 tests + 6 implementation/validation)
- **Polish Phase**: 6 tasks
- **Parallel Opportunities**: 14 tasks can run in parallel

## Suggested MVP Scope

**Minimum Viable Product**: User Story 1 only
- Fixes the core bug (false deletions)
- Delivers immediate value (remote workloads work correctly)
- Can be deployed and validated independently
- Total: 18 tasks (Setup + Foundational + US1)
