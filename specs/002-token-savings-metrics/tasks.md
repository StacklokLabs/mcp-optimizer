# Tasks: Token Savings Metrics for find_tool

**Input**: Design documents from `/specs/002-token-savings-metrics/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Organization**: Tasks are grouped by implementation phase to enable systematic delivery of the single user story.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[US1]**: User Story 1 - View Token Savings from Tool Filtering
- Include exact file paths in descriptions

## Path Conventions
- Single project structure: `src/mcp_optimizer/`, `tests/` at repository root
- Database migrations: `migrations/versions/`

---

## Phase 1: Setup (Dependencies & Infrastructure)

**Purpose**: Install required dependencies and prepare development environment

- [x] T001 Install tiktoken dependency using `uv add tiktoken`
- [x] T002 Verify tiktoken installation: `uv run python -c "import tiktoken; print(tiktoken.__version__)"`
- [x] T003 [P] Run baseline tests to ensure existing functionality: `task test`

**Checkpoint**: Development environment ready with tiktoken available

---

## Phase 2: Foundational (Core Components)

**Purpose**: Implement foundational components needed for token counting

**⚠️ CRITICAL**: These components must be complete before modifying ingestion or find_tool

### Database Schema Updates

- [x] T004 Add `token_count INTEGER NOT NULL DEFAULT 0` column to tool table in `migrations/versions/2025_08_18_0743-d2977d4c8c53_create_initial_tables.py`

### Token Counter Utility

- [x] T005 [P] Create TokenCounter class in `src/mcp_optimizer/token_counter.py` with `__init__(encoding_name="cl100k_base")`, `count_tokens(text: str) -> int`, and `count_tool_tokens(tool: McpTool) -> int` methods

### Data Models

- [x] T006 [P] Add `token_count: int = Field(default=0, ge=0)` field to Tool model in `src/mcp_optimizer/db/models.py`
- [x] T007 [P] Add `token_count: int | None = Field(default=None, ge=0)` to ToolUpdateDetails model in `src/mcp_optimizer/db/models.py`
- [x] T008 [P] Create TokenMetrics model in `src/mcp_optimizer/db/models.py` with fields: baseline_tokens, returned_tokens, tokens_saved, savings_percentage and validation logic

**Checkpoint**: Foundation ready - database schema updated, token counter utility available, models extended

---

## Phase 3: User Story 1 - View Token Savings from Tool Filtering (Priority: P1) 🎯 MVP

**Goal**: Users can see token savings metrics in find_tool response showing efficiency gains from tool filtering

**Independent Test**: Call find_tool with any query and verify response includes token_metrics with baseline_tokens, returned_tokens, tokens_saved, and savings_percentage fields

### Unit Tests for Core Components

- [x] T009 [P] [US1] Create test_token_counter.py in `tests/unit/` with tests for count_tokens() with simple text, count_tool_tokens() with McpTool object, and encoding initialization
- [x] T010 [P] [US1] Add test cases for TokenMetrics validation in `tests/unit/test_models.py`: valid metrics, tokens_saved calculation validation, savings_percentage calculation validation, zero baseline handling
- [x] T011 [P] [US1] Add test for Tool model with token_count field in `tests/unit/test_models.py`: non-negative validation, default value of 0

### CRUD Operations Updates

- [x] T012 [US1] Update create_tool method signature in `src/mcp_optimizer/db/crud.py` to accept `token_count: int` parameter
- [x] T013 [US1] Add token_count to INSERT statement and params in create_tool method in `src/mcp_optimizer/db/crud.py`
- [x] T014 [US1] Add sum_token_counts_for_running_servers method to ToolOps class in `src/mcp_optimizer/db/crud.py` that sums token_count for all tools from running servers, filtering by allowed_groups if provided
- [x] T015 [P] [US1] Add unit tests for create_tool with token_count parameter in `tests/unit/test_crud.py`
- [x] T016 [P] [US1] Add unit tests for sum_token_counts_for_running_servers in `tests/unit/test_crud.py` covering zero tools, single server, multiple servers, and group filtering

### Ingestion Updates

- [x] T017 [US1] Initialize TokenCounter instance in IngestionService.__init__ in `src/mcp_optimizer/ingestion.py`
- [x] T018 [US1] Update _sync_tools method in `src/mcp_optimizer/ingestion.py` to calculate token counts for each tool using token_counter.count_tool_tokens()
- [x] T019 [US1] Pass token_count to tool_ops.create_tool calls in _sync_tools method in `src/mcp_optimizer/ingestion.py`
- [x] T020 [US1] Update _ingest_registry_tools method in `src/mcp_optimizer/ingestion.py` to calculate and store token counts for registry tools
- [x] T021 [P] [US1] Add unit tests for token count calculation during ingestion in `tests/unit/test_ingestion.py` covering _sync_tools with token calculation and _ingest_registry_tools with token calculation

### find_tool Endpoint Updates

- [x] T022 [US1] Calculate baseline_tokens in find_tool using tool_ops.sum_token_counts_for_running_servers() in `src/mcp_optimizer/server.py`
- [x] T023 [US1] Calculate returned_tokens by summing token_count from filtered tools in find_tool in `src/mcp_optimizer/server.py`
- [x] T024 [US1] Calculate tokens_saved and savings_percentage in find_tool in `src/mcp_optimizer/server.py`
- [x] T025 [US1] Update find_tool return value to include token_metrics dictionary with baseline_tokens, returned_tokens, tokens_saved, savings_percentage in `src/mcp_optimizer/server.py`
- [x] T026 [US1] Handle edge case: zero baseline_tokens (no running servers) returns savings_percentage of 0.0 in `src/mcp_optimizer/server.py`
- [x] T027 [US1] Handle edge case: all tools returned (no filtering) shows zero or minimal savings in `src/mcp_optimizer/server.py`

### Integration Tests

- [ ] T028 [US1] Add integration test for find_tool response format in `tests/integration/test_server.py`: verify token_metrics present in response (SKIPPED - not implementing)
- [ ] T029 [US1] Add integration test for token savings calculation in `tests/integration/test_server.py`: with multiple servers, with filtered results, with no servers (zero baseline), with all tools returned (minimal savings) (SKIPPED - not implementing)
- [ ] T030 [US1] Add integration test for end-to-end flow in `tests/integration/test_end_to_end.py`: ingest tools with token counts, query find_tool, verify token metrics accuracy (SKIPPED - not implementing)

**Checkpoint**: User Story 1 complete - find_tool returns token savings metrics for all queries

---

## Phase 4: Polish & Cross-Cutting Concerns

**Purpose**: Quality improvements and documentation

- [x] T031 [P] Run `task format` to auto-format all modified code
- [x] T032 Run `task lint` and fix any linting issues
- [x] T033 Run `task typecheck` and fix any type errors
- [x] T034 Run `task test` and ensure all tests pass with adequate coverage
- [ ] T035 [P] Verify quickstart.md examples work as documented
- [ ] T036 [P] Update CLAUDE.md if any development patterns changed
- [ ] T037 Performance validation: Measure token calculation overhead during ingestion (should be <50ms for 200 tools)
- [ ] T038 Performance validation: Measure find_tool overhead for token metrics calculation (should be <10ms)

**Checkpoint**: All quality gates passed, feature ready for review

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup (T001-T003) completion - BLOCKS all user story work
- **User Story 1 (Phase 3)**: Depends on Foundational (T004-T008) completion
- **Polish (Phase 4)**: Depends on User Story 1 completion

### Within User Story 1

- **Unit tests (T009-T011)**: Can run in parallel after foundational phase, must FAIL before implementation
- **CRUD updates (T012-T016)**: T012-T014 must complete before T015-T016 (tests need implementation)
- **Ingestion updates (T017-T021)**: T017-T020 must complete before T021 (tests need implementation)
- **Endpoint updates (T022-T027)**: Must complete in order (each builds on previous)
- **Integration tests (T028-T030)**: Must wait for all implementation tasks (T012-T027) to complete

### Parallel Opportunities

**Phase 1 (Setup)**:
- T002 and T003 can run in parallel after T001

**Phase 2 (Foundational)**:
- T005, T006, T007, T008 can all run in parallel after T004 (different files)

**Phase 3 (User Story 1)**:
- T009, T010, T011 can run in parallel (different test files)
- T015, T016 can run in parallel (both are tests, different test cases)
- T021 is independent and can be done in parallel with T015-T016
- T028, T029, T030 can start together once implementation is done

**Phase 4 (Polish)**:
- T031, T035, T036 can run in parallel (different tasks/files)

---

## Parallel Example: Foundational Phase

```bash
# After T004 (database schema) completes, launch these together:
Task T005: "Create TokenCounter class in src/mcp_optimizer/token_counter.py"
Task T006: "Add token_count field to Tool model in src/mcp_optimizer/db/models.py"
Task T007: "Add token_count to ToolUpdateDetails in src/mcp_optimizer/db/models.py"
Task T008: "Create TokenMetrics model in src/mcp_optimizer/db/models.py"
```

## Parallel Example: User Story 1 Unit Tests

```bash
# After foundational components (T004-T008) complete, launch test creation together:
Task T009: "Create test_token_counter.py in tests/unit/"
Task T010: "Add TokenMetrics validation tests in tests/unit/test_models.py"
Task T011: "Add Tool model token_count tests in tests/unit/test_models.py"
```

## Parallel Example: User Story 1 Integration Tests

```bash
# After implementation (T012-T027) completes, launch integration tests together:
Task T028: "Add find_tool response format test in tests/integration/test_server.py"
Task T029: "Add token savings calculation test in tests/integration/test_server.py"
Task T030: "Add end-to-end flow test in tests/integration/test_end_to_end.py"
```

---

## Implementation Strategy

### MVP Delivery (User Story 1 Only)

1. **Complete Phase 1**: Setup (T001-T003) - ~5 minutes
2. **Complete Phase 2**: Foundational (T004-T008) - ~1 hour
   - Database schema update
   - Token counter utility
   - Data model extensions
3. **Complete Phase 3**: User Story 1 (T009-T030) - ~4-6 hours
   - Unit tests (write first, ensure they fail)
   - CRUD operations
   - Ingestion updates
   - Endpoint updates
   - Integration tests
4. **Complete Phase 4**: Polish (T031-T038) - ~30 minutes
5. **VALIDATE**: Test find_tool independently with various queries
6. Ready for review and deployment

### Incremental Validation Checkpoints

1. **After T008**: Verify database schema, models compile, token counter works in isolation
2. **After T016**: Verify CRUD operations handle token_count correctly
3. **After T021**: Verify tools are ingested with correct token counts
4. **After T027**: Verify find_tool returns token_metrics
5. **After T030**: Verify end-to-end accuracy of token savings calculation
6. **After T034**: All tests pass, ready for code review

---

## Task Summary

- **Total Tasks**: 38
- **Phase 1 (Setup)**: 3 tasks
- **Phase 2 (Foundational)**: 5 tasks
- **Phase 3 (User Story 1)**: 22 tasks
  - Unit tests: 8 tasks (T009-T016, T021)
  - Implementation: 11 tasks (T012-T014, T017-T020, T022-T027)
  - Integration tests: 3 tasks (T028-T030)
- **Phase 4 (Polish)**: 8 tasks
- **Parallelizable**: 15 tasks marked with [P]
- **User Story 1 Tasks**: 22 tasks marked with [US1]

## MVP Scope

**Recommended MVP**: Complete all phases (Phase 1-4)
- This is a single, focused user story that delivers complete value
- Token savings metrics are visible in every find_tool response
- Demonstrates efficiency gains from mcp-optimizer's intelligent tool filtering
- ~6-7 hours of total implementation time

---

## Notes

- [P] tasks = different files, no dependencies
- [US1] label = User Story 1 task for traceability
- User Story 1 is independently testable after Phase 3
- Write unit tests first (T009-T011), verify they fail before implementing
- Commit after logical groups (foundational components, CRUD updates, ingestion, endpoint, tests)
- Run quality gates (format, lint, typecheck, test) before final review
- The feature extends existing functionality without breaking changes to other components
