<!--
Sync Impact Report - Constitution Update

Version Change: NONE → 1.0.0 (Initial Constitution)
Ratification: 2025-10-20

Modified Principles:
- (New) I. Tooling Consistency
- (New) II. Centralized Configuration
- (New) III. Modern Python Standards
- (New) IV. Database Agnosticism
- (New) V. Quality Gates (NON-NEGOTIABLE)
- (New) VI. Testing Coverage
- (New) VII. CLI Design Patterns

Added Sections:
- Development Workflow
- Communication Standards
- Governance

Templates Requiring Updates:
- ✅ plan-template.md: Updated with Constitution Check reference
- ✅ spec-template.md: Aligned with requirements structure
- ✅ tasks-template.md: Aligned with quality gate task patterns

Follow-up TODOs: None
-->

# MCP Optimizer Constitution

## Core Principles

### I. Tooling Consistency

Every feature MUST adhere to the established toolchain:
- Package management via uv only (`uv add <package>` for dependencies, `uv add <package> --dev` for dev tools)
- Task automation via taskfile (`task format`, `task lint`, `task typecheck`, `task test`)
- Python execution via uv (`uv run python` instead of bare `python` command)

**Rationale**: Consistent tooling ensures reproducible builds, eliminates environment drift, and guarantees all developers use the correct Python version and dependencies.

### II. Centralized Configuration

pyproject.toml MUST be the single source of truth for all project configuration:
- Linters (ruff)
- Type checkers (ty)
- Test frameworks (pytest)
- Build systems (hatchling)
- Database migrations (alembic)
- Dependencies and dev dependencies

**Rationale**: Centralized configuration eliminates conflicts between tool config files, improves discoverability, and simplifies onboarding.

### III. Modern Python Standards

Code MUST follow modern Python practices:
- Native Python types instead of typing module (`list` not `List`, `dict` not `Dict`)
- Python 3.13+ features and syntax
- Pydantic models for ALL structured data validation and serialization
- Type hints for function signatures and complex data structures

**Rationale**: Modern standards improve code clarity, enable better IDE support, and leverage latest language features for cleaner, more maintainable code.

### IV. Database Agnosticism

Database interactions MUST remain vendor-agnostic:
- Use SQLAlchemy `text()` for raw SQL queries only
- NO ORM features or models (no declarative base, no relationship mappings)
- NO database-specific SQL dialects or proprietary functions
- UUID primary keys via `uuid.uuid4()` for all entities
- Execute queries via `Connection.execute()` with `text()` statements

**Rationale**: Database agnosticism enables deployment flexibility, simplifies testing with different backends, and prevents vendor lock-in.

### V. Quality Gates (NON-NEGOTIABLE)

Every code change MUST pass through the quality gate sequence in order:
1. `task format` - Auto-format code (ruff format + ruff check --fix)
2. `task lint` - Validate code quality (ruff check)
3. `task typecheck` - Verify type correctness (ty check)
4. `task test` - Run test suite with coverage (pytest with coverage reporting)

No code may be committed or submitted for review until ALL gates pass.

**Rationale**: Sequential quality gates catch issues early, enforce consistent standards, and prevent technical debt accumulation. Auto-formatting prevents style debates.

### VI. Testing Coverage

Testing MUST be comprehensive and measurable:
- pytest as the test framework
- Coverage tracking enabled (pytest-cov)
- Tests located in `tests/` directory at repository root
- Test file naming: `test_*.py`
- Async tests supported via pytest-asyncio
- Tests MUST cover new features, bug fixes, and edge cases

**Rationale**: High test coverage ensures reliability, enables confident refactoring, and serves as living documentation of system behavior.

### VII. CLI Design Patterns

CLI commands MUST follow consistent design:
- Group related functionality under single commands with parameters
- Use `logger` for output, NOT `click.echo`
- Commands should represent big functionality, parameters provide inputs
- Follow established patterns in `src/mcp_optimizer/cli.py`

**Rationale**: Consistent CLI design improves user experience, simplifies maintenance, and ensures proper logging integration.

### VIII. Batch Operations

Operations that process multiple items MUST be batched when possible:
- Embeddings via fastembed MUST process lists of texts in single calls
- Database operations SHOULD batch when feasible
- Avoid iterative single-item processing where batch operations exist

**Rationale**: Batch operations dramatically improve performance, reduce overhead, and scale better with increased load.

## Development Workflow

### Code Modification Workflow

When modifying or adding code:
1. Make changes to source files
2. Run `task format` to auto-format and auto-fix linting issues
3. Run `task lint` to identify remaining linting issues, fix manually
4. Run `task typecheck` to verify type correctness, fix type errors
5. Run `task test` to ensure all tests pass
6. Commit changes only after all quality gates pass

### Feature Development Workflow

When implementing new features:
1. Follow spec-driven development (use `/speckit.*` commands)
2. Prioritize user stories (P1 → P2 → P3)
3. Ensure each user story is independently testable
4. Write tests for new functionality (unit, integration as appropriate)
5. Implement minimal viable solution before optimization
6. Run full quality gate sequence before considering feature complete

## Communication Standards

### Code Documentation

- Use clear, descriptive variable and function names
- Add docstrings for complex functions and classes
- Keep comments focused on "why" not "what"
- Update documentation when behavior changes

### Commit Messages

Commit messages MUST follow project standards:
- Serve as PR descriptions (will be public-facing)
- Use simple, direct language without self-promotion
- Keep under 100 words (max 200 for complex changes)
- Use bullet points for clarity
- NO markdown in commit title
- Minimal markdown in commit body
- Focus on what changed in the code, not marketing language
- Must be understandable in under 1 minute

Example format:
```
Fix database connection handling

- Added connection pooling to prevent timeout errors
- Implemented retry logic for transient failures
- Updated error logging to include connection metadata
```

### Logging

- Use structured logging (structlog)
- Log at appropriate levels (debug, info, warning, error)
- Include relevant context in log messages
- NO use of print statements in production code

## Governance

### Constitution Authority

This constitution supersedes all other development practices and guidelines. When conflicts arise between this document and other guidance, this constitution takes precedence.

### Amendments

Amendments to this constitution require:
1. Documentation of proposed changes with rationale
2. Impact analysis on existing code and templates
3. Update to all dependent templates and documentation
4. Version bump following semantic versioning rules:
   - MAJOR: Backward incompatible governance/principle removals or redefinitions
   - MINOR: New principle/section added or materially expanded guidance
   - PATCH: Clarifications, wording, typo fixes, non-semantic refinements

### Compliance

- All code reviews MUST verify constitutional compliance
- Quality gate failures MUST block merges
- Complexity that violates principles MUST be justified in writing
- Violations require documentation in plan.md Complexity Tracking table

### Runtime Development Guidance

For agent-specific guidance during development, refer to:
- `CLAUDE.md` - Project-specific development instructions for AI assistants
- `README.md` - Project overview and setup instructions
- `.specify/templates/` - Feature specification templates

**Version**: 1.0.0 | **Ratified**: 2025-10-20 | **Last Amended**: 2025-10-20
