# Feature Specification: Fix Remote Workload Matching

**Feature Branch**: `001-fix-remote-workload-matching`
**Created**: 2025-10-20
**Status**: Draft
**Input**: User description: "Update the ingestion logic for remote MCP workloads. Right now the code is checking for MCP workloads relying that the workflow name will be the same as its name on the registry. However, this integration point is brittle. It would be better, that if the running workload is remote, we make a get request to get the detail of the workflow to get the `url` of it. If the `url` of a remote workload is the same as the `url` in the registry then it should be considered that the workload corresponds to the one in the registry. Right now the ingestion logic is broken and if there's a remote workload which doesn't match in name to an MCP server in the registry then the workload will be ingested and immediately after being ingested is deleted"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Stable Remote Workload Ingestion (Priority: P1)

When a remote MCP workload is running in ToolHive, the MCP-Optimizer system should correctly match it to its registry entry based on URL rather than name, preventing unintended deletion of correctly deployed workloads.

**Why this priority**: This is the core bug fix that prevents data loss and ensures system stability. Without this fix, remote workloads cannot be reliably used in production.

**Independent Test**: Deploy a remote MCP workload with a custom name (different from its registry name) and verify it remains ingested and functional after the ingestion cycle completes. The workload should not be deleted.

**Acceptance Scenarios**:

1. **Given** a remote MCP workload is running in ToolHive with name "custom-github-server" that corresponds to registry entry "github" with URL "https://api.github.com/mcp", **When** the ingestion process runs, **Then** the system fetches the workload details including its URL, matches it to the registry entry by URL, and keeps the workload in the database with status RUNNING

2. **Given** a remote MCP workload is running with a URL that matches a registry entry URL, **When** the ingestion cleanup phase runs, **Then** the workload is not marked for deletion because URL matching identifies it as a valid registry server

3. **Given** a remote MCP workload with URL "https://custom.api.com/mcp" that does not exist in the registry, **When** the ingestion process runs, **Then** the system creates a new server entry with from_registry=False and keeps it in the database

---

### User Story 2 - Registry Server URL Updates (Priority: P2)

When a registry entry's URL changes, running remote workloads should continue to be correctly matched to their registry entries as long as the workload URL hasn't changed.

**Why this priority**: This ensures the system handles registry metadata updates gracefully without disrupting running workloads.

**Independent Test**: Update a registry entry's URL, then run ingestion with a remote workload still using the old URL. The workload should be treated as a custom deployment (not matching the updated registry entry).

**Acceptance Scenarios**:

1. **Given** a running remote workload with URL "https://api.v1.service.com/mcp" and a registry entry that updates to URL "https://api.v2.service.com/mcp", **When** ingestion runs, **Then** the workload is treated as a custom deployment because URLs no longer match

2. **Given** a registry entry with URL "https://api.service.com/mcp" and a running workload with the same URL, **When** ingestion runs, **Then** the workload is correctly identified as matching the registry entry and metadata is synchronized

---

### Edge Cases

- What happens when a remote workload's URL is temporarily unavailable during detail fetch? The system should log the error and skip processing that workload for the current cycle, retrying on the next ingestion cycle.

- What happens when multiple remote workloads have the same URL? Each workload should be matched to the same registry entry, and both should be kept in the database with their respective workload-specific metadata.

- What happens when a workload is truly no longer in ToolHive and should be deleted? The cleanup logic should continue to work as before - workloads not found in the active workload list will be removed (or status changed to REGISTRY if from_registry=True).

- What happens when fetching workload details fails due to network or API errors? The system should log the error, treat it as a temporary failure, and skip that workload for the current ingestion cycle.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST fetch detailed workload information for remote workloads during ingestion to obtain the workload URL

- **FR-002**: System MUST match remote workloads to registry entries by comparing workload URL with registry server URL

- **FR-003**: System MUST preserve existing matching logic for container workloads (match by package name)

- **FR-004**: System MUST handle cases where fetching remote workload details fails by logging the error and skipping that workload for the current ingestion cycle

- **FR-005**: System MUST treat remote workloads with URLs not found in the registry as custom deployments (from_registry=False)

- **FR-006**: System MUST continue to prevent deletion of registry-sourced servers by changing their status to REGISTRY when the workload is no longer running

- **FR-007**: System MUST maintain existing workload cleanup behavior for non-matching workloads (delete custom workloads, change status for registry workloads)

### Key Entities

- **Remote Workload**: A running MCP server hosted at a remote URL, identified in ToolHive with tool_type="remote" and containing a URL property that points to the actual MCP server endpoint

- **Workload URL**: The endpoint URL where a remote MCP workload is accessible, used as the stable identifier for matching workloads to registry entries

- **Registry Entry**: Metadata for an MCP server stored in the registry, including server name, description, tags, tools, and most importantly the canonical URL for the server

- **Workload-Registry Matching**: The process of determining whether a running workload corresponds to a known registry entry, now based on URL comparison for remote workloads instead of name matching

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Remote workloads deployed with custom names remain ingested and accessible after ingestion completes (0% false deletion rate)

- **SC-002**: Remote workloads are correctly matched to their registry entries within one ingestion cycle (100% correct matching rate for URL-matched workloads)

- **SC-003**: Remote workloads with URLs not in the registry are successfully created as custom deployments (100% success rate for custom remote workloads)

- **SC-004**: Container workload ingestion continues to function identically to current behavior (0% regression in container workload handling)

- **SC-005**: Ingestion cycle completes within the same time bounds as current implementation (no performance degradation)
