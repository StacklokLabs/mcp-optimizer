# Feature Specification: Database Table Separation for Registry and Workload MCP Servers

**Feature Branch**: `002-split-db-tables`
**Created**: 2025-10-21
**Status**: Draft
**Input**: User description: "I want to add to simplify my ingestion with database operations. In the past I was avoiding to create separate database tables since it would involve a lot of duplicated information. However, it is getting hard to maintain. Now I want to: Create a separate DB table for MCP servers coming from the registry and Create a separate DB table for MCP servers coming from running workloads"

## Clarifications

### Session 2025-10-21

- Q: When a workload server has conflicting matching criteria (e.g., matches a remote registry server by URL but also a container registry server by package), how should the system respond? → A: Workload MCP servers can also be remote or come from a container. Remote workload servers match ONLY with remote registry servers by URL. Container workload servers match ONLY with container registry servers by package. This conflict cannot occur because matching is type-specific.
- Q: When a workload server finds multiple matching registry servers due to duplicate URLs or packages in the registry (data inconsistency), how should the system respond? → A: Reject ingestion with error indicating duplicate registry entries
- Q: When a registry server is updated with new tools, how should linked workload servers be affected? → A: Tools do not affect server-level description or embeddings. When registry server tools are updated, nothing happens to linked workloads. However, workload servers track the registry MCP server name, which is used as context when calculating embeddings for workload tools. If a registry server name changes, it triggers re-embedding of all linked workload servers' tools.
- Q: When embedding calculation fails for a workload server without a registry link, how should the system respond? → A: Create workload server and reflect the issue in server status (STOPPED or ERROR), matching existing behavior where tools are filtered by server status
- Q: When a workload server is re-ingested with different matching criteria (URL or package changed), how should the system respond? → A: Update existing workload server with new matching criteria and re-establish registry relationship

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Clean Separation of Registry and Workload Data (Priority: P1)

The system maintains MCP server information from two distinct sources: a registry of available servers and actual running workload instances. Currently both are stored in a single table making cleanup and maintenance difficult when servers are removed or changed.

**Why this priority**: This is the foundation of the entire refactoring. Separating registry and workload data enables proper lifecycle management and simplifies all subsequent operations. Without this separation, the other features cannot function correctly.

**Independent Test**: Can be fully tested by ingesting registry servers and workload servers separately, verifying they are stored in different tables, and confirming that deleting from one table does not affect the other.

**Acceptance Scenarios**:

1. **Given** a registry MCP server is ingested with remote URL, **When** the system processes it, **Then** it creates a record in the registry table using URL as unique identifier and stores its tools in the registry tools table
2. **Given** a registry MCP server is ingested with container package, **When** the system processes it, **Then** it creates a record in the registry table using package as unique identifier and stores its tools in the registry tools table
3. **Given** a workload MCP server is ingested, **When** the system processes it, **Then** it creates a record in the workload table using workload name as unique identifier and stores its tools in the workload tools table
4. **Given** a registry MCP server exists, **When** it is deleted, **Then** only the registry table record and its tools are removed, workload servers remain unaffected
5. **Given** a workload MCP server exists, **When** it is deleted, **Then** only the workload table record and its tools are removed, registry servers remain unaffected

---

### User Story 2 - Automatic Registry Matching for Workload Servers (Priority: P2)

When a workload MCP server is ingested, the system should automatically detect if it corresponds to a server in the registry and establish a relationship to reuse description and embeddings.

**Why this priority**: This enables data reuse and consistency across the system. It builds on the separated tables from P1 but isn't required for basic functionality.

**Independent Test**: Can be fully tested by first ingesting a registry server, then ingesting a matching workload server, and verifying the relationship is established and description/embeddings are inherited.

**Acceptance Scenarios**:

1. **Given** a remote registry MCP server exists with URL "https://example.com/mcp", **When** a remote workload server with the same URL is ingested, **Then** a relationship is established and workload server inherits registry description and embeddings
2. **Given** a container registry MCP server exists with package "mcp/awesome-tool", **When** a container workload server with the same package is ingested, **Then** a relationship is established and workload server inherits registry description and embeddings
3. **Given** a workload server with no matching registry entry (same type and identifier), **When** it is ingested, **Then** it has its own description and embeddings calculated from tool mean pooling
4. **Given** multiple workload servers of the same type, **When** they match the same registry server by identifier, **Then** all workload servers establish relationships to the same registry server
5. **Given** multiple remote registry servers with duplicate URL exist, **When** a remote workload server with that URL is ingested, **Then** ingestion fails with error indicating duplicate registry entries must be resolved
6. **Given** a workload server without registry link is ingested, **When** embedding calculation fails, **Then** workload server is created with status reflecting the issue (STOPPED or ERROR)

---

### User Story 3 - Relationship Management During Updates (Priority: P3)

The system maintains stable relationships between workload and registry servers even as their tool sets evolve, only breaking relationships when explicitly requested or when registry servers are deleted.

**Why this priority**: This provides stability and predictability to the system. It's important for production use but not critical for initial implementation.

**Independent Test**: Can be fully tested by establishing a relationship, modifying the workload server's tools, and verifying the relationship persists. Then delete the registry server and verify the workload server calculates its own embeddings.

**Acceptance Scenarios**:

1. **Given** a workload server linked to a registry server, **When** tools are added to the workload server, **Then** the registry relationship is maintained and description/embeddings remain inherited
2. **Given** a workload server linked to a registry server, **When** tools are removed from the workload server, **Then** the registry relationship is maintained and description/embeddings remain inherited
3. **Given** a workload server linked to a registry server, **When** the registry server is deleted, **Then** the workload server relationship is removed and it calculates its own description and embeddings from its tools
4. **Given** a workload server with its own embeddings (no registry link), **When** a matching registry server is later added, **Then** the workload server maintains its own embeddings until an explicit re-match operation
5. **Given** a workload server linked to a registry server, **When** tools are added or removed from the registry server, **Then** the workload server is not affected (tool changes don't impact server-level data)
6. **Given** a workload server linked to a registry server, **When** the registry server name is changed, **Then** all linked workload servers re-embed their tools using the new registry server name as context
7. **Given** an existing workload server, **When** it is re-ingested with a different URL or package, **Then** the existing record is updated with new matching criteria and registry relationship is re-evaluated and re-established if a match is found

---

### Edge Cases

- When a workload server matches multiple registry servers due to duplicate URLs or packages in registry, the system rejects ingestion with an error requiring resolution of duplicate registry entries
- When a registry server tools are updated, linked workload servers are not affected (tools don't impact server-level description/embeddings)
- When a registry server name changes, all linked workload servers must re-embed their tools using the new name as context
- When embedding calculation fails for workload servers without registry links, the server status is set to STOPPED or ERROR (tools are filtered by server status)
- When a workload server is re-ingested with different matching criteria (URL or package changed), the existing record is updated and registry relationship is re-evaluated

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST store registry MCP servers in a dedicated table separate from workload MCP servers
- **FR-002**: System MUST store workload MCP servers in a dedicated table separate from registry MCP servers
- **FR-003**: System MUST store registry MCP server tools in a dedicated table separate from workload tools
- **FR-004**: System MUST store workload MCP server tools in a dedicated table separate from registry tools
- **FR-005**: System MUST enforce unique constraint on URL for remote registry MCP servers
- **FR-006**: System MUST enforce unique constraint on package for container registry MCP servers
- **FR-007**: System MUST enforce unique constraint on workload name for workload MCP servers
- **FR-008**: System MUST automatically match remote workload servers to remote registry servers by URL only
- **FR-009**: System MUST automatically match container workload servers to container registry servers by package only
- **FR-010**: System MUST reject workload server ingestion with error when multiple registry servers match due to duplicate URLs or packages
- **FR-011**: System MUST allow multiple workload servers to reference the same registry server
- **FR-012**: System MUST maintain optional relationship between workload servers and registry servers
- **FR-013**: Workload servers MUST inherit description and embeddings from linked registry servers when relationship exists
- **FR-014**: Workload servers MUST calculate their own description and embeddings from tool mean pooling when no registry relationship exists
- **FR-015**: System MUST maintain registry relationship when workload server tool sets change (tools added or removed)
- **FR-016**: Workload servers MUST track and store the name of their linked registry server
- **FR-017**: System MUST use registry server name as context when calculating embeddings for workload server tools
- **FR-018**: System MUST NOT re-embed workload server tools when linked registry server tools are added or removed
- **FR-019**: System MUST re-embed all tools for linked workload servers when registry server name changes
- **FR-020**: System MUST remove registry relationship and recalculate embeddings when linked registry server is deleted
- **FR-021**: System MUST support deletion of registry servers without affecting unrelated workload servers
- **FR-022**: System MUST support deletion of workload servers without affecting registry servers
- **FR-023**: System MUST create workload server even when embedding calculation fails for servers without registry links
- **FR-024**: System MUST reflect embedding calculation failures in workload server status (STOPPED or ERROR)
- **FR-025**: System MUST filter tools by parent server status matching existing behavior (e.g., status = RUNNING)
- **FR-026**: System MUST support re-ingestion of existing workload servers with updated URL or package by updating the existing record
- **FR-027**: System MUST re-evaluate and re-establish registry relationships when workload server matching criteria change during re-ingestion
- **FR-028**: System MUST provide database migration to convert existing single table structure to separated tables
- **FR-029**: System MUST delete old tables (mcpserver, tool) during migration
- **FR-030**: System MUST create new tables (mcpservers_registry, mcpservers_workload, tools_registry, tools_workload) during migration

### Key Entities *(include if feature involves data)*

- **Registry MCP Server**: Represents an MCP server from the registry catalog. Can be either remote (identified by URL) or container (identified by package). Contains server metadata, description, and embeddings calculated from registry information.
- **Workload MCP Server**: Represents a running MCP server instance from a workload. Identified uniquely by workload name. Can be either remote or container type. May have an optional relationship to a registry server of the same type (remote-to-remote or container-to-container) to inherit description and embeddings, or calculate its own from tools. Tracks the linked registry server name for use as context in tool embedding calculations.
- **Registry Tool**: Represents a tool provided by a registry MCP server. Contains tool details and embeddings. Belongs to exactly one registry MCP server.
- **Workload Tool**: Represents a tool provided by a workload MCP server. Contains tool details and embeddings calculated using server name context (either linked registry server name or workload server's own name). Belongs to exactly one workload MCP server. May be marked as unavailable if embedding calculation fails, with automatic retry until successful.
- **Registry Relationship**: Optional relationship from workload MCP server to registry MCP server. Enables inheritance of description and embeddings. Multiple workload servers can link to same registry server.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: System can successfully ingest and store 1000 registry MCP servers without conflicts
- **SC-002**: System can successfully ingest and store 1000 workload MCP servers without conflicts
- **SC-003**: Deletion of registry servers completes in under 100ms and only affects registry tables
- **SC-004**: Deletion of workload servers completes in under 100ms and only affects workload tables
- **SC-005**: Workload server ingestion automatically matches to registry servers with 100% accuracy for identical URLs and packages
- **SC-006**: Database migration completes successfully with zero data loss for existing deployments
- **SC-007**: System maintains relationship stability - tool changes in workload servers do not break registry links
- **SC-008**: Workload servers calculate fallback embeddings within 200ms when registry relationship is removed
- **SC-009**: Registry server tool changes do not trigger re-embedding of linked workload server tools
- **SC-010**: Registry server name changes trigger complete re-embedding of all linked workload server tools within 500ms per workload
- **SC-011**: Workload servers are created successfully even when embedding calculation fails, with server status reflecting the issue
- **SC-012**: Workload servers can be re-ingested with updated matching criteria, automatically updating registry relationships within 300ms
- **SC-013**: Tools are correctly filtered by parent server status (RUNNING) matching existing tool query behavior

