# Data Model: Database Table Separation

**Feature**: 002-split-db-tables
**Date**: 2025-10-21
**Status**: Complete

## Overview

This document defines the database schema and Pydantic models for the separated registry and workload table structure. The model supports type-specific matching, optional relationships, and independent lifecycle management.

## Entity Relationship Diagram

```
┌─────────────────────────────┐
│   mcpservers_registry       │
├─────────────────────────────┤
│ id (UUID, PK)              │
│ name (TEXT, NOT NULL)      │
│ url (TEXT)                 │
│ package (TEXT)             │
│ remote (BOOLEAN)           │
│ transport (TEXT)           │
│ description (TEXT)         │
│ server_embedding (BLOB)    │
│ group (TEXT)               │
│ last_updated (TIMESTAMP)   │
│ created_at (TIMESTAMP)     │
├─────────────────────────────┤
│ UNIQUE(url) WHERE remote   │
│ UNIQUE(package) WHERE !remote│
└─────────────────────────────┘
         │
         │ 1
         │
         │ 0..*
         ▼
┌─────────────────────────────┐
│   mcpservers_workload       │
├─────────────────────────────┤
│ id (UUID, PK)              │
│ workload_name (TEXT,UNIQUE)│
│ name (TEXT, NOT NULL)      │
│ url (TEXT)                 │
│ package (TEXT)             │
│ remote (BOOLEAN)           │
│ transport (TEXT)           │
│ status (TEXT)              │
│ registry_server_id (UUID FK│  ────┐
│ registry_server_name (TEXT)│      │
│ description (TEXT, NULLABLE│      │ Optional
│ server_embedding (BLOB,NULL│      │ relationship
│ group (TEXT)               │      │
│ last_updated (TIMESTAMP)   │      │
│ created_at (TIMESTAMP)     │      │
└─────────────────────────────┘  ────┘
         │ 1                │ 1
         │                  │
         │ 0..*             │ 0..*
         ▼                  ▼
┌──────────────────────┐   ┌──────────────────────┐
│   tools_workload     │   │   tools_registry     │
├──────────────────────┤   ├──────────────────────┤
│ id (UUID, PK)       │   │ id (UUID, PK)       │
│ server_id (UUID FK) │   │ server_id (UUID FK) │
│ name (TEXT)         │   │ name (TEXT)         │
│ description (TEXT)  │   │ description (TEXT)  │
│ input_schema (JSON) │   │ input_schema (JSON) │
│ embedding (BLOB)    │   │ embedding (BLOB)    │
│ available (BOOLEAN) │   │ last_updated (TS)   │
│ last_updated (TS)   │   │ created_at (TS)     │
│ created_at (TS)     │   └──────────────────────┘
└──────────────────────┘
```

## Table Schemas

### mcpservers_registry

Stores MCP servers from the registry catalog (stable, curated metadata).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT (UUID) | PRIMARY KEY | Unique server identifier |
| name | TEXT | NOT NULL | Server name |
| url | TEXT | NULLABLE | Server URL (for remote servers) |
| package | TEXT | NULLABLE | Container package/image (for container servers) |
| remote | BOOLEAN | NOT NULL | True if remote server, False if container |
| transport | TEXT | NOT NULL | Transport protocol (stdio, sse, etc.) |
| description | TEXT | NULLABLE | Server description |
| server_embedding | BLOB | NULLABLE | Vector embedding of server description |
| group | TEXT | NOT NULL, DEFAULT 'default' | Server grouping |
| last_updated | TIMESTAMP | NOT NULL | Last update timestamp |
| created_at | TIMESTAMP | NOT NULL | Creation timestamp |

**Constraints**:
- `UNIQUE(url)` WHERE `remote = true` - Remote servers identified by URL
- `UNIQUE(package)` WHERE `remote = false` - Container servers identified by package
- CHECK: `(remote = true AND url IS NOT NULL) OR (remote = false AND package IS NOT NULL)`

**Indexes**:
- PRIMARY KEY on `id`
- INDEX on `remote`
- UNIQUE INDEX on `url` WHERE `remote = true`
- UNIQUE INDEX on `package` WHERE `remote = false`

### mcpservers_workload

Stores running MCP server instances from workloads (dynamic, runtime state).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT (UUID) | PRIMARY KEY | Unique workload server identifier |
| workload_name | TEXT | UNIQUE, NOT NULL | Workload identifier from ToolHive |
| name | TEXT | NOT NULL | Server name |
| url | TEXT | NULLABLE | Server URL (for remote servers) |
| package | TEXT | NULLABLE | Container package/image (for container servers) |
| remote | BOOLEAN | NOT NULL | True if remote server, False if container |
| transport | TEXT | NOT NULL | Transport protocol |
| status | TEXT | NOT NULL | Server status (running, stopped, error) |
| registry_server_id | TEXT (UUID) | NULLABLE, FOREIGN KEY | Link to registry server (if matched) |
| registry_server_name | TEXT | NULLABLE | Cached name of registry server (for tool embedding context) |
| description | TEXT | NULLABLE | Server description (only if no registry link) |
| server_embedding | BLOB | NULLABLE | Vector embedding (only if no registry link) |
| group | TEXT | NOT NULL, DEFAULT 'default' | Server grouping |
| last_updated | TIMESTAMP | NOT NULL | Last update timestamp |
| created_at | TIMESTAMP | NOT NULL | Creation timestamp |

**Constraints**:
- `UNIQUE(workload_name)` - Each workload instance has unique name
- `FOREIGN KEY(registry_server_id) REFERENCES mcpservers_registry(id) ON DELETE SET NULL` - Optional relationship to registry
- CHECK: `(remote = true AND url IS NOT NULL) OR (remote = false AND package IS NOT NULL)`
- CHECK: `(registry_server_id IS NULL AND description IS NOT NULL) OR (registry_server_id IS NOT NULL)` - Must have description if autonomous

**Indexes**:
- PRIMARY KEY on `id`
- UNIQUE INDEX on `workload_name`
- INDEX on `registry_server_id`
- INDEX on `remote`

### tools_registry

Stores tools for registry MCP servers (matches current tool table schema).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT (UUID) | PRIMARY KEY | Unique tool identifier |
| mcpserver_id | TEXT (UUID) | NOT NULL, FOREIGN KEY | Parent registry server ID |
| details | TEXT (JSON) | NOT NULL | MCP tool definition (name, description, inputSchema) |
| details_embedding | BLOB | NULLABLE | Vector embedding of tool description |
| last_updated | TIMESTAMP | NOT NULL | Last update timestamp |
| created_at | TIMESTAMP | NOT NULL | Creation timestamp |

**Constraints**:
- `FOREIGN KEY(mcpserver_id) REFERENCES mcpservers_registry(id) ON DELETE CASCADE` - Tools deleted with server
- Tool names unique within server (enforced via details JSON)

**Indexes**:
- PRIMARY KEY on `id`
- INDEX on `mcpserver_id`

### tools_workload

Stores tools for workload MCP servers (matches current tool table schema).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | TEXT (UUID) | PRIMARY KEY | Unique tool identifier |
| mcpserver_id | TEXT (UUID) | NOT NULL, FOREIGN KEY | Parent workload server ID |
| details | TEXT (JSON) | NOT NULL | MCP tool definition (name, description, inputSchema) |
| details_embedding | BLOB | NULLABLE | Vector embedding using server name context |
| last_updated | TIMESTAMP | NOT NULL | Last update timestamp |
| created_at | TIMESTAMP | NOT NULL | Creation timestamp |

**Constraints**:
- `FOREIGN KEY(mcpserver_id) REFERENCES mcpservers_workload(id) ON DELETE CASCADE` - Tools deleted with server
- Tool names unique within server (enforced via details JSON)

**Indexes**:
- PRIMARY KEY on `id`
- INDEX on `mcpserver_id`

**Note**: Embedding failures are reflected in the parent workload server's status, not individual tool availability. Tools are filtered by server status (e.g., `status = 'running'`).

## Virtual Tables

### Registry Virtual Tables

**registry_server_vector** (sqlite-vec):
- Materialized view of mcpservers_registry with vec0 module
- Synced when mcpservers_registry.server_embedding changes

**registry_tool_vectors** (sqlite-vec):
- Materialized view of tools_registry with vec0 module
- Synced when tools_registry.embedding changes

**registry_tool_fts** (FTS5):
- Full-text search index on tools_registry(name, description)
- Synced when tools_registry name or description changes

### Workload Virtual Tables

**workload_server_vector** (sqlite-vec):
- Materialized view of mcpservers_workload with vec0 module
- Only includes servers with non-NULL server_embedding (autonomous servers)
- Synced when mcpservers_workload.server_embedding changes

**workload_tool_vectors** (sqlite-vec):
- Materialized view of tools_workload with vec0 module
- Filtered by parent server status (only from RUNNING servers)
- Synced when tools_workload.details_embedding changes

**workload_tool_fts** (FTS5):
- Full-text search index on tools_workload details (name, description from JSON)
- Filtered by parent server status (only from RUNNING servers)
- Synced when tools_workload.details changes

## Pydantic Models

### Registry Server Models

```python
class RegistryServer(BaseModel):
    """Registry MCP server from catalog."""
    id: str
    name: str
    url: str | None
    package: str | None
    remote: bool
    transport: TransportType
    description: str | None
    server_embedding: np.ndarray | None
    group: str = "default"
    last_updated: datetime
    created_at: datetime

    @model_validator(mode='after')
    def validate_identifier(self) -> Self:
        """Ensure remote servers have URL, container servers have package."""
        if self.remote and not self.url:
            raise ValueError("Remote servers must have URL")
        if not self.remote and not self.package:
            raise ValueError("Container servers must have package")
        return self
```

### Workload Server Models

```python
class WorkloadServer(BaseModel):
    """Workload MCP server from running instance."""
    id: str
    workload_name: str
    name: str
    url: str | None
    package: str | None
    remote: bool
    transport: TransportType
    status: McpStatus
    registry_server_id: str | None  # NULL if autonomous
    registry_server_name: str | None  # Cached for tool embedding context
    description: str | None  # Only if autonomous (registry_server_id is NULL)
    server_embedding: np.ndarray | None  # Only if autonomous
    group: str = "default"
    last_updated: datetime
    created_at: datetime

    @model_validator(mode='after')
    def validate_identifier(self) -> Self:
        """Ensure remote servers have URL, container servers have package."""
        if self.remote and not self.url:
            raise ValueError("Remote servers must have URL")
        if not self.remote and not self.package:
            raise ValueError("Container servers must have package")
        return self

    @model_validator(mode='after')
    def validate_description(self) -> Self:
        """Ensure autonomous servers have description."""
        if self.registry_server_id is None and self.description is None:
            raise ValueError("Autonomous workload servers must have description")
        return self
```

### Tool Models

```python
class RegistryTool(BaseModel):
    """Tool from registry MCP server (matches current Tool schema)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    mcpserver_id: str  # References mcpservers_registry.id
    details: McpTool  # MCP tool definition with name, description, inputSchema
    details_embedding: np.ndarray | None = Field(default=None, exclude=True)
    last_updated: datetime
    created_at: datetime

class WorkloadTool(BaseModel):
    """Tool from workload MCP server (matches current Tool schema)."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    mcpserver_id: str  # References mcpservers_workload.id
    details: McpTool  # MCP tool definition with name, description, inputSchema
    details_embedding: np.ndarray | None = Field(default=None, exclude=True)
    last_updated: datetime
    created_at: datetime
```

**Note**: Both tool models match the existing `Tool` schema. Embedding failures affect the parent server's status, not individual tools. Tools are filtered by querying servers with `status = McpStatus.RUNNING`.

### Relationship Models

```python
class WorkloadWithRegistry(BaseModel):
    """Workload server with resolved registry relationship."""
    workload: WorkloadServer
    registry: RegistryServer | None  # None if autonomous

    @property
    def effective_description(self) -> str | None:
        """Get description (inherited from registry or own)."""
        if self.registry:
            return self.registry.description
        return self.workload.description

    @property
    def effective_embedding(self) -> np.ndarray | None:
        """Get embedding (inherited from registry or own)."""
        if self.registry:
            return self.registry.server_embedding
        return self.workload.server_embedding

    @property
    def server_name_for_tools(self) -> str:
        """Get server name to use as context for tool embeddings."""
        if self.registry:
            return self.registry.name
        return self.workload.name
```

## State Transitions

### Workload Server Lifecycle

```
┌────────────────────────────────────┐
│  Workload Server Ingested         │
└───────────────┬────────────────────┘
                │
                ▼
        ┌───────────────┐
        │ Match Registry│
        │  by URL/pkg   │
        └───────┬───────┘
                │
      ┌─────────┴──────────┐
      │                    │
      ▼                    ▼
┌──────────────┐    ┌────────────────┐
│ Match Found  │    │  No Match      │
│ (1 registry) │    │ OR Duplicates  │
└──────┬───────┘    └────────┬───────┘
       │                     │
       │                     ▼
       │              ┌──────────────┐
       │              │If Duplicates:│
       │              │ Reject Error │
       │              └──────────────┘
       │                     │
       │              ┌──────┴───────┐
       │              │If No Match:  │
       │              │ Create as    │
       │              │ Autonomous   │
       │              └──────┬───────┘
       │                     │
       ▼                     ▼
┌─────────────────┐   ┌─────────────────┐
│ Linked Workload │   │Autonomous       │
│ registry_id SET │   │registry_id NULL │
│ Inherits desc/  │   │Own desc/embed   │
│ embedding       │   │from tool pool   │
└─────────────────┘   └─────────────────┘
```

### Tool Ingestion State

```
┌──────────────────────────┐
│ Tool Ingested            │
└───────────┬──────────────┘
            │
            ▼
    ┌───────────────────┐
    │ Calculate         │
    │ Embedding         │
    │ (with server name)│
    └────────┬──────────┘
             │
     ┌───────┴────────┐
     │                │
     ▼                ▼
┌─────────────┐  ┌──────────────────┐
│ Success     │  │  Failure         │
│             │  │                  │
└──────┬──────┘  └────────┬─────────┘
       │                  │
       ▼                  ▼
embedding SET      Server status set
Tool stored        to STOPPED/ERROR
                   (affects all tools)
```

**Note**: Unlike individual tool availability tracking, embedding failures affect the entire server's status. This matches existing behavior where tools are filtered by server status rather than per-tool availability flags.

## Data Validation Rules

### Registry Server Validation

1. Exactly one of (url, package) must be non-NULL based on `remote` flag
2. If remote=true: url must be valid URL, package must be NULL
3. If remote=false: package must be non-empty, url must be NULL
4. Name must be non-empty string
5. Transport must be valid TransportType enum value

### Workload Server Validation

1. Same identifier validation as registry (url XOR package based on remote flag)
2. workload_name must be non-empty and unique
3. If registry_server_id is NULL: description and server_embedding must be non-NULL
4. If registry_server_id is SET: registry_server_name must also be SET (cached for tool context)
5. Status must be valid McpStatus enum value

### Tool Validation

1. mcpserver_id must reference existing server (registry or workload)
2. details must contain valid McpTool object with name
3. Tool names must be unique within server (enforced via details.name)
4. details.inputSchema must be valid JSON schema if present
5. details_embedding can be NULL (will be filtered by parent server status)

## References

- Existing models: `src/mcp_optimizer/db/models.py`
- Validation patterns: Pydantic `@model_validator` decorators
- Enum types: `McpStatus`, `TransportType` from existing models

