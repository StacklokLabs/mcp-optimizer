# Operations Interface Contracts

**Feature**: 002-split-db-tables
**Date**: 2025-10-21

## Overview

This document defines the programmatic interfaces for the four database operations classes. Each ops class provides CRUD operations for its table plus virtual table sync methods.

## Common Patterns

### Constructor Pattern

All ops classes follow this constructor pattern:

```python
class {Entity}Ops:
    def __init__(self, db: DatabaseConfig):
        """Initialize ops with database configuration.

        Args:
            db: Database configuration providing async connection access

        """
        self.db = db
```

### Connection Management Pattern

All methods accept optional `conn` parameter for transaction support:

```python
async def operation(
    self,
    ...,
    conn: AsyncConnection | None = None
) -> ReturnType:
    """Operation description.

    Args:
        conn: Optional existing connection for transaction support.
              If None, creates new connection.
    """
    async def _execute(connection: AsyncConnection) -> ReturnType:
        # Operation implementation
        pass

    if conn:
        return await _execute(conn)
    else:
        async with self.db.get_connection() as connection:
            return await _execute(connection)
```

## RegistryServerOps

**Purpose**: CRUD operations for mcpservers_registry table.

### Methods

#### create_server

```python
async def create_server(
    self,
    name: str,
    url: str | None,
    package: str | None,
    remote: bool,
    transport: TransportType,
    description: str | None = None,
    server_embedding: np.ndarray | None = None,
    group: str = "default",
    conn: AsyncConnection | None = None,
) -> RegistryServer:
    """Create a new registry server.

    Args:
        name: Server name
        url: Server URL (required if remote=True)
        package: Container package (required if remote=False)
        remote: True for remote server, False for container
        transport: Transport protocol type
        description: Optional server description
        server_embedding: Optional vector embedding
        group: Server grouping (default: "default")
        conn: Optional connection for transaction

    Returns:
        Created RegistryServer model

    Raises:
        ValueError: If identifier invalid (URL missing for remote, etc.)
        IntegrityError: If duplicate URL (remote) or package (container)
    """
```

#### get_server_by_id

```python
async def get_server_by_id(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> RegistryServer | None:
    """Get registry server by ID.

    Args:
        server_id: Server UUID
        conn: Optional connection

    Returns:
        RegistryServer if found, None otherwise
    """
```

#### get_server_by_url

```python
async def get_server_by_url(
    self,
    url: str,
    conn: AsyncConnection | None = None,
) -> RegistryServer | None:
    """Get remote registry server by URL.

    Args:
        url: Server URL
        conn: Optional connection

    Returns:
        RegistryServer if found (with remote=True), None otherwise
    """
```

#### get_server_by_package

```python
async def get_server_by_package(
    self,
    package: str,
    conn: AsyncConnection | None = None,
) -> RegistryServer | None:
    """Get container registry server by package.

    Args:
        package: Container package/image name
        conn: Optional connection

    Returns:
        RegistryServer if found (with remote=False), None otherwise
    """
```

#### find_matching_servers

```python
async def find_matching_servers(
    self,
    url: str | None,
    package: str | None,
    remote: bool,
    conn: AsyncConnection | None = None,
) -> list[RegistryServer]:
    """Find registry servers matching identifier.

    Args:
        url: Server URL (for remote servers)
        package: Container package (for container servers)
        remote: Server type to match
        conn: Optional connection

    Returns:
        List of matching RegistryServer (empty, 1, or multiple for duplicates)

    Note:
        Returns multiple servers only if duplicates exist in registry.
        Caller should detect duplicate case and handle appropriately.
    """
```

#### update_server

```python
async def update_server(
    self,
    server_id: str,
    name: str | None = None,
    description: str | None = None,
    server_embedding: np.ndarray | None = None,
    transport: TransportType | None = None,
    group: str | None = None,
    conn: AsyncConnection | None = None,
) -> RegistryServer:
    """Update registry server fields.

    Args:
        server_id: Server UUID
        name: Optional new name
        description: Optional new description
        server_embedding: Optional new embedding
        transport: Optional new transport
        group: Optional new group
        conn: Optional connection

    Returns:
        Updated RegistryServer

    Raises:
        DbNotFoundError: If server not found

    Note:
        URL and package are immutable (identifier fields).
        Name changes trigger workload tool re-embedding.
    """
```

#### delete_server

```python
async def delete_server(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> None:
    """Delete registry server (cascades to tools).

    Args:
        server_id: Server UUID
        conn: Optional connection

    Side Effects:
        - Deletes all registry tools for this server (ON DELETE CASCADE)
        - Sets registry_server_id to NULL for linked workload servers
        - Triggers workload server re-calculation of embeddings
    """
```

#### list_servers

```python
async def list_servers(
    self,
    group: str | None = None,
    remote: bool | None = None,
    limit: int | None = None,
    offset: int = 0,
    conn: AsyncConnection | None = None,
) -> list[RegistryServer]:
    """List registry servers with optional filtering.

    Args:
        group: Optional group filter
        remote: Optional remote filter (True=remote, False=container, None=all)
        limit: Optional result limit
        offset: Result offset (default: 0)
        conn: Optional connection

    Returns:
        List of RegistryServer ordered by created_at DESC
    """
```

#### sync_server_vectors

```python
async def sync_server_vectors(
    self,
    server_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync registry_server_vector virtual table.

    Args:
        server_id: Optional specific server to sync (None = sync all)
        conn: Optional connection

    Note:
        Triggered when server_embedding changes.
        Updates sqlite-vec virtual table for vector similarity search.
    """
```

## WorkloadServerOps

**Purpose**: CRUD operations for mcpservers_workload table with registry relationship management.

### Methods

#### create_server

```python
async def create_server(
    self,
    workload_name: str,
    name: str,
    url: str | None,
    package: str | None,
    remote: bool,
    transport: TransportType,
    status: McpStatus,
    registry_server_id: str | None = None,
    registry_server_name: str | None = None,
    description: str | None = None,
    server_embedding: np.ndarray | None = None,
    group: str = "default",
    conn: AsyncConnection | None = None,
) -> WorkloadServer:
    """Create a new workload server.

    Args:
        workload_name: Unique workload identifier
        name: Server name
        url: Server URL (required if remote=True)
        package: Container package (required if remote=False)
        remote: True for remote server, False for container
        transport: Transport protocol type
        status: Server status
        registry_server_id: Optional link to registry server
        registry_server_name: Optional registry name (for tool context)
        description: Server description (required if registry_server_id is None)
        server_embedding: Vector embedding (required if registry_server_id is None)
        group: Server grouping (default: "default")
        conn: Optional connection

    Returns:
        Created WorkloadServer model

    Raises:
        ValueError: If validation fails (e.g., autonomous without description)
        IntegrityError: If workload_name duplicate
    """
```

#### get_server_by_id

```python
async def get_server_by_id(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> WorkloadServer | None:
    """Get workload server by ID.

    Args:
        server_id: Server UUID
        conn: Optional connection

    Returns:
        WorkloadServer if found, None otherwise
    """
```

#### get_server_by_workload_name

```python
async def get_server_by_workload_name(
    self,
    workload_name: str,
    conn: AsyncConnection | None = None,
) -> WorkloadServer | None:
    """Get workload server by workload name.

    Args:
        workload_name: Unique workload identifier
        conn: Optional connection

    Returns:
        WorkloadServer if found, None otherwise
    """
```

#### get_server_with_registry

```python
async def get_server_with_registry(
    self,
    server_id: str,
    registry_ops: RegistryServerOps,
    conn: AsyncConnection | None = None,
) -> WorkloadWithRegistry | None:
    """Get workload server with resolved registry relationship.

    Args:
        server_id: Workload server UUID
        registry_ops: Registry server ops for JOIN
        conn: Optional connection

    Returns:
        WorkloadWithRegistry if found (with registry if linked), None otherwise
    """
```

#### list_servers_by_registry

```python
async def list_servers_by_registry(
    self,
    registry_server_id: str,
    conn: AsyncConnection | None = None,
) -> list[WorkloadServer]:
    """List all workload servers linked to a registry server.

    Args:
        registry_server_id: Registry server UUID
        conn: Optional connection

    Returns:
        List of WorkloadServer with matching registry_server_id
    """
```

#### update_server

```python
async def update_server(
    self,
    server_id: str,
    name: str | None = None,
    status: McpStatus | None = None,
    url: str | None = None,
    package: str | None = None,
    registry_server_id: str | None = ...,  # Sentinel: None=no change, NULL=remove link
    registry_server_name: str | None = None,
    description: str | None = None,
    server_embedding: np.ndarray | None = None,
    transport: TransportType | None = None,
    group: str | None = None,
    conn: AsyncConnection | None = None,
) -> WorkloadServer:
    """Update workload server fields.

    Args:
        server_id: Server UUID
        name: Optional new name
        status: Optional new status
        url: Optional new URL (triggers re-matching if changed)
        package: Optional new package (triggers re-matching if changed)
        registry_server_id: Optional new registry link (... = no change)
        registry_server_name: Optional new registry name
        description: Optional new description
        server_embedding: Optional new embedding
        transport: Optional new transport
        group: Optional new group
        conn: Optional connection

    Returns:
        Updated WorkloadServer

    Raises:
        DbNotFoundError: If server not found

    Note:
        Changing URL or package should trigger re-matching logic externally.
    """
```

#### remove_registry_relationship

```python
async def remove_registry_relationship(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> None:
    """Remove registry relationship and calculate autonomous embeddings.

    Args:
        server_id: Workload server UUID
        conn: Optional connection

    Side Effects:
        - Sets registry_server_id to NULL
        - Sets registry_server_name to NULL
        - Calculates description from tools
        - Calculates server_embedding from tool embeddings (mean pooling)
        - Triggers tool re-embedding with workload name as context
    """
```

#### delete_server

```python
async def delete_server(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> None:
    """Delete workload server (cascades to tools).

    Args:
        server_id: Server UUID
        conn: Optional connection

    Side Effects:
        - Deletes all workload tools for this server (ON DELETE CASCADE)
        - Does NOT affect registry servers
    """
```

#### list_servers

```python
async def list_servers(
    self,
    group: str | None = None,
    status: McpStatus | None = None,
    has_registry_link: bool | None = None,
    limit: int | None = None,
    offset: int = 0,
    conn: AsyncConnection | None = None,
) -> list[WorkloadServer]:
    """List workload servers with optional filtering.

    Args:
        group: Optional group filter
        status: Optional status filter
        has_registry_link: Optional filter (True=linked, False=autonomous, None=all)
        limit: Optional result limit
        offset: Result offset (default: 0)
        conn: Optional connection

    Returns:
        List of WorkloadServer ordered by created_at DESC
    """
```

#### sync_server_vectors

```python
async def sync_server_vectors(
    self,
    server_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync workload_server_vector virtual table.

    Args:
        server_id: Optional specific server to sync (None = sync all)
        conn: Optional connection

    Note:
        Only syncs autonomous servers (registry_server_id is NULL).
        Linked servers use registry embeddings (not in workload vector table).
    """
```

## RegistryToolOps

**Purpose**: CRUD operations for tools_registry table.

### Methods

#### create_tool

```python
async def create_tool(
    self,
    server_id: str,
    details: McpTool,
    details_embedding: np.ndarray | None = None,
    conn: AsyncConnection | None = None,
) -> RegistryTool:
    """Create a new registry tool.

    Args:
        server_id: Parent registry server UUID
        details: MCP tool definition (name, description, inputSchema)
        details_embedding: Optional vector embedding
        conn: Optional connection

    Returns:
        Created RegistryTool model

    Raises:
        IntegrityError: If tool with same name exists for server
        ForeignKeyError: If server_id doesn't exist
    """
```

#### get_tool_by_id

```python
async def get_tool_by_id(
    self,
    tool_id: str,
    conn: AsyncConnection | None = None,
) -> RegistryTool | None:
    """Get registry tool by ID.

    Args:
        tool_id: Tool UUID
        conn: Optional connection

    Returns:
        RegistryTool if found, None otherwise
    """
```

#### get_tool_by_name

```python
async def get_tool_by_name(
    self,
    server_id: str,
    name: str,
    conn: AsyncConnection | None = None,
) -> RegistryTool | None:
    """Get registry tool by server and name.

    Args:
        server_id: Parent registry server UUID
        name: Tool name
        conn: Optional connection

    Returns:
        RegistryTool if found, None otherwise
    """
```

#### list_tools_by_server

```python
async def list_tools_by_server(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> list[RegistryTool]:
    """List all tools for a registry server.

    Args:
        server_id: Registry server UUID
        conn: Optional connection

    Returns:
        List of RegistryTool ordered by name ASC
    """
```

#### update_tool

```python
async def update_tool(
    self,
    tool_id: str,
    details: McpTool | None = None,
    details_embedding: np.ndarray | None = None,
    conn: AsyncConnection | None = None,
) -> RegistryTool:
    """Update registry tool fields.

    Args:
        tool_id: Tool UUID
        details: Optional new MCP tool definition
        details_embedding: Optional new embedding
        conn: Optional connection

    Returns:
        Updated RegistryTool

    Raises:
        DbNotFoundError: If tool not found
    """
```

#### delete_tool

```python
async def delete_tool(
    self,
    tool_id: str,
    conn: AsyncConnection | None = None,
) -> None:
    """Delete registry tool.

    Args:
        tool_id: Tool UUID
        conn: Optional connection
    """
```

#### bulk_upsert_tools

```python
async def bulk_upsert_tools(
    self,
    server_id: str,
    tools: list[McpTool],
    embeddings: list[np.ndarray] | None = None,
    conn: AsyncConnection | None = None,
) -> list[RegistryTool]:
    """Upsert multiple tools efficiently.

    Args:
        server_id: Parent registry server UUID
        tools: List of MCP tool definitions
        embeddings: Optional list of embeddings (same order as tools)
        conn: Optional connection

    Returns:
        List of created/updated RegistryTool

    Note:
        Uses INSERT OR REPLACE for efficiency.
        If embeddings provided, must match len(tools).
    """
```

#### sync_tool_vectors

```python
async def sync_tool_vectors(
    self,
    tool_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync registry_tool_vectors virtual table.

    Args:
        tool_id: Optional specific tool to sync (None = sync all)
        conn: Optional connection

    Note:
        Updates sqlite-vec virtual table for vector similarity search.
    """
```

#### sync_tool_fts

```python
async def sync_tool_fts(
    self,
    tool_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync registry_tool_fts virtual table.

    Args:
        tool_id: Optional specific tool to sync (None = sync all)
        conn: Optional connection

    Note:
        Updates FTS5 virtual table for full-text search.
    """
```

## WorkloadToolOps

**Purpose**: CRUD operations for tools_workload table (matches current Tool schema).

### Methods

#### create_tool

```python
async def create_tool(
    self,
    server_id: str,
    details: McpTool,
    details_embedding: np.ndarray | None = None,
    conn: AsyncConnection | None = None,
) -> WorkloadTool:
    """Create a new workload tool.

    Args:
        server_id: Parent workload server UUID
        details: MCP tool definition (name, description, inputSchema)
        details_embedding: Optional vector embedding (calculated with server name context)
        conn: Optional connection

    Returns:
        Created WorkloadTool model

    Raises:
        IntegrityError: If tool with same name exists for server
        ForeignKeyError: If server_id doesn't exist

    Note:
        Embedding failures should update parent server status, not tool availability.
    """
```

#### get_tool_by_id

```python
async def get_tool_by_id(
    self,
    tool_id: str,
    conn: AsyncConnection | None = None,
) -> WorkloadTool | None:
    """Get workload tool by ID.

    Args:
        tool_id: Tool UUID
        conn: Optional connection

    Returns:
        WorkloadTool if found, None otherwise
    """
```

#### get_tool_by_name

```python
async def get_tool_by_name(
    self,
    server_id: str,
    name: str,
    conn: AsyncConnection | None = None,
) -> WorkloadTool | None:
    """Get workload tool by server and name.

    Args:
        server_id: Parent workload server UUID
        name: Tool name
        conn: Optional connection

    Returns:
        WorkloadTool if found, None otherwise
    """
```

#### list_tools_by_server

```python
async def list_tools_by_server(
    self,
    server_id: str,
    conn: AsyncConnection | None = None,
) -> list[WorkloadTool]:
    """List tools for a workload server.

    Args:
        server_id: Workload server UUID
        conn: Optional connection

    Returns:
        List of WorkloadTool ordered by name ASC

    Note:
        To filter usable tools, query by parent server status (McpStatus.RUNNING).
        This matches existing pattern where tool availability is determined by server state.
    """
```

#### update_tool

```python
async def update_tool(
    self,
    tool_id: str,
    details: McpTool | None = None,
    details_embedding: np.ndarray | None = None,
    conn: AsyncConnection | None = None,
) -> WorkloadTool:
    """Update workload tool fields.

    Args:
        tool_id: Tool UUID
        details: Optional new MCP tool definition
        details_embedding: Optional new embedding
        conn: Optional connection

    Returns:
        Updated WorkloadTool

    Raises:
        DbNotFoundError: If tool not found
    """
```

#### delete_tool

```python
async def delete_tool(
    self,
    tool_id: str,
    conn: AsyncConnection | None = None,
) -> None:
    """Delete workload tool.

    Args:
        tool_id: Tool UUID
        conn: Optional connection
    """
```

#### bulk_upsert_tools

```python
async def bulk_upsert_tools(
    self,
    server_id: str,
    tools: list[McpTool],
    embeddings: list[np.ndarray] | None = None,
    server_name_context: str,
    conn: AsyncConnection | None = None,
) -> list[WorkloadTool]:
    """Upsert multiple tools efficiently with server name context.

    Args:
        server_id: Parent workload server UUID
        tools: List of MCP tool definitions
        embeddings: Optional list of embeddings (same order as tools)
        server_name_context: Server name to use for embedding context
        conn: Optional connection

    Returns:
        List of created/updated WorkloadTool

    Raises:
        EmbeddingCalculationFailedError: If embedding calculation fails for any tool,
            should update parent server status to STOPPED/ERROR

    Note:
        Uses INSERT OR REPLACE for efficiency.
        If embeddings provided, must match len(tools).
        Embedding failures affect parent server status, not individual tool availability.
    """
```

#### sync_tool_vectors

```python
async def sync_tool_vectors(
    self,
    tool_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync workload_tool_vectors virtual table.

    Args:
        tool_id: Optional specific tool to sync (None = sync all)
        conn: Optional connection

    Note:
        Only syncs tools from RUNNING servers (filtered by parent server status).
        Updates sqlite-vec virtual table for vector similarity search.
    """
```

#### sync_tool_fts

```python
async def sync_tool_fts(
    self,
    tool_id: str | None = None,
    conn: AsyncConnection | None = None,
) -> None:
    """Sync workload_tool_fts virtual table.

    Args:
        tool_id: Optional specific tool to sync (None = sync all)
        conn: Optional connection

    Note:
        Only syncs tools from RUNNING servers (filtered by parent server status).
        Updates FTS5 virtual table for full-text search.
    """
```

## Error Types

### Common Exceptions

```python
class DbNotFoundError(Exception):
    """Raised when database record not found."""
    pass

class DuplicateRegistryServersError(Exception):
    """Raised when multiple registry servers match workload identifier."""
    def __init__(self, duplicates: list[RegistryServer]):
        self.duplicates = duplicates
        super().__init__(f"Found {len(duplicates)} duplicate registry servers")

class EmbeddingCalculationFailedError(Exception):
    """Raised when embedding calculation fails for tool."""
    pass
```

## Transaction Patterns

### Example: Workload Ingestion with Registry Matching

```python
async with db.get_connection() as conn:
    async with conn.begin():
        # 1. Find matching registry servers
        registry_ops = RegistryServerOps(db)
        matches = await registry_ops.find_matching_servers(
            url=workload_url,
            package=workload_package,
            remote=is_remote,
            conn=conn
        )

        # 2. Validate single match
        if len(matches) > 1:
            raise DuplicateRegistryServersError(matches)

        registry_server_id = matches[0].id if matches else None
        registry_server_name = matches[0].name if matches else None

        # 3. Create/update workload server
        workload_ops = WorkloadServerOps(db)
        server = await workload_ops.create_server(
            workload_name=name,
            ...,
            registry_server_id=registry_server_id,
            registry_server_name=registry_server_name,
            conn=conn
        )

        # 4. Create tools with context
        tool_ops = WorkloadToolOps(db)
        tools = await tool_ops.bulk_upsert_tools(
            server_id=server.id,
            tools=mcp_tools,
            server_name_context=registry_server_name or server.name,
            conn=conn
        )
```

### Example: Registry Server Deletion with Workload Update

```python
async with db.get_connection() as conn:
    async with conn.begin():
        registry_ops = RegistryServerOps(db)
        workload_ops = WorkloadServerOps(db)

        # 1. Find affected workload servers
        affected = await workload_ops.list_servers_by_registry(
            registry_server_id=registry_id,
            conn=conn
        )

        # 2. Delete registry server (sets registry_server_id to NULL via ON DELETE SET NULL)
        await registry_ops.delete_server(registry_id, conn=conn)

        # 3. Recalculate embeddings for affected workload servers
        for workload in affected:
            await workload_ops.remove_registry_relationship(
                server_id=workload.id,
                conn=conn
            )
```

## References

- Pydantic models: `data-model.md`
- Database schema: `data-model.md`
- Transaction patterns: SQLAlchemy async transactions

