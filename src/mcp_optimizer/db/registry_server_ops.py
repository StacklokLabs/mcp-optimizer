"""Registry server operations for database table separation feature."""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.base_server_ops import BaseServerOps
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import RegistryServer, RegistryServerUpdateDetails, TransportType

if TYPE_CHECKING:
    from mcp_optimizer.db.workload_server_ops import WorkloadServerOps

logger = structlog.get_logger(__name__)


class RegistryServerOps(BaseServerOps):
    """Repository for registry MCP server operations.

    Extends BaseServerOps with registry-specific table names and models.
    All common CRUD operations are inherited from the base class.
    """

    @property
    def server_table_name(self) -> str:
        """Name of the server table."""
        return "mcpservers_registry"

    @property
    def vector_table_name(self) -> str:
        """Name of the vector virtual table."""
        return "registry_server_vector"

    @property
    def server_model_class(self) -> type[RegistryServer]:
        """Server model class."""
        return RegistryServer

    @property
    def server_update_model_class(self) -> type[RegistryServerUpdateDetails]:
        """Server update details model class."""
        return RegistryServerUpdateDetails

    # Registry-specific methods
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
        # Validate the inputs using Pydantic model
        new_server = RegistryServer(
            id=str(uuid.uuid4()),
            name=name,
            url=url,
            package=package,
            remote=remote,
            transport=transport,
            description=description,
            server_embedding=server_embedding,
            group=group,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        query = """
        INSERT INTO mcpservers_registry (
            id, name, url, package, remote, transport, description,
            server_embedding, "group", last_updated, created_at
        )
        VALUES (
            :id, :name, :url, :package, :remote, :transport, :description,
            :server_embedding, :group, :last_updated, :created_at
        )
        """

        # Convert the model to dict and handle embedding serialization
        params = new_server.model_dump()
        if new_server.server_embedding is not None:
            params["server_embedding"] = new_server.server_embedding.tobytes()
        else:
            params["server_embedding"] = None

        await self.db.execute_non_query(query, params, conn=conn)
        logger.debug(
            "Created registry server",
            server_id=new_server.id,
            name=name,
            remote=remote,
        )
        return new_server

    async def get_server_by_url(
        self,
        url: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryServer:
        """Get remote registry server by URL.

        Args:
            url: Server URL
            conn: Optional connection

        Returns:
            RegistryServer (with remote=True)

        Raises:
            DbNotFoundError: If server not found
        """
        query = "SELECT * FROM mcpservers_registry WHERE url = :url AND remote = 1"
        results = await self.db.execute_query(query, {"url": url}, conn=conn)
        if not results:
            raise DbNotFoundError(f"Registry server with URL '{url}' not found.")

        server_data = dict(results[0]._mapping)
        # Deserialize server_embedding from bytes to numpy array
        if server_data["server_embedding"] is not None:
            server_data["server_embedding"] = np.frombuffer(
                server_data["server_embedding"], dtype=np.float32
            )
        return RegistryServer.model_validate(server_data)

    async def get_server_by_package(
        self,
        package: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryServer:
        """Get container registry server by package.

        Args:
            package: Container package/image name
            conn: Optional connection

        Returns:
            RegistryServer (with remote=False)

        Raises:
            DbNotFoundError: If server not found
        """
        query = "SELECT * FROM mcpservers_registry WHERE package = :package AND remote = 0"
        results = await self.db.execute_query(query, {"package": package}, conn=conn)
        if not results:
            raise DbNotFoundError(f"Registry server with package '{package}' not found.")

        server_data = dict(results[0]._mapping)
        # Deserialize server_embedding from bytes to numpy array
        if server_data["server_embedding"] is not None:
            server_data["server_embedding"] = np.frombuffer(
                server_data["server_embedding"], dtype=np.float32
            )
        return RegistryServer.model_validate(server_data)

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
        if remote:
            if url is None:
                return []
            query = "SELECT * FROM mcpservers_registry WHERE url = :url AND remote = 1"
            params = {"url": url}
        else:
            if package is None:
                return []
            query = "SELECT * FROM mcpservers_registry WHERE package = :package AND remote = 0"
            params = {"package": package}

        results = await self.db.execute_query(query, params, conn=conn)
        servers = []
        for row in results:
            server_data = dict(row._mapping)
            # Deserialize server_embedding from bytes to numpy array
            if server_data["server_embedding"] is not None:
                server_data["server_embedding"] = np.frombuffer(
                    server_data["server_embedding"], dtype=np.float32
                )
            servers.append(RegistryServer.model_validate(server_data))
        return servers

    async def delete_server(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete registry server and remove relationships from workload servers.

        Args:
            server_id: Registry server UUID
            conn: Optional connection
            **kwargs: Additional arguments including:
                - workload_ops: Optional WorkloadServerOps instance for handling relationships

        Side Effects:
            If workload_ops provided:
            - Finds all workload servers linked to this registry server
            - Removes registry relationship from each
              (sets to NULL, calculates autonomous embeddings)
            Then:
            - Deletes the registry server (cascades to tools)

        Note:
            If workload_ops is not provided, relationships are not cleaned up.
            This is for cases where the workload table doesn't exist yet or isn't being used.
        """
        # Extract workload_ops from kwargs if provided
        workload_ops: WorkloadServerOps | None = kwargs.get("workload_ops")

        # If workload_ops provided, clean up relationships first
        if workload_ops is not None:
            # Find all workload servers linked to this registry server
            linked_workloads = await workload_ops.list_servers_by_registry(server_id, conn=conn)

            # Remove registry relationship from each
            for workload_server in linked_workloads:
                await workload_ops.remove_registry_relationship(workload_server.id, conn=conn)
                logger.debug(
                    "Removed registry relationship from workload",
                    workload_id=workload_server.id,
                    workload_name=workload_server.name,
                    registry_id=server_id,
                )

            if linked_workloads:
                logger.info(
                    "Removed registry relationships before deletion",
                    registry_id=server_id,
                    affected_workloads=len(linked_workloads),
                )

        # Call parent's delete_server to perform the actual deletion
        await super().delete_server(server_id, conn=conn)

    async def update_server(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
        **kwargs: Any,
    ) -> RegistryServer:
        """Update registry server with name change detection.

        Args:
            server_id: Server UUID
            conn: Optional connection
            **kwargs: Update fields

        Returns:
            Updated registry server

        Side Effects:
            If name changes:
            - Logs warning about linked workload servers needing re-embedding
            - Updates workload_server.registry_server_name for linked servers

        Note:
            When registry server name changes, linked workload tools should be
            re-embedded with the new registry name context. This happens during
            the next ingestion cycle, not immediately.
        """
        # Get current server to check for name changes
        existing_server = await self.get_server_by_id(server_id, conn=conn)

        # Check if name is changing
        name_changed = "name" in kwargs and kwargs["name"] != existing_server.name

        # Call parent's update_server
        updated_server = cast(
            RegistryServer, await super().update_server(server_id, conn=conn, **kwargs)
        )

        # If name changed, update registry_server_name in linked workload servers
        if name_changed:
            new_name = kwargs["name"]
            logger.warning(
                "Registry server name changed - updating linked workload servers",
                registry_id=server_id,
                old_name=existing_server.name,
                new_name=new_name,
            )

            # Update registry_server_name in all linked workload servers
            update_query = """
            UPDATE mcpservers_workload
            SET registry_server_name = :new_name,
                last_updated = :last_updated
            WHERE registry_server_id = :registry_id
            """
            params = {
                "new_name": new_name,
                "registry_id": server_id,
                "last_updated": datetime.now(timezone.utc),
            }
            await self.db.execute_non_query(update_query, params, conn=conn)

            logger.info(
                "Updated registry_server_name in linked workload servers",
                registry_id=server_id,
                new_name=new_name,
            )

        return updated_server
