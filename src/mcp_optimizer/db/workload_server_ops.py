"""Workload server operations for database table separation feature."""

import uuid
from datetime import datetime, timezone
from typing import Any, cast

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.base_server_ops import BaseServerOps
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import (
    McpStatus,
    TransportType,
    WorkloadServer,
    WorkloadServerUpdateDetails,
    WorkloadWithRegistry,
)

logger = structlog.get_logger(__name__)


class WorkloadServerOps(BaseServerOps):
    """Repository for workload MCP server operations.

    Extends BaseServerOps with workload-specific table names and models.
    All common CRUD operations are inherited from the base class.
    """

    @property
    def server_table_name(self) -> str:
        """Name of the server table."""
        return "mcpservers_workload"

    @property
    def vector_table_name(self) -> str:
        """Name of the vector virtual table."""
        return "workload_server_vector"

    @property
    def server_model_class(self) -> type[WorkloadServer]:
        """Server model class."""
        return WorkloadServer

    @property
    def server_update_model_class(self) -> type[WorkloadServerUpdateDetails]:
        """Server update details model class."""
        return WorkloadServerUpdateDetails

    # Workload-specific methods
    async def create_server(
        self,
        name: str,
        url: str,
        workload_identifier: str,
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
            name: Unique workload identifier
            url: Server URL (required, NOT NULL)
            workload_identifier: Container package or remote URL identifier (NOT NULL)
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
            IntegrityError: If name duplicate
        """
        # Validate the inputs using Pydantic model
        new_server = WorkloadServer(
            id=str(uuid.uuid4()),
            name=name,
            url=url,
            workload_identifier=workload_identifier,
            remote=remote,
            transport=transport,
            status=status,
            registry_server_id=registry_server_id,
            registry_server_name=registry_server_name,
            description=description,
            server_embedding=server_embedding,
            group=group,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        query = """
        INSERT INTO mcpservers_workload (
            id, name, url, workload_identifier, remote, transport, status,
            registry_server_id, registry_server_name, description, server_embedding,
            "group", last_updated, created_at
        )
        VALUES (
            :id, :name, :url, :workload_identifier, :remote, :transport, :status,
            :registry_server_id, :registry_server_name, :description, :server_embedding,
            :group, :last_updated, :created_at
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
            "Created workload server",
            server_id=new_server.id,
            workload_name=name,
            has_registry_link=registry_server_id is not None,
        )
        return new_server

    async def get_server_by_workload_name(
        self,
        name: str,
        conn: AsyncConnection | None = None,
    ) -> WorkloadServer:
        """Get workload server by workload name.

        Args:
            name: Unique workload identifier
            conn: Optional connection

        Returns:
            WorkloadServer

        Raises:
            DbNotFoundError: If server not found
        """
        query = "SELECT * FROM mcpservers_workload WHERE name = :name"
        results = await self.db.execute_query(query, {"name": name}, conn=conn)
        if not results:
            raise DbNotFoundError(f"Workload server with name '{name}' not found.")

        server_data = dict(results[0]._mapping)
        # Deserialize server_embedding from bytes to numpy array
        if server_data["server_embedding"] is not None:
            server_data["server_embedding"] = np.frombuffer(
                server_data["server_embedding"], dtype=np.float32
            )
        return WorkloadServer.model_validate(server_data)

    async def get_server_with_registry(
        self,
        server_id: str,
        registry_ops: Any,  # RegistryServerOps to avoid circular import
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
        server = await self.get_server_by_id(server_id, conn=conn)
        if not server:
            return None

        # Cast to WorkloadServer since this is WorkloadServerOps
        workload = cast(WorkloadServer, server)

        # Get registry if linked
        registry = None
        if workload.registry_server_id:
            registry = await registry_ops.get_server_by_id(workload.registry_server_id, conn=conn)

        return WorkloadWithRegistry(workload=workload, registry=registry)

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
        query = """
        SELECT * FROM mcpservers_workload
        WHERE registry_server_id = :registry_server_id
        ORDER BY created_at DESC
        """
        results = await self.db.execute_query(
            query, {"registry_server_id": registry_server_id}, conn=conn
        )

        servers = []
        for row in results:
            server_data = dict(row._mapping)
            # Deserialize server_embedding from bytes to numpy array
            if server_data["server_embedding"] is not None:
                server_data["server_embedding"] = np.frombuffer(
                    server_data["server_embedding"], dtype=np.float32
                )
            servers.append(WorkloadServer.model_validate(server_data))
        return servers

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

        Note:
            This is a placeholder that sets registry fields to NULL.
            Full implementation of autonomous embedding calculation
            should be done in ingestion.py or a separate service.
        """
        query = """
        UPDATE mcpservers_workload
        SET registry_server_id = NULL,
            registry_server_name = NULL,
            last_updated = :last_updated
        WHERE id = :id
        """
        params = {
            "id": server_id,
            "last_updated": datetime.now(timezone.utc),
        }
        await self.db.execute_non_query(query, params, conn=conn)
        logger.info("Removed registry relationship", server_id=server_id)
