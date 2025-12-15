"""Base server operations with shared logic for registry and workload servers."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import (
    RegistryServer,
    RegistryServerUpdateDetails,
    WorkloadServer,
    WorkloadServerUpdateDetails,
)

logger = structlog.get_logger(__name__)


class BaseServerOps(ABC):
    """Base class for server operations with common CRUD logic.

    SECURITY NOTE - Table Name SQL Injection Risk:
    This class uses f-strings to construct SQL queries with table names from abstract
    properties (server_table_name, vector_table_name). While these are currently
    class-controlled constants, this pattern is fragile and could become a security
    vulnerability if refactored incorrectly.

    IMPORTANT: Table names MUST NEVER come from user input. They should only be
    hardcoded string literals in subclass implementations.

    Recommended improvements for additional safety:
    - Add validation in __init__ to verify table names match expected values
    - Use a table name registry/enum instead of abstract string properties
    - Consider using SQLAlchemy's table metadata for safer table references

    TRANSACTION HANDLING:
    All methods in this class accept an optional `conn` parameter for transaction support.
    If `conn` is provided, the method operates within that transaction. If `conn` is None,
    each database operation runs independently without explicit transaction boundaries.

    For multi-step operations that require atomicity, callers MUST establish a transaction
    using DatabaseConfig.begin_transaction() as shown in ingestion.py:

        async with self.db_config.begin_transaction() as conn:
            # Multiple operations within single transaction
            server = await server_ops.create_server(..., conn=conn)
            await tool_ops.create_tool(..., conn=conn)
            # Transaction commits automatically on success, rolls back on exception

    Without explicit transaction management, each operation commits independently,
    which may lead to partial updates on errors.
    """

    def __init__(self, db: DatabaseConfig):
        """Initialize ops with database configuration.

        Args:
            db: Database configuration providing async connection access

        Note:
            Table names from abstract properties are used in f-string SQL queries.
            Subclasses MUST return hardcoded string literals only - never user input.
        """
        self.db = db

        # Validate that abstract properties are implemented and return non-empty strings
        # This provides basic safety but does not prevent all SQL injection vectors
        if not self.server_table_name or not isinstance(self.server_table_name, str):
            raise ValueError("server_table_name must be a non-empty string")
        if not self.vector_table_name or not isinstance(self.vector_table_name, str):
            raise ValueError("vector_table_name must be a non-empty string")

    # Abstract properties that subclasses must implement
    @property
    @abstractmethod
    def server_table_name(self) -> str:
        """Name of the server table (e.g., 'mcpservers_registry', 'mcpservers_workload')."""

    @property
    @abstractmethod
    def vector_table_name(self) -> str:
        """Name of the vector virtual table (e.g., 'registry_server_vector')."""

    @property
    @abstractmethod
    def server_model_class(self) -> type[RegistryServer | WorkloadServer]:
        """Server model class (RegistryServer or WorkloadServer)."""

    @property
    @abstractmethod
    def server_update_model_class(
        self,
    ) -> type[RegistryServerUpdateDetails | WorkloadServerUpdateDetails]:
        """Server update details model class."""

    # Common CRUD operations
    async def get_server_by_id(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryServer | WorkloadServer:
        """Get server by ID.

        Args:
            server_id: Server UUID
            conn: Optional connection

        Returns:
            Server instance

        Raises:
            DbNotFoundError: If server not found
        """
        query = f"SELECT * FROM {self.server_table_name} WHERE id = :id"  # nosec B608 - Table name is code-controlled, params are safe
        results = await self.db.execute_query(query, {"id": server_id}, conn=conn)
        if not results:
            raise DbNotFoundError(
                f"Server with ID {server_id} not found in {self.server_table_name}."
            )

        server_data = dict(results[0]._mapping)
        # Deserialize server_embedding from bytes to numpy array
        if server_data["server_embedding"] is not None:
            server_data["server_embedding"] = np.frombuffer(
                server_data["server_embedding"], dtype=np.float32
            )
        return self.server_model_class.model_validate(server_data)

    async def get_server_by_name(
        self,
        name: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryServer | WorkloadServer:
        """Get server by name.

        Args:
            name: Server name
            conn: Optional connection

        Returns:
            Server instance

        Raises:
            DbNotFoundError: If server not found
        """
        query = f"SELECT * FROM {self.server_table_name} WHERE name = :name"  # nosec B608 - Table name is code-controlled, params are safe
        results = await self.db.execute_query(query, {"name": name}, conn=conn)
        if not results:
            raise DbNotFoundError(
                f"Server with name '{name}' not found in {self.server_table_name}."
            )

        server_data = dict(results[0]._mapping)
        # Deserialize server_embedding from bytes to numpy array
        if server_data["server_embedding"] is not None:
            server_data["server_embedding"] = np.frombuffer(
                server_data["server_embedding"], dtype=np.float32
            )
        return self.server_model_class.model_validate(server_data)

    async def update_server(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
        **kwargs: Any,
    ) -> RegistryServer | WorkloadServer:
        """Update server fields using Pydantic validation.

        Args:
            server_id: Server UUID
            conn: Optional connection
            **kwargs: Update fields

        Returns:
            Updated server

        Raises:
            DbNotFoundError: If server not found
            ValueError: If validation fails (from Pydantic model)
        """
        # First verify server exists (raises DbNotFoundError if not found)
        existing_server = await self.get_server_by_id(server_id, conn=conn)

        # Validate and get update fields using Pydantic
        server_update_details = self.server_update_model_class.model_validate(kwargs)
        if not server_update_details.needs_update():
            return existing_server

        # Build dynamic SET clause from validated fields
        update_fields = server_update_details.get_update_fields()
        # Quote 'group' field since it's a reserved keyword
        set_clauses = [
            f'"{field}" = :{field}' if field == "group" else f"{field} = :{field}"
            for field in update_fields.keys()
        ]
        query = f"UPDATE {self.server_table_name} SET {', '.join(set_clauses)} WHERE id = :id"  # nosec B608 - Table name is code-controlled, params are safe

        # Add server_id to params
        params = update_fields.copy()
        params["id"] = server_id

        await self.db.execute_non_query(query, params, conn=conn)
        logger.info(
            f"Updated {self.server_table_name} server",
            server_id=server_id,
            fields=list(update_fields.keys()),
        )

        # Return updated server (raises DbNotFoundError if not found)
        return await self.get_server_by_id(server_id, conn=conn)

    async def delete_server(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
        **kwargs: Any,
    ) -> None:
        """Delete server (cascades to tools).

        Args:
            server_id: Server UUID
            conn: Optional connection
            **kwargs: Additional subclass-specific parameters
        """
        query = f"DELETE FROM {self.server_table_name} WHERE id = :id"  # nosec B608 - Table name is code-controlled, params are safe
        await self.db.execute_non_query(query, {"id": server_id}, conn=conn)
        logger.info(f"Deleted {self.server_table_name} server", server_id=server_id)

    async def list_servers(
        self,
        group: str | None = None,
        remote: bool | None = None,
        limit: int | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryServer] | list[WorkloadServer]:
        """List servers with optional filtering.

        Args:
            group: Optional group filter
            remote: Optional remote filter (True=remote, False=container, None=all)
            limit: Optional result limit
            conn: Optional connection

        Returns:
            List of servers ordered by created_at DESC
        """
        # Build WHERE clause
        where_clauses = []
        params: dict[str, Any] = {}

        if group is not None:
            where_clauses.append('"group" = :group')
            params["group"] = group
        if remote is not None:
            where_clauses.append("remote = :remote")
            params["remote"] = 1 if remote else 0

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Build LIMIT clause
        limit_sql = ""
        if limit is not None:
            limit_sql = "LIMIT :limit"
            params["limit"] = limit

        query = f"""
        SELECT * FROM {self.server_table_name}
        {where_sql}
        ORDER BY created_at DESC
        {limit_sql}
        """  # nosec B608 - Table name is code-controlled, params are safe

        results = await self.db.execute_query(query, params, conn=conn)
        servers = []
        for row in results:
            server_data = dict(row._mapping)
            # Deserialize server_embedding from bytes to numpy array
            if server_data["server_embedding"] is not None:
                server_data["server_embedding"] = np.frombuffer(
                    server_data["server_embedding"], dtype=np.float32
                )
            servers.append(self.server_model_class.model_validate(server_data))
        return servers

    async def sync_server_vectors(
        self,
        server_id: str | None = None,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Sync server_vector virtual table.

        Args:
            server_id: Optional specific server to sync (None = sync all)
            conn: Optional connection

        Note:
            Updates sqlite-vec virtual table for vector similarity search.
        """
        if server_id is not None:
            # Sync specific server
            delete_query = f"DELETE FROM {self.vector_table_name} WHERE server_id = :server_id"  # nosec B608 - Table name is code-controlled, params are safe
            await self.db.execute_non_query(delete_query, {"server_id": server_id}, conn=conn)

            # Get server and insert if it has embedding
            server = await self.get_server_by_id(server_id, conn=conn)
            if server.server_embedding is not None:
                insert_query = f"""
                INSERT INTO {self.vector_table_name} (server_id, embedding)
                VALUES (:server_id, :embedding)
                """  # nosec B608 - Table name is code-controlled, params are safe
                params = {
                    "server_id": server_id,
                    "embedding": server.server_embedding.tobytes(),
                }
                await self.db.execute_non_query(insert_query, params, conn=conn)
                logger.debug(f"Synced {self.vector_table_name} server vector", server_id=server_id)
        else:
            # Sync all servers - rebuild entire virtual table
            delete_all_query = f"DELETE FROM {self.vector_table_name}"  # nosec B608 - Table name is code-controlled, params are safe
            await self.db.execute_non_query(delete_all_query, {}, conn=conn)

            # Get all servers with embeddings
            query = f"""
            SELECT id, server_embedding FROM {self.server_table_name}
            WHERE server_embedding IS NOT NULL
            """  # nosec B608 - Table name is code-controlled, params are safe
            results = await self.db.execute_query(query, {}, conn=conn)

            # Batch insert
            for row in results:
                server_id_val = row._mapping["id"]
                embedding_bytes = row._mapping["server_embedding"]
                insert_query = f"""
                INSERT INTO {self.vector_table_name} (server_id, embedding)
                VALUES (:server_id, :embedding)
                """  # nosec B608 - Table name is code-controlled, params are safe
                params = {
                    "server_id": server_id_val,
                    "embedding": embedding_bytes,
                }
                await self.db.execute_non_query(insert_query, params, conn=conn)

            logger.info(f"Synced all {self.vector_table_name} vectors")

    async def find_similar_servers(
        self,
        query_embedding: np.ndarray,
        limit: int,
        distance_threshold: float,
        allowed_groups: list[str] | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryServer] | list[WorkloadServer]:
        """Find servers similar to a query embedding using cosine distance search.

        Args:
            query_embedding: The embedding vector for the search query
            limit: Maximum number of similar servers to return
            distance_threshold: Maximum cosine distance (0=identical, 2=opposite)
            allowed_groups: Optional list of group names to filter by
            conn: Optional connection

        Returns:
            List of similar servers ordered by similarity (most similar first)

        Note:
            Uses server_vector virtual table with cosine distance metric.
        """
        # Convert query embedding to sqlite-vec format
        query_embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

        # Build group filter
        group_filter = ""
        if allowed_groups and len(allowed_groups) > 0:
            group_placeholders = ",".join([f":group{i}" for i in range(len(allowed_groups))])
            group_filter = f'AND s."group" IN ({group_placeholders})'

        params: dict[str, Any] = {
            "query_embedding": query_embedding_str,
            "limit": limit,
        }

        # Perform cosine similarity search
        similarity_query = f"""
        SELECT
            sv.server_id,
            sv.distance
        FROM {self.vector_table_name} sv
        JOIN {self.server_table_name} s ON sv.server_id = s.id
        WHERE sv.embedding MATCH :query_embedding
        {group_filter}
        AND k = :limit
        ORDER BY sv.distance
        """  # nosec B608 - Table names are code-controlled, params are safe

        # Add group parameters
        if allowed_groups:
            for i, group in enumerate(allowed_groups):
                params[f"group{i}"] = group

        similarity_results = await self.db.execute_query(similarity_query, params, conn=conn)

        # Log results
        distances = [result._mapping["distance"] for result in similarity_results]
        server_ids = [result._mapping["server_id"] for result in similarity_results]
        logger.debug(
            f"{self.server_table_name} similarity search",
            distances=distances,
            server_ids=server_ids,
            distance_threshold=distance_threshold,
            result_count=len(similarity_results),
        )

        # Filter by distance threshold
        filtered_results = [
            result
            for result in similarity_results
            if result._mapping["distance"] is not None
            and result._mapping["distance"] < distance_threshold
        ]

        if not filtered_results:
            return []

        # Get full server details
        server_ids_filtered = [result._mapping["server_id"] for result in filtered_results]
        server_distances = {
            result._mapping["server_id"]: result._mapping["distance"] for result in filtered_results
        }

        placeholders = ",".join([f":id{i}" for i in range(len(server_ids_filtered))])
        details_query = f"""
        SELECT * FROM {self.server_table_name}
        WHERE id IN ({placeholders})
        ORDER BY created_at
        """  # nosec B608 - Table name is code-controlled, params are safe

        params_details = {f"id{i}": server_id for i, server_id in enumerate(server_ids_filtered)}
        server_results = await self.db.execute_query(details_query, params_details, conn=conn)

        servers = []
        for row in server_results:
            server_data = dict(row._mapping)

            # Deserialize embedding
            if server_data["server_embedding"] is not None:
                server_data["server_embedding"] = np.frombuffer(
                    server_data["server_embedding"], dtype=np.float32
                )

            server = self.server_model_class.model_validate(server_data)
            servers.append(server)

        # Sort by distance (most similar first)
        servers.sort(key=lambda s: server_distances[s.id])

        logger.info(
            f"Found similar {self.server_table_name} servers",
            result_count=len(servers),
        )

        return servers
