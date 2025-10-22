"""Base tool operations with shared logic for registry and workload tools."""

from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any

import numpy as np
import structlog
from mcp.types import Tool as McpTool
from sqlalchemy import Row, Sequence
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import (
    BaseToolWithMetadata,
    McpStatus,
    RegistryTool,
    RegistryToolUpdateDetails,
    RegistryToolWithMetadata,
    WorkloadTool,
    WorkloadToolUpdateDetails,
    WorkloadToolWithMetadata,
)

logger = structlog.get_logger(__name__)

# FTS query sanitization
# Words that FTS5 might interpret as column names, causing "no such column" errors
DEFAULT_FTS_PROBLEMATIC_WORDS = {
    "name",
    "description",
    "schema",
    "input",
    "output",
    "type",
    "properties",
    "required",
    "title",
    "id",
    "tool",
    "server",
    "meta",
    "data",
    "content",
    "text",
    "value",
    "field",
    "column",
    "table",
    "index",
    "key",
    "primary",
}


class BaseToolOps(ABC):
    """Base class for tool operations with common CRUD logic.

    SECURITY NOTE - Table Name SQL Injection Risk:
    This class uses f-strings to construct SQL queries with table names from abstract
    properties (tool_table_name, server_table_name, vector_table_name, fts_table_name).
    While these are currently class-controlled constants, this pattern is fragile and
    could become a security vulnerability if refactored incorrectly.

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
        if not self.tool_table_name or not isinstance(self.tool_table_name, str):
            raise ValueError("tool_table_name must be a non-empty string")
        if not self.server_table_name or not isinstance(self.server_table_name, str):
            raise ValueError("server_table_name must be a non-empty string")
        if not self.vector_table_name or not isinstance(self.vector_table_name, str):
            raise ValueError("vector_table_name must be a non-empty string")
        if not self.fts_table_name or not isinstance(self.fts_table_name, str):
            raise ValueError("fts_table_name must be a non-empty string")

    # Abstract properties that subclasses must implement
    @property
    @abstractmethod
    def tool_table_name(self) -> str:
        """Name of the main tool table (e.g., 'tools_registry', 'tools_workload')."""

    @property
    @abstractmethod
    def server_table_name(self) -> str:
        """Name of the server table (e.g., 'mcpservers_registry', 'mcpservers_workload')."""

    @property
    @abstractmethod
    def vector_table_name(self) -> str:
        """Name of the vector virtual table (e.g., 'registry_tool_vectors')."""

    @property
    @abstractmethod
    def fts_table_name(self) -> str:
        """Name of the FTS virtual table (e.g., 'registry_tool_fts')."""

    @property
    @abstractmethod
    def tool_model_class(self) -> type[RegistryTool | WorkloadTool]:
        """Tool model class (RegistryTool or WorkloadTool)."""

    @property
    @abstractmethod
    def tool_update_model_class(
        self,
    ) -> type[RegistryToolUpdateDetails | WorkloadToolUpdateDetails]:
        """Tool update details model class."""

    @property
    @abstractmethod
    def tool_with_metadata_model_class(
        self,
    ) -> type[RegistryToolWithMetadata | WorkloadToolWithMetadata]:
        """Tool with metadata model class."""

    def should_filter_by_status(self) -> bool:
        """Whether to filter by server status. Override in subclasses."""
        return False

    def get_status_filter(self) -> McpStatus | None:
        """Get status to filter by. Returns None if no filtering needed."""
        return McpStatus.RUNNING if self.should_filter_by_status() else None

    async def get_tool_by_id(
        self,
        tool_id: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryTool | WorkloadTool:
        """Get tool by ID.

        Args:
            tool_id: Tool UUID
            conn: Optional connection

        Returns:
            Tool instance

        Raises:
            DbNotFoundError: If tool not found
        """
        query = f"SELECT * FROM {self.tool_table_name} WHERE id = :id"
        results = await self.db.execute_query(query, {"id": tool_id}, conn=conn)
        if not results:
            raise DbNotFoundError(f"Tool with ID {tool_id} not found in {self.tool_table_name}.")

        row_data = dict(results[0]._mapping)
        # Parse JSON details back to McpTool
        row_data["details"] = McpTool.model_validate_json(row_data["details"])
        # Deserialize embedding from bytes to numpy array
        if row_data["details_embedding"] is not None:
            row_data["details_embedding"] = np.frombuffer(
                row_data["details_embedding"], dtype=np.float32
            )
        return self.tool_model_class.model_validate(row_data)

    async def get_tool_by_server_and_name(
        self,
        server_name: str,
        name: str,
        conn: AsyncConnection | None = None,
    ) -> RegistryTool | WorkloadTool:
        """Get tool by server and name.

        Args:
            server_name: Server name
            name: Tool name
            conn: Optional connection

        Returns:
            Tool instance

        Raises:
            DbNotFoundError: If tool not found
        """
        query = f"""
        SELECT * FROM {self.tool_table_name} t
        JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
        WHERE s.name = :server_name
        AND json_extract(t.details, '$.name') = :name
        """
        results = await self.db.execute_query(
            query, {"server_name": server_name, "name": name}, conn=conn
        )
        if not results:
            raise DbNotFoundError(
                f"Tool with name '{name}' not found for server '{server_name}' "
                f"in {self.tool_table_name}."
            )

        row_data = dict(results[0]._mapping)
        # Parse JSON details back to McpTool
        row_data["details"] = McpTool.model_validate_json(row_data["details"])
        # Deserialize embedding from bytes to numpy array
        if row_data["details_embedding"] is not None:
            row_data["details_embedding"] = np.frombuffer(
                row_data["details_embedding"], dtype=np.float32
            )
        return self.tool_model_class.model_validate(row_data)

    async def list_tools_by_server(
        self,
        server_id: str,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryTool | WorkloadTool]:
        """List all tools for a server.

        Args:
            server_id: Server UUID
            conn: Optional connection

        Returns:
            List of tools ordered by name ASC
        """
        query = f"""
        SELECT * FROM {self.tool_table_name}
        WHERE mcpserver_id = :server_id
        ORDER BY json_extract(details, '$.name') ASC
        """
        results = await self.db.execute_query(query, {"server_id": server_id}, conn=conn)

        tools = []
        for row in results:
            row_data = dict(row._mapping)
            # Parse JSON details back to McpTool
            row_data["details"] = McpTool.model_validate_json(row_data["details"])
            # Deserialize embedding from bytes to numpy array
            if row_data["details_embedding"] is not None:
                row_data["details_embedding"] = np.frombuffer(
                    row_data["details_embedding"], dtype=np.float32
                )
            tools.append(self.tool_model_class.model_validate(row_data))
        return tools

    async def get_all_tools(
        self,
        server_statuses: list[str] | None = None,
        allowed_groups: list[str] | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Get all tools from all servers with their metadata.

        Args:
            server_statuses: Optional list of status values to filter servers by
            allowed_groups: Optional list of group names to filter by
            conn: Optional connection

        Returns:
            List of ToolWithMetadata objects containing tools with server info
        """
        # Build status filter
        status_filter = ""
        if server_statuses and len(server_statuses) > 0:
            status_placeholders = ",".join([f":status{i}" for i in range(len(server_statuses))])
            status_filter = f"AND s.status IN ({status_placeholders})"

        # Build group filter
        group_filter = ""
        if allowed_groups and len(allowed_groups) > 0:
            group_placeholders = ",".join([f":group{i}" for i in range(len(allowed_groups))])
            group_filter = f'AND s."group" IN ({group_placeholders})'

        query = f"""
        SELECT t.*, s.name as server_name, s.description as server_description
        FROM {self.tool_table_name} t
        JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
        WHERE 1=1
        {status_filter}
        {group_filter}
        ORDER BY s.name, t.created_at
        """

        params: dict[str, Any] = {}
        if server_statuses:
            for i, status in enumerate(server_statuses):
                params[f"status{i}"] = status
        if allowed_groups:
            for i, group in enumerate(allowed_groups):
                params[f"group{i}"] = group

        results = await self.db.execute_query(query, params=params, conn=conn)

        tools_with_metadata = []
        for row in results:
            row_data = dict(row._mapping)

            # Extract server info
            server_name = row_data.pop("server_name")
            server_description = row_data.pop("server_description")

            # Parse JSON details back to McpTool
            row_data["details"] = McpTool.model_validate_json(row_data["details"])
            # Deserialize embedding from bytes to numpy array
            if row_data["details_embedding"] is not None:
                row_data["details_embedding"] = np.frombuffer(
                    row_data["details_embedding"], dtype=np.float32
                )

            tool = self.tool_model_class.model_validate(row_data)
            tool_with_metadata = self.tool_with_metadata_model_class(
                tool=tool,
                server_name=server_name,
                server_description=server_description,
                distance=0.0,  # Distance not applicable for get_all
            )
            tools_with_metadata.append(tool_with_metadata)

        return tools_with_metadata

    async def update_tool(
        self,
        tool_id: str,
        conn: AsyncConnection | None = None,
        **kwargs: Any,
    ) -> RegistryTool | WorkloadTool:
        """Update tool fields using Pydantic validation.

        Args:
            tool_id: Tool UUID
            conn: Optional connection
            **kwargs: Update fields (details, details_embedding)

        Returns:
            Updated tool

        Raises:
            DbNotFoundError: If tool not found
            ValueError: If validation fails (from Pydantic model)
        """
        # First verify tool exists (raises DbNotFoundError if not found)
        existing_tool = await self.get_tool_by_id(tool_id, conn=conn)

        # Validate and get update fields using Pydantic
        tool_update_details = self.tool_update_model_class.model_validate(kwargs)
        if not tool_update_details.needs_update():
            return existing_tool

        # Build dynamic SET clause from validated fields
        update_fields = tool_update_details.get_update_fields()
        set_clauses = [f"{field} = :{field}" for field in update_fields.keys()]
        query = f"UPDATE {self.tool_table_name} SET {', '.join(set_clauses)} WHERE id = :id"

        # Add tool_id to params
        params = update_fields.copy()
        params["id"] = tool_id

        await self.db.execute_non_query(query, params, conn=conn)
        logger.debug(f"Updated {self.tool_table_name} tool", tool_id=tool_id)

        # Return updated tool (raises DbNotFoundError if not found)
        return await self.get_tool_by_id(tool_id, conn=conn)

    async def delete_tool(
        self,
        tool_id: str,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Delete tool.

        Args:
            tool_id: Tool UUID
            conn: Optional connection
        """
        query = f"DELETE FROM {self.tool_table_name} WHERE id = :id"
        await self.db.execute_non_query(query, {"id": tool_id}, conn=conn)
        logger.debug(f"Deleted {self.tool_table_name} tool", tool_id=tool_id)

    async def delete_tools_by_server(
        self, server_id: str, conn: AsyncConnection | None = None
    ) -> int:
        """Delete all tools for a specific server.

        Args:
            server_id: Server UUID
            conn: Optional connection

        Returns:
            Number of tools deleted
        """
        # First get count of tools to be deleted
        count_query = (
            f"SELECT COUNT(*) as count FROM {self.tool_table_name} WHERE mcpserver_id = :server_id"
        )
        results = await self.db.execute_query(count_query, {"server_id": server_id}, conn=conn)
        count = results[0]._mapping["count"] if results else 0

        # Delete the tools
        query = f"DELETE FROM {self.tool_table_name} WHERE mcpserver_id = :server_id"
        await self.db.execute_non_query(query, {"server_id": server_id}, conn=conn)
        logger.debug(
            f"Deleted {count} tools from {self.tool_table_name}",
            server_id=server_id,
            count=count,
        )
        return count

    async def sync_tool_vectors(
        self,
        tool_id: str | None = None,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Sync tool_vectors virtual table.

        Args:
            tool_id: Optional specific tool to sync (None = sync all)
            conn: Optional connection

        Note:
            Updates sqlite-vec virtual table for vector similarity search.
            May filter by server status if should_filter_by_status() is True.
        """
        status_filter = self.get_status_filter()

        if tool_id is not None:
            # Sync specific tool
            delete_query = f"DELETE FROM {self.vector_table_name} WHERE tool_id = :tool_id"
            await self.db.execute_non_query(delete_query, {"tool_id": tool_id}, conn=conn)

            # Get tool and check if it has embedding and passes status filter
            tool = await self.get_tool_by_id(tool_id, conn=conn)
            if tool.details_embedding is not None:
                # Check if parent server passes status filter
                if status_filter:
                    server_query = (
                        f"SELECT status FROM {self.server_table_name} WHERE id = :server_id"
                    )
                    server_result = await self.db.execute_query(
                        server_query, {"server_id": tool.mcpserver_id}, conn=conn
                    )
                    if not server_result or server_result[0]._mapping["status"] != status_filter:
                        return  # Don't sync if status doesn't match

                insert_query = f"""
                INSERT INTO {self.vector_table_name} (tool_id, embedding)
                VALUES (:tool_id, :embedding)
                """
                params = {
                    "tool_id": tool_id,
                    "embedding": tool.details_embedding.tobytes(),
                }
                await self.db.execute_non_query(insert_query, params, conn=conn)
                logger.debug(f"Synced {self.vector_table_name} tool vector", tool_id=tool_id)
        else:
            # Sync all tools - rebuild entire virtual table
            delete_all_query = f"DELETE FROM {self.vector_table_name}"
            await self.db.execute_non_query(delete_all_query, {}, conn=conn)

            # Bulk insert using INSERT..SELECT pattern with optional status filtering
            if status_filter:
                sync_query = f"""
                INSERT INTO {self.vector_table_name}(
                    tool_id, embedding
                )
                SELECT
                    t.id as tool_id,
                    t.details_embedding as embedding
                FROM {self.tool_table_name} t
                JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
                WHERE s.status = :status
                AND t.details_embedding IS NOT NULL
                """
                params = {"status": status_filter}
            else:
                sync_query = f"""
                INSERT INTO {self.vector_table_name}(
                    tool_id, embedding
                )
                SELECT
                    t.id as tool_id,
                    t.details_embedding as embedding
                FROM {self.tool_table_name} t
                WHERE t.details_embedding IS NOT NULL
                """
                params = {}

            await self.db.execute_non_query(sync_query, params, conn=conn)
            status_msg = f" ({status_filter} servers only)" if status_filter else ""
            logger.info(f"Synced all {self.vector_table_name} vectors{status_msg}")

    async def sync_tool_fts(
        self,
        tool_id: str | None = None,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Sync tool_fts virtual table.

        Args:
            tool_id: Optional specific tool to sync (None = sync all)
            conn: Optional connection

        Note:
            Updates FTS5 virtual table for full-text search.
            Uses bulk INSERT..SELECT pattern for efficiency.
            May filter by server status if should_filter_by_status() is True.
        """
        status_filter = self.get_status_filter()

        if tool_id is not None:
            # Sync specific tool
            delete_query = f"DELETE FROM {self.fts_table_name} WHERE tool_id = :tool_id"
            await self.db.execute_non_query(delete_query, {"tool_id": tool_id}, conn=conn)

            # Insert using JOIN to get server name, with optional status filter
            if status_filter:
                insert_query = f"""
                INSERT INTO {self.fts_table_name}(
                    tool_id, mcp_server_name, tool_name, tool_description
                )
                SELECT
                    t.id as tool_id,
                    s.name as mcp_server_name,
                    JSON_EXTRACT(t.details, '$.name') as tool_name,
                    COALESCE(JSON_EXTRACT(t.details, '$.description'), '') as tool_description
                FROM {self.tool_table_name} t
                JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
                WHERE t.id = :tool_id AND s.status = :status
                """
                params = {"tool_id": tool_id, "status": status_filter}
            else:
                insert_query = f"""
                INSERT INTO {self.fts_table_name}(
                    tool_id, mcp_server_name, tool_name, tool_description
                )
                SELECT
                    t.id as tool_id,
                    s.name as mcp_server_name,
                    JSON_EXTRACT(t.details, '$.name') as tool_name,
                    COALESCE(JSON_EXTRACT(t.details, '$.description'), '') as tool_description
                FROM {self.tool_table_name} t
                JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
                WHERE t.id = :tool_id
                """
                params = {"tool_id": tool_id}

            await self.db.execute_non_query(insert_query, params, conn=conn)
            logger.debug(f"Synced {self.fts_table_name} tool FTS", tool_id=tool_id)
        else:
            # Sync all tools - rebuild with bulk INSERT..SELECT
            delete_all_query = f"DELETE FROM {self.fts_table_name}"
            await self.db.execute_non_query(delete_all_query, {}, conn=conn)

            # Bulk insert using INSERT..SELECT pattern with optional status filtering
            if status_filter:
                sync_query = f"""
                INSERT INTO {self.fts_table_name}(
                    tool_id, mcp_server_name, tool_name, tool_description
                )
                SELECT
                    t.id as tool_id,
                    s.name as mcp_server_name,
                    JSON_EXTRACT(t.details, '$.name') as tool_name,
                    COALESCE(JSON_EXTRACT(t.details, '$.description'), '') as tool_description
                FROM {self.tool_table_name} t
                JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
                WHERE s.status = :status
                """
                params = {"status": status_filter}
            else:
                sync_query = f"""
                INSERT INTO {self.fts_table_name}(
                    tool_id, mcp_server_name, tool_name, tool_description
                )
                SELECT
                    t.id as tool_id,
                    s.name as mcp_server_name,
                    JSON_EXTRACT(t.details, '$.name') as tool_name,
                    COALESCE(JSON_EXTRACT(t.details, '$.description'), '') as tool_description
                FROM {self.tool_table_name} t
                JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
                """
                params = {}

            await self.db.execute_non_query(sync_query, params, conn=conn)
            status_msg = f" ({status_filter} servers only)" if status_filter else ""
            logger.info(f"Synced all {self.fts_table_name} FTS{status_msg}")

    def _build_status_filter(
        self, server_statuses: list[McpStatus] | None, params: dict[str, str | int]
    ) -> str:
        """Build SQL filter for server statuses.

        Args:
            server_statuses: List of status values to filter by
            params: Parameters dict to add status values to

        Returns:
            SQL filter string (empty if no statuses)
        """
        if not server_statuses or len(server_statuses) == 0:
            return ""

        status_placeholders = ",".join([f":status{i}" for i in range(len(server_statuses))])
        for i, status in enumerate(server_statuses):
            params[f"status{i}"] = status

        return f"AND s.status IN ({status_placeholders})"

    async def _execute_hybrid_search_tasks(
        self,
        semantic_task: Awaitable,
        bm25_task: Awaitable,
    ) -> tuple[list[BaseToolWithMetadata], list[BaseToolWithMetadata]]:
        """Execute hybrid search tasks concurrently.

        Args:
            semantic_task: Semantic search coroutine
            bm25_task: BM25 search coroutine

        Returns:
            Tuple of (semantic_results, bm25_results)

        Raises:
            Exception: If either search task fails. Exceptions are propagated to the caller
                      for proper error handling. This ensures search failures are visible
                      to users rather than returning incomplete results silently.
        """
        import asyncio

        # Execute both searches concurrently without catching exceptions
        # If either fails, the exception propagates to the caller
        semantic_results, bm25_results = await asyncio.gather(semantic_task, bm25_task)

        return semantic_results, bm25_results

    async def find_similar_tools(
        self,
        query_embedding: np.ndarray,
        limit: int,
        distance_threshold: float,
        server_ids: list[str] | None = None,
        server_statuses: list[McpStatus] | None = None,
        allowed_groups: list[str] | None = None,
        query_text: str | None = None,
        hybrid_search_semantic_ratio: float | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Find tools using hybrid search combining semantic and BM25 text search.

        Args:
            query_embedding: The embedding vector for the search query
            limit: Maximum number of similar tools to return
            distance_threshold: Maximum cosine distance (0=identical, 2=opposite)
            server_ids: Optional list of server IDs to filter by
            server_statuses: Optional list of status values to filter servers by
            allowed_groups: Optional list of group names to filter by
            query_text: Optional text query for BM25 search (enables hybrid)
            hybrid_search_semantic_ratio: Ratio of semantic vs BM25 results
            conn: Optional connection

        Returns:
            List of tool with metadata instances ordered by similarity

        Note:
            If query_text and hybrid_search_semantic_ratio provided, performs hybrid search.
            Otherwise falls back to semantic-only search.
        """
        # Fall back to semantic-only if no hybrid search params
        if query_text is None or hybrid_search_semantic_ratio is None:
            return await self._find_similar_tools_semantic_only(
                query_embedding=query_embedding,
                limit=limit,
                distance_threshold=distance_threshold,
                server_ids=server_ids,
                server_statuses=server_statuses,
                allowed_groups=allowed_groups,
                conn=conn,
            )

        # Hybrid search: combine semantic and BM25
        semantic_limit = max(1, int(limit * hybrid_search_semantic_ratio))
        bm25_limit = max(1, limit - semantic_limit)

        logger.debug(
            "Performing hybrid search", semantic_limit=semantic_limit, bm25_limit=bm25_limit
        )

        # Create search tasks
        semantic_task = self._find_similar_tools_semantic_only(
            query_embedding=query_embedding,
            limit=semantic_limit,
            distance_threshold=distance_threshold,
            server_ids=server_ids,
            server_statuses=server_statuses,
            allowed_groups=allowed_groups,
            conn=conn,
        )
        bm25_task = self._find_tools_bm25(
            query=query_text,
            limit=bm25_limit,
            server_ids=server_ids,
            server_statuses=server_statuses,
            allowed_groups=allowed_groups,
            conn=conn,
        )

        # Execute with error handling
        semantic_results, bm25_results = await self._execute_hybrid_search_tasks(
            semantic_task, bm25_task
        )

        # Combine and deduplicate
        combined_results = self._combine_search_results(semantic_results, bm25_results, limit)

        logger.info(
            "Hybrid search completed",
            semantic_count=len(semantic_results),
            bm25_count=len(bm25_results),
            combined_count=len(combined_results),
            final_limit=limit,
        )

        return combined_results

    async def _find_similar_tools_semantic_only(
        self,
        query_embedding: np.ndarray,
        limit: int,
        distance_threshold: float,
        server_ids: list[str] | None = None,
        server_statuses: list[McpStatus] | None = None,
        allowed_groups: list[str] | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Perform semantic similarity search using vector embeddings."""
        # Convert query embedding to sqlite-vec format
        query_embedding_str = "[" + ",".join(map(str, query_embedding.tolist())) + "]"

        params: dict[str, str | int] = {
            "query_embedding": query_embedding_str,
            "limit": limit,
        }

        # Build filters
        status_filter = self._build_status_filter(server_statuses, params)
        group_filter = ""
        if allowed_groups and len(allowed_groups) > 0:
            group_placeholders = ",".join([f":group{i}" for i in range(len(allowed_groups))])
            group_filter = f'AND s."group" IN ({group_placeholders})'

        # Build query based on server filter
        if server_ids and len(server_ids) > 0:
            server_placeholders = ",".join([f":server_id{i}" for i in range(len(server_ids))])
            similarity_query = f"""
            SELECT
                tv.tool_id,
                tv.distance
            FROM {self.vector_table_name} tv
            JOIN {self.tool_table_name} t ON tv.tool_id = t.id
            JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
            WHERE tv.embedding MATCH :query_embedding
            AND t.mcpserver_id IN ({server_placeholders})
            {status_filter}
            {group_filter}
            AND k = :limit
            ORDER BY tv.distance
            """
            for i, server_id in enumerate(server_ids):
                params[f"server_id{i}"] = server_id
        else:
            similarity_query = f"""
            SELECT
                tv.tool_id,
                tv.distance
            FROM {self.vector_table_name} tv
            JOIN {self.tool_table_name} t ON tv.tool_id = t.id
            JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
            WHERE tv.embedding MATCH :query_embedding
            {status_filter}
            {group_filter}
            AND k = :limit
            ORDER BY tv.distance
            """

        # Add group parameters
        if allowed_groups:
            for i, group in enumerate(allowed_groups):
                params[f"group{i}"] = group

        # Execute similarity search
        similarity_results = await self.db.execute_query(similarity_query, params, conn=conn)

        # Log results
        distances = [result._mapping["distance"] for result in similarity_results]
        tool_ids = [result._mapping["tool_id"] for result in similarity_results]
        logger.debug(
            f"{self.tool_table_name} semantic search",
            distances=distances,
            tool_ids=tool_ids,
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

        # Get full tool details with server information
        return await self._fetch_tools_with_metadata(
            filtered_results, search_type="semantic", conn=conn
        )

    async def _find_tools_bm25(
        self,
        query: str,
        limit: int,
        server_ids: list[str] | None = None,
        server_statuses: list[McpStatus] | None = None,
        allowed_groups: list[str] | None = None,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Perform BM25 full-text search using FTS5."""
        # Sanitize query for FTS5 - escape special characters and handle phrases
        fts_sanitized_query = self._sanitize_fts_query(query)
        if not fts_sanitized_query.strip():
            logger.warning("Empty or invalid FTS query after sanitization", original_query=query)
            return []

        params: dict[str, Any] = {
            "query": fts_sanitized_query,
            "limit": limit,
        }

        # Build filters
        status_filter = self._build_status_filter(server_statuses, params)
        group_filter = ""
        if allowed_groups and len(allowed_groups) > 0:
            group_placeholders = ",".join([f":group{i}" for i in range(len(allowed_groups))])
            group_filter = f'AND s."group" IN ({group_placeholders})'

        # Build FTS query
        if server_ids and len(server_ids) > 0:
            server_placeholders = ",".join([f":server_id{i}" for i in range(len(server_ids))])
            fts_query = f"""
            SELECT
                fts.tool_id,
                fts.rank
            FROM {self.fts_table_name} fts
            JOIN {self.tool_table_name} t ON fts.tool_id = t.id
            JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
            WHERE {self.fts_table_name} MATCH :query
            AND t.mcpserver_id IN ({server_placeholders})
            {status_filter}
            {group_filter}
            ORDER BY fts.rank
            LIMIT :limit
            """
            for i, server_id in enumerate(server_ids):
                params[f"server_id{i}"] = server_id
        else:
            fts_query = f"""
            SELECT
                fts.tool_id,
                fts.rank
            FROM {self.fts_table_name} fts
            JOIN {self.tool_table_name} t ON fts.tool_id = t.id
            JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
            WHERE {self.fts_table_name} MATCH :query
            {status_filter}
            {group_filter}
            ORDER BY fts.rank
            LIMIT :limit
            """

        # Add group parameters
        if allowed_groups:
            for i, group in enumerate(allowed_groups):
                params[f"group{i}"] = group

        # Execute BM25 search
        fts_results = await self.db.execute_query(fts_query, params, conn=conn)

        logger.debug(
            f"{self.tool_table_name} BM25 search",
            query=query,
            result_count=len(fts_results),
        )

        if not fts_results:
            return []

        # Convert to format matching semantic search (with distance=0 for BM25 results)
        bm25_as_similarity = [
            type("Row", (), {"_mapping": {"tool_id": r._mapping["tool_id"], "distance": 0.0}})()
            for r in fts_results
        ]

        # Fetch full tool details
        return await self._fetch_tools_with_metadata(
            bm25_as_similarity, search_type="bm25", conn=conn
        )

    async def _fetch_tools_with_metadata(
        self,
        similarity_results: Sequence[Row],
        search_type: str,
        conn: AsyncConnection | None = None,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Fetch full tool details with server information."""
        tool_ids_filtered = [result._mapping["tool_id"] for result in similarity_results]
        tool_distances = {
            result._mapping["tool_id"]: result._mapping["distance"] for result in similarity_results
        }

        placeholders = ",".join([f":id{i}" for i in range(len(tool_ids_filtered))])
        details_query = f"""
        SELECT
            t.*,
            s.name as server_name,
            s.description as server_description
        FROM {self.tool_table_name} t
        JOIN {self.server_table_name} s ON t.mcpserver_id = s.id
        WHERE t.id IN ({placeholders})
        ORDER BY t.created_at
        """

        params_details = {f"id{i}": tool_id for i, tool_id in enumerate(tool_ids_filtered)}
        tool_results = await self.db.execute_query(details_query, params_details, conn=conn)

        tools_with_metadata = []
        for row in tool_results:
            row_data = dict(row._mapping)

            # Extract server info
            server_name = row_data.pop("server_name")
            server_description = row_data.pop("server_description")

            # Parse tool details from JSON
            row_data["details"] = McpTool.model_validate_json(row_data["details"])

            # Deserialize embedding
            if row_data["details_embedding"] is not None:
                row_data["details_embedding"] = np.frombuffer(
                    row_data["details_embedding"], dtype=np.float32
                )

            # Create tool
            tool = self.tool_model_class.model_validate(row_data)

            # Get distance for this tool
            distance = tool_distances[tool.id]

            # Create metadata object
            tool_with_metadata = self.tool_with_metadata_model_class(
                tool=tool,
                server_name=server_name,
                server_description=server_description,
                distance=distance,
            )
            tools_with_metadata.append(tool_with_metadata)

        # Sort by distance (most similar first)
        tools_with_metadata.sort(key=lambda x: x.distance)
        tool_names = [t.tool.details.name for t in tools_with_metadata]
        tool_distances = [t.distance for t in tools_with_metadata]
        logger.info(
            f"Fetched tools with metadata ({search_type} search)",
            tool_names=tool_names,
            tool_distances=tool_distances,
        )

        return tools_with_metadata

    def _combine_search_results(
        self,
        semantic_results: list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata],
        bm25_results: list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata],
        limit: int,
    ) -> list[RegistryToolWithMetadata] | list[WorkloadToolWithMetadata]:
        """Combine and deduplicate semantic and BM25 results."""
        # Deduplicate by tool ID, keeping the first occurrence
        seen_tool_ids = set()
        combined = []

        # Process semantic results first (they have distance scores)
        for result in semantic_results:
            tool_id = result.tool.id
            if tool_id not in seen_tool_ids:
                seen_tool_ids.add(tool_id)
                combined.append(result)

        # Add BM25 results that aren't already included
        for result in bm25_results:
            tool_id = result.tool.id
            if tool_id not in seen_tool_ids:
                seen_tool_ids.add(tool_id)
                combined.append(result)

        # Return up to limit results
        return combined[:limit]

    def _sanitize_fts_query(self, query: str) -> str:
        """
        Sanitize query for SQLite FTS5 to prevent syntax errors.

        FTS5 can interpret certain words as column names, causing "no such column" errors.
        We use a hybrid approach: first try individual words joined with OR, and if that
        contains problematic terms, fall back to phrase search.

        Args:
            query: Raw query string to sanitize

        Returns:
            Sanitized FTS5 query string
        """
        if not query or not query.strip():
            return ""

        # Clean and normalize the query
        sanitized = query.strip()
        sanitized = " ".join(sanitized.split())

        if not sanitized:
            return ""

        # Split into individual words and filter out empty strings
        words = [word.strip() for word in sanitized.split() if word.strip()]
        if not words:
            return ""

        # Check if any word could be problematic (common column-like words)
        # These are words that FTS5 might interpret as column names
        has_problematic_words = any(word.lower() in DEFAULT_FTS_PROBLEMATIC_WORDS for word in words)

        if has_problematic_words or len(words) == 1:
            # Use phrase search for safety
            escaped_query = sanitized.replace('"', '""')
            return f'"{escaped_query}"'
        else:
            # Use OR search for better recall with multiple words
            # Quote each individual word to prevent interpretation issues
            quoted_words = [f'"{word.replace('"', '""')}"' for word in words]
            return " OR ".join(quoted_words)
