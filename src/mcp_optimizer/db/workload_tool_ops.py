"""Workload tool operations for database table separation feature."""

import uuid
from datetime import datetime, timezone

import numpy as np
import structlog
from mcp.types import Tool as McpTool
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.base_tool_ops import BaseToolOps
from mcp_optimizer.db.models import (
    McpStatus,
    WorkloadTool,
    WorkloadToolUpdateDetails,
    WorkloadToolWithMetadata,
)

logger = structlog.get_logger(__name__)


class WorkloadToolOps(BaseToolOps):
    """Repository for workload MCP tool operations.

    Extends BaseToolOps with workload-specific table names and models.
    All CRUD operations and search methods are inherited from the base class.

    Key difference from RegistryToolOps: Filters tools by RUNNING server status.
    """

    @property
    def tool_table_name(self) -> str:
        """Name of the main tool table."""
        return "tools_workload"

    @property
    def server_table_name(self) -> str:
        """Name of the server table."""
        return "mcpservers_workload"

    @property
    def vector_table_name(self) -> str:
        """Name of the vector virtual table."""
        return "workload_tool_vectors"

    @property
    def fts_table_name(self) -> str:
        """Name of the FTS virtual table."""
        return "workload_tool_fts"

    @property
    def tool_model_class(self) -> type[WorkloadTool]:
        """Tool model class."""
        return WorkloadTool

    @property
    def tool_update_model_class(self) -> type[WorkloadToolUpdateDetails]:
        """Tool update details model class."""
        return WorkloadToolUpdateDetails

    @property
    def tool_with_metadata_model_class(self) -> type[WorkloadToolWithMetadata]:
        """Tool with metadata model class."""
        return WorkloadToolWithMetadata

    def should_filter_by_status(self) -> bool:
        """Workload tools filter by RUNNING server status."""
        return True

    async def create_tool(
        self,
        server_id: str,
        details: McpTool,
        token_count: int,
        details_embedding: np.ndarray | None = None,
        conn: AsyncConnection | None = None,
    ) -> WorkloadTool:
        """Create a new tool.

        Args:
            server_id: Parent server UUID
            details: MCP tool definition (name, description, inputSchema)
            details_embedding: Optional vector embedding
            conn: Optional connection

        Returns:
            Created tool model instance
        """
        new_tool = WorkloadTool(
            id=str(uuid.uuid4()),
            mcpserver_id=server_id,
            details=details,
            details_embedding=details_embedding,
            token_count=token_count,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        query = """
        INSERT INTO tools_workload (
            id, mcpserver_id, details, details_embedding,
            last_updated, created_at, token_count
        )
        VALUES (
            :id, :mcpserver_id, :details, :details_embedding,
            :last_updated, :created_at, :token_count
        )
        """

        params = {
            "id": new_tool.id,
            "mcpserver_id": server_id,
            "details": details.model_dump_json(),
            "details_embedding": (
                details_embedding.tobytes() if details_embedding is not None else None
            ),
            "token_count": token_count,
            "last_updated": new_tool.last_updated,
            "created_at": new_tool.created_at,
        }

        await self.db.execute_non_query(query, params, conn=conn)
        logger.debug(
            f"Created {self.tool_table_name} tool",
            tool_id=new_tool.id,
            server_id=server_id,
            tool_name=details.name,
        )
        return new_tool

    async def sum_token_counts_for_running_servers(
        self,
        allowed_groups: list[str] | None = None,
        conn: AsyncConnection | None = None,
    ) -> int:
        """Sum token counts for all tools from running servers.

        Uses the same virtual MCP filtering logic as find_similar_tools to ensure
        consistent token metrics. For groups with virtual MCP servers, only counts
        tokens from those virtual servers. For groups without virtual MCP servers,
        counts tokens from all servers in the group.

        Args:
            allowed_groups: Optional list of group names to filter by.
                If None, sums tokens from all groups.
                If provided, only sums tokens from servers in the specified groups.
            conn: Database connection
        Returns:
            Total token count for all tools from running servers
        """
        params: dict = {"status": McpStatus.RUNNING}

        # Build group filter using the same virtual MCP logic as find_similar_tools
        group_filter = ""
        if allowed_groups and len(allowed_groups) > 0:
            group_filter, group_params = self._get_group_server_filter(
                allowed_groups, [McpStatus.RUNNING]
            )
            params.update(group_params)

        query = f"""
        SELECT COALESCE(SUM(t.token_count), 0) as total_tokens
        FROM tools_workload t
        JOIN mcpservers_workload s ON t.mcpserver_id = s.id
        WHERE s.status = :status
        {group_filter}
        """  # nosec B608 - Table names are code-controlled, params are safe

        results = await self.db.execute_query(query, params, conn=conn)
        total_tokens = results[0]._mapping["total_tokens"] if results else 0

        logger.debug(
            "Calculated total token count for running servers",
            total_tokens=total_tokens,
            allowed_groups=allowed_groups,
        )

        return int(total_tokens)
