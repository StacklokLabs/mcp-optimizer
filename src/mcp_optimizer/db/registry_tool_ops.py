"""Registry tool operations for database table separation feature."""

import uuid
from datetime import datetime, timezone

import numpy as np
import structlog
from mcp.types import Tool as McpTool
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.base_tool_ops import BaseToolOps
from mcp_optimizer.db.models import (
    RegistryTool,
    RegistryToolUpdateDetails,
    RegistryToolWithMetadata,
)

logger = structlog.get_logger(__name__)


class RegistryToolOps(BaseToolOps):
    """Repository for registry MCP tool operations.

    Extends BaseToolOps with registry-specific table names and models.
    All CRUD operations and search methods are inherited from the base class.
    """

    @property
    def tool_table_name(self) -> str:
        """Name of the main tool table."""
        return "tools_registry"

    @property
    def server_table_name(self) -> str:
        """Name of the server table."""
        return "mcpservers_registry"

    @property
    def vector_table_name(self) -> str:
        """Name of the vector virtual table."""
        return "registry_tool_vectors"

    @property
    def fts_table_name(self) -> str:
        """Name of the FTS virtual table."""
        return "registry_tool_fts"

    @property
    def tool_model_class(self) -> type[RegistryTool]:
        """Tool model class."""
        return RegistryTool

    @property
    def tool_update_model_class(self) -> type[RegistryToolUpdateDetails]:
        """Tool update details model class."""
        return RegistryToolUpdateDetails

    @property
    def tool_with_metadata_model_class(self) -> type[RegistryToolWithMetadata]:
        """Tool with metadata model class."""
        return RegistryToolWithMetadata

    def should_filter_by_status(self) -> bool:
        """Registry tools don't filter by server status."""
        return False

    async def create_tool(
        self,
        server_id: str,
        details: McpTool,
        details_embedding: np.ndarray | None = None,
        conn: AsyncConnection | None = None,
    ) -> RegistryTool:
        """Create a new tool.

        Args:
            server_id: Parent server UUID
            details: MCP tool definition (name, description, inputSchema)
            details_embedding: Optional vector embedding
            conn: Optional connection

        Returns:
            Created tool model instance
        """
        new_tool = RegistryTool(
            id=str(uuid.uuid4()),
            mcpserver_id=server_id,
            details=details,
            details_embedding=details_embedding,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        query = """
        INSERT INTO tools_registry (
            id, mcpserver_id, details, details_embedding,
            last_updated, created_at
        )
        VALUES (
            :id, :mcpserver_id, :details, :details_embedding,
            :last_updated, :created_at
        )
        """

        params = {
            "id": new_tool.id,
            "mcpserver_id": server_id,
            "details": details.model_dump_json(),
            "details_embedding": (
                details_embedding.tobytes() if details_embedding is not None else None
            ),
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
