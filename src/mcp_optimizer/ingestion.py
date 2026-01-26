"""
Core ingestion service for MCP Optimizer.

This module provides functionality to ingest workloads (MCP servers) and their tools
from Toolhive into the MCP Optimizer database with semantic embeddings.
"""

import asyncio
from typing import Any, NamedTuple, cast

import numpy as np
import structlog
from mcp.types import ListToolsResult, Tool
from more_itertools import batched
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError, DuplicateRegistryServersError
from mcp_optimizer.db.models import (
    McpStatus,
    RegistryServer,
    TransportType,
    WorkloadServer,
)
from mcp_optimizer.db.registry_server_ops import RegistryServerOps
from mcp_optimizer.db.registry_tool_ops import RegistryToolOps
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.mcp_client import (
    MCPServerClient,
    WorkloadConnectionError,
    determine_transport_type,
)
from mcp_optimizer.token_counter import TokenCounter
from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.api_models.registry import ImageMetadata, Registry, RemoteServerMetadata
from mcp_optimizer.toolhive.k8s_client import K8sClient
from mcp_optimizer.toolhive.toolhive_client import (
    ToolhiveClient,
    ToolhiveConnectionError,
    ToolhiveScanError,
)

logger = structlog.get_logger(__name__)


class IngestionError(Exception):
    """Custom exception for ingestion errors."""

    pass


class WorkloadRetrievalError(Exception):
    """Custom exception for workload retrieval errors."""

    pass


class ToolHiveUnavailable(Exception):
    """Exception raised when ToolHive is unavailable.

    This exception is raised when ToolHive cannot be reached or is not available,
    allowing callers to handle this case explicitly rather than checking for None/empty returns.
    """

    pass


class SeparatedWorkloads(NamedTuple):
    """Named tuple for separated workloads."""

    container_workloads: list[Workload]
    remote_workloads: list[Workload]
    all_workloads: list[Workload]
    workload_details: list[str]


class IngestionService:
    """Service for ingesting workloads and tools from Toolhive into the database."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        embedding_manager: EmbeddingManager,
        mcp_timeout: int,
        registry_ingestion_batch_size: int,
        workload_ingestion_batch_size: int,
        encoding: str,
        skipped_workloads: list[str] | None = None,
        runtime_mode: str = "docker",
        k8s_api_server_url: str = "http://127.0.0.1:8001",
        k8s_namespace: str | None = None,
        k8s_all_namespaces: bool = True,
    ):
        """Initialize the ingestion service.

        Args:
            db_config: Database configuration.
            embedding_manager: Embedding manager for generating embeddings.
            mcp_timeout: Timeout for MCP operations in seconds.
            registry_ingestion_batch_size: Batch size for parallel registry server ingestion.
            workload_ingestion_batch_size: Batch size for parallel workload ingestion.
            encoding: Tiktoken encoding to use for token counting.
            skipped_workloads: List of workload names to skip during ingestion.
                              Defaults to ["inspector", "mcp-optimizer"] if not provided.
            runtime_mode: Runtime mode ("docker" or "k8s").
            k8s_api_server_url: Kubernetes API server URL (used when runtime_mode is "k8s").
            k8s_namespace: Kubernetes namespace to query (used when runtime_mode is "k8s").
            k8s_all_namespaces: Whether to query all namespaces (used when runtime_mode is "k8s").
        """
        self.db_config = db_config
        # Separated ops classes for registry and workload tables
        self.registry_server_ops = RegistryServerOps(self.db_config)
        self.registry_tool_ops = RegistryToolOps(self.db_config)
        self.workload_server_ops = WorkloadServerOps(self.db_config)
        self.workload_tool_ops = WorkloadToolOps(self.db_config)
        self.embedding_manager = embedding_manager
        self.token_counter = TokenCounter(encoding_name=encoding)
        self.mcp_timeout = mcp_timeout
        self.registry_ingestion_batch_size = registry_ingestion_batch_size
        self.workload_ingestion_batch_size = workload_ingestion_batch_size
        self.skipped_workloads = skipped_workloads or ["inspector", "mcp-optimizer"]
        self.runtime_mode = runtime_mode
        self.k8s_api_server_url = k8s_api_server_url
        self.k8s_namespace = k8s_namespace
        self.k8s_all_namespaces = k8s_all_namespaces

    def _should_skip_workload(self, workload: Workload) -> bool:
        """Check if a workload should be skipped during ingestion.

        Args:
            workload: Workload to check

        Returns:
            True if workload should be skipped, False otherwise
        """
        if not workload.name:
            return True

        # Check if workload name matches or contains any skipped workload identifier
        for skipped_name in self.skipped_workloads:
            if skipped_name in workload.name:
                return True

        return False

    def _get_skip_reason(self, workload: Workload) -> str | None:
        """Get the reason why a workload should be skipped.

        Args:
            workload: Workload to check

        Returns:
            Skip reason string, or None if workload should not be skipped
        """
        if not workload.name:
            return "missing or empty name"

        # Check if workload name matches or contains any skipped workload identifier
        for skipped_name in self.skipped_workloads:
            if skipped_name in workload.name:
                return f"auxiliary workload ({skipped_name})"

        return None

    async def _batch_gather(self, tasks: list[Any], batch_size: int) -> list[Any]:
        """Execute async tasks in batches using asyncio.gather.

        Args:
            tasks: List of coroutines/tasks to execute
            batch_size: Number of tasks to execute in parallel per batch

        Returns:
            List of results from all tasks
        """
        results = []
        for batch_num, batch in enumerate(batched(tasks, batch_size), start=1):
            batch = list(batch)  # batched returns an iterator
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            results.extend(batch_results)
            logger.debug(
                "Completed batch of tasks",
                batch_number=batch_num,
                batch_size=len(batch),
                total_batches=(len(tasks) + batch_size - 1) // batch_size,
            )
        return results

    def _map_workload_status(self, workload_status: str | None) -> McpStatus:
        """Map workload status to McpStatus enum.

        Args:
            workload_status: Status from Toolhive workload

        Returns:
            Mapped McpStatus enum value
        """
        if workload_status is None:
            raise IngestionError("Workload status cannot be None")

        # Map running workloads to RUNNING, everything else to STOPPED
        if workload_status.lower() == "running":
            return McpStatus.RUNNING
        else:
            return McpStatus.STOPPED

    def _create_tool_text_to_embed(self, tool: Tool, server_name: str) -> str:
        """Create text representation of tool for embedding generation.

        Args:
            tool: Tool object from MCP server
            server_name: Name of the MCP server hosting this tool

        Returns:
            Text representation suitable for embedding
        """
        parts = []

        # Add tool name if available
        if tool.name:
            parts.append(f"Tool: {tool.name}")

        # Add description if available
        if tool.description:
            embed_description = tool.description.replace("\n", " ").strip().split(". ")[0]
            parts.append(f"Description: {embed_description}")

        # Add title if available and different from name
        if tool.title and tool.title != tool.name:
            parts.append(f"Title: {tool.title}")

        text_to_embed = " | ".join(parts) if parts else tool.model_dump_json()
        # Add server name first for better context
        text_to_embed = f"Server: {server_name} | {text_to_embed}"
        logger.debug("Generated tool text for embedding", text=text_to_embed)
        return text_to_embed

    def _create_server_text_to_embed(
        self, server_name: str, description: str | None, tags: list[str] | None
    ) -> str:
        """Create text representation of server for embedding generation.

        Args:
            server_name: Name of the server
            description: Server description from registry
            tags: Server tags from registry

        Returns:
            Text representation suitable for embedding
        """
        # Add server name
        parts = [f"Server: {server_name}"]

        # Add description if available
        if description:
            parts.append(f"Description: {description}")

        # Add tags if available
        if tags:
            parts.append(f"Tags: {', '.join(tags)}")

        server_text_to_embed = " | ".join(parts)
        logger.debug("Created server text for embedding", server_text=server_text_to_embed)
        return server_text_to_embed

    def _compare_embeddings(
        self, embedding1: np.ndarray | None, embedding2: np.ndarray | None
    ) -> bool:
        """Compare two embeddings for equality.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            True if embeddings are equal (within tolerance), False otherwise
        """
        # Both None is considered equal
        if embedding1 is None and embedding2 is None:
            return True

        # One None and one not is not equal
        if embedding1 is None or embedding2 is None:
            return False

        # Compare using np.allclose with default tolerances
        return np.allclose(embedding1, embedding2, rtol=1e-05, atol=1e-08)

    def _compare_tools(self, tool1: Tool, tool2: Tool) -> bool:
        """Compare two McpTool objects for equality.

        Args:
            tool1: First MCP Tool
            tool2: Second MCP Tool

        Returns:
            True if tools are equal, False otherwise
        """
        # Compare name
        if tool1.name != tool2.name:
            return False

        # Compare description
        if tool1.description != tool2.description:
            return False

        # Compare inputSchema - serialize to JSON for comparison
        schema1 = tool1.model_dump().get("inputSchema", {})
        schema2 = tool2.model_dump().get("inputSchema", {})
        if schema1 != schema2:
            return False

        return True

    def _has_registry_server_changed(
        self,
        existing_server: RegistryServer,
        new_name: str | None = None,
        new_url: str | None = None,
        new_package: str | None = None,
        new_transport: TransportType | None = None,
        new_description: str | None = None,
        new_embedding: np.ndarray | None = None,
        new_group: str | None = None,
    ) -> bool:
        """Check if registry server attributes have changed (US3).

        Args:
            existing_server: Existing registry server from database
            new_name: New server name
            new_url: New URL (for remote servers)
            new_package: New package (for container servers)
            new_transport: New transport type
            new_description: New description
            new_embedding: New embedding
            new_group: New group

        Returns:
            True if any attribute has changed, False otherwise
        """

        if new_name is not None and existing_server.name != new_name:
            return True
        if new_url is not None and existing_server.url != new_url:
            return True
        if new_package is not None and existing_server.package != new_package:
            return True
        if new_transport is not None and existing_server.transport != new_transport:
            return True
        if new_description is not None and existing_server.description != new_description:
            return True
        if new_group is not None and existing_server.group != new_group:
            return True
        if new_embedding is not None and not self._compare_embeddings(
            existing_server.server_embedding, new_embedding
        ):
            return True

        return False

    def _has_workload_server_changed(  # noqa: C901
        self,
        existing_server: "WorkloadServer",
        new_url: str | None = None,
        new_workload_identifier: str | None = None,
        new_transport: TransportType | None = None,
        new_status: McpStatus | None = None,
        new_group: str | None = None,
        new_registry_server_id: str | None = None,
        new_description: str | None = None,
        new_embedding: np.ndarray | None = None,
        new_virtual_mcp: bool | None = None,
    ) -> bool:
        """Check if workload server attributes have changed (US3).

        Args:
            existing_server: Existing workload server from database
            new_url: New URL
            new_workload_identifier: New workload identifier
            new_transport: New transport type
            new_status: New status
            new_group: New group
            new_registry_server_id: New registry server ID (None means unlinked)
            new_description: New description
            new_embedding: New embedding
            new_virtual_mcp: New virtual_mcp flag

        Returns:
            True if any attribute has changed, False otherwise
        """

        if new_url is not None and existing_server.url != new_url:
            return True
        if (
            new_workload_identifier is not None
            and existing_server.workload_identifier != new_workload_identifier
        ):
            return True
        if new_transport is not None and existing_server.transport != new_transport:
            return True
        if new_status is not None and existing_server.status != new_status:
            return True
        if new_group is not None and existing_server.group != new_group:
            return True
        # Special handling for registry_server_id - detect linking, unlinking, and re-linking
        # Unlike other fields, we want to detect changes even when new value is None (unlinking)
        # Only skip check if both are None
        if not (new_registry_server_id is None and existing_server.registry_server_id is None):
            if new_registry_server_id != existing_server.registry_server_id:
                return True
        if new_description is not None and existing_server.description != new_description:
            return True
        if new_embedding is not None and not self._compare_embeddings(
            existing_server.server_embedding, new_embedding
        ):
            return True
        if new_virtual_mcp is not None and existing_server.virtual_mcp != new_virtual_mcp:
            return True

        return False

    async def _sync_registry_tools(
        self,
        server_id: str,
        server_name: str,
        tool_names: list[str],
        conn: AsyncConnection,
    ) -> tuple[int, bool]:
        """Synchronize tools for a registry server using RegistryToolOps.

        Registry provides tool names but not full MCP Tool details.
        Creates Tool records with minimal information for embedding-based search.

        Args:
            server_id: Registry server ID
            server_name: Server name for embedding context
            tool_names: List of tool names from registry
            conn: Database connection

        Returns:
            Tuple of (Number of tools processed, were_updated (bool))
        """
        # Handle empty tools case
        if not tool_names:
            # Check if there are existing tools to delete
            existing_tools = await self.registry_tool_ops.list_tools_by_server(server_id, conn=conn)
            if not existing_tools:
                logger.debug("No tools to sync for registry", server_id=server_id)
                return (0, False)

            # Delete existing tools
            await self.registry_tool_ops.delete_tools_by_server(server_id, conn=conn)
            logger.debug("Deleted all existing registry tools", server_id=server_id)
            return (0, True)

        # Convert tool names to minimal Tool objects
        tools = [Tool(name=tool_name, description=None, inputSchema={}) for tool_name in tool_names]

        # Generate texts for all tools
        tool_texts = [self._create_tool_text_to_embed(tool, server_name) for tool in tools]

        # Generate embeddings for all tools at once
        try:
            embeddings = self.embedding_manager.generate_embedding(tool_texts)
            if len(embeddings.shape) != 2:
                raise IngestionError("Embeddings did not return a 2D array as expected")

            if embeddings.shape[0] != len(tools):
                raise IngestionError("Embeddings shape does not match number of tools")
        except Exception as e:
            error_msg = f"Failed to generate registry tool embeddings: {e}"
            logger.error(error_msg, server_id=server_id, error=str(e))
            raise IngestionError(error_msg) from e

        # Check if tools have changed
        tools_changed = await self._tools_have_changed(
            self.registry_tool_ops, server_id, tools, embeddings, conn=conn
        )

        if not tools_changed:
            logger.debug("Registry tools unchanged, skipping sync", server_id=server_id)
            return (len(tools), False)

        # Delete existing tools for this server
        await self.registry_tool_ops.delete_tools_by_server(server_id, conn=conn)

        # Create tool records
        create_tasks = []
        for tool, embedding in zip(tools, embeddings, strict=True):
            task = self.registry_tool_ops.create_tool(
                server_id=server_id, details=tool, details_embedding=embedding, conn=conn
            )
            create_tasks.append(task)

        # Execute all create operations in parallel
        await asyncio.gather(*create_tasks)
        tools_count = len(tools)

        logger.debug(
            "Synchronized registry tools",
            server_id=server_id,
            new_count=tools_count,
            server_name=server_name,
        )
        return (tools_count, True)

    async def _tools_have_changed(
        self,
        tool_ops,  # RegistryToolOps or WorkloadToolOps
        server_id: str,
        new_tools: list[Tool],
        new_embeddings: np.ndarray,
        conn: AsyncConnection,
    ) -> bool:
        """Check if tools have changed compared to existing tools.

        Generic method that works for both registry and workload tools.

        Args:
            tool_ops: Tool operations instance (RegistryToolOps or WorkloadToolOps)
            server_id: Server ID
            new_tools: New tools from MCP server or registry
            new_embeddings: New tool embeddings
            conn: Database connection

        Returns:
            True if tools have changed, False otherwise
        """
        # Get existing tools
        existing_tools = await tool_ops.list_tools_by_server(server_id, conn=conn)

        # If counts differ, tools have changed
        if len(existing_tools) != len(new_tools):
            return True

        # Build lookup of existing tools by name
        existing_by_name = {tool.details.name: tool for tool in existing_tools}

        # Check each new tool
        for new_tool, new_embedding in zip(new_tools, new_embeddings, strict=True):
            if new_tool.name not in existing_by_name:
                return True

            existing_tool = existing_by_name[new_tool.name]

            # Compare tool properties
            if not self._compare_tools(new_tool, existing_tool.details):
                return True

            # Compare embeddings
            if not self._compare_embeddings(new_embedding, existing_tool.details_embedding):
                return True

        return False

    async def _find_and_link_registry_server(
        self,
        url: str | None,
        package: str | None,
        remote: bool,
        conn: AsyncConnection,
    ) -> tuple[str | None, str | None, str | None, np.ndarray | None]:
        """Find matching registry server and return relationship info.

        Args:
            url: Server URL (for remote servers)
            package: Container package (for container servers)
            remote: Server type
            conn: Database connection

        Returns:
            Tuple of (registry_server_id, registry_server_name, description, embedding)
            Returns (None, None, None, None) if no match found

        Raises:
            DuplicateRegistryServersError: If multiple matching servers found
        """
        # Find matching registry servers
        matching_servers = await self.registry_server_ops.find_matching_servers(
            url=url, package=package, remote=remote, conn=conn
        )

        if not matching_servers:
            logger.info(
                "No matching registry server found",
                url=url,
                package=package,
                remote=remote,
            )
            return (None, None, None, None)

        if len(matching_servers) > 1:
            server_ids = [s.id for s in matching_servers]
            logger.error(
                f"Found {len(matching_servers)} duplicate registry servers "
                f"for {'URL' if remote else 'package'} '{url if remote else package}'",
                server_ids=server_ids,
            )
            raise DuplicateRegistryServersError(matching_servers)

        # Single match found - return relationship info
        registry_server = matching_servers[0]
        logger.info(
            "Found matching registry server",
            registry_server_id=registry_server.id,
            registry_server_name=registry_server.name,
            url=url,
            package=package,
        )
        return (
            registry_server.id,
            registry_server.name,
            registry_server.description,
            registry_server.server_embedding,
        )

    async def _calculate_autonomous_embedding(
        self, server_id: str, conn: AsyncConnection
    ) -> np.ndarray | None:
        """Calculate autonomous server embedding from tool embeddings using mean pooling.

        Args:
            server_id: Server ID
            conn: Database connection

        Returns:
            Mean-pooled embedding or None if no tools
        """
        # Get all tool embeddings for this server
        tools = await self.workload_tool_ops.list_tools_by_server(server_id, conn=conn)

        if not tools:
            logger.debug("No tools found for autonomous embedding", server_id=server_id)
            return None

        # Extract embeddings (filter out None values)
        embeddings = [
            tool.details_embedding for tool in tools if tool.details_embedding is not None
        ]

        if not embeddings:
            logger.debug("No tool embeddings found for autonomous embedding", server_id=server_id)
            return None

        # Calculate mean embedding
        embeddings_array = np.array(embeddings)
        mean_embedding = np.mean(embeddings_array, axis=0)

        logger.info(
            "Calculated autonomous embedding from tool embeddings",
            server_id=server_id,
            tool_count=len(embeddings),
            embedding_shape=mean_embedding.shape,
        )

        return mean_embedding

    async def _upsert_workload_server(
        self, workload: Workload, conn: AsyncConnection
    ) -> tuple[str, bool]:
        """Create or update workload server with registry matching.

        This method implements US2 registry matching:
        1. Check for matching registry server
        2. If found: link and inherit metadata
        3. If not found: create with autonomous embeddings

        Args:
            workload: Workload information from Toolhive
            conn: Database connection

        Returns:
            Tuple of (Server ID (UUID), was_updated (bool))

        Raises:
            ValueError: If workload data is invalid
            DuplicateRegistryServersError: If multiple matching registry servers found
        """
        # Cast to TransportType (DB enum) from ToolHiveTransportType
        transport = cast(TransportType, determine_transport_type(workload, self.runtime_mode))
        status = self._map_workload_status(workload.status)

        if not workload.name:
            raise IngestionError("Workload name is required")

        if not workload.url:
            raise IngestionError("Workload URL is required")

        if not workload.package:
            raise IngestionError("Workload package is required")

        # Determine if this is a remote workload
        remote = workload.remote or False

        try:
            # Try to find existing workload server by workload_name
            existing_server = await self.workload_server_ops.get_server_by_workload_name(
                workload.name, conn=conn
            )

            # Find matching registry server
            try:
                (
                    registry_server_id,
                    registry_server_name,
                    registry_description,
                    registry_embedding,
                ) = await self._find_and_link_registry_server(
                    url=workload.package if remote else None,
                    package=workload.package if not remote else None,
                    remote=remote,
                    conn=conn,
                )
            except DuplicateRegistryServersError:
                # Log error but continue - set status to STOPPED
                logger.error(
                    "Multiple registry servers found, setting workload status to STOPPED",
                    workload_name=workload.name,
                )
                status = McpStatus.STOPPED
                registry_server_id = None
                registry_server_name = None
                registry_description = None
                registry_embedding = None

            # Check if server has changed using unified change detection method
            # Note: Description and embedding changes are handled separately
            # based on registry linkage
            server_has_changed = self._has_workload_server_changed(
                existing_server=existing_server,
                new_url=workload.url,
                new_transport=transport,
                new_status=status,
                new_group=workload.group or "default",
                new_registry_server_id=registry_server_id,
                new_workload_identifier=workload.package,
                new_virtual_mcp=workload.virtual_mcp or False,
            )

            if not server_has_changed:
                logger.debug(
                    "Workload server unchanged, skipping update",
                    server_id=existing_server.id,
                    workload_name=workload.name,
                )
                return (existing_server.id, False)

            # Update server
            update_kwargs = {
                "status": status,
                "transport": transport,
                "url": workload.url,
                "group": workload.group or "default",
                "registry_server_id": registry_server_id,
                "registry_server_name": registry_server_name,
                "virtual_mcp": workload.virtual_mcp or False,
            }

            # If linked to registry, use registry metadata
            if registry_server_id:
                update_kwargs["description"] = registry_description
                update_kwargs["server_embedding"] = registry_embedding
                logger.info(
                    "Updating workload server with registry metadata",
                    server_id=existing_server.id,
                    workload_name=workload.name,
                    registry_server_id=registry_server_id,
                )
            # Otherwise, keep existing embeddings (will calculate after tools sync if needed)

            updated_server = await self.workload_server_ops.update_server(
                existing_server.id, conn=conn, **update_kwargs
            )

            logger.info(
                "Updated workload server",
                server_id=updated_server.id,
                workload_name=workload.name,
                status=status,
                linked_to_registry=registry_server_id is not None,
            )
            return (updated_server.id, True)

        except DbNotFoundError:
            # Create new workload server
            # Find matching registry server first
            try:
                (
                    registry_server_id,
                    registry_server_name,
                    registry_description,
                    registry_embedding,
                ) = await self._find_and_link_registry_server(
                    url=workload.package if remote else None,
                    package=workload.package if not remote else None,
                    remote=remote,
                    conn=conn,
                )
            except DuplicateRegistryServersError:
                # Log error but continue - set status to STOPPED
                logger.error(
                    "Multiple registry servers found, setting workload status to STOPPED",
                    workload_name=workload.name,
                )
                status = McpStatus.STOPPED
                registry_server_id = None
                registry_server_name = None
                registry_description = None
                registry_embedding = None

            # Create with registry metadata if linked, otherwise None
            new_server = await self.workload_server_ops.create_server(
                name=workload.name,
                url=workload.url,
                workload_identifier=workload.package,
                remote=remote,
                transport=transport,
                status=status,
                description=registry_description,
                server_embedding=registry_embedding,
                group=workload.group or "default",
                registry_server_id=registry_server_id,
                registry_server_name=registry_server_name,
                virtual_mcp=workload.virtual_mcp or False,
                conn=conn,
            )

            logger.info(
                "Created new workload server",
                server_id=new_server.id,
                workload_name=workload.name,
                linked_to_registry=registry_server_id is not None,
            )
            return (new_server.id, True)

    async def _sync_workload_tools(
        self,
        server_id: str,
        server_name_context: str,
        tools_result: ListToolsResult,
        conn: AsyncConnection,
    ) -> tuple[int, bool]:
        """Synchronize tools for a workload server using WorkloadToolOps.

        This method implements US2/US3 tool embedding with registry context:
        - If linked to registry: uses registry_server_name for context
        - If autonomous: uses workload_name for context

        US3 Registry Name Changes:
        When a registry server name changes, this method automatically re-embeds
        tools with the new registry name context during the next ingestion cycle.
        The RegistryServerOps.update_server() method updates the registry_server_name
        field in linked workload servers, so this method will use the updated name.

        Args:
            server_id: Workload server ID
            server_name_context: Server name to use for embedding context
                                (registry_server_name if linked, workload_name otherwise)
            tools_result: Tools result from MCP server
            conn: Database connection

        Returns:
            Tuple of (Number of tools processed, were_updated (bool))

        Raises:
            IngestionError: If embedding generation fails
        """
        # Handle empty tools case
        if not tools_result.tools:
            # Check if there are existing tools to delete
            existing_tools = await self.workload_tool_ops.list_tools_by_server(server_id, conn=conn)
            if not existing_tools:
                logger.debug("No tools to sync for workload", server_id=server_id)
                return (0, False)

            # Delete existing tools
            await self.workload_tool_ops.delete_tools_by_server(server_id, conn=conn)
            logger.debug("Deleted all existing workload tools", server_id=server_id)
            return (0, True)

        # Generate texts for all tools using the context
        tool_texts = [
            self._create_tool_text_to_embed(tool, server_name_context)
            for tool in tools_result.tools
        ]

        # Generate embeddings for all tools at once
        try:
            embeddings = self.embedding_manager.generate_embedding(tool_texts)
            if len(embeddings.shape) != 2:
                raise IngestionError("Embeddings did not return a 2D array as expected")

            if embeddings.shape[0] != len(tools_result.tools):
                raise IngestionError("Embeddings shape does not match number of tools")
        except Exception as e:
            error_msg = f"Failed to generate tool embeddings: {e}"
            logger.error(error_msg, server_id=server_id, error=str(e))
            raise IngestionError(error_msg) from e

        # Calculate token counts for all tools
        token_counts = [self.token_counter.count_tool_tokens(tool) for tool in tools_result.tools]

        # Check if tools have changed
        tools_changed = await self._tools_have_changed(
            self.workload_tool_ops, server_id, tools_result.tools, embeddings, conn=conn
        )

        if not tools_changed:
            logger.debug("Workload tools unchanged, skipping sync", server_id=server_id)
            return (len(tools_result.tools), False)

        # Delete existing tools for this server
        await self.workload_tool_ops.delete_tools_by_server(server_id, conn=conn)

        # Create tool records using bulk upsert
        # Note: We're not using bulk_upsert here because we already deleted old tools
        # and need to create new ones
        create_tasks = []
        for tool, embedding, token_count in zip(
            tools_result.tools, embeddings, token_counts, strict=True
        ):
            task = self.workload_tool_ops.create_tool(
                server_id=server_id,
                details=tool,
                details_embedding=embedding,
                token_count=token_count,
                conn=conn,
            )
            create_tasks.append(task)

        # Execute all create operations in parallel
        await asyncio.gather(*create_tasks)
        tools_count = len(tools_result.tools)

        logger.debug(
            "Synchronized workload tools",
            server_id=server_id,
            new_count=tools_count,
            context=server_name_context,
        )
        return (tools_count, True)

    async def _process_workload(self, workload: Workload, conn: AsyncConnection) -> dict[str, Any]:
        """Process a single workload with registry matching (US2).

        This method implements US2 functionality:
        1. Creates/updates workload server with registry matching
        2. Syncs tools with appropriate context (registry or workload name)
        3. Calculates autonomous embeddings if not linked to registry

        Args:
            workload: Workload to process
            conn: Database connection

        Returns:
            Processing result with status and counts
        """
        result = {
            "name": workload.name,
            "status": "failed",
            "error": None,
            "tools_count": 0,
            "was_updated": False,
        }

        try:
            # Upsert workload server with registry matching
            server_id, server_was_updated = await self._upsert_workload_server(workload, conn)

            # Get the server to determine context for tool embeddings
            workload_server = cast(
                WorkloadServer,
                await self.workload_server_ops.get_server_by_id(server_id, conn=conn),
            )

            # Determine server name context for tool embeddings
            # Use registry_server_name if linked, otherwise use workload name
            server_name_context = (
                workload_server.registry_server_name
                if workload_server.registry_server_name
                else (workload.name or "unknown")
            )

            # Get tools from MCP server
            mcp_client = MCPServerClient(
                workload, timeout=self.mcp_timeout, runtime_mode=self.runtime_mode
            )
            tools_result = await mcp_client.list_tools()

            # Sync tools with appropriate context
            tools_count, tools_were_updated = await self._sync_workload_tools(
                server_id, server_name_context, tools_result, conn
            )

            # Track if anything was updated
            was_updated = server_was_updated or tools_were_updated

            logger.info(
                "Processed workload",
                server_id=server_id,
                workload_name=workload.name,
                url=workload.url,
                transport_type=workload.transport_type,
                group=workload.group,
                tools_count=tools_count,
                server_was_updated=server_was_updated,
                tools_were_updated=tools_were_updated,
            )

            # Calculate autonomous embedding if:
            # 1. Not linked to registry (registry_server_id is None)
            # 2. No server embedding exists
            # 3. Tools were updated and count > 0
            if (
                workload_server.registry_server_id is None
                and workload_server.server_embedding is None
                and tools_were_updated
                and tools_count > 0
            ):
                try:
                    autonomous_embedding = await self._calculate_autonomous_embedding(
                        server_id, conn=conn
                    )
                    if autonomous_embedding is not None:
                        await self.workload_server_ops.update_server(
                            server_id,
                            conn=conn,
                            server_embedding=autonomous_embedding,
                            description="Autonomous embedding from tool mean pooling",
                        )
                        logger.info(
                            "Updated workload server with autonomous embedding",
                            server_id=server_id,
                            workload_name=workload.name,
                            embedding_shape=autonomous_embedding.shape,
                        )
                        was_updated = True
                except Exception as e:
                    # Log error but don't fail the entire ingestion
                    # Set status to STOPPED to indicate embedding calculation failed
                    logger.error(
                        "Failed to calculate autonomous embedding, setting status to STOPPED",
                        server_id=server_id,
                        workload_name=workload.name,
                        error=str(e),
                    )
                    await self.workload_server_ops.update_server(
                        server_id, conn=conn, status=McpStatus.STOPPED
                    )
                    was_updated = True

            result.update(
                {
                    "status": "success",
                    "server_id": server_id,
                    "tools_count": tools_count,
                    "was_updated": was_updated,
                }
            )

        except DuplicateRegistryServersError as e:
            # This should have been handled in _upsert_workload_server,
            # but catch here for safety
            error_msg = f"Duplicate registry servers found for workload '{workload.name}': {e}"
            logger.error(
                error_msg,
                workload_name=workload.name,
                workload_url=workload.url,
                workload_package=workload.package,
            )
            result["error"] = str(e)

        except (WorkloadConnectionError, ValueError) as e:
            error_msg = f"Failed to process workload '{workload.name}': {e}"
            logger.warning(
                error_msg,
                workload_name=workload.name,
                workload_url=workload.url,
                workload_package=workload.package,
                error_type=type(e).__name__,
            )
            result["error"] = str(e)

        except Exception as e:
            error_msg = f"Unexpected error processing workload '{workload.name}': {e}"
            logger.exception(
                error_msg,
                workload_name=workload.name,
                workload_url=workload.url,
                workload_package=workload.package,
                error_type=type(e).__name__,
            )
            result["error"] = str(e)

        return result

    async def _upsert_registry_server(
        self,
        server_metadata: ImageMetadata | RemoteServerMetadata,
        package: str,
        remote: bool,
        conn: AsyncConnection,
    ) -> bool:
        """
        Upsert a registry server: update metadata if exists, create if not.

        With the new separated table design, registry servers are always in the
        mcpservers_registry table. This method only manages registry metadata.
        Workload-to-registry linking happens in _upsert_workload_server.

        Args:
            server_metadata: Metadata of the server to ingest
            package: Package/image for containers or URL for remote servers
            remote: Whether this is a remote server
            conn: Database connection

        Returns:
            True if server was updated, False otherwise
        """
        try:
            # Generate server embedding from metadata
            server_text = self._create_server_text_to_embed(
                server_metadata.name or "unknown",
                server_metadata.description,
                server_metadata.tags,
            )
            embedding = self.embedding_manager.generate_embedding([server_text])
            server_embedding = embedding[0]
            server_description = server_text

            try:
                # Try to find existing registry server by package (container) or URL (remote)
                if remote:
                    existing_server = await self.registry_server_ops.get_server_by_url(
                        package, conn=conn
                    )
                else:
                    existing_server = await self.registry_server_ops.get_server_by_package(
                        package, conn=conn
                    )

                # Check if server metadata has changed using unified change detection
                server_has_changed = self._has_registry_server_changed(
                    existing_server=existing_server,
                    new_name=server_metadata.name,
                    new_description=server_description,
                    new_embedding=server_embedding,
                )

                # Check if tools need updating
                tools_count, tools_were_updated = await self._sync_registry_tools(
                    existing_server.id,
                    server_metadata.name or "unknown",
                    server_metadata.tools or [],
                    conn,
                )

                # Update server if metadata changed
                if server_has_changed:
                    await self.registry_server_ops.update_server(
                        existing_server.id,
                        conn=conn,
                        name=server_metadata.name,
                        server_embedding=server_embedding,
                        description=server_description,
                    )
                    logger.debug(
                        "Updated existing registry server metadata",
                        server_name=server_metadata.name,
                        server_id=existing_server.id,
                        tools_count=tools_count,
                    )
                    return True
                elif tools_were_updated:
                    logger.debug(
                        "Updated tools for registry server",
                        server_name=server_metadata.name,
                        server_id=existing_server.id,
                        tools_count=tools_count,
                    )
                    return True
                else:
                    logger.debug(
                        "Registry server unchanged, skipping update",
                        server_name=server_metadata.name,
                        server_id=existing_server.id,
                    )
                    return False

            except DbNotFoundError:
                # Create new registry server
                # For remote servers, package contains the URL
                # For container servers, package contains the image name
                url = package if remote else None

                new_server = await self.registry_server_ops.create_server(
                    name=server_metadata.name or "unknown",
                    url=url,
                    package=package if not remote else None,
                    remote=remote,
                    transport=TransportType.SSE,  # Default transport for registry
                    server_embedding=server_embedding,
                    description=server_description,
                    group="default",  # Registry servers use default group
                    conn=conn,
                )

                # Ingest tool names if available
                tools_count, _ = await self._sync_registry_tools(
                    new_server.id,
                    server_metadata.name or "unknown",
                    server_metadata.tools or [],
                    conn,
                )

                logger.debug(
                    "Created new registry server",
                    server_name=server_metadata.name,
                    server_id=new_server.id,
                    tools_count=tools_count,
                )
                return True

        except Exception as e:
            logger.exception(
                "Failed to upsert registry server",
                server_name=server_metadata.name,
                error=str(e),
            )
            raise e

    async def _ingest_registry_servers(  # noqa: C901
        self, registry: Registry, conn: AsyncConnection
    ) -> int:
        """
        Ingest all servers from registry.

        For each server:
        - If exists in DB: update metadata (preserving status)
        - If not exists: create with status=REGISTRY

        Args:
            registry: Registry containing server definitions
            conn: Database connection

        Returns:
            Number of registry servers ingested and deleted
        """
        ingested_count = 0

        # Track all registry server identifiers for cleanup
        registry_server_identifiers = set()

        # Process container servers
        tasks = []
        if registry.servers and isinstance(registry.servers, dict):
            for _, server_metadata in registry.servers.items():
                # Skip container servers without an image (invalid)
                if not server_metadata.image:
                    logger.warning(
                        "Skipping container server without image",
                        server_name=server_metadata.name,
                    )
                    continue

                # Use image field from metadata as identifier for container servers
                registry_server_identifiers.add(
                    (server_metadata.image, False)
                )  # (package, is_remote)

                tasks.append(
                    self._upsert_registry_server(
                        server_metadata, package=server_metadata.image, remote=False, conn=conn
                    )
                )

        # Process remote servers
        if registry.remote_servers and isinstance(registry.remote_servers, dict):
            for _, server_metadata in registry.remote_servers.items():
                # Skip remote servers without a URL (invalid)
                if not server_metadata.url:
                    logger.warning(
                        "Skipping remote server without URL",
                        server_name=server_metadata.name,
                    )
                    continue

                # Use URL as identifier for remote servers (stable across registry updates)
                # This allows remote workloads to match registry entries even if names differ
                registry_server_identifiers.add((server_metadata.url, True))  # (url, is_remote)

                tasks.append(
                    self._upsert_registry_server(
                        server_metadata, package=server_metadata.url, remote=True, conn=conn
                    )
                )

        task_outcomes = await self._batch_gather(tasks, self.registry_ingestion_batch_size)
        for outcome in task_outcomes:
            if isinstance(outcome, Exception):
                logger.error("Error ingesting registry server", error=str(outcome))
            elif outcome:  # outcome is True if server was updated
                ingested_count += 1

        # Clean up registry servers that no longer exist in the registry
        deleted_count = await self._cleanup_removed_registry_servers(
            registry_server_identifiers, conn
        )
        if deleted_count > 0:
            logger.info(
                "Cleaned up removed registry servers",
                deleted_count=deleted_count,
            )

        return ingested_count + deleted_count

    async def _cleanup_removed_servers(
        self, active_workload_identifiers: set[str], conn=None
    ) -> set[str]:
        """Remove workload servers from database that are no longer in ToolHive.

        With the new separated table design, workload servers are ephemeral - they represent
        actual running/stopped workloads in ToolHive. When a workload is removed from ToolHive,
        we delete it from the workload table. Registry metadata remains in the registry table.

        Args:
            active_workload_identifiers: Set of workload names for workloads in ToolHive
            conn: Database connection

        Returns:
            Set of workload names that were deleted
        """
        # Get all workload servers from database
        all_workload_servers = cast(
            list[WorkloadServer],
            await self.workload_server_ops.list_servers(conn=conn),
        )
        deleted_workload_names = set()

        for workload_server in all_workload_servers:
            if workload_server.name not in active_workload_identifiers:
                # Workload no longer exists in ToolHive - delete from database
                logger.info(
                    "Workload no longer exists in ToolHive, deleting from database",
                    workload_name=workload_server.name,
                    server_id=workload_server.id,
                    workload_identifier=workload_server.workload_identifier,
                    remote=workload_server.remote,
                )
                try:
                    # Delete workload server; tools auto-deleted due to ON DELETE CASCADE
                    await self.workload_server_ops.delete_server(workload_server.id, conn=conn)
                    deleted_workload_names.add(workload_server.name)

                    logger.info(
                        "Successfully deleted workload from database",
                        workload_name=workload_server.name,
                        server_id=workload_server.id,
                    )
                except Exception as e:
                    logger.exception(
                        "Failed to delete workload server",
                        workload_name=workload_server.name,
                        server_id=workload_server.id,
                        error=str(e),
                    )

        if deleted_workload_names:
            logger.info("Workload cleanup completed", workloads_deleted=len(deleted_workload_names))

        return deleted_workload_names

    async def _cleanup_removed_registry_servers(
        self, active_registry_server_identifiers: set[tuple[str, bool]], conn=None
    ) -> int:
        """Remove registry servers from database that are no longer in the registry.

        This will also remove registry relationships from any linked workload servers,
        triggering autonomous embedding calculation for those workloads.

        Args:
            active_registry_server_identifiers: Set of (package, remote) tuples for registry servers
            conn: Database connection

        Returns:
            Number of registry servers deleted
        """
        # Get all registry servers from database
        all_registry_servers = cast(
            list[RegistryServer],
            await self.registry_server_ops.list_servers(conn=conn),
        )
        deleted_count = 0

        for registry_server in all_registry_servers:
            # Check if this registry server still exists in the registry
            if registry_server.remote:
                server_identifier = (registry_server.url, True)
            else:
                server_identifier = (registry_server.package, False)

            if server_identifier not in active_registry_server_identifiers:
                logger.info(
                    "Registry server no longer exists in registry, deleting from database",
                    server_name=registry_server.name,
                    server_id=registry_server.id,
                    package=registry_server.package,
                    remote=registry_server.remote,
                )
                try:
                    # Delete registry server with relationship cleanup
                    # The overridden delete_server method will:
                    # 1. Remove registry relationships from linked workloads
                    # 2. Delete the registry server and its tools (CASCADE)
                    await self.registry_server_ops.delete_server(
                        registry_server.id, workload_ops=self.workload_server_ops, conn=conn
                    )
                    deleted_count += 1

                    logger.info(
                        "Successfully deleted registry server from database",
                        server_name=registry_server.name,
                        server_id=registry_server.id,
                    )
                except Exception as e:
                    logger.exception(
                        "Failed to delete registry server",
                        server_name=registry_server.name,
                        server_id=registry_server.id,
                        error=str(e),
                    )

        if deleted_count > 0:
            logger.info("Registry cleanup completed", registry_servers_deleted=deleted_count)

        return deleted_count

    def _validate_and_separate_workloads(
        self, workloads: list[Workload], source: str
    ) -> SeparatedWorkloads:
        """Validate remote field and filter out workloads with remote=None.

        Args:
            workloads: List of workloads to validate
            source: Source name for logging (e.g., "Docker", "Kubernetes")

        Returns:
            List of valid workloads with remote field properly set
        """
        container_workloads = []
        remote_workloads = []
        workload_details = []
        for workload in workloads:
            # workload.remote == None for non-remote workloads
            is_remote = workload.remote or False
            if is_remote:
                remote_workloads.append(workload)
            else:
                container_workloads.append(workload)

            workload_details.append(f"{workload.name} (url: {workload.url or 'MISSING'})")
        return SeparatedWorkloads(
            container_workloads=container_workloads,
            remote_workloads=remote_workloads,
            all_workloads=workloads,
            workload_details=workload_details,
        )

    async def _get_all_workloads_from_docker(
        self, toolhive_client: ToolhiveClient
    ) -> list[Workload]:
        """Fetch all MCP workloads from ToolHive in Docker mode.
        Args:
            toolhive_client: Connected ToolhiveClient instance
        Returns:
            List of all running MCP workloads
        """
        try:
            async with toolhive_client as client:
                # Get ALL workloads (running and stopped) to detect deletions
                all_workloads_response = await client.list_workloads(all_workloads=True)
                running_workloads = [
                    workload
                    for workload in (all_workloads_response.workloads or [])
                    if workload.status == "running"
                ]

                # Validate remote field and filter out workloads with remote=None
                separated_workloads = self._validate_and_separate_workloads(
                    running_workloads, "Docker"
                )

                logger.info(
                    "Found running workloads from Docker",
                    container_workloads=len(separated_workloads.container_workloads),
                    container_names=[w.name for w in separated_workloads.container_workloads],
                    remote_workloads=len(separated_workloads.remote_workloads),
                    remote_names=[w.name for w in separated_workloads.remote_workloads],
                    total_workloads=len(separated_workloads.all_workloads),
                    workload_details=separated_workloads.workload_details,
                )
                return separated_workloads.all_workloads
        except Exception as e:
            # If ToolHive is unavailable, raise ToolHiveUnavailable exception
            if isinstance(e, (ToolhiveConnectionError, ToolhiveScanError, ConnectionError)):
                logger.info(
                    "ToolHive server unavailable - cannot fetch workloads. "
                    "Will retry on next polling cycle.",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ToolHiveUnavailable("ToolHive server unavailable") from e
            # Re-raise unexpected errors
            logger.warning(
                "Unexpected error fetching workloads from ToolHive",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise WorkloadRetrievalError("Failed to fetch workloads from ToolHive") from e

    async def _get_all_workloads_from_k8s(self) -> list[Workload]:
        """Fetch all MCP workloads from Kubernetes in K8s mode.

        Args:
            toolhive_client: Connected ToolhiveClient instance (used for registry only)
        Returns:
            Tuple of (list of all running MCP workloads, registry or None if fetch failed)
        """
        # Fetch workloads from Kubernetes
        try:
            k8s_client = K8sClient(
                api_server_url=self.k8s_api_server_url,
                namespace=self.k8s_namespace,
                timeout=self.mcp_timeout,
            )
            async with k8s_client:
                workloads_from_k8s = await k8s_client.list_mcpservers(
                    all_namespaces=self.k8s_all_namespaces
                )

                # Filter to only running workloads
                running_workloads = [
                    workload for workload in workloads_from_k8s if workload.status == "running"
                ]

                # Validate remote field and filter out workloads with remote=None
                separated_workloads = self._validate_and_separate_workloads(
                    running_workloads, "Kubernetes"
                )

                logger.info(
                    "Found running workloads from Kubernetes",
                    container_workloads=len(separated_workloads.container_workloads),
                    container_names=[w.name for w in separated_workloads.container_workloads],
                    remote_workloads=len(separated_workloads.remote_workloads),
                    remote_names=[w.name for w in separated_workloads.remote_workloads],
                    total_workloads=len(separated_workloads.all_workloads),
                    workload_details=separated_workloads.workload_details,
                )
                return separated_workloads.all_workloads
        except Exception as e:
            logger.exception("Failed to connect to Toolhive or fetch workloads")
            raise WorkloadRetrievalError("Failed to fetch workloads from ToolHive") from e

    async def _get_registry(self, toolhive_client: ToolhiveClient) -> Registry:
        """Fetch registry from ToolHive.

        Args:
            toolhive_client: Connected ToolhiveClient instance
        Returns:
            Registry from ToolHive
        Raises:
            ToolHiveUnavailable: If ToolHive is unavailable
        """
        try:
            async with toolhive_client as client:
                registry = await client.get_registry()
                logger.info("Successfully fetched registry for server embeddings")
                return registry
        except Exception as e:
            # If ToolHive is unavailable, raise ToolHiveUnavailable exception
            if isinstance(e, (ToolhiveConnectionError, ToolhiveScanError, ConnectionError)):
                logger.info(
                    "ToolHive server unavailable - cannot fetch registry. "
                    "Will retry on next polling cycle.",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ToolHiveUnavailable("ToolHive server unavailable") from e
            # For other errors, log warning and re-raise
            logger.warning(
                "Failed to fetch registry",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def _get_all_workloads(self, toolhive_client: ToolhiveClient) -> list[Workload]:
        """Fetch all MCP workloads based on runtime mode.

        Args:
            toolhive_client: Connected ToolhiveClient instance
        Returns:
            Tuple of (list of all running MCP workloads, registry or None if fetch failed)
        """
        if self.runtime_mode == "k8s":
            logger.info("Using Kubernetes mode to fetch workloads")
            return await self._get_all_workloads_from_k8s()
        else:
            logger.info("Using Docker mode to fetch workloads")
            return await self._get_all_workloads_from_docker(toolhive_client)

    async def ingest_workloads(self, toolhive_client: ToolhiveClient) -> None:  # noqa: C901
        """Ingest running workloads from Toolhive with cleanup of removed servers.

        This method fetches ALL workloads (running and stopped) from ToolHive,
        updates the database accordingly, and removes servers that no longer exist.
        All database operations are wrapped in a single transaction for atomicity.

        Note: This method does NOT ingest registry servers. Use ingest_registry() for that.

        Args:
            toolhive_client: Connected ToolhiveClient instance

        Returns:
            List of MCP servers with their associated tools
        """
        logger.info(
            "Starting workload ingestion with cleanup",
            host=toolhive_client.thv_host,
            port=toolhive_client.thv_port,
        )

        # Statistics for logging
        num_ingested = 0
        failed = 0
        total_tools = 0
        deleted_server_names: set[str] = set()

        # Fetch workloads outside transaction (read-only operations)
        try:
            all_workloads = await self._get_all_workloads(toolhive_client)
        except ToolHiveUnavailable:
            logger.info(
                "ToolHive unavailable - skipping workload ingestion. "
                "Will retry on next polling cycle."
            )
            return

        # Fetch workload details for remote workloads to get accurate URLs
        # This is critical for URL-based matching instead of package-based matching
        for workload in all_workloads:
            is_remote = workload.remote or False
            if is_remote and workload.name:
                try:
                    detailed_workload = await toolhive_client.get_workload_details(workload.name)
                    # Store the URL in the package field for remote workloads
                    # This makes the URL the stable identifier for remote workloads
                    workload.package = detailed_workload.url
                    logger.debug(
                        "Fetched URL for remote workload",
                        name=workload.name,
                        url=workload.package,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to fetch workload details, will skip this workload",
                        name=workload.name,
                        error=str(e),
                    )
                    # Set package to None to trigger skip in later processing
                    workload.package = None

        # Wrap all database operations in a single transaction for atomicity.
        # This ensures all workload updates, tool syncs, and vector table updates
        # either complete successfully together or roll back entirely on error.
        #
        # TRANSACTION PATTERN: All database operations requiring atomicity should
        # follow this pattern. Pass the `conn` object to all ops methods to ensure
        # they participate in the same transaction. Without this pattern, operations
        # execute independently and may result in partial updates on failure.
        async with self.db_config.begin_transaction() as conn:
            try:
                if not all_workloads:
                    logger.warning("No MCP workloads found in ToolHive")
                    # Clean up all servers since none exist in ToolHive
                    deleted_server_names = await self._cleanup_removed_servers(set(), conn)
                else:
                    # Extract workload identifiers for cleanup
                    workload_identifiers = set()
                    for workload in all_workloads:
                        workload_identifiers.add(workload.name)

                    # Filter out workloads that should be skipped
                    workloads_to_process = [
                        workload
                        for workload in all_workloads
                        if not self._should_skip_workload(workload)
                    ]

                    # Log skipped workloads
                    for workload in all_workloads:
                        skip_reason = self._get_skip_reason(workload)
                        if skip_reason:
                            log_level = (
                                logger.warning
                                if "missing or empty name" in skip_reason
                                else logger.debug
                            )
                            log_level(
                                "Skipping workload during ingestion",
                                workload_name=workload.name or "<no name>",
                                reason=skip_reason,
                            )

                    # Process all workloads in batches
                    process_tasks = [
                        self._process_workload(workload, conn) for workload in workloads_to_process
                    ]
                    results = await self._batch_gather(
                        process_tasks, self.workload_ingestion_batch_size
                    )

                    # Process results
                    for workload, result in zip(workloads_to_process, results, strict=True):
                        if isinstance(result, Exception):
                            failed += 1
                            logger.exception(
                                f"Failed to process workload '{workload.name}' in transaction",
                                workload_name=workload.name,
                                workload_remote=workload.remote or False,
                                workload_url=workload.url,
                                workload_package=workload.package,
                                error=str(result),
                                error_type=type(result).__name__,
                            )
                        elif result["status"] == "success":
                            num_ingested += result["was_updated"]
                            total_tools += result["tools_count"]
                            logger.debug(
                                "Successfully processed workload",
                                workload_name=workload.name,
                                workload_remote=workload.remote or False,
                                tools_count=result["tools_count"],
                            )
                        else:
                            failed += 1
                            logger.warning(
                                "Failed to process workload - see details above",
                                workload_name=workload.name,
                                workload_remote=workload.remote or False,
                                workload_url=workload.url,
                                error=result.get("error"),
                            )

                    # Clean up servers that are no longer in ToolHive
                    deleted_server_names = await self._cleanup_removed_servers(
                        workload_identifiers, conn
                    )

                # Only sync vector tables if there were changes
                # (successful ingestions or deleted servers indicate changes)
                if num_ingested > 0 or deleted_server_names:
                    logger.info(
                        "Synchronizing workload vector tables after ingestion",
                        successful_ingestions=num_ingested,
                        deleted_servers=len(deleted_server_names),
                    )
                    # Sync workload server and tool vectors
                    await self.workload_server_ops.sync_server_vectors(conn=conn)
                    await self.workload_tool_ops.sync_tool_vectors(conn=conn)

                    # Sync FTS table for BM25 search (workload tools only)
                    await self.workload_tool_ops.sync_tool_fts(conn=conn)
                    logger.info("Workload vector and FTS synchronization completed")
                else:
                    logger.info("Workload ingestion: no changes detected, skipping vector sync")

            except Exception as e:
                logger.exception("Transaction failed, rolling back all changes", error=str(e))
                raise  # Re-raise to trigger rollback

    async def ingest_registry(self, toolhive_client: ToolhiveClient) -> None:
        """Ingest registry servers from Toolhive.

        This method fetches the registry from ToolHive and ingests all registry servers.
        For each server:
        - If it exists in DB: update metadata (preserving status)
        - If it doesn't exist: create with status=REGISTRY

        All database operations are wrapped in a single transaction for atomicity.

        Args:
            toolhive_client: Connected ToolhiveClient instance

        Returns:
            List of MCP servers with their associated tools
        """
        # Skip registry ingestion in K8s mode - registry is not available via ToolHive HTTP API
        # In K8s mode, workloads come from Kubernetes CRDs, not ToolHive's registry API
        if self.runtime_mode == "k8s":
            logger.info(
                "Skipping registry ingestion in K8s mode "
                "(registry not available via ToolHive HTTP API)",
                runtime_mode=self.runtime_mode,
            )
            return

        logger.info(
            "Starting registry ingestion",
            host=toolhive_client.thv_host,
            port=toolhive_client.thv_port,
        )

        # Fetch registry outside transaction (read-only operation)
        try:
            registry = await self._get_registry(toolhive_client)
        except ToolHiveUnavailable:
            logger.info(
                "ToolHive unavailable - skipping registry ingestion. "
                "Will retry on next polling cycle."
            )
            return

        # Wrap all database operations in a single transaction for atomicity.
        # This ensures all registry server ingestions and vector table updates
        # either complete successfully together or roll back entirely on error.
        #
        # TRANSACTION PATTERN: All database operations requiring atomicity should
        # follow this pattern. Pass the `conn` object to all ops methods to ensure
        # they participate in the same transaction.
        async with self.db_config.begin_transaction() as conn:
            try:
                # Ingest all registry servers
                registry_servers_count = 0
                if registry:
                    registry_servers_count = await self._ingest_registry_servers(registry, conn)
                    logger.info(
                        "Ingested registry servers", registry_servers_count=registry_servers_count
                    )
                else:
                    logger.warning("No registry available to ingest")

                # Only sync vector tables if there were changes
                # (successful registry server ingestions indicate changes)
                if registry_servers_count > 0:
                    logger.info(
                        "Synchronizing registry vector tables after ingestion",
                        registry_servers_ingested=registry_servers_count,
                    )
                    # Sync registry server and tool vectors
                    await self.registry_server_ops.sync_server_vectors(conn=conn)
                    await self.registry_tool_ops.sync_tool_vectors(conn=conn)

                    # Sync FTS table for BM25 search (registry tools)
                    await self.registry_tool_ops.sync_tool_fts(conn=conn)
                    logger.info("Registry vector and FTS synchronization completed")
                else:
                    logger.info("Registry ingestion: no changes detected, skipping vector sync")

            except Exception as e:
                logger.exception("Transaction failed, rolling back all changes", error=str(e))
                raise  # Re-raise to trigger rollback
