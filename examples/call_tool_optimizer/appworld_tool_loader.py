"""Loads tools from AppWorld MCP server and ingests them into MCP Optimizer database.

This module follows the pattern from examples/anthropic_comparison/ingest_test_data.py
but fetches tools from a running AppWorld MCP server instead of a JSON file.
"""

import asyncio
import os
from pathlib import Path

import structlog
from mcp.types import Tool

from mcp_optimizer.db.config import DatabaseConfig, run_migrations
from mcp_optimizer.db.models import McpStatus, TransportType
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionService
from mcp_optimizer.mcp_client import MCPServerClient
from mcp_optimizer.response_optimizer.token_counter import TokenCounter
from mcp_optimizer.toolhive.api_models.core import Workload

logger = structlog.get_logger(__name__)


class AppWorldToolLoader:
    """Loads AppWorld tools from MCP server and ingests them into database."""

    def __init__(
        self,
        appworld_mcp_url: str,
        db_path: Path,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        mcp_timeout: float = 60.0,
    ):
        """Initialize loader with AppWorld MCP server URL and database path.

        Args:
            appworld_mcp_url: URL of the AppWorld MCP server
            db_path: Path to the SQLite database file
            embedding_model: Embedding model to use
            mcp_timeout: Timeout for MCP operations in seconds
        """
        if not appworld_mcp_url.endswith("/mcp"):
            appworld_mcp_url += "/mcp"
        self.appworld_mcp_url = appworld_mcp_url
        self.db_path = db_path
        self.embedding_model = embedding_model
        self.mcp_timeout = mcp_timeout

        # Will be initialized in setup()
        self.db_config: DatabaseConfig | None = None
        self.embedding_manager: EmbeddingManager | None = None
        self.token_counter: TokenCounter | None = None
        self.workload_server_ops: WorkloadServerOps | None = None
        self.workload_tool_ops: WorkloadToolOps | None = None
        self.ingestion_service: IngestionService | None = None

    async def setup(self) -> None:
        """Initialize database and components."""
        logger.info("Setting up AppWorld tool loader", db_path=str(self.db_path))

        # Set environment variables for migrations
        async_db_url = f"sqlite+aiosqlite:///{self.db_path}"
        sync_db_url = f"sqlite:///{self.db_path}"

        os.environ["ASYNC_DB_URL"] = async_db_url
        os.environ["DB_URL"] = sync_db_url

        # Run migrations
        logger.info("Running database migrations")
        try:
            run_migrations()
            logger.info("Migrations completed successfully")
        except Exception as e:
            logger.warning("Migration error (may be expected if schema exists)", error=str(e))

        # Initialize database config
        self.db_config = DatabaseConfig(database_url=async_db_url)

        # Initialize embedding manager
        logger.info("Initializing embedding manager", model=self.embedding_model)
        self.embedding_manager = EmbeddingManager(
            model_name=self.embedding_model,
            enable_cache=True,
            threads=2,
            fastembed_cache_path=None,
        )

        # Initialize token counter
        self.token_counter = TokenCounter(encoding_name="cl100k_base")

        # Initialize ops classes
        self.workload_server_ops = WorkloadServerOps(self.db_config)
        self.workload_tool_ops = WorkloadToolOps(self.db_config)

        # Create IngestionService to reuse its _create_tool_text_to_embed method
        self.ingestion_service = IngestionService(
            db_config=self.db_config,
            embedding_manager=self.embedding_manager,
            mcp_timeout=self.mcp_timeout,
            registry_ingestion_batch_size=5,
            workload_ingestion_batch_size=5,
            encoding="cl100k_base",
            skipped_workloads=[],
            runtime_mode="docker",
            k8s_api_server_url="http://127.0.0.1:8001",
            k8s_namespace=None,
            k8s_all_namespaces=True,
        )

        logger.info("Setup complete")

    async def fetch_tools_from_mcp(self) -> list[Tool]:
        """Fetch tools from AppWorld MCP server.

        Returns:
            List of MCP Tool objects
        """
        logger.info("Fetching tools from AppWorld MCP server", url=self.appworld_mcp_url)

        # Create workload pointing to AppWorld MCP server
        workload = Workload(
            name="appworld",
            url=self.appworld_mcp_url,
            proxy_mode="streamable-http",
        )

        # Create MCP client
        client = MCPServerClient(workload, timeout=self.mcp_timeout, runtime_mode="docker")

        # Fetch tools
        result = await client.list_tools()
        tools = list(result.tools)

        logger.info("Fetched tools from AppWorld", count=len(tools))
        return tools

    async def ingest_tools(self, tools: list[Tool]) -> dict:
        """Ingest tools into MCP Optimizer database.

        Args:
            tools: List of MCP Tool objects to ingest

        Returns:
            dict with ingestion statistics (tools_count, server_id, errors)
        """
        if not self.db_config or not self.ingestion_service:
            raise RuntimeError("Loader not setup. Call setup() first.")

        logger.info("Ingesting tools into database", count=len(tools))

        server_name = "appworld"
        errors = []

        async with self.workload_server_ops.db.begin_transaction() as conn:
            try:
                # Check if server already exists and delete it
                try:
                    existing_server = await self.workload_server_ops.get_server_by_name(
                        server_name, conn=conn
                    )
                    if existing_server:
                        logger.info("Deleting existing AppWorld server", id=existing_server.id)
                        await self.workload_server_ops.delete_server(existing_server.id, conn=conn)
                except Exception:
                    pass  # Server doesn't exist, continue

                # Create workload server for AppWorld
                server = await self.workload_server_ops.create_server(
                    name=server_name,
                    url=self.appworld_mcp_url,
                    workload_identifier="appworld-mcp",
                    remote=False,
                    transport=TransportType.STREAMABLE,
                    status=McpStatus.RUNNING,
                    description="AppWorld MCP server with 457 APIs across 9 applications",
                    conn=conn,
                )
                server_id = server.id
                logger.info("Created server", name=server_name, id=server_id)

                # Generate texts for all tools using the server name as context
                tool_texts = [
                    self.ingestion_service._create_tool_text_to_embed(tool, server_name)
                    for tool in tools
                ]

                # Generate embeddings for all tools at once (batch)
                logger.info("Generating embeddings for tools", count=len(tools))
                embeddings = self.embedding_manager.generate_embedding(tool_texts)

                # Calculate token counts for all tools
                token_counts = [self.token_counter.count_tool_tokens(tool) for tool in tools]

                # Create tool records
                create_tasks = []
                for tool, embedding, token_count in zip(
                    tools, embeddings, token_counts, strict=True
                ):
                    task = self.workload_tool_ops.create_tool(
                        server_id=server_id,
                        details=tool,
                        details_embedding=embedding,
                        token_count=token_count,
                        conn=conn,
                    )
                    create_tasks.append(task)

                await asyncio.gather(*create_tasks)

                # Sync vector tables after successful ingestion
                logger.info("Synchronizing vector tables")
                await self.workload_tool_ops.sync_tool_vectors(conn=conn)
                await self.workload_tool_ops.sync_tool_fts(conn=conn)
                logger.info("Vector synchronization completed")

                return {
                    "tools_count": len(tools),
                    "server_id": str(server_id),
                    "errors": errors,
                }

            except Exception as e:
                logger.exception("Failed to ingest tools", error=str(e))
                errors.append({"error": str(e)})
                raise

    async def load_and_ingest(self) -> dict:
        """Main method: fetch tools from AppWorld MCP and ingest them.

        Returns:
            dict with tools_count, server_id, errors
        """
        await self.setup()
        tools = await self.fetch_tools_from_mcp()
        return await self.ingest_tools(tools)


async def main():
    """Main entry point for standalone tool loading."""
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    )

    # Default paths
    db_path = Path(__file__).parent / "appworld_experiment.db"
    appworld_mcp_url = "http://localhost:10000/mcp"

    loader = AppWorldToolLoader(appworld_mcp_url=appworld_mcp_url, db_path=db_path)

    stats = await loader.load_and_ingest()

    logger.info(
        "Tool loading complete",
        tools_count=stats["tools_count"],
        server_id=stats["server_id"],
        error_count=len(stats["errors"]),
    )

    return stats


if __name__ == "__main__":
    asyncio.run(main())
