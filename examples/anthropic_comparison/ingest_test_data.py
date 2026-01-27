"""Standalone ingestion script for loading test data into MCP Optimizer database.

This script loads tools from mcp_tools_cleaned.json and creates a testing database
that enables find_tool to work without a running ToolHive server.

Uses core ingestion logic from IngestionService for consistency.
"""

import asyncio
import json
import os
from pathlib import Path

import structlog
from mcp.types import Tool

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import McpStatus, TransportType
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionService
from mcp_optimizer.response_optimizer.token_counter import TokenCounter

logger = structlog.get_logger(__name__)


class TestDataLoader:
    """Loads test data using core ingestion logic for consistency."""

    def __init__(
        self,
        ingestion_service: IngestionService,
        workload_server_ops: WorkloadServerOps,
        workload_tool_ops: WorkloadToolOps,
        token_counter: TokenCounter,
    ):
        """Initialize with dependencies.

        Args:
            ingestion_service: IngestionService instance (for _create_tool_text_to_embed)
            workload_server_ops: WorkloadServerOps instance
            workload_tool_ops: WorkloadToolOps instance
            token_counter: TokenCounter instance
        """
        self.ingestion_service = ingestion_service
        self.workload_server_ops = workload_server_ops
        self.workload_tool_ops = workload_tool_ops
        self.token_counter = token_counter
        # Get embedding_manager from ingestion_service for consistency
        self.embedding_manager = ingestion_service.embedding_manager

    def _convert_tool_to_mcp_format(self, tool_dict: dict) -> Tool:
        """Convert tool from mcp_tools_cleaned.json format to MCP Tool format.

        Args:
            tool_dict: Tool dictionary from JSON

        Returns:
            MCP Tool instance
        """
        return Tool(
            name=tool_dict["name"],
            description=tool_dict.get("description", ""),
            inputSchema={"properties": tool_dict.get("parameter", {})},
        )

    async def load_from_json(self, json_path: Path) -> dict:
        """Load tools from mcp_tools_cleaned.json using core ingestion logic.

        Uses the same logic as IngestionService._sync_workload_tools for consistency.

        Args:
            json_path: Path to mcp_tools_cleaned.json

        Returns:
            Dictionary with ingestion statistics
        """
        logger.info("Loading test data", json_path=str(json_path))

        # Load JSON
        with open(json_path) as f:
            servers_data = json.load(f)

        logger.info("Loaded servers from JSON", server_count=len(servers_data))

        total_servers = 0
        total_tools = 0
        errors = []

        # Get database connection and start transaction
        async with self.workload_server_ops.db.begin_transaction() as conn:
            for server_dict in servers_data:
                server_name = server_dict["name"]

                try:
                    logger.info("Processing server", server_name=server_name)

                    # Convert tools to MCP format
                    tools = [
                        self._convert_tool_to_mcp_format(tool_dict)
                        for tool_dict in server_dict.get("tools", [])
                    ]

                    if not tools:
                        logger.warning("No tools found for server", server_name=server_name)
                        continue

                    # Create workload server
                    server = await self.workload_server_ops.create_server(
                        name=server_name,
                        url=f"http://test-{server_name.lower().replace(' ', '-')}.local",
                        workload_identifier=f"test-{server_name.lower().replace(' ', '-')}",
                        remote=False,
                        transport=TransportType.SSE,
                        status=McpStatus.RUNNING,
                        description=server_dict.get(
                            "description", f"Test server for {server_name}"
                        ),
                        conn=conn,
                    )
                    server_id = server.id

                    # Sync tools using same logic as IngestionService._sync_workload_tools
                    # Generate texts for all tools using the server name as context
                    # Use IngestionService._create_tool_text_to_embed for consistency
                    tool_texts = [
                        self.ingestion_service._create_tool_text_to_embed(tool, server_name)
                        for tool in tools
                    ]

                    # Generate embeddings for all tools at once
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

                    total_servers += 1
                    total_tools += len(tools)

                    logger.info(
                        "Server processed successfully",
                        server_name=server_name,
                        tools_synced=len(tools),
                    )

                except Exception as e:
                    logger.exception(
                        "Failed to process server",
                        server_name=server_name,
                        error=str(e),
                    )
                    errors.append({"server": server_name, "error": str(e)})

            # Sync vector tables after successful ingestion
            # This populates the virtual vector tables used for semantic search
            if total_servers > 0:
                logger.info(
                    "Synchronizing vector tables after ingestion",
                    total_servers=total_servers,
                    total_tools=total_tools,
                )
                await self.workload_tool_ops.sync_tool_vectors(conn=conn)
                await self.workload_tool_ops.sync_tool_fts(conn=conn)
                logger.info("Vector synchronization completed")

        return {
            "total_servers": total_servers,
            "total_tools": total_tools,
            "errors": errors,
        }


async def main():
    """Main entry point for standalone ingestion."""
    # Setup logging
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    )

    # Path to test data
    json_path = Path(__file__).parent / Path("mcp_tools_cleaned.json")

    if not json_path.exists():
        raise FileNotFoundError(f"Test data not found: {json_path}")

    # Initialize database config (use test database in current directory)
    test_db_path = Path("./mcp_optimizer_test.db").resolve()

    # Set environment variables for migrations
    async_db_url = f"sqlite+aiosqlite:///{test_db_path}"
    sync_db_url = f"sqlite:///{test_db_path}"

    os.environ["ASYNC_DB_URL"] = async_db_url
    os.environ["DB_URL"] = sync_db_url

    db_config = DatabaseConfig(database_url=async_db_url)

    # Run migrations to create schema if needed
    logger.info("Running database migrations", db_path=str(test_db_path))
    from mcp_optimizer.db.config import run_migrations

    try:
        run_migrations()
        logger.info("Migrations completed successfully")
    except Exception as e:
        logger.warning("Migration error (may be expected if schema exists)", error=str(e))

    # Initialize embedding manager
    logger.info("Initializing embedding manager")
    embedding_manager = EmbeddingManager(
        model_name="BAAI/bge-small-en-v1.5",
        enable_cache=True,
        threads=2,
        fastembed_cache_path=None,
    )

    # Initialize token counter
    token_counter = TokenCounter(encoding_name="cl100k_base")

    # Initialize ops classes
    workload_server_ops = WorkloadServerOps(db_config)
    workload_tool_ops = WorkloadToolOps(db_config)

    # Create IngestionService to reuse its _create_tool_text_to_embed method
    ingestion_service = IngestionService(
        db_config=db_config,
        embedding_manager=embedding_manager,
        mcp_timeout=10,
        registry_ingestion_batch_size=5,
        workload_ingestion_batch_size=5,
        encoding="cl100k_base",
        skipped_workloads=[],
        runtime_mode="docker",
        k8s_api_server_url="http://127.0.0.1:8001",
        k8s_namespace=None,
        k8s_all_namespaces=True,
    )

    # Create test data loader with dependencies
    test_loader = TestDataLoader(
        ingestion_service=ingestion_service,
        workload_server_ops=workload_server_ops,
        workload_tool_ops=workload_tool_ops,
        token_counter=token_counter,
    )

    # Run ingestion
    logger.info("Starting ingestion")
    stats = await test_loader.load_from_json(json_path)

    # Print results
    logger.info(
        "Ingestion complete",
        total_servers=stats["total_servers"],
        total_tools=stats["total_tools"],
        error_count=len(stats["errors"]),
    )

    if stats["errors"]:
        logger.warning("Errors during ingestion", errors=stats["errors"])

    return stats


if __name__ == "__main__":
    asyncio.run(main())
