"""Unit tests for CRUD operations."""

import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest_asyncio
from alembic import command
from alembic.config import Config
from mcp.types import Tool as McpTool

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import McpStatus, TransportType
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps


@pytest_asyncio.fixture
async def db():
    """Create a temporary SQLite database and run migrations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = Path(tmp_file.name)

    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    project_root = Path(__file__).parent.parent.parent
    migrations_dir = project_root / "migrations"

    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(migrations_dir))
    alembic_config.set_main_option("db_url", f"sqlite:///{test_db_path}")

    command.upgrade(alembic_config, "head")

    # Create DatabaseConfig instance with test database
    db_config = DatabaseConfig(test_db_url)
    yield db_config

    # Cleanup
    try:
        await db_config.close()
    except Exception:
        pass
    try:
        test_db_path.unlink()
    except Exception:
        pass


@pytest_asyncio.fixture
async def server_ops(db):
    """Create WorkloadServerOps instance for testing."""
    return WorkloadServerOps(db)


@pytest_asyncio.fixture
async def tool_ops(db):
    """Create WorkloadToolOps instance for testing."""
    return WorkloadToolOps(db)


@pytest_asyncio.fixture
async def sample_server(server_ops):
    """Create a sample MCP server for testing."""
    return await server_ops.create_server(
        name="test-server",
        url="http://localhost:8080",
        workload_identifier="test-package",
        remote=False,
        transport=TransportType.SSE,
        status=McpStatus.RUNNING,
        description="Test server for unit tests",
        server_embedding=np.random.rand(384).astype(np.float32),
        group="default",
    )


class TestCreateToolWithTokenCount:
    """Test create_tool method with token_count parameter."""

    async def test_create_tool_with_token_count(self, tool_ops, sample_server):
        """Test creating a tool with explicit token_count."""
        tool = await tool_ops.create_tool(
            server_id=sample_server.id,
            details=McpTool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=150,
        )

        assert tool.token_count == 150
        assert tool.id is not None
        assert uuid.UUID(tool.id)

    async def test_create_tool_with_zero_token_count(self, tool_ops, sample_server):
        """Test creating a tool with zero token_count."""
        tool = await tool_ops.create_tool(
            server_id=sample_server.id,
            details=McpTool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=0,
        )

        assert tool.token_count == 0

    async def test_create_tool_with_large_token_count(self, tool_ops, sample_server):
        """Test creating a tool with large token_count."""
        tool = await tool_ops.create_tool(
            server_id=sample_server.id,
            details=McpTool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100000,
        )

        assert tool.token_count == 100000

    async def test_tool_token_count_persists_in_database(self, tool_ops, sample_server):
        """Test that token_count persists correctly in database."""
        # Create tool
        created_tool = await tool_ops.create_tool(
            server_id=sample_server.id,
            details=McpTool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {}},
            ),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=250,
        )

        # Retrieve tool from database
        retrieved_tool = await tool_ops.get_tool_by_id(created_tool.id)

        assert retrieved_tool.token_count == 250
        assert retrieved_tool.id == created_tool.id


class TestSumTokenCountsForRunningServers:
    """Test sum_token_counts_for_running_servers method."""

    async def test_sum_with_zero_tools(self, tool_ops):
        """Test summing token counts when no tools exist."""
        total = await tool_ops.sum_token_counts_for_running_servers()
        assert total == 0

    async def test_sum_with_single_running_server(self, tool_ops, server_ops):
        """Test summing token counts with single running server."""
        # Create running server
        server = await server_ops.create_server(
            name="server1",
            url="http://localhost:8080",
            workload_identifier="package1",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="default",
        )

        # Create tools with different token counts
        await tool_ops.create_tool(
            server_id=server.id,
            details=McpTool(name="tool1", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )
        await tool_ops.create_tool(
            server_id=server.id,
            details=McpTool(name="tool2", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=150,
        )
        await tool_ops.create_tool(
            server_id=server.id,
            details=McpTool(name="tool3", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )

        # Sum should be 100 + 150 + 200 = 450
        total = await tool_ops.sum_token_counts_for_running_servers()
        assert total == 450

    async def test_sum_with_multiple_running_servers(self, tool_ops, server_ops):
        """Test summing token counts with multiple running servers."""
        # Create multiple running servers
        server1 = await server_ops.create_server(
            name="server1",
            url="http://localhost:8080",
            workload_identifier="package1",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="default",
        )
        server2 = await server_ops.create_server(
            name="server2",
            url="http://localhost:8081",
            workload_identifier="package2",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="default",
        )

        # Create tools for server1
        await tool_ops.create_tool(
            server_id=server1.id,
            details=McpTool(name="tool1", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )
        await tool_ops.create_tool(
            server_id=server1.id,
            details=McpTool(name="tool2", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )

        # Create tools for server2
        await tool_ops.create_tool(
            server_id=server2.id,
            details=McpTool(name="tool3", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=300,
        )

        # Total should be 100 + 200 + 300 = 600
        total = await tool_ops.sum_token_counts_for_running_servers()
        assert total == 600

    async def test_sum_excludes_stopped_servers(self, tool_ops, server_ops):
        """Test that stopped servers are excluded from token count sum."""
        # Create running server
        running_server = await server_ops.create_server(
            name="running-server",
            url="http://localhost:8080",
            workload_identifier="package1",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Running server",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="default",
        )

        # Create stopped server
        stopped_server = await server_ops.create_server(
            name="stopped-server",
            url="http://localhost:8081",
            workload_identifier="package2",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.STOPPED,
            description="Stopped server",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="default",
        )

        # Create tools for running server
        await tool_ops.create_tool(
            server_id=running_server.id,
            details=McpTool(name="tool1", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )

        # Create tools for stopped server (should not be counted)
        await tool_ops.create_tool(
            server_id=stopped_server.id,
            details=McpTool(name="tool2", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=500,
        )

        # Total should only include running server: 100
        total = await tool_ops.sum_token_counts_for_running_servers()
        assert total == 100

    async def test_sum_with_group_filtering_single_group(self, tool_ops, server_ops):
        """Test summing token counts with group filtering for single group."""
        # Create servers in different groups
        server_a = await server_ops.create_server(
            name="server-a",
            url="http://localhost:8080",
            workload_identifier="package-a",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server A",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-a",
        )
        server_b = await server_ops.create_server(
            name="server-b",
            url="http://localhost:8081",
            workload_identifier="package-b",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server B",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-b",
        )

        # Create tools for group-a
        await tool_ops.create_tool(
            server_id=server_a.id,
            details=McpTool(name="tool-a", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )

        # Create tools for group-b
        await tool_ops.create_tool(
            server_id=server_b.id,
            details=McpTool(name="tool-b", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )

        # Filter by group-a only
        total_a = await tool_ops.sum_token_counts_for_running_servers(allowed_groups=["group-a"])
        assert total_a == 100

        # Filter by group-b only
        total_b = await tool_ops.sum_token_counts_for_running_servers(allowed_groups=["group-b"])
        assert total_b == 200

    async def test_sum_with_group_filtering_multiple_groups(self, tool_ops, server_ops):
        """Test summing token counts with group filtering for multiple groups."""
        # Create servers in different groups
        server_a = await server_ops.create_server(
            name="server-a-multi",
            url="http://localhost:8080",
            workload_identifier="package-a",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server A",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-a",
        )
        server_b = await server_ops.create_server(
            name="server-b-multi",
            url="http://localhost:8081",
            workload_identifier="package-b",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server B",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-b",
        )
        server_c = await server_ops.create_server(
            name="server-c-multi",
            url="http://localhost:8082",
            workload_identifier="package-c",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server C",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-c",
        )

        # Create tools for each group
        await tool_ops.create_tool(
            server_id=server_a.id,
            details=McpTool(name="tool-a", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )
        await tool_ops.create_tool(
            server_id=server_b.id,
            details=McpTool(name="tool-b", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )
        await tool_ops.create_tool(
            server_id=server_c.id,
            details=McpTool(name="tool-c", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=300,
        )

        # Filter by group-a and group-b
        total_ab = await tool_ops.sum_token_counts_for_running_servers(
            allowed_groups=["group-a", "group-b"]
        )
        assert total_ab == 300  # 100 + 200

    async def test_sum_with_empty_group_filter_includes_all(self, tool_ops, server_ops):
        """Test that empty group filter includes all groups."""
        # Create servers in different groups
        server_a = await server_ops.create_server(
            name="server-a-empty",
            url="http://localhost:8080",
            workload_identifier="package-a",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server A",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-a",
        )
        server_b = await server_ops.create_server(
            name="server-b-empty",
            url="http://localhost:8081",
            workload_identifier="package-b",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server B",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-b",
        )

        # Create tools
        await tool_ops.create_tool(
            server_id=server_a.id,
            details=McpTool(name="tool-a", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )
        await tool_ops.create_tool(
            server_id=server_b.id,
            details=McpTool(name="tool-b", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )

        # Empty group filter should include all
        total = await tool_ops.sum_token_counts_for_running_servers(allowed_groups=[])
        assert total == 300

    async def test_sum_with_none_group_filter_includes_all(self, tool_ops, server_ops):
        """Test that None group filter includes all groups."""
        # Create servers in different groups
        server_a = await server_ops.create_server(
            name="server-a-none",
            url="http://localhost:8080",
            workload_identifier="package-a",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server A",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-a",
        )
        server_b = await server_ops.create_server(
            name="server-b-none",
            url="http://localhost:8081",
            workload_identifier="package-b",
            remote=False,
            transport=TransportType.SSE,
            status=McpStatus.RUNNING,
            description="Server B",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="group-b",
        )

        # Create tools
        await tool_ops.create_tool(
            server_id=server_a.id,
            details=McpTool(name="tool-a", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=100,
        )
        await tool_ops.create_tool(
            server_id=server_b.id,
            details=McpTool(name="tool-b", inputSchema={}),
            details_embedding=np.random.rand(384).astype(np.float32),
            token_count=200,
        )

        # None group filter should include all
        total = await tool_ops.sum_token_counts_for_running_servers(allowed_groups=None)
        assert total == 300
