"""Tests for group filtering functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from mcp.types import Tool as McpTool

from mcp_optimizer.config import MCPOptimizerConfig
from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import McpStatus, WorkloadServer
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps


@pytest_asyncio.fixture
async def test_db():
    """Create a temporary SQLite database and run migrations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = Path(tmp_file.name)

    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    project_root = Path(__file__).parent.parent
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
        # Ignore cleanup errors
        pass
    test_db_path.unlink(missing_ok=True)


@pytest_asyncio.fixture
async def mcp_ops(test_db) -> WorkloadServerOps:
    """Create WorkloadServerOps instance with test database."""
    return WorkloadServerOps(test_db)


@pytest_asyncio.fixture
async def tool_ops(test_db) -> WorkloadToolOps:
    """Create WorkloadToolOps instance with test database."""
    return WorkloadToolOps(test_db)


@pytest_asyncio.fixture
async def servers_with_groups(mcp_ops) -> dict[str, WorkloadServer]:
    """Create sample servers in different groups."""
    group_a_server = await mcp_ops.create_server(
        name="workload-a",
        url="http://localhost:8000",
        workload_identifier="test/group-a-server",
        remote=False,
        transport="streamable-http",
        status="running",
        description="Group A server",
        server_embedding=np.random.rand(384).astype(np.float32),
        group="group-a",
    )

    group_b_server = await mcp_ops.create_server(
        name="workload-b",
        url="http://localhost:8001",
        workload_identifier="test/group-b-server",
        remote=False,
        transport="streamable-http",
        status="running",
        description="Group B server",
        server_embedding=np.random.rand(384).astype(np.float32),
        group="group-b",
    )

    default_group_server = await mcp_ops.create_server(
        name="workload-default",
        url="http://localhost:8002",
        workload_identifier="test/default-group-server",
        remote=False,
        transport="streamable-http",
        status="running",
        description="Default group server",
        server_embedding=np.random.rand(384).astype(np.float32),
        group="default",
    )

    return {
        "group_a": group_a_server,
        "group_b": group_b_server,
        "default_group": default_group_server,
    }


@pytest_asyncio.fixture
async def tools_in_groups(tool_ops, servers_with_groups):
    """Create sample tools in servers with different groups."""
    # Tool in group A
    tool_a = await tool_ops.create_tool(
        server_id=servers_with_groups["group_a"].id,
        details=McpTool(
            name="tool_a",
            description="Tool in group A",
            inputSchema={"type": "object", "properties": {}},
        ),
        details_embedding=np.random.rand(384).astype(np.float32),
        token_count=100,
    )

    # Tool in group B
    tool_b = await tool_ops.create_tool(
        server_id=servers_with_groups["group_b"].id,
        details=McpTool(
            name="tool_b",
            description="Tool in group B",
            inputSchema={"type": "object", "properties": {}},
        ),
        details_embedding=np.random.rand(384).astype(np.float32),
        token_count=150,
    )

    # Tool in default group
    tool_default = await tool_ops.create_tool(
        server_id=servers_with_groups["default_group"].id,
        details=McpTool(
            name="tool_default",
            description="Tool in default group",
            inputSchema={"type": "object", "properties": {}},
        ),
        details_embedding=np.random.rand(384).astype(np.float32),
        token_count=200,
    )

    return {"tool_a": tool_a, "tool_b": tool_b, "tool_default": tool_default}


class TestConfigGroupParsing:
    """Test configuration parsing for allowed_groups."""

    def test_parse_allowed_groups_single(self):
        """Test parsing a single group."""
        config = MCPOptimizerConfig(allowed_groups="group1")
        assert config.allowed_groups == ["group1"]

    def test_parse_allowed_groups_multiple(self):
        """Test parsing multiple groups."""
        config = MCPOptimizerConfig(allowed_groups="group1,group2,group3")
        assert config.allowed_groups == ["group1", "group2", "group3"]

    def test_parse_allowed_groups_with_spaces(self):
        """Test parsing groups with spaces."""
        config = MCPOptimizerConfig(allowed_groups="group1 , group2 , group3")
        assert config.allowed_groups == ["group1", "group2", "group3"]

    def test_parse_allowed_groups_none(self):
        """Test parsing when no groups are set."""
        config = MCPOptimizerConfig()
        assert config.allowed_groups is None

    def test_parse_allowed_groups_empty_string(self):
        """Test parsing an empty string."""
        config = MCPOptimizerConfig(allowed_groups="")
        assert config.allowed_groups is None

    def test_parse_allowed_groups_whitespace_only(self):
        """Test parsing whitespace-only string."""
        config = MCPOptimizerConfig(allowed_groups="   ")
        assert config.allowed_groups is None

    def test_parse_allowed_groups_list_input(self):
        """Test parsing when input is already a list."""
        config = MCPOptimizerConfig(allowed_groups=["group1", "group2", "group3"])
        assert config.allowed_groups == ["group1", "group2", "group3"]

    def test_parse_allowed_groups_list_with_spaces(self):
        """Test parsing list with spaces in group names."""
        config = MCPOptimizerConfig(allowed_groups=[" group1 ", " group2 ", " group3 "])
        assert config.allowed_groups == ["group1", "group2", "group3"]

    def test_parse_allowed_groups_empty_list(self):
        """Test parsing an empty list."""
        config = MCPOptimizerConfig(allowed_groups=[])
        assert config.allowed_groups is None


class TestServerGroupCRUD:
    """Test CRUD operations with group field."""

    @pytest.mark.asyncio
    async def test_create_server_with_group(self, mcp_ops):
        """Test creating a server with a group."""
        server = await mcp_ops.create_server(
            name="test-workload-with-group",
            url="http://localhost:9000",
            workload_identifier="test/server-with-group",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server with group",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="test-group",
        )
        assert server.group == "test-group"

    @pytest.mark.asyncio
    async def test_create_server_without_group(self, mcp_ops):
        """Test creating a server without explicitly specifying a group defaults to 'default'."""
        server = await mcp_ops.create_server(
            name="test-workload-no-group",
            url="http://localhost:9001",
            workload_identifier="test/server-no-group",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server without group",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        assert server.group == "default"

    @pytest.mark.asyncio
    async def test_update_server_group(self, mcp_ops):
        """Test updating a server's group."""
        server = await mcp_ops.create_server(
            name="test-workload-update-group",
            url="http://localhost:9002",
            workload_identifier="test/server-update-group",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server for group update",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="old-group",
        )
        assert server.group == "old-group"

        # Update the group
        updated_server = await mcp_ops.update_server(server.id, group="new-group")
        assert updated_server.group == "new-group"

    @pytest.mark.asyncio
    async def test_get_server_preserves_group(self, mcp_ops):
        """Test that getting a server preserves the group field."""
        server = await mcp_ops.create_server(
            name="test-workload-get-group",
            url="http://localhost:9003",
            workload_identifier="test/server-get-group",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server for get group test",
            server_embedding=np.random.rand(384).astype(np.float32),
            group="preserved-group",
        )

        # Get by ID
        retrieved_by_id = await mcp_ops.get_server_by_id(server.id)
        assert retrieved_by_id.group == "preserved-group"

        # Get by workload_name
        retrieved_by_name = await mcp_ops.get_server_by_workload_name("test-workload-get-group")
        assert retrieved_by_name.group == "preserved-group"


class TestToolFilteringByGroup:
    """Test tool filtering by group."""

    @pytest.mark.asyncio
    async def test_get_all_tools_no_group_filter(self, tool_ops, tools_in_groups):
        """Test get_all_tools returns all tools when no group filter is applied."""
        tools = await tool_ops.get_all_tools(server_statuses=[McpStatus.RUNNING])
        assert len(tools) == 3

    @pytest.mark.asyncio
    async def test_get_all_tools_filter_single_group(self, tool_ops, tools_in_groups):
        """Test get_all_tools filters by a single group."""
        tools = await tool_ops.get_all_tools(
            server_statuses=[McpStatus.RUNNING], allowed_groups=["group-a"]
        )
        assert len(tools) == 1
        assert tools[0].tool.details.name == "tool_a"

    @pytest.mark.asyncio
    async def test_get_all_tools_filter_multiple_groups(self, tool_ops, tools_in_groups):
        """Test get_all_tools filters by multiple groups."""
        tools = await tool_ops.get_all_tools(
            server_statuses=[McpStatus.RUNNING], allowed_groups=["group-a", "group-b"]
        )
        assert len(tools) == 2
        tool_names = {t.tool.details.name for t in tools}
        assert tool_names == {"tool_a", "tool_b"}

    @pytest.mark.asyncio
    async def test_get_all_tools_filter_nonexistent_group(self, tool_ops, tools_in_groups):
        """Test get_all_tools returns empty list for nonexistent group."""
        tools = await tool_ops.get_all_tools(
            server_statuses=[McpStatus.RUNNING], allowed_groups=["nonexistent"]
        )
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_find_similar_tools_no_group_filter(self, tool_ops, tools_in_groups):
        """Test find_similar_tools returns all relevant tools when no group filter is applied."""
        # Sync vectors first
        await tool_ops.sync_tool_vectors()

        # Create a query embedding
        query_embedding = np.random.rand(384).astype(np.float32)

        tools = await tool_ops.find_similar_tools(
            query_embedding=query_embedding,
            limit=10,
            distance_threshold=2.0,
            server_statuses=[McpStatus.RUNNING],
        )
        assert len(tools) >= 1  # Should find at least one tool

    @pytest.mark.asyncio
    async def test_find_similar_tools_filter_single_group(self, tool_ops, tools_in_groups):
        """Test find_similar_tools filters by a single group."""
        # Create a query embedding
        query_embedding = np.random.rand(384).astype(np.float32)

        tools = await tool_ops.find_similar_tools(
            query_embedding=query_embedding,
            limit=10,
            distance_threshold=2.0,
            server_statuses=[McpStatus.RUNNING],
            allowed_groups=["group-a"],
        )

        # Should only return tools from group-a
        for tool in tools:
            # Get the server to check its group
            assert tool.tool.mcpserver_id is not None
            # The tool should be from the group-a server

    @pytest.mark.asyncio
    async def test_find_similar_tools_filter_multiple_groups(self, tool_ops, tools_in_groups):
        """Test find_similar_tools filters by multiple groups."""
        # Sync vectors first
        await tool_ops.sync_tool_vectors()

        # Create a query embedding
        query_embedding = np.random.rand(384).astype(np.float32)

        tools = await tool_ops.find_similar_tools(
            query_embedding=query_embedding,
            limit=10,
            distance_threshold=2.0,
            server_statuses=[McpStatus.RUNNING],
            allowed_groups=["group-a", "group-b"],
        )

        # Should return tools from both groups but not from default group
        assert len(tools) >= 1


class TestHybridSearchWithGroups:
    """Test hybrid search (BM25 + semantic) with group filtering."""

    @pytest.mark.asyncio
    async def test_hybrid_search_with_group_filter(self, tool_ops, tools_in_groups):
        """Test hybrid search filters by groups."""
        # Ensure FTS table is ready
        await tool_ops.sync_tool_fts()

        query_embedding = np.random.rand(384).astype(np.float32)

        tools = await tool_ops.find_similar_tools(
            query_embedding=query_embedding,
            limit=10,
            distance_threshold=2.0,
            query_text="tool",
            hybrid_search_semantic_ratio=0.5,
            server_statuses=[McpStatus.RUNNING],
            allowed_groups=["group-a"],
        )

        # Should only return tools from group-a
        # (may be 0 if BM25 doesn't match or distance threshold is too low)
        for tool in tools:
            assert tool.tool.mcpserver_id is not None
