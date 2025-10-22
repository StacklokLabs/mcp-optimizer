import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pytest
from mcp.types import Tool as McpTool

from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import WorkloadServer
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps


@pytest.fixture
def sample_server_data() -> dict[str, Any]:
    """Sample server data for testing."""
    return {
        "name": "test-workload",
        "url": "http://localhost",
        "workload_identifier": "test/server",
        "remote": False,
        "transport": "streamable-http",
        "status": "running",
        "description": "Test server",
        "server_embedding": np.random.rand(384).astype(np.float32),
    }


class TestMCPServerOpsCreateServer:
    """Test cases for the create_server method."""

    @pytest.mark.asyncio
    async def test_create_server_with_all_parameters(self, mcp_ops, sample_server_data):
        """Test creating a server with all parameters provided."""
        result = await mcp_ops.create_server(**sample_server_data)

        # Verify the returned object
        assert isinstance(result, WorkloadServer)
        assert result.name == sample_server_data["name"]
        assert result.url == sample_server_data["url"]
        assert result.transport == sample_server_data["transport"]
        assert result.status == sample_server_data["status"]
        assert isinstance(result.id, str)
        assert uuid.UUID(result.id)  # Verify it's a valid UUID
        assert isinstance(result.created_at, datetime)
        assert isinstance(result.last_updated, datetime)

        # Verify data was persisted in database
        retrieved_server = await mcp_ops.get_server_by_id(result.id)
        assert retrieved_server.name == sample_server_data["name"]
        assert retrieved_server.url == sample_server_data["url"]

    @pytest.mark.asyncio
    async def test_create_server_generates_unique_ids(self, mcp_ops):
        """Test that multiple servers get unique IDs."""
        server1 = await mcp_ops.create_server(
            name="workload1",
            url="http://example1.com:8001",
            workload_identifier="test/server1",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload2",
            url="http://example2.com:8002",
            workload_identifier="test/server2",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        assert server1.id != server2.id
        assert uuid.UUID(server1.id)
        assert uuid.UUID(server2.id)


class TestMCPServerOpsGetServer:
    """Test cases for the get_server_by_id method."""

    @pytest.mark.asyncio
    async def test_get_existing_server_by_id(self, mcp_ops, sample_server_data):
        """Test retrieving an existing server by ID."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Retrieve it by ID
        retrieved_server = await mcp_ops.get_server_by_id(created_server.id)

        assert retrieved_server.id == created_server.id
        assert retrieved_server.name == created_server.name
        assert retrieved_server.url == created_server.url
        assert retrieved_server.transport == created_server.transport
        assert retrieved_server.status == created_server.status

    @pytest.mark.asyncio
    async def test_get_nonexistent_server_by_id_raises_error(self, mcp_ops):
        """Test that getting a non-existent server raises DbNotFoundError."""
        non_existent_id = str(uuid.uuid4())

        with pytest.raises(DbNotFoundError) as exc_info:
            await mcp_ops.get_server_by_id(non_existent_id)

        assert f"Server with ID {non_existent_id} not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_existing_server_by_name(self, mcp_ops, sample_server_data):
        """Test retrieving an existing server by workload_name."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Retrieve it by workload_name
        retrieved_server = await mcp_ops.get_server_by_workload_name(created_server.name)

        assert retrieved_server.id == created_server.id
        assert retrieved_server.name == created_server.name
        assert retrieved_server.url == created_server.url
        assert retrieved_server.transport == created_server.transport
        assert retrieved_server.status == created_server.status

    @pytest.mark.asyncio
    async def test_get_nonexistent_server_by_name_raises_error(self, mcp_ops):
        """Test that getting a non-existent server raises DbNotFoundError."""
        non_existent_name = "non-existent-server"

        with pytest.raises(DbNotFoundError) as exc_info:
            await mcp_ops.get_server_by_workload_name(non_existent_name)

        assert f"with name '{non_existent_name}' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_server_by_workload_name_case_sensitive(self, mcp_ops, sample_server_data):
        """Test that workload_name lookup is case sensitive."""
        # Create a server
        await mcp_ops.create_server(**sample_server_data)

        # Try to retrieve with different case
        with pytest.raises(DbNotFoundError):
            await mcp_ops.get_server_by_workload_name(sample_server_data["name"].upper())

    @pytest.mark.asyncio
    async def test_multiple_servers_get_correct_one_by_name(self, mcp_ops):
        """Test retrieving the correct server when multiple exist."""
        # Create multiple servers
        server1 = await mcp_ops.create_server(
            name="workload-one",
            url="http://one.com:8001",
            workload_identifier="test/server-one",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server one",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload-two",
            url="http://two.com:8002",
            workload_identifier="test/server-two",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server two",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Retrieve each by workload_name
        retrieved1 = await mcp_ops.get_server_by_workload_name("workload-one")
        retrieved2 = await mcp_ops.get_server_by_workload_name("workload-two")

        assert retrieved1.id == server1.id
        assert retrieved1.name == "workload-one"
        assert retrieved2.id == server2.id
        assert retrieved2.name == "workload-two"


class TestMCPServerOpsUpdateServer:
    """Test cases for the update_server method."""

    @pytest.mark.asyncio
    async def test_update_single_field(self, mcp_ops, sample_server_data):
        """Test updating a single field."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)
        original_updated_time = created_server.last_updated

        # Update only the description
        updated_server = await mcp_ops.update_server(
            created_server.id, description="updated description"
        )

        assert updated_server.id == created_server.id
        assert updated_server.description == "updated description"
        assert updated_server.url == created_server.url
        assert updated_server.transport == created_server.transport
        assert updated_server.status == created_server.status
        assert updated_server.last_updated > original_updated_time

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, mcp_ops, sample_server_data):
        """Test updating multiple fields."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Update multiple fields
        updated_server = await mcp_ops.update_server(
            created_server.id,
            description="new description",
            url="https://new-url.com:9090",
            status="stopped",
        )

        assert updated_server.id == created_server.id
        assert updated_server.description == "new description"
        assert updated_server.url == "https://new-url.com:9090"
        assert updated_server.status == "stopped"
        assert updated_server.transport == created_server.transport  # Unchanged

    @pytest.mark.asyncio
    async def test_update_nonexistent_server_raises_error(self, mcp_ops):
        """Test that updating a non-existent server raises DbNotFoundError."""
        non_existent_id = str(uuid.uuid4())

        with pytest.raises(DbNotFoundError) as exc_info:
            await mcp_ops.update_server(non_existent_id, description="new description")

        assert f"Server with ID {non_existent_id} not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_no_fields_returns_unchanged_server(self, mcp_ops, sample_server_data):
        """Test updating with no fields returns the server unchanged."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Update with no fields
        updated_server = await mcp_ops.update_server(created_server.id)

        assert updated_server.id == created_server.id
        assert updated_server.name == created_server.name
        assert updated_server.url == created_server.url
        assert updated_server.transport == created_server.transport
        assert updated_server.status == created_server.status


class TestMCPServerOpsDeleteServer:
    """Test cases for the delete server methods."""

    @pytest.mark.asyncio
    async def test_delete_server_by_id_success(self, mcp_ops, sample_server_data):
        """Test successfully deleting a server by ID."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Delete the server
        await mcp_ops.delete_server(created_server.id)

        # Verify server is deleted
        with pytest.raises(DbNotFoundError):
            await mcp_ops.get_server_by_id(created_server.id)

    @pytest.mark.asyncio
    async def test_delete_server_by_name_success(self, mcp_ops, sample_server_data):
        """Test successfully deleting a server by workload_name."""
        # Create a server first
        created_server = await mcp_ops.create_server(**sample_server_data)

        # Get the server by workload_name, then delete it by ID
        server = await mcp_ops.get_server_by_workload_name(created_server.name)
        await mcp_ops.delete_server(server.id)

        # Verify server is deleted
        with pytest.raises(DbNotFoundError):
            await mcp_ops.get_server_by_workload_name(created_server.name)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_server_by_id_does_nothing(self, mcp_ops):
        """Test that deleting a non-existent server by ID does nothing."""
        non_existent_id = str(uuid.uuid4())

        # Delete should not raise an error, just delete nothing
        await mcp_ops.delete_server(non_existent_id)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_server_by_name_raises_error(self, mcp_ops):
        """Test that getting a non-existent server by workload_name raises DbNotFoundError."""
        non_existent_name = "non-existent-server"

        # Try to get the server by workload_name first (which should fail)
        with pytest.raises(DbNotFoundError) as exc_info:
            server = await mcp_ops.get_server_by_workload_name(non_existent_name)
            await mcp_ops.delete_server(server.id)

        assert f"with name '{non_existent_name}' not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_delete_server_by_id_leaves_other_servers_intact(self, mcp_ops):
        """Test that deleting one server leaves other servers intact."""
        # Create multiple servers
        server1 = await mcp_ops.create_server(
            name="workload-one",
            url="http://one.com:8001",
            workload_identifier="test/server-one",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server one",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload-two",
            url="http://two.com:8002",
            workload_identifier="test/server-two",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server two",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Delete first server
        await mcp_ops.delete_server(server1.id)

        # Verify first server is deleted
        with pytest.raises(DbNotFoundError):
            await mcp_ops.get_server_by_id(server1.id)

        # Verify second server still exists
        retrieved_server2 = await mcp_ops.get_server_by_id(server2.id)
        assert retrieved_server2.id == server2.id
        assert retrieved_server2.name == "workload-two"

    @pytest.mark.asyncio
    async def test_delete_server_by_name_leaves_other_servers_intact(self, mcp_ops):
        """Test that deleting one server by name leaves other servers intact."""
        # Create multiple servers
        _ = await mcp_ops.create_server(
            name="workload-one",
            url="http://one.com:8001",
            workload_identifier="test/server-one",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server one",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload-two",
            url="http://two.com:8002",
            workload_identifier="test/server-two",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server two",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Delete first server by getting it by workload_name and then deleting by ID
        server = await mcp_ops.get_server_by_workload_name("workload-one")
        await mcp_ops.delete_server(server.id)

        # Verify first server is deleted
        with pytest.raises(DbNotFoundError):
            await mcp_ops.get_server_by_workload_name("workload-one")

        # Verify second server still exists
        retrieved_server2 = await mcp_ops.get_server_by_workload_name("workload-two")
        assert retrieved_server2.id == server2.id
        assert retrieved_server2.name == "workload-two"


class TestMCPServerOpsGetServersWithTools:
    """Test cases for getting servers with their tools."""

    @pytest.mark.asyncio
    async def test_list_servers_empty(self, mcp_ops):
        """Test listing servers when no servers exist."""
        servers = await mcp_ops.list_servers()
        assert len(servers) == 0

    @pytest.mark.asyncio
    async def test_list_servers_no_tools(self, mcp_ops):
        """Test listing servers when servers have no tools."""
        # Create some servers
        server1 = await mcp_ops.create_server(
            name="workload1",
            url="http://localhost:8001",
            workload_identifier="test/server1",
            remote=False,
            transport="sse",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload2",
            url="http://localhost:8002",
            workload_identifier="test/server2",
            remote=False,
            transport="sse",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        servers = await mcp_ops.list_servers()

        assert len(servers) == 2
        # Check that all servers exist
        for server in servers:
            assert isinstance(server, WorkloadServer)
            assert server.id in [server1.id, server2.id]

    @pytest.mark.asyncio
    async def test_list_servers_with_tools(self, mcp_ops, test_db):
        """Test listing servers when servers have tools."""
        # Create servers
        server1 = await mcp_ops.create_server(
            name="workload1",
            url="http://localhost:8001",
            workload_identifier="test/server1",
            remote=False,
            transport="sse",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload2",
            url="http://localhost:8002",
            workload_identifier="test/server2",
            remote=False,
            transport="sse",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Create tools for servers
        tool_ops = WorkloadToolOps(test_db)

        # Server 1 gets 2 tools
        await tool_ops.create_tool(
            server_id=server1.id,
            details=McpTool(name="tool1", description="Tool 1", inputSchema={"type": "object"}),
            details_embedding=np.random.rand(100).astype(np.float32),
            token_count=50,
        )
        await tool_ops.create_tool(
            server_id=server1.id,
            details=McpTool(name="tool2", description="Tool 2", inputSchema={"type": "object"}),
            details_embedding=np.random.rand(100).astype(np.float32),
            token_count=50,
        )

        # Server 2 gets 1 tool
        await tool_ops.create_tool(
            server_id=server2.id,
            details=McpTool(name="tool3", description="Tool 3", inputSchema={"type": "object"}),
            details_embedding=np.random.rand(100).astype(np.float32),
            token_count=50,
        )

        servers = await mcp_ops.list_servers()

        assert len(servers) == 2

        # Check servers are included
        server_ids = {s.id for s in servers}
        assert server1.id in server_ids
        assert server2.id in server_ids
