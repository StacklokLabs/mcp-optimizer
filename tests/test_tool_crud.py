import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pytest
import pytest_asyncio
from mcp.types import Tool as McpTool

from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import McpStatus, WorkloadServer, WorkloadTool
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps


@pytest_asyncio.fixture
async def tool_ops(test_db) -> WorkloadToolOps:
    """Create WorkloadToolOps instance with test database."""
    return WorkloadToolOps(test_db)


@pytest_asyncio.fixture
async def sample_server(mcp_ops) -> WorkloadServer:
    """Create a sample server for tool testing."""
    return await mcp_ops.create_server(
        name="test-workload",
        url="http://localhost",
        workload_identifier="test/server",
        remote=False,
        transport="streamable-http",
        status="running",
        description="Test server for tool testing",
        server_embedding=np.random.rand(384).astype(np.float32),
    )


@pytest.fixture
def sample_tool_details() -> dict[str, Any]:
    """Sample tool details for testing."""
    mcp_tool = McpTool(
        name="sample_tool",
        description="This is a sample tool for testing purposes.",
        inputSchema={"type": "object", "properties": {}},
    )
    return {
        "details": mcp_tool,
        "details_embedding": np.random.rand(384).astype(
            np.float32
        ),  # Example embedding as 1D numpy array
        "token_count": 42,  # Default token count for testing
    }


class TestToolOpsCreateTool:
    """Test cases for the create_tool method."""

    @pytest.mark.asyncio
    async def test_create_tool_with_valid_data(self, tool_ops, sample_server, sample_tool_details):
        """Test creating a tool with valid data."""
        sample_tool_details["server_id"] = sample_server.id
        result = await tool_ops.create_tool(**sample_tool_details)

        # Verify the returned object
        assert isinstance(result, WorkloadTool)
        assert result.mcpserver_id == sample_server.id
        assert result.details == sample_tool_details["details"]
        assert isinstance(result.id, str)
        assert uuid.UUID(result.id)  # Verify it's a valid UUID
        assert isinstance(result.created_at, datetime)
        assert isinstance(result.last_updated, datetime)
        assert isinstance(result.details_embedding, np.ndarray)
        assert result.details_embedding.ndim == 1
        assert len(result.details_embedding) > 0

        # Verify data was persisted in database
        retrieved_tool = await tool_ops.get_tool_by_id(result.id)
        assert retrieved_tool.mcpserver_id == sample_server.id
        # Compare serialized forms due to MCP Tool model serialization issue
        assert retrieved_tool.details.model_dump() == sample_tool_details["details"].model_dump()

    @pytest.mark.asyncio
    async def test_create_tool_generates_unique_ids(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test that multiple tools get unique IDs."""
        sample_tool_details["server_id"] = sample_server.id

        details1 = sample_tool_details.copy()
        details1["details"] = McpTool(
            name="tool1",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        details2 = sample_tool_details.copy()
        details2["details"] = McpTool(
            name="tool2",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        tool1 = await tool_ops.create_tool(**details1)
        tool2 = await tool_ops.create_tool(**details2)

        assert tool1.id != tool2.id
        assert uuid.UUID(tool1.id)
        assert uuid.UUID(tool2.id)


class TestToolOpsGetTool:
    """Test cases for getting tools."""

    @pytest.mark.asyncio
    async def test_get_existing_tool_by_id(self, tool_ops, sample_server, sample_tool_details):
        """Test retrieving an existing tool by ID."""
        # Create a tool first
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Retrieve it by ID
        retrieved_tool = await tool_ops.get_tool_by_id(created_tool.id)

        assert retrieved_tool.id == created_tool.id
        assert retrieved_tool.mcpserver_id == created_tool.mcpserver_id
        # Compare serialized forms due to MCP Tool model serialization issue
        assert retrieved_tool.details.model_dump() == created_tool.details.model_dump()
        assert np.array_equal(retrieved_tool.details_embedding, created_tool.details_embedding)

    @pytest.mark.asyncio
    async def test_get_nonexistent_tool_by_id_raises_error(self, tool_ops):
        """Test that getting a non-existent tool raises DbNotFoundError."""
        non_existent_id = str(uuid.uuid4())

        with pytest.raises(DbNotFoundError) as exc_info:
            await tool_ops.get_tool_by_id(non_existent_id)

        assert f"Tool with ID {non_existent_id} not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_tools_by_server(self, tool_ops, sample_server, sample_tool_details):
        """Test getting all tools for a server."""
        # Create multiple tools for the server
        sample_tool_details["server_id"] = sample_server.id
        details1 = sample_tool_details.copy()
        details1["details"] = McpTool(
            name="tool1",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        details2 = sample_tool_details.copy()
        details2["details"] = McpTool(
            name="tool2",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        tool1 = await tool_ops.create_tool(**details1)
        tool2 = await tool_ops.create_tool(**details2)

        # Get all tools for server
        tools = await tool_ops.list_tools_by_server(sample_server.id)

        assert len(tools) == 2
        tool_ids = [t.id for t in tools]
        assert tool1.id in tool_ids
        assert tool2.id in tool_ids

    @pytest.mark.asyncio
    async def test_list_tools_by_server_empty_result(self, tool_ops, sample_server):
        """Test getting tools for a server with no tools."""
        tools = await tool_ops.list_tools_by_server(sample_server.id)
        assert len(tools) == 0


class TestToolOpsUpdateTool:
    """Test cases for updating tools."""

    @pytest.mark.asyncio
    async def test_update_tool_details(self, tool_ops, sample_server, sample_tool_details):
        """Test updating tool details."""
        # Create a tool first
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)
        original_updated_time = created_tool.last_updated
        original_embedding = created_tool.details_embedding

        # Update the details
        new_mcp_tool = McpTool(
            name="updated_tool",
            description="Updated description for testing",
            inputSchema={"type": "object", "properties": {}},
        )
        new_details_embedding = np.random.rand(384).astype(np.float32)

        updated_tool = await tool_ops.update_tool(
            created_tool.id,
            details=new_mcp_tool,
            details_embedding=new_details_embedding,
        )

        assert updated_tool.id == created_tool.id
        # Compare serialized forms due to MCP Tool model serialization issue
        assert updated_tool.details.model_dump() == new_mcp_tool.model_dump()
        assert updated_tool.details.model_dump() != created_tool.details.model_dump()
        assert updated_tool.last_updated > original_updated_time
        # Embedding should change since details changed
        assert not np.array_equal(updated_tool.details_embedding, original_embedding)

    @pytest.mark.asyncio
    async def test_update_tool_server_id(self, tool_ops, mcp_ops, sample_tool_details):
        """Test updating tool's server assignment."""
        # Create two servers
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

        # Create tool for first server
        sample_tool_details["server_id"] = server1.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Update to second server
        updated_tool = await tool_ops.update_tool(created_tool.id, mcpserver_id=server2.id)

        assert updated_tool.id == created_tool.id
        assert updated_tool.mcpserver_id == server2.id
        assert updated_tool.mcpserver_id != created_tool.mcpserver_id

    @pytest.mark.asyncio
    async def test_update_nonexistent_tool_raises_error(self, tool_ops):
        """Test that updating a non-existent tool raises DbNotFoundError."""
        non_existent_id = str(uuid.uuid4())

        with pytest.raises(DbNotFoundError) as exc_info:
            await tool_ops.update_tool(non_existent_id, details={"name": "new"})

        assert f"Tool with ID {non_existent_id} not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_update_no_fields_returns_unchanged_tool(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test updating with no fields returns the tool unchanged."""
        # Create a tool first
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Update with no fields
        updated_tool = await tool_ops.update_tool(created_tool.id)

        assert updated_tool.id == created_tool.id
        # Compare serialized forms due to MCP Tool model serialization issue
        assert updated_tool.details.model_dump() == created_tool.details.model_dump()
        assert updated_tool.mcpserver_id == created_tool.mcpserver_id
        assert np.array_equal(updated_tool.details_embedding, created_tool.details_embedding)


class TestToolOpsDeleteTool:
    """Test cases for deleting tools."""

    @pytest.mark.asyncio
    async def test_delete_tool_success(self, tool_ops, sample_server, sample_tool_details):
        """Test successfully deleting a tool by ID."""
        # Create a tool first
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Delete the tool (returns None)
        await tool_ops.delete_tool(created_tool.id)

        # Verify tool is deleted
        with pytest.raises(DbNotFoundError):
            await tool_ops.get_tool_by_id(created_tool.id)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_tool_raises_error(self, tool_ops):
        """Test that deleting a non-existent tool does not raise an error."""
        non_existent_id = str(uuid.uuid4())

        # Should not raise an error (idempotent delete)
        await tool_ops.delete_tool(non_existent_id)

    @pytest.mark.asyncio
    async def test_delete_tools_by_server(self, tool_ops, sample_server, sample_tool_details):
        """Test deleting all tools for a server."""
        sample_tool_details["server_id"] = sample_server.id

        # Create multiple tools for the server
        details1 = sample_tool_details.copy()
        details1["details"] = McpTool(
            name="tool1",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        details2 = sample_tool_details.copy()
        details2["details"] = McpTool(
            name="tool2",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )

        await tool_ops.create_tool(**details1)
        await tool_ops.create_tool(**details2)

        # Delete all tools for server
        count = await tool_ops.delete_tools_by_server(sample_server.id)
        assert count == 2

        # Verify no tools remain for server
        tools = await tool_ops.list_tools_by_server(sample_server.id)
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_delete_tools_by_server_empty(self, tool_ops, sample_server):
        """Test deleting tools for a server with no tools."""
        count = await tool_ops.delete_tools_by_server(sample_server.id)
        assert count == 0


class TestToolOpsForeignKeyConstraints:
    """Test foreign key constraints and cascading deletes."""

    @pytest.mark.asyncio
    async def test_cascade_delete_when_server_deleted(self, tool_ops, mcp_ops, sample_tool_details):
        """Test that tools are deleted when their server is deleted."""
        # Create a server and tools
        server = await mcp_ops.create_server(
            name="test-workload",
            url="http://localhost:8080",
            workload_identifier="test/server",
            remote=False,
            transport="sse",
            status="running",
            description="Test server",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        sample_tool_details["server_id"] = server.id
        tool1 = await tool_ops.create_tool(**sample_tool_details)

        details2 = sample_tool_details.copy()
        details2["details"] = McpTool(
            name="tool2",
            description="This is a sample tool for testing purposes.",
            inputSchema={"type": "object", "properties": {}},
        )
        tool2 = await tool_ops.create_tool(**details2)

        # Verify tools exist
        tools_before = await tool_ops.list_tools_by_server(server.id)
        assert len(tools_before) == 2

        # Delete the server
        await mcp_ops.delete_server(server.id)

        # Verify tools are also deleted (cascade)
        with pytest.raises(DbNotFoundError):
            await tool_ops.get_tool_by_id(tool1.id)

        with pytest.raises(DbNotFoundError):
            await tool_ops.get_tool_by_id(tool2.id)


class TestToolOpsSimilaritySearch:
    """Test similarity search functionality for tools."""

    @pytest.mark.asyncio
    async def test_find_similar_tools_empty_database(self, tool_ops):
        """Test similarity search on empty database returns empty list."""
        query_embedding = np.random.random(384).astype(np.float32)
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_single_tool(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with single tool in database."""
        # Create a tool
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # Create query embedding (same dimension as the tool embedding)
        query_embedding = np.random.random(384).astype(np.float32)

        # Search for similar tools
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return the single tool
        assert len(results) == 1
        assert results[0].tool.id == created_tool.id
        assert results[0].tool.details.name == created_tool.details.name

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_multiple_tools(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with multiple tools in database."""
        sample_tool_details["server_id"] = sample_server.id

        # Create multiple tools with different embeddings
        tools_created = []
        for i in range(3):
            details = sample_tool_details.copy()
            details["details"] = McpTool(
                name=f"test_tool_{i}",
                description=f"Test tool number {i} for similarity testing",
                inputSchema={"type": "object", "properties": {}},
            )
            details["details_embedding"] = np.random.random(384).astype(np.float32)

            tool = await tool_ops.create_tool(**details)
            tools_created.append(tool)

        # Create query embedding
        query_embedding = np.random.random(384).astype(np.float32)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # Search for similar tools with higher threshold for testing
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return all 3 tools
        assert len(results) == 3

        # Verify all created tools are in results
        result_ids = {tool.tool.id for tool in results}
        created_ids = {tool.id for tool in tools_created}
        assert result_ids == created_ids

    @pytest.mark.asyncio
    async def test_find_similar_tools_respects_limit(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test that similarity search respects the limit parameter."""
        sample_tool_details["server_id"] = sample_server.id

        # Create 5 tools
        for i in range(5):
            details = sample_tool_details.copy()
            details["details"] = McpTool(
                name=f"tool_{i}",
                description=f"Tool {i} for limit testing",
                inputSchema={"type": "object", "properties": {}},
            )
            details["details_embedding"] = np.random.random(384).astype(np.float32)

            await tool_ops.create_tool(**details)

        # Create query embedding
        query_embedding = np.random.random(384).astype(np.float32)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # Search with limit of 3 and higher threshold for testing
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=3,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return exactly 3 tools
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_find_similar_tools_virtual_table_created_once(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test that virtual table is created only once and reused."""
        sample_tool_details["server_id"] = sample_server.id

        # Create a tool
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Create query embedding
        query_embedding = np.random.random(384).astype(np.float32)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # First search - should create virtual table
        results1 = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )
        assert len(results1) == 1

        # Second search - should reuse existing virtual table
        results2 = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )
        assert len(results2) == 1

        # Both searches should return the same tool
        assert results1[0].tool.id == results2[0].tool.id == created_tool.id

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_different_servers(
        self, tool_ops, mcp_ops, sample_tool_details
    ):
        """Test similarity search with tools from different servers."""
        # Create two servers
        server1 = await mcp_ops.create_server(
            name="workload1",
            url="http://localhost:8080",
            workload_identifier="test/server1",
            remote=False,
            transport="sse",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload2",
            url="http://localhost:8081",
            workload_identifier="test/server2",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Create tools for each server
        details1 = sample_tool_details.copy()
        details1["server_id"] = server1.id
        details1["details"] = McpTool(
            name="server1_tool",
            description="Tool from server 1",
            inputSchema={"type": "object", "properties": {}},
        )
        tool1 = await tool_ops.create_tool(**details1)

        details2 = sample_tool_details.copy()
        details2["server_id"] = server2.id
        details2["details"] = McpTool(
            name="server2_tool",
            description="Tool from server 2",
            inputSchema={"type": "object", "properties": {}},
        )
        tool2 = await tool_ops.create_tool(**details2)

        # Search for similar tools with higher threshold for testing
        query_embedding = np.random.random(384).astype(np.float32)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING.value],
        )

        # Should return tools from both servers
        assert len(results) == 2
        result_ids = {tool.tool.id for tool in results}
        assert result_ids == {tool1.id, tool2.id}

    @pytest.mark.asyncio
    async def test_find_similar_tools_ordering_by_similarity(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test that similarity search returns results ordered by similarity (distance)."""
        sample_tool_details["server_id"] = sample_server.id

        # Create a query embedding
        query_embedding = np.array([1.0] * 384, dtype=np.float32)

        # Create tools with known embeddings at different distances from query
        # Tool 1: Close to query (small distance)
        details1 = sample_tool_details.copy()
        details1["details"] = McpTool(
            name="close_tool",
            description="Tool close to query",
            inputSchema={"type": "object", "properties": {}},
        )
        details1["details_embedding"] = np.array(
            [0.9] * 384, dtype=np.float32
        )  # Very similar to query
        await tool_ops.create_tool(**details1)

        # Tool 2: Far from query (large distance)
        details2 = sample_tool_details.copy()
        details2["details"] = McpTool(
            name="far_tool",
            description="Tool far from query",
            inputSchema={"type": "object", "properties": {}},
        )
        details2["details_embedding"] = np.array(
            [-1.0] * 384, dtype=np.float32
        )  # Very different from query
        await tool_ops.create_tool(**details2)

        # Tool 3: Medium distance from query
        details3 = sample_tool_details.copy()
        details3["details"] = McpTool(
            name="medium_tool",
            description="Tool medium distance from query",
            inputSchema={"type": "object", "properties": {}},
        )
        details3["details_embedding"] = np.array([0.1] * 384, dtype=np.float32)  # Medium similarity
        await tool_ops.create_tool(**details3)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # Search for similar tools with higher threshold for testing
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return all 3 tools
        assert len(results) == 3

        # Results should be ordered by cosine distance (smallest first)
        # medium_tool ([0.1] vs [1.0]) has smaller distance than close_tool ([0.9] vs [1.0])
        assert results[0].tool.details.name == "medium_tool"  # Smallest distance
        assert results[1].tool.details.name == "close_tool"  # Second smallest
        assert results[2].tool.details.name == "far_tool"  # Largest distance

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_zero_limit(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with limit of 0."""
        sample_tool_details["server_id"] = sample_server.id

        # Create a tool
        await tool_ops.create_tool(**sample_tool_details)

        # Create query embedding
        query_embedding = np.random.random(384).astype(np.float32)

        # Search with limit of 0
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=0,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return empty list
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_find_similar_tools_handles_large_embeddings(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with standard embedding dimensions."""
        sample_tool_details["server_id"] = sample_server.id

        # Create tool with standard embedding (384 dimensions - BGE small model size)
        large_embedding = np.random.random(384).astype(np.float32)
        sample_tool_details["details_embedding"] = large_embedding
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        # Create query embedding with same dimension
        query_embedding = np.random.random(384).astype(np.float32)

        # Sync vector tables after creating tools
        await tool_ops.sync_tool_vectors()

        # Search for similar tools
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return the tool
        assert len(results) == 1
        assert results[0].tool.id == created_tool.id

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_empty_server_list(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with empty server list behaves like no filtering."""
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        query_embedding = np.random.random(384).astype(np.float32)
        await tool_ops.sync_tool_vectors()

        # Search with empty server list
        results_empty_list = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            server_ids=[],
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Search with no server filtering
        results_no_filter = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Both should return the same results
        assert len(results_empty_list) == len(results_no_filter) == 1
        assert results_empty_list[0].tool.id == results_no_filter[0].tool.id == created_tool.id

    @pytest.mark.asyncio
    async def test_find_similar_tools_with_none_server_list(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with None server list behaves like no filtering."""
        sample_tool_details["server_id"] = sample_server.id
        created_tool = await tool_ops.create_tool(**sample_tool_details)

        query_embedding = np.random.random(384).astype(np.float32)
        await tool_ops.sync_tool_vectors()

        # Search with None server list (explicit)
        results_none = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            server_ids=None,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Search with no server filtering (default)
        results_default = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Both should return the same results
        assert len(results_none) == len(results_default) == 1
        assert results_none[0].tool.id == results_default[0].tool.id == created_tool.id

    @pytest.mark.asyncio
    async def test_find_similar_tools_filter_by_single_server(
        self, tool_ops, mcp_ops, sample_tool_details
    ):
        """Test similarity search filtered by single specific server."""
        # Create two servers
        server1 = await mcp_ops.create_server(
            name="workload-server1",
            url="http://localhost:8080",
            workload_identifier="test/server1",
            remote=False,
            transport="sse",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload-server2",
            url="http://localhost:8081",
            workload_identifier="test/server2",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Create tools for each server
        tool1_details = sample_tool_details.copy()
        tool1_details["server_id"] = server1.id
        tool1_details["details"] = McpTool(
            name="server1_tool",
            description="Tool from server 1",
            inputSchema={"type": "object", "properties": {}},
        )
        tool1 = await tool_ops.create_tool(**tool1_details)

        tool2_details = sample_tool_details.copy()
        tool2_details["server_id"] = server2.id
        tool2_details["details"] = McpTool(
            name="server2_tool",
            description="Tool from server 2",
            inputSchema={"type": "object", "properties": {}},
        )
        tool2 = await tool_ops.create_tool(**tool2_details)

        query_embedding = np.random.random(384).astype(np.float32)
        await tool_ops.sync_tool_vectors()

        # Search only in server1
        results_server1 = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
            server_ids=[server1.id],
        )

        # Should only return tool from server1
        assert len(results_server1) == 1
        assert results_server1[0].tool.id == tool1.id
        assert results_server1[0].tool.details.name == "server1_tool"

        # Search only in server2
        results_server2 = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
            server_ids=[server2.id],
        )

        # Should only return tool from server2
        assert len(results_server2) == 1
        assert results_server2[0].tool.id == tool2.id
        assert results_server2[0].tool.details.name == "server2_tool"

    @pytest.mark.asyncio
    async def test_find_similar_tools_filter_by_multiple_servers(
        self, tool_ops, mcp_ops, sample_tool_details
    ):
        """Test similarity search filtered by multiple specific servers."""
        # Create three servers
        server1 = await mcp_ops.create_server(
            name="workload-server1",
            url="http://localhost:8080",
            workload_identifier="test/server1",
            remote=False,
            transport="sse",
            status="running",
            description="Server 1",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server2 = await mcp_ops.create_server(
            name="workload-server2",
            url="http://localhost:8081",
            workload_identifier="test/server2",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 2",
            server_embedding=np.random.rand(384).astype(np.float32),
        )
        server3 = await mcp_ops.create_server(
            name="workload-server3",
            url="http://localhost:8082",
            workload_identifier="test/server3",
            remote=False,
            transport="streamable-http",
            status="running",
            description="Server 3",
            server_embedding=np.random.rand(384).astype(np.float32),
        )

        # Create tools for each server
        tool1_details = sample_tool_details.copy()
        tool1_details["server_id"] = server1.id
        tool1_details["details"] = McpTool(
            name="server1_tool",
            description="Tool from server 1",
            inputSchema={"type": "object", "properties": {}},
        )
        tool1 = await tool_ops.create_tool(**tool1_details)

        tool2_details = sample_tool_details.copy()
        tool2_details["server_id"] = server2.id
        tool2_details["details"] = McpTool(
            name="server2_tool",
            description="Tool from server 2",
            inputSchema={"type": "object", "properties": {}},
        )
        tool2 = await tool_ops.create_tool(**tool2_details)

        tool3_details = sample_tool_details.copy()
        tool3_details["server_id"] = server3.id
        tool3_details["details"] = McpTool(
            name="server3_tool",
            description="Tool from server 3",
            inputSchema={"type": "object", "properties": {}},
        )
        tool3 = await tool_ops.create_tool(**tool3_details)

        query_embedding = np.random.random(384).astype(np.float32)
        await tool_ops.sync_tool_vectors()

        # Search in server1 and server2 only
        results_filtered = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            distance_threshold=2.1,
            server_statuses=[McpStatus.RUNNING],
            server_ids=[server1.id, server2.id],
        )

        # Should return tools from server1 and server2, but not server3
        assert len(results_filtered) == 2
        result_ids = {tool.tool.id for tool in results_filtered}
        assert result_ids == {tool1.id, tool2.id}

        # Verify server3 tool is not included
        assert tool3.id not in result_ids

    @pytest.mark.asyncio
    async def test_find_similar_tools_filter_by_nonexistent_server(
        self, tool_ops, sample_server, sample_tool_details
    ):
        """Test similarity search with nonexistent server ID returns empty results."""
        sample_tool_details["server_id"] = sample_server.id
        await tool_ops.create_tool(**sample_tool_details)

        query_embedding = np.random.random(384).astype(np.float32)
        await tool_ops.sync_tool_vectors()

        # Search with nonexistent server ID
        fake_server_id = str(uuid.uuid4())
        results = await tool_ops.find_similar_tools(
            query_embedding,
            limit=5,
            server_ids=[fake_server_id],
            distance_threshold=1.0,
            server_statuses=[McpStatus.RUNNING],
        )

        # Should return empty list
        assert len(results) == 0
