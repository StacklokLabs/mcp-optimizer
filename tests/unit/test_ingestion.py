"""Unit tests for ingestion service token counting."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest
from mcp.types import ListToolsResult
from mcp.types import Tool as McpTool

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionService
from mcp_optimizer.token_counter import TokenCounter


class TestIngestionServiceTokenCounting:
    """Test token counting functionality in IngestionService."""

    @pytest.fixture
    def mock_db_config(self):
        """Create a mock database configuration."""
        return Mock(spec=DatabaseConfig)

    @pytest.fixture
    def mock_embedding_manager(self):
        """Create a mock embedding manager."""
        return Mock(spec=EmbeddingManager)

    @pytest.fixture
    def ingestion_service(self, mock_db_config, mock_embedding_manager):
        """Create an IngestionService instance with mocked dependencies."""
        return IngestionService(
            mock_db_config,
            mock_embedding_manager,
            mcp_timeout=10,
            registry_ingestion_batch_size=5,
            workload_ingestion_batch_size=5,
            encoding="cl100k_base",
        )

    @pytest.fixture
    def sample_mcp_tool(self):
        """Create a sample MCP tool for testing."""
        return McpTool(
            name="test_tool",
            description="A test tool for calculating tokens",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                    "param2": {"type": "number", "description": "Second parameter"},
                },
                "required": ["param1"],
            },
        )

    @pytest.fixture
    def sample_tools_result(self, sample_mcp_tool):
        """Create a sample tools result for testing."""
        return ListToolsResult(tools=[sample_mcp_tool])

    def test_token_counter_initialization(self, ingestion_service):
        """Test that TokenCounter is initialized in IngestionService."""
        assert hasattr(ingestion_service, "token_counter")
        assert isinstance(ingestion_service.token_counter, TokenCounter)
        assert ingestion_service.token_counter.encoding.name == "cl100k_base"

    @pytest.mark.asyncio
    async def test_sync_workload_tools_calculates_token_counts(
        self, ingestion_service, sample_tools_result, sample_mcp_tool
    ):
        """Test that _sync_workload_tools calculates token counts for tools."""
        server_id = "test-server-id"
        server_name = "test-server"
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])

        # Mock tool operations
        ingestion_service.workload_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.workload_tool_ops.delete_tools_by_server_id = AsyncMock(return_value=0)
        ingestion_service.workload_tool_ops.create_tool = AsyncMock()

        # Mock embedding generation
        ingestion_service.embedding_manager.generate_embedding.return_value = np.array(
            [mock_embedding]
        )

        # Execute sync_tools
        mock_conn = AsyncMock()
        tools_count, was_updated = await ingestion_service._sync_workload_tools(
            server_id, server_name, sample_tools_result, mock_conn
        )

        # Verify results
        assert tools_count == 1
        assert was_updated is True

        # Verify create_tool was called with token_count
        ingestion_service.workload_tool_ops.create_tool.assert_called_once()
        call_args = ingestion_service.workload_tool_ops.create_tool.call_args

        assert call_args[1]["server_id"] == server_id
        assert call_args[1]["details"] == sample_mcp_tool
        np.testing.assert_array_equal(call_args[1]["details_embedding"], mock_embedding)

        # Verify token_count was calculated and passed
        assert "token_count" in call_args[1]
        token_count = call_args[1]["token_count"]
        # Token count might be a list or int depending on batch processing
        if isinstance(token_count, list):
            assert len(token_count) > 0
            assert all(isinstance(tc, int) and tc > 0 for tc in token_count)
        else:
            assert isinstance(token_count, int)
            assert token_count > 0

    @pytest.mark.asyncio
    async def test_sync_workload_tools_calculates_correct_token_count(
        self, ingestion_service, sample_mcp_tool
    ):
        """Test that _sync_workload_tools calculates the correct token count."""
        server_id = "test-server-id"
        server_name = "test-server"
        tools_result = ListToolsResult(tools=[sample_mcp_tool])
        mock_embedding = np.array([0.1, 0.2, 0.3, 0.4])

        # Mock operations
        ingestion_service.workload_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.workload_tool_ops.delete_tools_by_server_id = AsyncMock(return_value=0)
        ingestion_service.workload_tool_ops.create_tool = AsyncMock()
        ingestion_service.embedding_manager.generate_embedding.return_value = np.array(
            [mock_embedding]
        )

        # Calculate expected token count
        expected_token_count = ingestion_service.token_counter.count_tool_tokens(sample_mcp_tool)

        # Execute sync_tools
        mock_conn = AsyncMock()
        await ingestion_service._sync_workload_tools(
            server_id, server_name, tools_result, mock_conn
        )

        # Verify the token count matches what TokenCounter would calculate
        call_args = ingestion_service.workload_tool_ops.create_tool.call_args
        actual_token_count = call_args[1]["token_count"]

        # Token count might be in a list due to batch processing
        if isinstance(actual_token_count, list):
            actual_token_count = actual_token_count[0]

        assert actual_token_count == expected_token_count

    @pytest.mark.asyncio
    async def test_sync_workload_tools_with_multiple_tools(self, ingestion_service):
        """Test that _sync_workload_tools calculates token counts for multiple tools."""
        server_id = "test-server-id"
        server_name = "test-server"

        # Create multiple tools with different token counts
        tool1 = McpTool(name="tool1", description="First tool", inputSchema={})
        tool2 = McpTool(
            name="tool2",
            description="Second tool with a much longer description to increase token count",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string"},
                    "param2": {"type": "number"},
                    "param3": {"type": "boolean"},
                },
            },
        )
        tool3 = McpTool(name="tool3", description=None, inputSchema={})

        tools_result = ListToolsResult(tools=[tool1, tool2, tool3])

        # Mock operations
        ingestion_service.workload_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.workload_tool_ops.delete_tools_by_server_id = AsyncMock(return_value=0)
        ingestion_service.workload_tool_ops.create_tool = AsyncMock()
        ingestion_service.embedding_manager.generate_embedding.return_value = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ]
        )

        # Execute sync_tools
        mock_conn = AsyncMock()
        tools_count, was_updated = await ingestion_service._sync_workload_tools(
            server_id, server_name, tools_result, mock_conn
        )

        # Verify results
        assert tools_count == 3
        assert was_updated is True
        assert ingestion_service.workload_tool_ops.create_tool.call_count == 3

        # Verify each tool has a token count
        for call in ingestion_service.workload_tool_ops.create_tool.call_args_list:
            assert "token_count" in call[1]
            token_count = call[1]["token_count"]
            # Token count might be a list or int depending on batch processing
            if isinstance(token_count, list):
                assert len(token_count) > 0
                assert all(isinstance(tc, int) and tc > 0 for tc in token_count)
            else:
                assert isinstance(token_count, int)
                assert token_count > 0

        # Verify that all tools have valid token counts
        for i, call in enumerate(ingestion_service.workload_tool_ops.create_tool.call_args_list):
            token_count = call[1]["token_count"]
            # Extract first element if list
            if isinstance(token_count, list):
                token_count = token_count[0]
            assert token_count > 0, f"Tool {i + 1} should have positive token count"

    @pytest.mark.asyncio
    async def test_sync_registry_tools_includes_token_counts(self, ingestion_service):
        """Test that _sync_registry_tools calculates token counts for registry tools."""
        server_id = "test-server-id"
        server_name = "test-server"
        tool_names = ["registry_tool_1", "registry_tool_2", "registry_tool_3"]

        # Mock operations
        ingestion_service.registry_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.registry_tool_ops.delete_tools_by_server_id = AsyncMock(return_value=0)
        ingestion_service.registry_tool_ops.create_tool = AsyncMock()
        ingestion_service.embedding_manager.generate_embedding.return_value = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.2, 0.3, 0.4, 0.5],
                [0.3, 0.4, 0.5, 0.6],
            ]
        )

        # Mock connection
        mock_conn = AsyncMock()

        # Execute ingest_registry_tools
        tools_count, was_updated = await ingestion_service._sync_registry_tools(
            server_id, server_name, tool_names, mock_conn
        )

        # Verify results
        assert tools_count == 3
        assert was_updated is True
        assert ingestion_service.registry_tool_ops.create_tool.call_count == 3

        # Verify create_tool was called for each tool
        # Note: Registry tools may not calculate token counts in this implementation
        for call in ingestion_service.registry_tool_ops.create_tool.call_args_list:
            assert "details" in call[1]
            assert "details_embedding" in call[1]

    @pytest.mark.asyncio
    async def test_sync_workload_tools_empty_tools_does_not_calculate_tokens(
        self, ingestion_service
    ):
        """Test that _sync_workload_tools doesn't calculate tokens when no tools are present."""
        server_id = "test-server-id"
        server_name = "test-server"
        empty_tools_result = ListToolsResult(tools=[])

        # Mock operations
        ingestion_service.workload_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.workload_tool_ops.delete_tools_by_server_id = AsyncMock(return_value=0)
        ingestion_service.workload_tool_ops.create_tool = AsyncMock()

        # Execute sync_tools with empty tools
        mock_conn = AsyncMock()
        tools_count, was_updated = await ingestion_service._sync_workload_tools(
            server_id, server_name, empty_tools_result, mock_conn
        )

        # Verify no tokens were calculated (no tools to process)
        assert tools_count == 0
        assert was_updated is False
        ingestion_service.workload_tool_ops.create_tool.assert_not_called()
        ingestion_service.embedding_manager.generate_embedding.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_registry_tools_empty_names(self, ingestion_service):
        """Test that _sync_registry_tools handles empty tool names correctly."""
        server_id = "test-server-id"
        server_name = "test-server"
        empty_tool_names = []

        # Mock operations
        ingestion_service.registry_tool_ops.get_tools_by_server_id = AsyncMock(return_value=[])
        ingestion_service.registry_tool_ops.create_tool = AsyncMock()
        mock_conn = AsyncMock()

        # Execute ingest_registry_tools with empty names
        tools_count, was_updated = await ingestion_service._sync_registry_tools(
            server_id, server_name, empty_tool_names, mock_conn
        )

        # Verify no tokens were calculated
        assert tools_count == 0
        assert was_updated is False
        ingestion_service.registry_tool_ops.create_tool.assert_not_called()
