"""Unit tests for ingestion service."""

import json
from unittest.mock import Mock

import numpy as np
import pytest
from mcp.types import Tool as McpTool

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import McpStatus
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionError, IngestionService


class TestIngestionServiceMapping:
    """Test cases for IngestionService mapping methods."""

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

    def test_map_workload_status_running(self, ingestion_service):
        """Test mapping running status to McpStatus.RUNNING."""
        result = ingestion_service._map_workload_status("running")
        assert result == McpStatus.RUNNING

    def test_map_workload_status_stopped(self, ingestion_service):
        """Test mapping stopped status to McpStatus.STOPPED."""
        result = ingestion_service._map_workload_status("stopped")
        assert result == McpStatus.STOPPED

    def test_map_workload_status_other(self, ingestion_service):
        """Test mapping other statuses to McpStatus.STOPPED."""
        for status in ["pending", "failed", "unknown"]:
            result = ingestion_service._map_workload_status(status)
            assert result == McpStatus.STOPPED

    def test_map_workload_status_none_raises_error(self, ingestion_service):
        """Test that None status raises IngestionError."""
        with pytest.raises(IngestionError, match="Workload status cannot be None"):
            ingestion_service._map_workload_status(None)

    def test_create_tool_text_to_embed_full_tool(self, ingestion_service):
        """Test creating text representation for tool with all fields."""
        tool = McpTool(
            name="test_tool",
            description="A test tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "First parameter"},
                },
            },
        )
        text = ingestion_service._create_tool_text_to_embed(tool, "test_server")
        expected = "Server: test_server | Tool: test_tool | Description: A test tool"
        assert text == expected

    def test_create_tool_text_to_embed_minimal_tool(self, ingestion_service):
        """Test creating text representation for tool with minimal fields."""
        tool = McpTool(name="minimal_tool", inputSchema={})
        text = ingestion_service._create_tool_text_to_embed(tool, "test_server")
        assert text == "Server: test_server | Tool: minimal_tool"

    def test_create_tool_text_to_embed_no_name(self, ingestion_service):
        """Test creating text representation for tool with no name but with description."""
        tool = McpTool(name="", description="Tool without name", inputSchema={})
        text = ingestion_service._create_tool_text_to_embed(tool, "test_server")
        assert text == "Server: test_server | Description: Tool without name"

    def test_create_tool_text_to_embed_fallback_to_json(self, ingestion_service):
        """Test creating text representation includes server name even with minimal tool info."""
        tool = McpTool(name="", inputSchema={})
        text = ingestion_service._create_tool_text_to_embed(tool, "test_server")
        assert text == f"Server: test_server | {tool.model_dump_json()}"
        # Assert that is valid JSON
        assert json.loads(text.split(" | ")[1])


class TestChangeDetection:
    """Test cases for change detection functionality."""

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

    def test_compare_embeddings_both_none(self, ingestion_service):
        """Test comparing embeddings when both are None."""
        result = ingestion_service._compare_embeddings(None, None)
        assert result is True

    def test_compare_embeddings_one_none(self, ingestion_service):
        """Test comparing embeddings when one is None."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert ingestion_service._compare_embeddings(embedding, None) is False
        assert ingestion_service._compare_embeddings(None, embedding) is False

    def test_compare_embeddings_identical(self, ingestion_service):
        """Test comparing identical embeddings."""
        embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert ingestion_service._compare_embeddings(embedding1, embedding2) is True

    def test_compare_embeddings_different(self, ingestion_service):
        """Test comparing different embeddings."""
        embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding2 = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        assert ingestion_service._compare_embeddings(embedding1, embedding2) is False

    def test_compare_embeddings_within_tolerance(self, ingestion_service):
        """Test comparing embeddings within floating point tolerance."""
        embedding1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        embedding2 = np.array([1.0, 2.0, 3.0 + 1e-9], dtype=np.float32)
        assert ingestion_service._compare_embeddings(embedding1, embedding2) is True

    def test_compare_tools_identical(self, ingestion_service):
        """Test comparing identical tools."""
        tool1 = McpTool(
            name="test_tool",
            description="Test description",
            inputSchema={"type": "object", "properties": {}},
        )
        tool2 = McpTool(
            name="test_tool",
            description="Test description",
            inputSchema={"type": "object", "properties": {}},
        )
        assert ingestion_service._compare_tools(tool1, tool2) is True

    def test_compare_tools_different_name(self, ingestion_service):
        """Test comparing tools with different names."""
        tool1 = McpTool(name="tool1", description="Test", inputSchema={})
        tool2 = McpTool(name="tool2", description="Test", inputSchema={})
        assert ingestion_service._compare_tools(tool1, tool2) is False

    def test_compare_tools_different_description(self, ingestion_service):
        """Test comparing tools with different descriptions."""
        tool1 = McpTool(name="test", description="Description 1", inputSchema={})
        tool2 = McpTool(name="test", description="Description 2", inputSchema={})
        assert ingestion_service._compare_tools(tool1, tool2) is False

    def test_compare_tools_different_schema(self, ingestion_service):
        """Test comparing tools with different input schemas."""
        tool1 = McpTool(name="test", description="Test", inputSchema={"type": "object"})
        tool2 = McpTool(name="test", description="Test", inputSchema={"type": "string"})
        assert ingestion_service._compare_tools(tool1, tool2) is False
