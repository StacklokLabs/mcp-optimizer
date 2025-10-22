import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from mcp.types import Tool as McpTool
from starlette.testclient import TestClient

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import WorkloadTool, WorkloadToolWithMetadata
from mcp_optimizer.server import call_tool, find_tool, mcp, starlette_app


@pytest.fixture
def client():
    """Create a test client for the MCP server."""
    return TestClient(starlette_app)


def test_server_startup():
    """Test that the MCP server can be instantiated successfully."""
    assert mcp is not None
    assert mcp.name == "mcp-optimizer"
    assert mcp.settings.host == "0.0.0.0"


def test_health_check_endpoint(client):
    """Test that the health check endpoint returns OK status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_health_check_endpoint(client):
    """Test that the root endpoint returns OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


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


class TestFindToolFunction:
    """Test the find_tool server function."""

    @pytest.mark.asyncio
    async def test_find_tool_empty_database(self):
        """Test find_tool with empty database returns empty result."""
        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,
            patch("mcp_optimizer.server._config") as mock_config,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops
            mock_tool_ops.find_similar_tools = AsyncMock(return_value=[])

            # Mock embedding manager
            mock_embedding_manager.generate_embedding.return_value = [np.random.random(384)]

            # Mock config
            mock_config.max_tools_to_return = 8
            mock_config.tool_distance_threshold = 1.0
            mock_config.hybrid_search_semantic_ratio = 0.5

            # Mock global tool ops
            mock_tool_ops_global.find_similar_tools = AsyncMock(return_value=[])
            mock_tool_ops_global.sum_token_counts_for_running_servers = AsyncMock(return_value=0)

            result = await find_tool("test query", "test query")

            assert isinstance(result, dict)
            assert "tools" in result
            assert "token_metrics" in result
            assert result["tools"] == []
            assert result["token_metrics"]["baseline_tokens"] == 0
            assert result["token_metrics"]["returned_tokens"] == 0
            assert result["token_metrics"]["tokens_saved"] == 0
            assert result["token_metrics"]["savings_percentage"] == 0.0

    @pytest.mark.asyncio
    async def test_find_tool_with_results(self):
        """Test find_tool with database results."""
        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,
            patch("mcp_optimizer.server._config") as mock_config,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
        ):
            # Mock tool ops with a sample tool
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            # Create a mock tool
            mock_tool_details = McpTool(
                name="test_tool",
                description="A test tool",
                inputSchema={"type": "object", "properties": {"param": {"type": "string"}}},
            )

            # Create a ToolWithMetadata mock
            mock_tool_with_metadata = WorkloadToolWithMetadata(
                tool=WorkloadTool(
                    id="test-tool-id",
                    mcpserver_id="test-server-id",
                    details=mock_tool_details,
                    token_count=42,
                    last_updated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                ),
                server_name="test_server",
                server_description="http://test.com",
                distance=0.1,
            )

            mock_tool_ops.find_similar_tools = AsyncMock(return_value=[mock_tool_with_metadata])

            # Mock embedding manager
            mock_embedding_manager.generate_embedding.return_value = [np.random.random(384)]

            # Mock config
            mock_config.max_tools_to_return = 8
            mock_config.tool_distance_threshold = 1.0
            mock_config.hybrid_search_semantic_ratio = 0.5

            # Mock global tool ops
            mock_tool_ops_global.find_similar_tools = AsyncMock(
                return_value=[mock_tool_with_metadata]
            )
            mock_tool_ops_global.sum_token_counts_for_running_servers = AsyncMock(return_value=100)

            result = await find_tool("weather query", "weather")

            assert isinstance(result, dict)
            assert "tools" in result
            assert "token_metrics" in result
            assert len(result["tools"]) == 1
            assert result["token_metrics"]["baseline_tokens"] == 100
            assert result["token_metrics"]["returned_tokens"] == 42
            assert result["token_metrics"]["tokens_saved"] == 58
            assert result["token_metrics"]["savings_percentage"] == pytest.approx(58.0)

            tool = result["tools"][0]
            assert tool.name == "test_tool"
            assert tool.description == "A test tool"
            assert hasattr(tool, "mcp_server_name")
            assert tool.mcp_server_name == "test_server"

    @pytest.mark.asyncio
    async def test_find_tool_handles_exceptions(self):
        """Test find_tool handles exceptions gracefully."""
        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,  # noqa: F841
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,
            patch("mcp_optimizer.server._config") as mock_config,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
        ):
            # Mock config
            mock_config.max_tools_to_return = 8
            mock_config.tool_distance_threshold = 1.0
            mock_config.hybrid_search_semantic_ratio = 0.5

            # Mock global tool ops to throw an error
            mock_tool_ops_global.find_similar_tools = AsyncMock(
                side_effect=Exception("Database error")
            )

            # Mock embedding manager
            mock_embedding_manager.generate_embedding.return_value = [np.random.random(384)]

            with pytest.raises(Exception) as exc_info:
                await find_tool("test query", "test query")

            assert "Database error" in str(exc_info.value)


class TestCallToolFunction:
    """Test the call_tool server function."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """Test successful tool calling."""
        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Mock successful MCP result
            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text="Tool executed successfully")], isError=False
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = 4000

            # Mock global ops - these need to be AsyncMock since they're awaited
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is False
            assert len(result.content) == 1
            assert result.content[0].text == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_call_tool_mcp_error(self):
        """Test tool calling with MCP error response."""
        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client with error result
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text="Tool execution failed")], isError=True
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = 4000

            # Mock global ops - these need to be AsyncMock since they're awaited
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is True
            assert len(result.content) == 1
            assert result.content[0].text == "Tool execution failed"

    @pytest.mark.asyncio
    async def test_call_tool_handles_exceptions(self):
        """Test call_tool handles exceptions gracefully."""
        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,  # noqa: F841
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,  # noqa: F841
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,  # noqa: F841
        ):
            # Mock the global tool ops to raise an exception when called
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(
                side_effect=Exception("Database error")
            )

            with pytest.raises(Exception) as exc_info:
                await call_tool("test-server", "test_tool", {"param": "value"})

            assert "Database error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_call_tool_with_large_response_truncation(self):
        """Test that large tool responses are truncated."""
        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client with very large response
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Create a large response that exceeds token limit
            large_text = "A" * 20000  # ~5000 tokens
            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text=large_text)], isError=False
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config with small token limit
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = 100

            # Mock global ops
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is False

            # Response should be truncated
            # Should only have the truncation notice, large content omitted
            assert len(result.content) == 1
            assert "truncated" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_large_json_list(self):
        """Test that large JSON list responses are omitted."""
        import json

        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client with JSON list response
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Create a large JSON list
            data = [{"id": i, "value": f"item_{i}" * 10} for i in range(200)]
            json_text = json.dumps(data, indent=2)

            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text=json_text)], isError=False
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config with moderate token limit
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = 500

            # Mock global ops
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is False

            # Should only have truncation notice, large JSON omitted
            assert len(result.content) == 1
            assert "truncated" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_response_under_limit_not_truncated(self):
        """Test that responses under token limit are not truncated."""
        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client with small response
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            small_text = "Short response"
            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text=small_text)], isError=False
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config with large token limit
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = 4000

            # Mock global ops
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is False

            # Response should not be modified - should be exactly as returned
            assert len(result.content) == 1
            assert result.content[0].text == small_text
            assert "truncated" not in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_none_limit_no_truncation(self):
        """Test that responses are not truncated when max_tool_response_tokens is None."""
        from mcp.types import CallToolResult, TextContent

        with (
            patch("mcp_optimizer.server.WorkloadToolOps") as mock_tool_ops_class,
            patch("mcp_optimizer.server.WorkloadServerOps") as mock_server_ops_class,
            patch("mcp_optimizer.server.MCPServerClient") as mock_client_class,
            patch("mcp_optimizer.server.workload_tool_ops") as mock_tool_ops_global,
            patch("mcp_optimizer.server.workload_server_ops") as mock_server_ops_global,
            patch("mcp_optimizer.server.embedding_manager") as mock_embedding_manager,  # noqa: F841
            patch("mcp_optimizer.server._config") as mock_config,
        ):
            # Mock tool ops
            mock_tool_ops = AsyncMock()
            mock_tool_ops_class.return_value = mock_tool_ops

            mock_tool = AsyncMock()
            mock_tool.id = "test-tool-id"
            mock_tool.mcpserver_id = "test-server-id"
            mock_tool.details.name = "test_tool"
            mock_tool_ops.get_tool_by_server_and_name.return_value = mock_tool

            # Mock server ops
            mock_server_ops = AsyncMock()
            mock_server_ops_class.return_value = mock_server_ops

            mock_server = AsyncMock()
            mock_server.id = "test-server-id"
            mock_server.name = "test-server"
            mock_server.url = "http://localhost:8080/mcp"
            mock_server.transport.value = "sse"
            mock_server_ops.get_server_by_name.return_value = mock_server

            # Mock MCP client with very large response
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client

            # Create a large response
            large_text = "A" * 50000  # ~12500 tokens
            mock_mcp_result = CallToolResult(
                content=[TextContent(type="text", text=large_text)], isError=False
            )
            mock_client.call_tool.return_value = mock_mcp_result

            # Mock config with None token limit (disabled)
            mock_config.mcp_timeout = 10
            mock_config.max_tool_response_tokens = None

            # Mock global ops
            mock_tool_ops_global.get_tool_by_server_and_name = AsyncMock(return_value=mock_tool)
            mock_server_ops_global.get_server_by_name = AsyncMock(return_value=mock_server)

            result = await call_tool("test-server", "test_tool", {"param": "value"})

            assert isinstance(result, CallToolResult)
            assert result.isError is False

            # Response should NOT be truncated even though it's large
            assert len(result.content) == 1
            assert result.content[0].text == large_text
            assert "truncated" not in result.content[0].text.lower()
