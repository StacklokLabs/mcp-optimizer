"""
Tests for the MCP client.
"""

from unittest.mock import AsyncMock, patch

import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from mcp_optimizer.mcp_client import MCPServerClient, WorkloadConnectionError
from mcp_optimizer.toolhive.api_models.core import Workload


@pytest.fixture
def mock_workload():
    """Create a mock workload for testing."""
    return Workload(
        name="test-server",
        package="test/mcp-server:latest",
        url="http://127.0.0.1:8080/sse#test-server",
        port=8080,
        tool_type="mcp",
        transport_type="sse",
        status="running",
        status_context="Up 1 hour",
    )


@pytest.mark.asyncio
async def test_mcp_server_client_no_url():
    """Test MCP server client with no URL."""
    workload = Workload(
        name="test-server",
        status="running",
        tool_type="mcp",
    )

    client = MCPServerClient(workload, timeout=10)
    with pytest.raises(WorkloadConnectionError, match="Workload has no URL"):
        await client.list_tools()


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_no_url():
    """Test that MCPServerClient.call_tool raises error when workload has no URL."""
    workload = Workload(
        name="test-server",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    with pytest.raises(WorkloadConnectionError, match="Workload has no URL"):
        await client.call_tool("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_streamable():
    """Test MCPServerClient.call_tool with streamable-http transport."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    # Mock the MCP client session and result
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock(text="Tool result")]

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = mock_result

    with (
        patch("mcp_optimizer.mcp_client.streamablehttp_client") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_session

        result = await client.call_tool("test_tool", {"param": "value"})

        assert result == mock_result
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_sse():
    """Test MCPServerClient.call_tool with SSE transport."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/sse/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    # Mock the MCP client session and result
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock(text="Tool result")]

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = mock_result

    with (
        patch("mcp_optimizer.mcp_client.sse_client") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_session

        result = await client.call_tool("test_tool", {"param": "value"})

        assert result == mock_result
        mock_session.initialize.assert_called_once()
        mock_session.call_tool.assert_called_once_with("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_unsupported_transport():
    """Test that MCPServerClient.call_tool raises error for unsupported transport."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/unknown/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    with pytest.raises(ValueError, match="Unsupported ToolHive URL"):
        await client.call_tool("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_handles_exception_group():
    """Test that MCPServerClient properly handles ExceptionGroup from TaskGroups."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    # Create an ExceptionGroup with a nested McpError (simulating Python 3.13+ TaskGroup behavior)
    error_data = ErrorData(code=1, message="Session terminated")
    mcp_error = McpError(error_data)
    exception_group = ExceptionGroup("unhandled errors in a TaskGroup", [mcp_error])

    mock_session = AsyncMock()
    mock_session.initialize.side_effect = exception_group

    with (
        patch("mcp_optimizer.mcp_client.streamablehttp_client") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Should convert ExceptionGroup to WorkloadConnectionError
        with pytest.raises(WorkloadConnectionError, match="MCP protocol error"):
            await client.list_tools()


@pytest.mark.asyncio
async def test_mcp_server_client_handles_mcp_error():
    """Test that MCPServerClient properly handles direct McpError."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    # Create a direct McpError
    error_data = ErrorData(code=1, message="Connection refused")
    mcp_error = McpError(error_data)

    mock_session = AsyncMock()
    mock_session.initialize.side_effect = mcp_error

    with (
        patch("mcp_optimizer.mcp_client.streamablehttp_client") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Should convert McpError to WorkloadConnectionError
        with pytest.raises(WorkloadConnectionError, match="MCP protocol error"):
            await client.list_tools()


def test_extract_error_from_exception_group():
    """Test the _extract_error_from_exception_group method."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
        tool_type="mcp",
    )
    client = MCPServerClient(workload, timeout=10)

    # Test with McpError
    error_data = ErrorData(code=1, message="Test error")
    mcp_error = McpError(error_data)
    eg = ExceptionGroup("test", [mcp_error])
    result = client._extract_error_from_exception_group(eg)
    assert "MCP protocol error" in result
    assert "Test error" in result

    # Test with nested ExceptionGroup
    nested_eg = ExceptionGroup("outer", [ExceptionGroup("inner", [mcp_error])])
    result = client._extract_error_from_exception_group(nested_eg)
    assert "MCP protocol error" in result

    # Test with non-McpError
    value_error = ValueError("Some error")
    eg = ExceptionGroup("test", [value_error])
    result = client._extract_error_from_exception_group(eg)
    assert "ValueError" in result
    assert "Some error" in result
