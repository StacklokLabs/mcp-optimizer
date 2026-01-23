"""
Tests for the MCP client.
"""

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData

from mcp_optimizer.mcp_client import (
    MCPServerClient,
    WorkloadConnectionError,
    _create_tolerant_httpx_client,
    _TolerantStream,
    determine_transport_type,
)
from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.enums import ToolHiveTransportMode


@pytest.fixture
def mock_workload():
    """Create a mock workload for testing."""
    return Workload(
        name="test-server",
        package="test/mcp-server:latest",
        url="http://127.0.0.1:8080/sse#test-server",
        port=8080,
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
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")
    with pytest.raises(WorkloadConnectionError, match="Workload has no URL"):
        await client.list_tools()


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_no_url():
    """Test that MCPServerClient.call_tool raises error when workload has no URL."""
    workload = Workload(
        name="test-server",
        status="running",
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with pytest.raises(WorkloadConnectionError, match="Workload has no URL"):
        await client.call_tool("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_call_tool_streamable():
    """Test MCPServerClient.call_tool with streamable-http transport."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    # Mock the MCP client session and result
    mock_result = AsyncMock()
    mock_result.content = [AsyncMock(text="Tool result")]

    mock_session = AsyncMock()
    mock_session.call_tool.return_value = mock_result

    with (
        patch("mcp_optimizer.mcp_client.streamable_http_client") as mock_client,
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
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

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
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with pytest.raises(ValueError, match="Unsupported ToolHive URL"):
        await client.call_tool("test_tool", {"param": "value"})


@pytest.mark.asyncio
async def test_mcp_server_client_handles_exception_group():
    """Test that MCPServerClient properly handles ExceptionGroup from TaskGroups."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test",
        status="running",
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    # Create an ExceptionGroup with a nested McpError (simulating Python 3.13+ TaskGroup behavior)
    error_data = ErrorData(code=1, message="Session terminated")
    mcp_error = McpError(error_data)
    exception_group = ExceptionGroup("unhandled errors in a TaskGroup", [mcp_error])

    mock_session = AsyncMock()
    mock_session.initialize.side_effect = exception_group

    with (
        patch("mcp_optimizer.mcp_client.streamable_http_client") as mock_client,
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
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    # Create a direct McpError
    error_data = ErrorData(code=1, message="Connection refused")
    mcp_error = McpError(error_data)

    mock_session = AsyncMock()
    mock_session.initialize.side_effect = mcp_error

    with (
        patch("mcp_optimizer.mcp_client.streamable_http_client") as mock_client,
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
            )
    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

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


@pytest.fixture
def mock_mcp_session():
    """Create a mock MCP session for testing."""
    mock_session = AsyncMock()
    mock_list_result = AsyncMock()
    mock_list_result.tools = []
    mock_session.list_tools.return_value = mock_list_result

    mock_call_result = AsyncMock()
    mock_call_result.content = [AsyncMock(text="Tool result")]
    mock_session.call_tool.return_value = mock_call_result

    return mock_session


@pytest.mark.parametrize(
    "url,transport_type",
    [
        ("http://localhost:8080/sse/test-server", None),
        ("http://localhost:8080/mcp/test-server", "streamable-http"),
        ("http://localhost:8080/custom/endpoint", "sse"),
    ],
)
def test_workload_url_unchanged_after_init(url, transport_type):
    """Test that workload URL is not modified during MCPServerClient initialization."""
    workload = Workload(
        name="test-server",
        url=url,
        transport_type=transport_type,
        status="running",
            )

    # Create client
    _client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    # Verify URL is unchanged
    assert workload.url == url


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,client_mock_name,context_return",
    [
        (
            "http://localhost:8080/mcp/test-server",
            "streamable_http_client",
            (AsyncMock(), AsyncMock(), AsyncMock()),
        ),
        ("http://localhost:8080/sse/test-server", "sse_client", (AsyncMock(), AsyncMock())),
    ],
)
async def test_workload_url_unchanged_during_list_tools(
    url, client_mock_name, context_return, mock_mcp_session
):
    """Test that workload URL remains unchanged during list_tools for both transport types."""
    workload = Workload(
        name="test-server",
        url=url,
        status="running",
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with (
        patch(f"mcp_optimizer.mcp_client.{client_mock_name}") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_mcp_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = context_return
        mock_session_class.return_value.__aenter__.return_value = mock_mcp_session

        # Call list_tools
        await client.list_tools()

        # Verify URL is unchanged in workload
        assert workload.url == url

        # Verify the client was called with the original URL
        # SSE client uses keyword arguments including httpx_client_factory
        if client_mock_name == "sse_client":
            mock_client.assert_called_once()
            assert mock_client.call_args.kwargs["url"] == url
            assert "httpx_client_factory" in mock_client.call_args.kwargs
        else:
            mock_client.assert_called_once_with(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url,proxy_mode,client_mock_name,context_return",
    [
        (
            "http://localhost:8080/mcp/test-server",
            None,
            "streamable_http_client",
            (AsyncMock(), AsyncMock(), AsyncMock()),
        ),
        (
            "http://localhost:8080/custom/endpoint",
            "streamable-http",
            "streamable_http_client",
            (AsyncMock(), AsyncMock(), AsyncMock()),
        ),
    ],
)
async def test_workload_url_unchanged_during_call_tool(
    url, proxy_mode, client_mock_name, context_return, mock_mcp_session
):
    """Test that workload URL remains unchanged during call_tool in docker mode."""
    workload = Workload(
        name="test-server",
        url=url,
        proxy_mode=proxy_mode,
        status="running",
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with (
        patch(f"mcp_optimizer.mcp_client.{client_mock_name}") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_mcp_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = context_return
        mock_session_class.return_value.__aenter__.return_value = mock_mcp_session

        # Call tool
        await client.call_tool("test_tool", {"param": "value"})

        # Verify URL is unchanged in workload (we don't mutate the workload object)
        assert workload.url == url

        # Verify the client was called with the original URL (no normalization)
        mock_client.assert_called_once_with(url)


@pytest.mark.asyncio
async def test_workload_url_unchanged_multiple_operations(mock_mcp_session):
    """Test that workload URL remains unchanged across multiple operations."""
    original_url = "http://localhost:8080/mcp/test-server"
    workload = Workload(
        name="test-server",
        url=original_url,
        status="running",
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with (
        patch("mcp_optimizer.mcp_client.streamable_http_client") as mock_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_mcp_session
        ) as mock_session_class,
    ):
        # Mock the context manager
        mock_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_mcp_session

        # Perform multiple operations
        await client.list_tools()
        assert workload.url == original_url

        await client.call_tool("test_tool", {"param": "value"})
        assert workload.url == original_url

        await client.list_tools()
        assert workload.url == original_url

        # Verify the client was always called with the original URL
        assert mock_client.call_count == 3
        for call in mock_client.call_args_list:
            assert call[0][0] == original_url


# Unit tests for determine_transport_type function


def test_determine_transport_type_streamable_http():
    """Test determine_transport_type with transport_type set to 'streamable-http' in k8s mode."""
    workload = Workload(
        name="test-workload",
        transport_type="streamable-http",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_sse():
    """Test determine_transport_type with transport_type set to 'sse' in k8s mode."""
    workload = Workload(
        name="test-workload",
        transport_type="sse",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_streamable_http_value():
    """Test determine_transport_type with 'streamable-http' transport in k8s mode."""
    workload = Workload(
        name="test-workload",
        transport_type="streamable-http",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_sse_value():
    """Test determine_transport_type with 'sse' transport in k8s mode."""
    workload = Workload(
        name="test-workload",
        transport_type="sse",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_stdio_value():
    """Test determine_transport_type with 'stdio' transport in k8s mode falls back to URL."""
    workload = Workload(
        name="test-workload",
        transport_type="stdio",
        url="http://localhost:8080/mcp/path",
    )
    # stdio is not a valid HTTP transport, so it should fall back to URL detection
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_fallback_to_url_mcp():
    """Test determine_transport_type falls back to URL detection for /mcp path in docker mode."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/mcp/test-server",
    )
    result = determine_transport_type(workload, "docker")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_fallback_to_url_sse():
    """Test determine_transport_type falls back to URL detection for /sse path in docker mode."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/sse/test-server",
    )
    result = determine_transport_type(workload, "docker")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_stdio_fallback_to_url_mcp():
    """Test determine_transport_type with stdio transport_type falls back to URL with /mcp
    in k8s mode (stdio is not an HTTP transport)."""
    workload = Workload(
        name="test-workload",
        transport_type="stdio",
        url="http://localhost:8080/mcp/test-server",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_stdio_fallback_to_url_sse():
    """Test determine_transport_type with stdio transport_type falls back to URL with /sse
    in k8s mode (stdio is not an HTTP transport)."""
    workload = Workload(
        name="test-workload",
        transport_type="stdio",
        url="http://localhost:8080/sse/test-server",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_no_transport_no_url():
    """Test determine_transport_type raises error when no transport_type and no URL
    in docker mode."""
    workload = Workload(
        name="test-workload",
    )
    with pytest.raises(WorkloadConnectionError, match="No transport type or URL available"):
        determine_transport_type(workload, "docker")


def test_determine_transport_type_stdio_transport_no_url():
    """Test determine_transport_type raises error when stdio transport_type and no URL
    in k8s mode (stdio is not an HTTP transport and needs URL fallback)."""
    workload = Workload(
        name="test-workload",
        transport_type="stdio",
    )
    with pytest.raises(WorkloadConnectionError, match="No transport type or URL available"):
        determine_transport_type(workload, "k8s")


def test_determine_transport_type_stdio_transport_unsupported_url():
    """Test determine_transport_type raises ValueError when stdio transport_type
    and unsupported URL in k8s mode."""
    workload = Workload(
        name="test-workload",
        transport_type="stdio",
        url="http://localhost:8080/unsupported/path",
    )
    with pytest.raises(ValueError, match="Unsupported ToolHive URL"):
        determine_transport_type(workload, "k8s")


def test_determine_transport_type_no_transport_unsupported_url():
    """Test determine_transport_type raises ValueError when no transport_type
    and unsupported URL in docker mode."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/unknown/path",
    )
    with pytest.raises(ValueError, match="Unsupported ToolHive URL"):
        determine_transport_type(workload, "docker")


def test_determine_transport_type_docker_mode_proxy_mode_streamable():
    """Test docker mode uses proxy_mode field for streamable-http."""
    workload = Workload(
        name="test-workload",
        proxy_mode="streamable-http",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "docker")
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_docker_mode_proxy_mode_sse():
    """Test docker mode uses proxy_mode field for sse."""
    workload = Workload(
        name="test-workload",
        proxy_mode="sse",
        url="http://localhost:8080/some/path",
    )
    result = determine_transport_type(workload, "docker")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_docker_ignores_transport_type():
    """Test that docker mode ignores transport_type field when proxy_mode is set."""
    workload = Workload(
        name="test-workload",
        transport_type="sse",
        proxy_mode="streamable-http",
        url="http://localhost:8080/mcp/path",
    )
    result = determine_transport_type(workload, "docker")
    # Should use proxy_mode, not transport_type
    assert result == ToolHiveTransportMode.STREAMABLE


def test_determine_transport_type_k8s_ignores_proxy_mode():
    """Test that k8s mode ignores proxy_mode field when transport_type is set."""
    workload = Workload(
        name="test-workload",
        proxy_mode="streamable-http",
        transport_type="sse",
        url="http://localhost:8080/sse/path",
    )
    result = determine_transport_type(workload, "k8s")
    # Should use transport_type, not proxy_mode
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_k8s_fallback_to_url():
    """Test k8s mode falls back to URL when transport_type not set."""
    workload = Workload(
        name="test-workload",
        url="http://localhost:8080/sse/test-server",
    )
    result = determine_transport_type(workload, "k8s")
    assert result == ToolHiveTransportMode.SSE


def test_determine_transport_type_docker_fallback_when_no_proxy_mode():
    """Test docker mode falls back to URL when proxy_mode not set but transport_type is."""
    workload = Workload(
        name="test-workload",
        transport_type="streamable-http",
        url="http://localhost:8080/mcp/test-server",
    )
    result = determine_transport_type(workload, "docker")
    # Docker mode should ignore transport_type and fallback to URL
    assert result == ToolHiveTransportMode.STREAMABLE


# Unit tests for SSE tolerant client behavior


@pytest.mark.asyncio
async def test_sse_session_propagates_errors(mock_mcp_session):
    """Test that SSE session propagates errors as WorkloadConnectionError."""
    workload = Workload(
        name="test-server",
        url="http://localhost:8080/sse/test-server",
        status="running",
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    class FailingCM:
        async def __aenter__(self):
            raise ExceptionGroup("errors", [ValueError("some error")])

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return False

    with (
        patch("mcp_optimizer.mcp_client.sse_client") as mock_sse_client,
        patch("mcp_optimizer.mcp_client.ClientSession", return_value=mock_mcp_session),
    ):
        mock_sse_client.return_value = FailingCM()

        # Should raise WorkloadConnectionError
        with pytest.raises(WorkloadConnectionError):
            await client.list_tools()

        # Verify sse_client was called once
        assert mock_sse_client.call_count == 1


@pytest.mark.asyncio
async def test_sse_session_uses_tolerant_client(mock_mcp_session):
    """Test that SSE session always uses the tolerant httpx client."""
    workload = Workload(
        name="test-server",
        url="http://localhost:8080/sse/test-server",
        status="running",
            )

    client = MCPServerClient(workload, timeout=10, runtime_mode="docker")

    with (
        patch("mcp_optimizer.mcp_client.sse_client") as mock_sse_client,
        patch(
            "mcp_optimizer.mcp_client.ClientSession", return_value=mock_mcp_session
        ) as mock_session_class,
    ):
        # Mock successful connection
        mock_sse_client.return_value.__aenter__.return_value = (AsyncMock(), AsyncMock())
        mock_session_class.return_value.__aenter__.return_value = mock_mcp_session

        # Call list_tools
        await client.list_tools()

        # Verify sse_client was called with httpx_client_factory
        assert mock_sse_client.call_count == 1
        call_kwargs = mock_sse_client.call_args.kwargs
        assert "httpx_client_factory" in call_kwargs
        assert call_kwargs["httpx_client_factory"] == _create_tolerant_httpx_client


# Unit tests for _TolerantStream class


class MockAsyncByteStream(httpx.AsyncByteStream):
    """Mock async byte stream for testing _TolerantStream."""

    def __init__(self, chunks: list[bytes], exception: Exception | None = None):
        self.chunks = chunks
        self.exception = exception
        self._closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk
        if self.exception:
            raise self.exception

    async def aclose(self):
        self._closed = True


@pytest.mark.asyncio
async def test_tolerant_stream_swallows_remote_protocol_error():
    """Test that _TolerantStream catches and ignores RemoteProtocolError."""
    # Create a stream that raises RemoteProtocolError after yielding some data
    error = httpx.RemoteProtocolError("Server disconnected")
    mock_stream = MockAsyncByteStream(chunks=[b"chunk1", b"chunk2"], exception=error)

    tolerant_stream = _TolerantStream(mock_stream)

    # Should not raise, just stop iterating
    chunks = []
    async for chunk in tolerant_stream:
        chunks.append(chunk)

    # Should have received the chunks before the error
    assert chunks == [b"chunk1", b"chunk2"]


@pytest.mark.asyncio
async def test_tolerant_stream_propagates_other_errors():
    """Test that _TolerantStream does not swallow non-RemoteProtocolError exceptions."""
    # Create a stream that raises a different error
    error = ValueError("Some other error")
    mock_stream = MockAsyncByteStream(chunks=[b"chunk1"], exception=error)

    tolerant_stream = _TolerantStream(mock_stream)

    # Should raise ValueError, not swallow it
    with pytest.raises(ValueError, match="Some other error"):
        async for _ in tolerant_stream:
            pass


@pytest.mark.asyncio
async def test_tolerant_stream_passes_through_chunks():
    """Test that _TolerantStream correctly passes through all chunks when no error."""
    mock_stream = MockAsyncByteStream(chunks=[b"chunk1", b"chunk2", b"chunk3"])

    tolerant_stream = _TolerantStream(mock_stream)

    chunks = []
    async for chunk in tolerant_stream:
        chunks.append(chunk)

    assert chunks == [b"chunk1", b"chunk2", b"chunk3"]


@pytest.mark.asyncio
async def test_tolerant_stream_aclose():
    """Test that _TolerantStream properly closes the underlying stream."""
    mock_stream = MockAsyncByteStream(chunks=[])

    tolerant_stream = _TolerantStream(mock_stream)

    assert not mock_stream._closed
    await tolerant_stream.aclose()
    assert mock_stream._closed
