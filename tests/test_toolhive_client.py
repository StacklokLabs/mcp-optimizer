"""
Tests for the Toolhive client.
"""

from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from semver import Version

from mcp_optimizer.toolhive.api_models.v1 import WorkloadListResponse
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient, ToolhiveScanError


@pytest.fixture
def mock_workload_response():
    """Mock workload response data."""
    return {
        "workloads": [
            {
                "name": "test-server",
                "package": "test/mcp-server:latest",
                "url": "http://127.0.0.1:8080/sse#test-server",
                "port": 8080,
                "tool_type": "mcp",
                "transport_type": "stdio",
                "status": "running",
                "status_context": "Up 1 hour",
                "created_at": "2025-08-19T06:00:00Z",
            }
        ]
    }


@pytest.fixture
def toolhive_client(monkeypatch):
    """Create a ToolhiveClient for testing."""

    async def mock_scan_for_toolhive(self, host, start_port, end_port):
        return 8080  # Force return of 8080 for testing

    async def mock_is_toolhive_available(self, host, port):
        # Return (Version, port) tuple as per new signature
        if port == 8080:
            return (Version.parse("1.0.0"), 8080)
        raise ToolhiveScanError(f"Port {port} not available")

    # Mock the methods before creating the client
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive",
        mock_scan_for_toolhive,
    )
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    # Mock the port scanning to avoid network calls during testing
    client = ToolhiveClient(
        host="localhost",  # Use localhost instead of 127.0.0.1 for URL replacement tests
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )
    return client


@pytest.mark.asyncio
async def test_list_workloads(toolhive_client, mock_workload_response):
    """Test listing workloads."""
    async with toolhive_client as client:
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_workload_response
        mock_response.raise_for_status = Mock()

        client._client.get = AsyncMock(return_value=mock_response)

        # Call the method
        result = await client.list_workloads()

        # Verify the result
        assert isinstance(result, WorkloadListResponse)
        assert len(result.workloads) == 1

        workload = result.workloads[0]
        assert workload.name == "test-server"
        assert workload.status == "running"
        assert workload.transport_type == "stdio"
        assert workload.tool_type == "mcp"

        # Verify the HTTP call
        expected_url = (
            f"http://{toolhive_client.thv_host}:{toolhive_client.thv_port}/api/v1beta/workloads"
        )
        toolhive_client._client.get.assert_called_once_with(expected_url, params={})


@pytest.mark.asyncio
async def test_get_running_mcp_workloads(toolhive_client, mock_workload_response):
    """Test getting running MCP workloads."""
    async with toolhive_client as client:
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_workload_response
        mock_response.raise_for_status = Mock()

        client._client.get = AsyncMock(return_value=mock_response)

        # Call the method
        result = await client.get_running_mcp_workloads()

        # Verify the result
        assert len(result) == 1
        assert result[0].name == "test-server"
        assert result[0].status == "running"
        assert result[0].tool_type == "mcp"


@pytest.mark.asyncio
async def test_list_workloads_with_all_flag(toolhive_client, mock_workload_response):
    """Test listing workloads with all flag."""
    async with toolhive_client as client:
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_workload_response
        mock_response.raise_for_status = Mock()

        client._client.get = AsyncMock(return_value=mock_response)

        # Call the method with all_workloads=True
        await client.list_workloads(all_workloads=True)

        # Verify the HTTP call includes the 'all' parameter
        expected_url = (
            f"http://{toolhive_client.thv_host}:{toolhive_client.thv_port}/api/v1beta/workloads"
        )
        toolhive_client._client.get.assert_called_once_with(expected_url, params={"all": "true"})


@pytest.mark.asyncio
async def test_http_error_handling(toolhive_client):
    """Test HTTP error handling."""
    async with toolhive_client as c:
        # Mock the HTTP client to raise an HTTP error
        c._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError("404 Not Found", request=Mock(), response=Mock())
        )

        # Call the method and expect an exception
        with pytest.raises(httpx.HTTPStatusError):
            await c.list_workloads()


@pytest.mark.asyncio
async def test_context_manager(monkeypatch):
    """Test using ToolhiveClient as context manager (backward compatibility)."""

    def mock_discover_port(self, port):
        # Skip the async discovery during init
        self.thv_port = 8080
        self.base_url = f"http://{self.thv_host}:{self.thv_port}"

    # Mock the discovery to avoid asyncio.run in async context
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._discover_port",
        mock_discover_port,
    )

    async with ToolhiveClient(
        host="127.0.0.1",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    ) as client:
        assert client._client is not None

    # Client should be closed after exiting context
    assert client._client is None


@pytest.mark.asyncio
async def test_connect_disconnect(monkeypatch):
    """Test explicit connect and disconnect methods."""

    def mock_discover_port(self, port):
        # Skip the async discovery during init
        self.thv_port = 8080
        self.base_url = f"http://{self.thv_host}:{self.thv_port}"

    # Mock the discovery to avoid asyncio.run in async context
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._discover_port",
        mock_discover_port,
    )

    client = ToolhiveClient(
        host="127.0.0.1",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )

    # Initially not connected
    assert client._client is None

    # Connect
    async with client as c:
        assert c._client is not None

    # Disconnect
    assert client._client is None


@pytest.mark.asyncio
async def test_client_lazy_initialization(monkeypatch):
    """Test that accessing client property creates client lazily."""

    def mock_discover_port(self, port):
        # Skip the async discovery during init
        self.thv_port = 8080
        self.base_url = f"http://{self.thv_host}:{self.thv_port}"

    # Mock the discovery to avoid asyncio.run in async context
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._discover_port",
        mock_discover_port,
    )

    client = ToolhiveClient(
        host="127.0.0.1",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )

    # Client should be None initially
    assert client._client is None

    # Accessing the client property should create it
    http_client = client.client
    assert http_client is not None
    assert client._client is not None


@pytest.mark.asyncio
async def test_port_scanning_not_available(monkeypatch):
    """Test port scanning when no ToolHive is available."""

    async def mock_is_toolhive_available(self, host, port):
        raise ToolhiveScanError(f"Port {port} not available")

    # Mock the port checking method
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    client = ToolhiveClient(
        host="127.0.0.1",
        port=None,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )
    # Port discovery is lazy, so exception should occur when ensure_connected is called
    with pytest.raises(
        ConnectionError, match="ToolHive not found on 127.0.0.1 in port range 50000-50100"
    ):
        await client.ensure_connected()


@pytest.mark.asyncio
async def test_port_scanning_finds_port(monkeypatch):
    """Test port scanning when ToolHive is found."""

    async def mock_is_toolhive_available(self, host, port):
        # Simulate finding ToolHive at port 50050
        if port == 50050:
            return (Version.parse("1.0.0"), 50050)
        raise ToolhiveScanError(f"Port {port} not available")

    # Mock the port checking method
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    client = ToolhiveClient(
        host="127.0.0.1",
        port=None,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )
    # Port discovery is lazy, so we need to await ensure_connected first
    await client.ensure_connected()
    assert client.thv_port == 50050
    assert client.base_url == "http://127.0.0.1:50050"


@pytest.mark.asyncio
async def test_fallback_to_port_scanning(monkeypatch):
    """Test fallback to port scanning when provided port is not available."""

    async def mock_is_toolhive_available(self, host, port):
        if port == 8080:  # Provided port is not available
            raise ToolhiveScanError(f"Port {port} not available")
        # But ToolHive is found at port 50075
        if port == 50075:
            return (Version.parse("1.0.0"), 50075)
        raise ToolhiveScanError(f"Port {port} not available")

    # Mock the port checking method
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    client = ToolhiveClient(
        host="127.0.0.1",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )
    # Port discovery is lazy, so we need to await ensure_connected first
    await client.ensure_connected()
    assert client.thv_port == 50075
    assert client.base_url == "http://127.0.0.1:50075"


@pytest.mark.asyncio
async def test_is_toolhive_available_validates_response_format(monkeypatch):  # noqa: C901
    """Test that _is_toolhive_available properly validates the response is from ToolHive.

    This prevents false positives when another service (like Ubuntu Multipass)
    is running on a port in the scan range and returns a 200 OK but with
    a different response format.
    """
    client = ToolhiveClient.__new__(ToolhiveClient)
    client.thv_host = "localhost"
    client.timeout = 5  # Set the timeout attribute that _is_toolhive_available needs

    # Test case 1: Service returns 200 OK but response has no 'version' field
    mock_response_wrong_format = Mock()
    mock_response_wrong_format.raise_for_status = Mock()
    mock_response_wrong_format.json.return_value = {"status": "ok", "service": "multipass"}

    class MockAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return mock_response_wrong_format

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient)
    with pytest.raises(ToolhiveScanError, match="did not respond with valid JSON"):
        await client._is_toolhive_available("localhost", 50051)

    # Test case 2: Service returns 200 OK with proper ToolHive version format
    mock_response_toolhive = Mock()
    mock_response_toolhive.raise_for_status = Mock()
    mock_response_toolhive.json.return_value = {"version": "v0.1.0"}

    class MockAsyncClient2:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return mock_response_toolhive

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient2)
    version, port = await client._is_toolhive_available("localhost", 50056)
    assert port == 50056
    assert version == Version.parse("0.1.0")

    # Test case 3: Service returns 200 OK but not valid JSON
    mock_response_not_json = Mock()
    mock_response_not_json.raise_for_status = Mock()
    mock_response_not_json.json.side_effect = ValueError("Invalid JSON")

    class MockAsyncClient3:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return mock_response_not_json

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient3)
    with pytest.raises(ToolhiveScanError, match="did not respond with valid JSON"):
        await client._is_toolhive_available("localhost", 50051)

    # Test case 4: Service returns 200 OK but response is not a dict (e.g., a list)
    mock_response_not_dict = Mock()
    mock_response_not_dict.raise_for_status = Mock()
    mock_response_not_dict.json.return_value = ["version", "v0.1.0"]

    class MockAsyncClient4:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            return mock_response_not_dict

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient4)
    with pytest.raises(ToolhiveScanError, match="did not respond with valid JSON"):
        await client._is_toolhive_available("localhost", 50051)

    # Test case 5: Connection error (port not listening)
    class MockAsyncClient5:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient5)
    with pytest.raises(ToolhiveScanError, match="Error checking ToolHive availability"):
        await client._is_toolhive_available("localhost", 50051)

    # Test case 6: HTTP error (404, 500, etc.)
    class MockAsyncClient6:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, *args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 404
            raise httpx.HTTPStatusError("404 Not Found", request=Mock(), response=mock_response)

    monkeypatch.setattr(httpx, "AsyncClient", MockAsyncClient6)
    with pytest.raises(ToolhiveScanError, match="Error checking ToolHive availability"):
        await client._is_toolhive_available("localhost", 50051)


# Tests for get_workload_details (T003)


@pytest.fixture
def mock_workload_detail_response():
    """Mock workload detail response data."""
    return {
        "name": "custom-github-server",
        "package": "mcp-github",
        "url": "https://api.github.com/mcp",
        "remote": True,
        "status": "running",
        "tool_type": "remote",
        "proxy_mode": "sse",
        "port": 8081,
        "group": "production",
        "created_at": "2025-10-20T10:00:00Z",
        "transport_type": "sse",
    }


@pytest.mark.asyncio
async def test_get_workload_details_success(toolhive_client, mock_workload_detail_response):
    """Test successful fetch of workload details."""
    async with toolhive_client as client:
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_workload_detail_response
        mock_response.raise_for_status = Mock()

        client._client.get = AsyncMock(return_value=mock_response)

        # Call the method
        result = await client.get_workload_details("custom-github-server")

        # Verify the result
        assert result.name == "custom-github-server"
        assert result.url == "https://api.github.com/mcp"
        assert result.remote is True
        assert result.status == "running"
        assert result.tool_type == "remote"

        # Verify the HTTP call
        expected_url = (
            f"http://{toolhive_client.thv_host}:{toolhive_client.thv_port}/"
            "api/v1beta/workloads/custom-github-server"
        )
        toolhive_client._client.get.assert_called_once_with(expected_url)


@pytest.mark.asyncio
async def test_get_workload_details_replaces_localhost(mock_workload_detail_response, monkeypatch):
    """Test that localhost URLs are replaced with the toolhive host."""
    # Mock _is_running_in_docker to return True so URL replacement happens
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client._is_running_in_docker",
        lambda: True,
    )

    async def mock_scan_for_toolhive(self, host, start_port, end_port):
        return 8080

    async def mock_is_toolhive_available(self, host, port):
        if port == 8080:
            return (Version.parse("1.0.0"), 8080)
        raise ToolhiveScanError(f"Port {port} not available")

    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive",
        mock_scan_for_toolhive,
    )
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    # Use a host that's NOT localhost/127.0.0.1 so replacement happens
    client = ToolhiveClient(
        host="toolhive-host",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )

    # Modify the mock response to have a localhost URL
    mock_workload_detail_response["url"] = "http://localhost:8080/mcp"

    async with client:
        # Ensure connected first (lazy port discovery)
        await client.ensure_connected()
        # Mock the HTTP client
        mock_response = Mock()
        mock_response.json.return_value = mock_workload_detail_response
        mock_response.raise_for_status = Mock()

        client._client.get = AsyncMock(return_value=mock_response)

        # Call the method
        result = await client.get_workload_details("test-server")

        # Verify that localhost was replaced with the toolhive host
        assert result.url == "http://toolhive-host:8080/mcp"


@pytest.mark.asyncio
async def test_get_workload_details_404_error(toolhive_client):
    """Test 404 error when workload doesn't exist."""
    async with toolhive_client as client:
        # Mock the HTTP client to raise a 404 error
        mock_response = Mock()
        mock_response.status_code = 404
        client._client.get = AsyncMock(
            side_effect=httpx.HTTPStatusError(
                "404 Not Found", request=Mock(), response=mock_response
            )
        )

        # Call the method and expect an exception
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_workload_details("nonexistent-workload")


@pytest.mark.asyncio
async def test_get_workload_details_network_error(toolhive_client):
    """Test network error handling when fetching workload details."""
    async with toolhive_client as client:
        # Mock the HTTP client to raise a network error
        client._client.get = AsyncMock(side_effect=httpx.RequestError("Network error"))

        # Call the method and expect an exception
        with pytest.raises(httpx.RequestError):
            await client.get_workload_details("test-server")


@pytest.mark.asyncio
async def test_get_workload_details_timeout(toolhive_client):
    """Test timeout error handling when fetching workload details."""
    async with toolhive_client as client:
        # Mock the HTTP client to raise a timeout error
        client._client.get = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

        # Call the method and expect an exception
        with pytest.raises(httpx.TimeoutException):
            await client.get_workload_details("test-server")
