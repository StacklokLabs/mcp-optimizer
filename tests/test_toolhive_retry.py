"""
Tests for the ToolHive client retry logic and connection resilience.
"""

import asyncio
from unittest.mock import AsyncMock, Mock

import httpx
import pytest

from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient, ToolhiveConnectionError


@pytest.fixture
def retry_client(monkeypatch):
    """Create a ToolhiveClient with retry configuration for testing."""

    def mock_scan_for_toolhive(self, host, start_port, end_port):
        return 50001

    def mock_is_toolhive_available(self, host, port):
        # Return (version, bool) tuple as per new signature
        if port == 50001:
            return ("1.0.0", True)
        return ("", False)

    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive",
        mock_scan_for_toolhive,
    )
    monkeypatch.setattr(
        "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
        mock_is_toolhive_available,
    )

    client = ToolhiveClient(
        host="localhost",
        port=50001,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,  # Use fewer retries for faster tests
        initial_backoff=0.1,  # Shorter backoff for tests
        max_backoff=0.5,
    )
    return client


class TestPortRediscovery:
    """Tests for port rediscovery functionality."""

    @pytest.mark.asyncio
    async def test_rediscover_port_success(self, monkeypatch):
        """Test successful port rediscovery to a new port."""
        call_count = 0

        def mock_is_toolhive_available(self, host, port):
            nonlocal call_count
            call_count += 1
            # During init: port 50001 is available (first call)
            if call_count == 1:
                if port == 50001:
                    return ("1.0.0", True)
                return ("", False)
            return ("", False)

        async def mock_scan_for_toolhive_async(self, host, start_port, end_port):
            # During rediscovery: return port 50002
            return 50002

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )
        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive_async",
            mock_scan_for_toolhive_async,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=3,
            initial_backoff=0.1,
            max_backoff=0.5,
        )

        # Initially on port 50001
        assert client.thv_port == 50001
        old_port = client.thv_port

        # Trigger rediscovery - should find new port
        assert await client._rediscover_port() is True
        assert client.thv_port == 50002
        assert client.thv_port != old_port
        assert client.base_url == "http://localhost:50002"

    @pytest.mark.asyncio
    async def test_rediscover_port_same_port(self, monkeypatch):
        """Test rediscovery when ToolHive is still on the same port."""

        def mock_is_toolhive_available(self, host, port):
            if port == 50001:
                return ("1.0.0", True)
            return ("", False)

        async def mock_scan_for_toolhive_async(self, host, start_port, end_port):
            # During rediscovery: return same port 50001
            return 50001

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )
        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive_async",
            mock_scan_for_toolhive_async,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=3,
            initial_backoff=0.1,
            max_backoff=0.5,
        )

        assert await client._rediscover_port() is True
        assert client.thv_port == 50001

    @pytest.mark.asyncio
    async def test_rediscover_port_failure(self, monkeypatch):
        """Test failed port rediscovery when ToolHive is not available."""
        call_count = 0

        def mock_is_toolhive_available(self, host, port):
            nonlocal call_count
            call_count += 1
            # During init: port 50001 is available
            if call_count <= 1:
                if port == 50001:
                    return ("1.0.0", True)
                return ("", False)
            return ("", False)

        async def mock_scan_for_toolhive_async(self, host, start_port, end_port):
            # During rediscovery: no ports available - raise ConnectionError
            raise ConnectionError(
                f"ToolHive not found on {host} in port range {start_port}-{end_port}"
            )

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )
        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._scan_for_toolhive_async",
            mock_scan_for_toolhive_async,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=3,
            initial_backoff=0.1,
            max_backoff=0.5,
        )

        old_port = client.thv_port
        assert await client._rediscover_port() is False
        # Port should be restored to old value on failure
        assert client.thv_port == old_port

    @pytest.mark.asyncio
    async def test_rediscover_port_exception_handling(self, monkeypatch):
        """Test that exceptions during rediscovery are handled gracefully."""
        call_count = 0

        def mock_discover_port_with_exception(self, port):
            nonlocal call_count
            call_count += 1
            # Allow first call (init) to succeed
            if call_count == 1:
                self.thv_port = 50001
                self.base_url = f"http://{self.thv_host}:{self.thv_port}"
                return
            # Subsequent calls (rediscovery) raise exception
            raise Exception("Scan failed")

        async def mock_discover_port_async_with_exception(self, port):
            # Async version for rediscovery
            raise Exception("Scan failed")

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._discover_port",
            mock_discover_port_with_exception,
        )
        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._discover_port_async",
            mock_discover_port_async_with_exception,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=3,
            initial_backoff=0.1,
            max_backoff=0.5,
        )

        old_port = client.thv_port
        assert await client._rediscover_port() is False
        # Port should be restored on error
        assert client.thv_port == old_port


class TestRetryLogic:
    """Tests for retry logic with exponential backoff."""

    @pytest.mark.asyncio
    async def test_successful_operation_no_retry(self, retry_client):
        """Test that successful operations don't trigger retries."""
        async with retry_client as client:
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()

            client._client.get = AsyncMock(return_value=mock_response)

            result = await client.list_workloads()

            # Should only call once, no retries
            assert client._client.get.call_count == 1
            assert result.workloads == []

    @pytest.mark.asyncio
    async def test_retry_on_connect_error(self, retry_client, monkeypatch):
        """Test retry logic on connection errors."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("Connection refused")
            # Success on third attempt
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        # Mock rediscovery to succeed
        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get

            result = await client.list_workloads()

            assert call_count == 3
            assert result.workloads == []

    @pytest.mark.asyncio
    async def test_retry_on_timeout_error(self, retry_client, monkeypatch):
        """Test retry logic on timeout errors."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ReadTimeout("Request timed out")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get

            await client.list_workloads()

            assert call_count == 2

    @pytest.mark.asyncio
    async def test_ultimate_failure_raises_exception(self, retry_client, monkeypatch):
        """Test that ultimate failure after all retries raises ToolhiveConnectionError."""

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=False))

        async with retry_client as client:
            client._client.get = mock_get

            with pytest.raises(ToolhiveConnectionError) as exc_info:
                await client.list_workloads()

            assert "Failed to connect to ToolHive after 3 attempts" in str(exc_info.value)
            assert "Connection refused" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exponential_backoff_timing(self, retry_client, monkeypatch):
        """Test that exponential backoff delays are applied correctly."""
        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=False))

        async with retry_client as client:
            client._client.get = mock_get

            with pytest.raises(ToolhiveConnectionError):
                await client.list_workloads()

            # Should have backoff delays: 0.1, 0.2, then no more (3 attempts total)
            # Last attempt doesn't sleep
            assert len(sleep_calls) == 2
            assert sleep_calls[0] == 0.1  # initial_backoff
            assert sleep_calls[1] == 0.2  # doubled

    @pytest.mark.asyncio
    async def test_backoff_capped_at_max(self, retry_client, monkeypatch):
        """Test that backoff is capped at max_backoff."""
        # Use more retries to test capping
        retry_client.max_retries = 10

        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        async def mock_get(*args, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=False))

        async with retry_client as client:
            client._client.get = mock_get

            with pytest.raises(ToolhiveConnectionError):
                await client.list_workloads()

            # Check that backoff is capped at max_backoff (0.5)
            assert all(delay <= 0.5 for delay in sleep_calls)
            # Should reach max backoff
            assert 0.5 in sleep_calls

    @pytest.mark.asyncio
    async def test_backoff_reset_after_successful_rediscovery(self, monkeypatch):
        """Test that backoff resets after successful port rediscovery."""

        # Create a client with more retries for this test
        def mock_is_toolhive_available(self, host, port):
            if port == 50001:
                return ("1.0.0", True)
            return ("", False)

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=10,  # More retries for this test
            initial_backoff=0.1,
            max_backoff=0.5,
        )

        call_count = 0
        rediscover_call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        async def mock_rediscover():
            nonlocal rediscover_call_count
            rediscover_call_count += 1
            # Succeed on second rediscovery attempt
            return rediscover_call_count >= 2

        sleep_calls = []

        async def mock_sleep(delay):
            sleep_calls.append(delay)

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(client, "_rediscover_port", AsyncMock(side_effect=mock_rediscover))

        async with client as c:
            c._client.get = mock_get

            result = await c.list_workloads()

            # Should succeed eventually
            assert result.workloads == []

            # Verify backoff was reset after successful rediscovery
            # First failure: backoff 0.1, no rediscovery
            # Second failure: rediscovery succeeds, backoff reset
            # Should see initial backoff value again after reset
            assert 0.1 in sleep_calls

    @pytest.mark.asyncio
    async def test_immediate_retry_after_successful_rediscovery(self, retry_client, monkeypatch):
        """Test that successful rediscovery triggers immediate retry without backoff."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        sleep_called = False

        async def mock_sleep(delay):
            nonlocal sleep_called
            sleep_called = True

        monkeypatch.setattr(asyncio, "sleep", mock_sleep)
        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get

            result = await client.list_workloads()

            assert result.workloads == []
            # Sleep should not be called when rediscovery succeeds
            assert not sleep_called


class TestRetryOnAllMethods:
    """Test that retry logic is applied to all HTTP methods."""

    @pytest.mark.asyncio
    async def test_list_workloads_retry(self, retry_client, monkeypatch):
        """Test retry on list_workloads."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get
            await client.list_workloads()
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_workload_details_retry(self, retry_client, monkeypatch):
        """Test retry on get_workload_details."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {
                "name": "test",
                "url": "http://localhost:8080",
                "status": "running",
                "package": "test/pkg",
            }
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get
            await client.get_workload_details("test")
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_get_registry_retry(self, retry_client, monkeypatch):
        """Test retry on get_registry."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {"registry": {"servers": {}, "remote_servers": {}}}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get
            await client.get_registry()
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_install_server_retry(self, retry_client, monkeypatch):
        """Test retry on install_server."""
        from mcp_optimizer.toolhive.api_models.v1 import CreateRequest

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection refused")
            mock_response = Mock()
            mock_response.json.return_value = {"name": "test", "port": 8080}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.post = mock_post
            request = CreateRequest(name="test", image="test/image")
            await client.install_server(request)
            assert call_count == 2


class TestConnectionErrorTypes:
    """Test that different connection error types trigger retries."""

    @pytest.mark.asyncio
    async def test_connect_timeout_triggers_retry(self, retry_client, monkeypatch):
        """Test that ConnectTimeout errors trigger retry."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectTimeout("Connection timed out")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get
            await client.list_workloads()
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_remote_protocol_error_triggers_retry(self, retry_client, monkeypatch):
        """Test that RemoteProtocolError triggers retry."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.RemoteProtocolError("Protocol error")
            mock_response = Mock()
            mock_response.json.return_value = {"workloads": []}
            mock_response.raise_for_status = Mock()
            return mock_response

        monkeypatch.setattr(retry_client, "_rediscover_port", AsyncMock(return_value=True))

        async with retry_client as client:
            client._client.get = mock_get
            await client.list_workloads()
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_http_status_error_no_retry(self, retry_client):
        """Test that HTTP status errors (404, 500) do not trigger retry."""
        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise httpx.HTTPStatusError("404 Not Found", request=Mock(), response=Mock())

        async with retry_client as client:
            client._client.get = mock_get

            with pytest.raises(httpx.HTTPStatusError):
                await client.list_workloads()

            # Should not retry on HTTP status errors
            assert call_count == 1


class TestRetryConfiguration:
    """Test retry configuration parameters."""

    def test_custom_max_retries(self, monkeypatch):
        """Test custom max_retries configuration."""

        def mock_is_toolhive_available(self, host, port):
            if port == 50001:
                return ("1.0.0", True)
            return ("", False)

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=25,
            initial_backoff=1.0,
            max_backoff=60.0,
        )

        assert client.max_retries == 25

    def test_custom_backoff_values(self, monkeypatch):
        """Test custom backoff configuration."""

        def mock_is_toolhive_available(self, host, port):
            if port == 50001:
                return ("1.0.0", True)
            return ("", False)

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=10,
            initial_backoff=2.5,
            max_backoff=120.0,
        )

        assert client.initial_backoff == 2.5
        assert client.max_backoff == 120.0

    def test_initial_port_stored(self, monkeypatch):
        """Test that initial port is stored for rediscovery."""

        def mock_is_toolhive_available(self, host, port):
            if port == 50001:
                return ("1.0.0", True)
            return ("", False)

        monkeypatch.setattr(
            "mcp_optimizer.toolhive.toolhive_client.ToolhiveClient._is_toolhive_available",
            mock_is_toolhive_available,
        )

        client = ToolhiveClient(
            host="localhost",
            port=50001,
            scan_port_start=50000,
            scan_port_end=50100,
            timeout=5.0,
            max_retries=10,
            initial_backoff=1.0,
            max_backoff=60.0,
        )

        assert client._initial_port == 50001
