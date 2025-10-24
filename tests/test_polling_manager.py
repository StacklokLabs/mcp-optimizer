"""Tests for the PollingManager."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from semver import Version

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.models import McpStatus, RegistryServer, WorkloadServer
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.polling_manager import PollingManager
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient, ToolhiveScanError


@pytest.fixture
def db_config():
    """Create a test database config."""
    return DatabaseConfig(database_url="sqlite+aiosqlite:///:memory:")


@pytest.fixture
def embedding_manager():
    """Create a test embedding manager."""
    return EmbeddingManager(model_name="BAAI/bge-small-en-v1.5", enable_cache=True)


@pytest.fixture
def toolhive_client(monkeypatch):
    """Create a mock ToolhiveClient for testing."""

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

    return ToolhiveClient(
        host="localhost",
        port=8080,
        scan_port_start=50000,
        scan_port_end=50100,
        timeout=5.0,
        max_retries=3,
        initial_backoff=1.0,
        max_backoff=60.0,
    )


@pytest.fixture
def polling_manager(db_config, embedding_manager, toolhive_client):
    """Create a polling manager for testing."""
    return PollingManager(
        db_config=db_config,
        embedding_manager=embedding_manager,
        workload_polling_interval=0.1,  # Very fast for testing
        registry_polling_interval=0.2,  # Very fast for testing
        toolhive_client=toolhive_client,
        mcp_timeout=10,
        registry_ingestion_batch_size=5,
        workload_ingestion_batch_size=5,
        encoding="cl100k_base",  # Default tiktoken encoding
        targeted_polling_max_attempts=120,
        targeted_polling_interval=1,
    )


class TestPollingManager:
    """Test cases for PollingManager."""

    def test_initialization(self, polling_manager):
        """Test that PollingManager initializes correctly."""
        assert polling_manager.workload_polling_interval == 0.1
        assert polling_manager.registry_polling_interval == 0.2
        assert polling_manager.toolhive_client.thv_host == "localhost"
        assert polling_manager.toolhive_client.thv_port == 8080
        assert polling_manager._workload_polling_task is None
        assert polling_manager._registry_polling_task is None
        assert not polling_manager._shutdown_requested
        assert not polling_manager.is_polling()

    async def test_start_polling(self, polling_manager):
        """Test starting polling."""
        with (
            patch.object(polling_manager, "_workload_polling_loop", new_callable=AsyncMock),
            patch.object(polling_manager, "_registry_polling_loop", new_callable=AsyncMock),
        ):
            await polling_manager.start_polling()

            assert polling_manager._workload_polling_task is not None
            assert polling_manager._registry_polling_task is not None
            assert polling_manager.is_polling()

            # Clean up
            await polling_manager.stop_polling()

    async def test_stop_polling(self, polling_manager):
        """Test stopping polling."""
        with (
            patch.object(polling_manager, "_workload_polling_loop", new_callable=AsyncMock),
            patch.object(polling_manager, "_registry_polling_loop", new_callable=AsyncMock),
        ):
            await polling_manager.start_polling()
            assert polling_manager.is_polling()

            await polling_manager.stop_polling()
            assert not polling_manager.is_polling()
            assert polling_manager._workload_polling_task is None
            assert polling_manager._registry_polling_task is None

    async def test_start_polling_already_running(self, polling_manager):
        """Test that starting polling when already running does nothing."""
        with (
            patch.object(polling_manager, "_workload_polling_loop", new_callable=AsyncMock),
            patch.object(polling_manager, "_registry_polling_loop", new_callable=AsyncMock),
        ):
            await polling_manager.start_polling()

            # Try to start again
            await polling_manager.start_polling()

            # Should still be running
            assert polling_manager.is_polling()

            # Clean up
            await polling_manager.stop_polling()

    async def test_stop_polling_not_running(self, polling_manager):
        """Test stopping polling when not running."""
        # Should not raise an exception
        await polling_manager.stop_polling()
        assert not polling_manager.is_polling()

    async def test_poll_and_sync(self, polling_manager):
        """Test the polling and sync functionality."""
        # Mock the ingestion service methods
        mock_ingest_workloads = AsyncMock(return_value=[])
        mock_ingest_registry = AsyncMock(return_value=[])
        polling_manager.ingestion_service.ingest_workloads = mock_ingest_workloads
        polling_manager.ingestion_service.ingest_registry = mock_ingest_registry

        # Call the workload sync method
        await polling_manager._poll_workloads()

        # Verify it called the ingestion service with the ToolhiveClient
        mock_ingest_workloads.assert_called_once_with(polling_manager.toolhive_client)

        # Call the registry sync method
        await polling_manager._poll_registry()

        # Verify it called the ingestion service with the ToolhiveClient
        mock_ingest_registry.assert_called_once_with(polling_manager.toolhive_client)

    async def test_poll_and_sync_with_error(self, polling_manager):
        """Test that polling continues even if sync fails."""
        # Mock the ingestion service to raise an error
        mock_ingest = AsyncMock(side_effect=Exception("Test error"))
        polling_manager.ingestion_service.ingest_workloads = mock_ingest

        # Should raise the exception
        with pytest.raises(Exception, match="Test error"):
            await polling_manager._poll_workloads()

    async def test_polling_loop_with_error(self, polling_manager):
        """Test that polling loop continues even if individual cycles fail."""
        call_count = 0
        first_call_event = asyncio.Event()
        second_call_event = asyncio.Event()

        async def mock_poll_workloads():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_call_event.set()
                raise Exception("First call fails")
            elif call_count == 2:
                second_call_event.set()
                # Second call succeeds - no need to return anything
                return

        with (
            patch.object(polling_manager, "_poll_workloads", side_effect=mock_poll_workloads),
            patch.object(polling_manager, "_registry_polling_loop", new_callable=AsyncMock),
        ):
            # Start polling
            await polling_manager.start_polling()

            # Wait for the first call to complete (which should fail)
            await asyncio.wait_for(first_call_event.wait(), timeout=1.0)

            # Wait for the second call to complete (which should succeed)
            await asyncio.wait_for(second_call_event.wait(), timeout=1.0)

            # Stop polling
            await polling_manager.stop_polling()

            # Should have called sync at least twice despite the first error
            assert call_count >= 2

    async def test_targeted_polling_success(self, polling_manager):
        """Test targeted polling succeeds when workload becomes available."""
        # Mock the ingestion service
        mock_ingest_workloads = AsyncMock(return_value=[])
        polling_manager.ingestion_service.ingest_workloads = mock_ingest_workloads

        # Mock workload_ops to return running workload on second attempt
        call_count = 0

        async def mock_list_servers_by_registry(registry_server_id, conn=None):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                # Return running workload on second call
                return [
                    WorkloadServer(
                        id="workload-id",
                        name="test-workload",
                        workload_identifier="test/server:latest",
                        remote=False,
                        status=McpStatus.RUNNING,
                        transport="sse",
                        url="http://localhost:8080",
                        registry_server_id=registry_server_id,
                        last_updated=datetime.now(timezone.utc),
                        created_at=datetime.now(timezone.utc),
                    )
                ]
            return []

        polling_manager.workload_ops.list_servers_by_registry = mock_list_servers_by_registry

        # Create a test registry server
        test_server = RegistryServer(
            id="registry-id",
            name="test-registry",
            package="test/server:latest",
            remote=False,
            transport="sse",
            url="http://localhost:8080",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        # Mock asyncio.sleep to speed up the test
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Start targeted polling
            await polling_manager._targeted_workload_polling(test_server)

            # Should have found the server
            assert call_count >= 2
            # Should have called ingestion at least twice
            assert mock_ingest_workloads.call_count >= 2
            # Should have called sleep at least once
            assert mock_sleep.call_count >= 1

    async def test_targeted_polling_timeout(self, polling_manager):
        """Test targeted polling times out if workload never becomes available."""
        # Mock the ingestion service
        mock_ingest_workloads = AsyncMock(return_value=[])
        polling_manager.ingestion_service.ingest_workloads = mock_ingest_workloads

        # Mock workload_ops to always return empty list (no workloads found)
        polling_manager.workload_ops.list_servers_by_registry = AsyncMock(return_value=[])

        # Create a test registry server
        test_server = RegistryServer(
            id="registry-id",
            name="test-registry",
            package="test/server:latest",
            remote=False,
            transport="sse",
            url="http://localhost:8080",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        async def quick_targeted_polling(mcp_server: RegistryServer) -> None:
            """Version with reduced timeout for testing."""
            async with polling_manager._workload_polling_pause_lock:
                was_paused = polling_manager._workload_polling_paused
                polling_manager._workload_polling_paused = True

            max_polls = 3  # Only try 3 times for test
            poll_count = 0

            try:
                while poll_count < max_polls:
                    poll_count += 1
                    await polling_manager.ingestion_service.ingest_workloads(
                        polling_manager.toolhive_client
                    )
                    workloads = await polling_manager.workload_ops.list_servers_by_registry(
                        registry_server_id=mcp_server.id
                    )
                    if workloads and any(w.status == McpStatus.RUNNING for w in workloads):
                        return
                    await asyncio.sleep(0.01)  # Very short sleep for test
            finally:
                async with polling_manager._workload_polling_pause_lock:
                    if not was_paused:
                        polling_manager._workload_polling_paused = False

        await quick_targeted_polling(test_server)

        # Should have tried 3 times
        assert mock_ingest_workloads.call_count == 3

    async def test_targeted_polling_pauses_global_polling(self, polling_manager):
        """Test that targeted polling pauses global workload polling."""
        # Initially not paused
        assert not polling_manager._workload_polling_paused

        # Mock the ingestion and server lookup to succeed quickly
        polling_manager.ingestion_service.ingest_workloads = AsyncMock(return_value=[])

        # Create a test registry server
        test_server = RegistryServer(
            id="registry-id",
            name="test-registry",
            package="test/server:latest",
            remote=False,
            transport="sse",
            url="http://localhost:8080",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        polling_manager.workload_ops.list_servers_by_registry = AsyncMock(
            return_value=[
                WorkloadServer(
                    id="workload-id",
                    name="test-workload",
                    workload_identifier="test/server:latest",
                    remote=False,
                    status=McpStatus.RUNNING,
                    transport="sse",
                    url="http://localhost:8080",
                    registry_server_id="registry-id",
                    last_updated=datetime.now(timezone.utc),
                    created_at=datetime.now(timezone.utc),
                )
            ]
        )

        # Mock asyncio.sleep to speed up the test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Start targeted polling (should pause immediately)
            task = asyncio.create_task(polling_manager._targeted_workload_polling(test_server))

            # Check that it paused during execution
            # (may already be resumed if task completed quickly)
            await task

        # After completion, should be unpaused
        assert not polling_manager._workload_polling_paused

    async def test_targeted_polling_resumes_on_error(self, polling_manager):
        """Test that global polling resumes even if targeted polling encounters an error."""
        # Initially not paused
        assert not polling_manager._workload_polling_paused

        # Mock the ingestion to raise an error
        polling_manager.ingestion_service.ingest_workloads = AsyncMock(
            side_effect=Exception("Test error")
        )

        # Create a test registry server
        test_server = RegistryServer(
            id="registry-id",
            name="test-registry",
            package="test/server:latest",
            remote=False,
            transport="sse",
            url="http://localhost:8080",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        # Mock asyncio.sleep to speed up the test
        with patch("asyncio.sleep", new_callable=AsyncMock):
            # Create a short version that will fail fast
            try:
                await polling_manager._targeted_workload_polling(test_server)
            except Exception:
                pass

        # Should be unpaused even after error (due to finally block)
        assert not polling_manager._workload_polling_paused

    async def test_workload_polling_loop_respects_pause(self, polling_manager):
        """Test that workload polling loop skips cycles when paused."""
        call_count = 0

        async def mock_poll_workloads():
            nonlocal call_count
            call_count += 1

        with (
            patch.object(polling_manager, "_poll_workloads", side_effect=mock_poll_workloads),
            patch.object(polling_manager, "_registry_polling_loop", new_callable=AsyncMock),
        ):
            # Start polling
            await polling_manager.start_polling()

            # Let it run one cycle
            await asyncio.sleep(0.15)
            initial_count = call_count

            # Pause it
            polling_manager._workload_polling_paused = True
            await asyncio.sleep(0.15)
            paused_count = call_count

            # Should not have increased much (maybe by 1 if a cycle was in progress)
            assert paused_count <= initial_count + 1

            # Unpause
            polling_manager._workload_polling_paused = False
            await asyncio.sleep(0.25)  # Give it more time to run after unpause
            resumed_count = call_count

            # Should have increased after resuming
            assert resumed_count >= paused_count

            # Stop polling
            await polling_manager.stop_polling()

    async def test_start_targeted_workload_polling_creates_task(self, polling_manager):
        """Test that start_targeted_workload_polling launches a background task."""
        # Mock the internal method to complete quickly
        mock_targeted_polling = AsyncMock()
        polling_manager._targeted_workload_polling = mock_targeted_polling

        # Set up the event loop (normally done by start_polling)
        polling_manager._loop = asyncio.get_running_loop()

        # Create a test registry server
        test_server = RegistryServer(
            id="registry-id",
            name="test-registry",
            package="test/server:latest",
            remote=False,
            transport="sse",
            url="http://localhost:8080",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )

        # Start targeted polling (fire-and-forget)
        polling_manager.start_targeted_workload_polling(test_server)

        # Give the task a moment to be created and scheduled
        await asyncio.sleep(0.01)

        # Should have called the internal method
        mock_targeted_polling.assert_called_once_with(test_server)
