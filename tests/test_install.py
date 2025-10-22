"""Tests for the McpServerInstaller and targeted polling integration."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest

from mcp_optimizer.db.models import RegistryServer
from mcp_optimizer.install import McpServerInstaller, _trigger_targeted_polling
from mcp_optimizer.toolhive.api_models.registry import ImageMetadata
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient


def create_test_registry_server(
    name: str = "test-server",
    package: str = "test/server",
    url: str | None = None,
    remote: bool = False,
) -> RegistryServer:
    """Helper to create a test registry server with all required fields."""
    return RegistryServer(
        id="test-id",
        name=name,
        transport="sse",
        url=url,
        package=package,
        remote=remote,
        last_updated=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        group="default",
    )


@pytest.fixture
def mock_toolhive_client():
    """Create a mock ToolhiveClient."""
    client = Mock(spec=ToolhiveClient)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_workload_server_ops():
    """Create a mock WorkloadServerOps."""
    ops = Mock()
    ops.list_servers = AsyncMock(return_value=[])
    return ops


@pytest.fixture
def installer(mock_toolhive_client, mock_workload_server_ops):
    """Create an installer with a mock client."""
    return McpServerInstaller(
        toolhive_client=mock_toolhive_client, workload_server_ops=mock_workload_server_ops
    )


class TestTriggerTargetedPolling:
    """Test cases for _trigger_targeted_polling helper function."""

    def test_trigger_with_available_polling_manager(self):
        """Test triggering targeted polling when polling manager is available."""
        # Mock the polling manager
        mock_polling_manager = Mock()
        mock_polling_manager.start_targeted_workload_polling = Mock()

        test_server = create_test_registry_server()

        with patch("mcp_optimizer.install.get_polling_manager", return_value=mock_polling_manager):
            _trigger_targeted_polling(test_server)

        # Should have called start_targeted_workload_polling
        mock_polling_manager.start_targeted_workload_polling.assert_called_once_with(test_server)

    def test_trigger_with_no_polling_manager(self):
        """Test triggering targeted polling when polling manager is not available."""
        test_server = create_test_registry_server()
        with patch("mcp_optimizer.install.get_polling_manager", return_value=None):
            # Should not raise an exception
            _trigger_targeted_polling(test_server)

    def test_trigger_handles_exceptions(self):
        """Test that exceptions in targeted polling trigger are handled gracefully."""
        test_server = create_test_registry_server()
        with patch(
            "mcp_optimizer.install.get_polling_manager", side_effect=Exception("Test error")
        ):
            # Should not raise an exception
            _trigger_targeted_polling(test_server)


class TestMcpServerInstaller:
    """Test cases for McpServerInstaller."""

    async def test_install_already_running_server(self, installer, mock_workload_server_ops):
        """Test installing a server that is already running."""
        from mcp_optimizer.db.models import McpStatus, WorkloadServer

        server = create_test_registry_server()

        # Mock an existing workload with RUNNING status
        existing_workload = WorkloadServer(
            id="workload-id",
            name="test-workload",
            url="http://localhost:8080",
            workload_identifier="test/server",
            remote=False,
            transport="sse",
            status=McpStatus.RUNNING,
            registry_server_id=server.id,
            group="default",
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        mock_workload_server_ops.list_servers = AsyncMock(return_value=[existing_workload])

        result = await installer.install(server)

        assert "already running" in result.lower()
        assert "find_tool()" in result

    async def test_install_server_not_found_in_registry(self, installer, mock_toolhive_client):
        """Test installing a server that exists in DB but not in registry."""
        server = create_test_registry_server()

        # Mock get_server_from_registry to return None
        mock_toolhive_client.get_server_from_registry = AsyncMock(return_value=None)

        result = await installer.install(server)

        assert "not found in registry" in result.lower()

    async def test_install_server_requires_env_vars(self, installer, mock_toolhive_client):
        """Test installing a server that requires environment variables."""
        server = create_test_registry_server()

        # Mock registry metadata with required env vars
        from mcp_optimizer.toolhive.api_models.registry import EnvVar

        metadata = ImageMetadata(
            name="test-server",
            image="test/image:latest",
            transport="sse",
            env_vars=[
                EnvVar(name="API_KEY", required=True, secret=False, description="API key required")
            ],
        )

        mock_toolhive_client.get_server_from_registry = AsyncMock(return_value=metadata)

        result = await installer.install(server)

        assert "environment variables" in result.lower()
        assert "API_KEY" in result

    async def test_install_server_requires_secrets(self, installer, mock_toolhive_client):
        """Test installing a server that requires secrets."""
        server = create_test_registry_server()

        # Mock registry metadata with required secrets
        from mcp_optimizer.toolhive.api_models.registry import EnvVar

        metadata = ImageMetadata(
            name="test-server",
            image="test/image:latest",
            transport="sse",
            env_vars=[
                EnvVar(name="SECRET_TOKEN", required=True, secret=True, description="Secret token")
            ],
        )

        mock_toolhive_client.get_server_from_registry = AsyncMock(return_value=metadata)

        result = await installer.install(server)

        assert "secrets" in result.lower()
        assert "SECRET_TOKEN" in result

    async def test_install_server_success(self, installer, mock_toolhive_client):
        """Test successful server installation."""
        server = create_test_registry_server()

        # Mock registry metadata without env vars
        metadata = ImageMetadata(
            name="test-server", image="test/image:latest", transport="sse", env_vars=None
        )

        mock_toolhive_client.get_server_from_registry = AsyncMock(return_value=metadata)
        mock_toolhive_client.install_server = AsyncMock(return_value={"port": 8080})

        # Mock the targeted polling trigger
        with patch("mcp_optimizer.install._trigger_targeted_polling") as mock_trigger:
            result = await installer.install(server)

            # Should have triggered targeted polling with the server object
            mock_trigger.assert_called_once_with(server)

        assert "successfully installed" in result.lower()
        assert "8080" in result
        assert "being polled" in result.lower()
        assert "find_tool()" in result

    async def test_install_server_failure(self, installer, mock_toolhive_client):
        """Test server installation failure."""
        server = create_test_registry_server()

        # Mock registry metadata
        metadata = ImageMetadata(
            name="test-server", image="test/image:latest", transport="sse", env_vars=None
        )

        mock_toolhive_client.get_server_from_registry = AsyncMock(return_value=metadata)
        mock_toolhive_client.install_server = AsyncMock(
            side_effect=Exception("Installation failed")
        )

        result = await installer.install(server)

        assert "failed to install" in result.lower()
        assert "installation failed" in result.lower()
