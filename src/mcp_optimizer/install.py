from typing import cast

import structlog

from mcp_optimizer.db.models import McpStatus, RegistryServer, WorkloadServer
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.polling_manager import get_polling_manager
from mcp_optimizer.toolhive.api_models.registry import ImageMetadata, RemoteServerMetadata
from mcp_optimizer.toolhive.api_models.v1 import CreateRequest
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient

logger = structlog.get_logger(__name__)


def _trigger_targeted_polling(registry_server: RegistryServer) -> None:
    """Trigger targeted polling for a newly installed server.

    This is a helper function that attempts to launch targeted polling
    if a polling manager is available. It handles import and availability
    gracefully to avoid circular dependencies.

    Args:
        registry_server: Registry server to poll for
    """
    try:
        polling_manager = get_polling_manager()
        if polling_manager is not None:
            polling_manager.start_targeted_workload_polling(registry_server)
            logger.info(
                "Triggered targeted polling for new workload", server_name=registry_server.name
            )
        else:
            logger.debug(
                "Polling manager not available, skipping targeted polling",
                server_name=registry_server.name,
            )
    except Exception as e:
        logger.warning(
            "Failed to trigger targeted polling", server_name=registry_server.name, error=str(e)
        )


class McpServerInstaller:
    def __init__(
        self, toolhive_client: ToolhiveClient, workload_server_ops: WorkloadServerOps
    ) -> None:
        self.toolhive_client = toolhive_client
        self.workload_server_ops = workload_server_ops

    @classmethod
    def _needs_env_vars(
        cls, mcp_registry_metadata: ImageMetadata | RemoteServerMetadata
    ) -> list[str]:
        env_vars_needed = []
        if not mcp_registry_metadata.env_vars:
            return env_vars_needed

        for env_var in mcp_registry_metadata.env_vars:
            if env_var.required and not env_var.secret:
                env_vars_needed.append(f"{env_var.name}. {env_var.description or ''}".strip())

        return env_vars_needed

    @classmethod
    def _needs_secrets(
        cls, mcp_registry_metadata: ImageMetadata | RemoteServerMetadata
    ) -> list[str]:
        secrets_needed = []
        if not mcp_registry_metadata.env_vars:
            return secrets_needed

        for env_var in mcp_registry_metadata.env_vars:
            if env_var.secret:
                secrets_needed.append(f"{env_var.name}. {env_var.description or ''}".strip())

        return secrets_needed

    @classmethod
    def _format_help_message(cls, env_vars_needed: list[str], secrets_needed: list[str]) -> str:
        messages = []
        if env_vars_needed:
            env_vars_list = "\n  - ".join(env_vars_needed)
            messages.append(
                f"The server requires the following environment variables to be "
                f"set:\n  - {env_vars_list}"
            )

        if secrets_needed:
            secrets_list = "\n  - ".join(secrets_needed)
            messages.append(
                f"The server requires the following secrets to be set:\n  - {secrets_list}"
            )

        help_message = (
            "Please install the MCP server with the required configuration in "
            "[ToolHive UI](https://docs.stacklok.com/toolhive/guides-ui/run-mcp-servers#configure-the-server) and "  # noqa: E501
            " when done, try find_tool() to discover the tools."
        )

        return "\n\n".join(messages) + "\n\n" + help_message

    async def install(self, registry_server: RegistryServer) -> str:
        """Install the given registry server as a workload.

        Args:
            registry_server: Registry server to install

        Returns:
            Success or error message.
        """
        # Check if server already exists as a running workload
        # Try to find by package by matching the registry_server_id
        try:
            existing_workloads = cast(
                list[WorkloadServer],
                await self.workload_server_ops.list_servers(),
            )
            for workload in existing_workloads:
                if (
                    workload.registry_server_id == registry_server.id
                    and workload.status == McpStatus.RUNNING
                ):
                    return (
                        f"Server '{registry_server.name}' is already running. "
                        "Use find_tool() to discover its tools."
                    )
        except Exception as e:
            logger.warning(
                "Failed to check for existing workload",
                server_name=registry_server.name,
                error=str(e),
            )

        async with self.toolhive_client as client:
            server_metadata = await self.toolhive_client.get_server_from_registry(
                registry_server.name
            )
            if server_metadata is None:
                return (
                    f"Server '{registry_server.name}' not found in registry. Cannot install."
                    " Please check the registry with search_registry() or tools with find_tool()."
                )

        # Check if server requires env vars or secrets
        env_vars_needed = self._needs_env_vars(server_metadata)
        secrets_needed = self._needs_secrets(server_metadata)

        if env_vars_needed or secrets_needed:
            return self._format_help_message(env_vars_needed, secrets_needed)

        # Prepare minimal CreateRequest
        create_request_dict = {
            "name": registry_server.name,
            "transport": server_metadata.transport,
        }

        # Add image for container servers
        if isinstance(server_metadata, ImageMetadata):
            create_request_dict["image"] = server_metadata.image

        create_request = CreateRequest.model_validate(create_request_dict)

        # Install server using ToolhiveClient
        logger.info(
            "Installing server via ToolHive API", create_request=create_request.model_dump()
        )

        try:
            async with self.toolhive_client as client:
                result = await client.install_server(create_request)

                # Trigger targeted polling for the newly installed server
                _trigger_targeted_polling(registry_server)

                success_msg = (
                    f"Successfully installed server '{registry_server.name}'!\n"
                    f"Workload created on port {result.get('port', 'N/A')}.\n\n"
                    f"The server is being polled and will be available shortly "
                    f"(typically within 1-2 minutes).\n"
                    f"Use find_tool() to discover the server's tools once available."
                )

                logger.info(
                    "Server installation successful",
                    server_name=registry_server.name,
                    port=result.get("port"),
                )

                return success_msg
        except Exception as e:
            error_msg = f"Failed to install server '{registry_server.name}': {str(e)}"
            logger.error(
                "Server installation failed", server_name=registry_server.name, error=str(e)
            )
            return error_msg
