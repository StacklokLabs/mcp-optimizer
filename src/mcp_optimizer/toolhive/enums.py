from enum import Enum


class ToolHiveProxyMode(str, Enum):
    """
    Enum for ToolHive proxy modes.
    Proxy modes is the MCP transport type in which ToolHive's proxy operates for a workload.
    """

    STREAMABLE = "streamable-http"
    SSE = "sse"

    def __str__(self) -> str:
        """Return the string representation of the transport type."""
        return self.value


class ToolHiveWorkloadStatus(str, Enum):
    """Enum for ToolHive workload statuses."""

    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    STARTING = "starting"
    STOPPING = "stopping"
    UNHEALTHY = "unhealthy"
    REMOVING = "removing"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        """Return the string representation of the status."""
        return self.value


def url_to_toolhive_proxy_mode(toolhive_url: str) -> ToolHiveProxyMode:
    """Map ToolHive URL to ToolHiveProxyMode enum.

    Args:
        toolhive_url: ToolHive URL

    Returns:
        Corresponding ToolHiveProxyMode enum value

    Raises:
        IngestionError: If URL is not supported
    """
    if "/mcp" in toolhive_url:
        return ToolHiveProxyMode.STREAMABLE
    elif "/sse" in toolhive_url:
        return ToolHiveProxyMode.SSE
    else:
        raise ValueError(f"Unsupported ToolHive URL: {toolhive_url}")
