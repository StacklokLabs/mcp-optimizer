from enum import Enum


class ToolHiveTransportMode(str, Enum):
    """
    Enum for ToolHive proxy modes.
    Proxy modes represent the MCP transport type in which ToolHive's proxy operates for a workload.
    """

    STREAMABLE = "streamable-http"
    SSE = "sse"

    def __str__(self) -> str:
        """Return the string representation of the proxy mode."""
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


def url_to_toolhive_transport_mode(toolhive_url: str) -> ToolHiveTransportMode:
    """Map ToolHive URL to ToolHiveTransportMode enum.

    Args:
        toolhive_url: ToolHive URL

    Returns:
        Corresponding ToolHiveProxyMode enum value

    Raises:
        IngestionError: If URL is not supported
    """
    if "/mcp" in toolhive_url:
        return ToolHiveTransportMode.STREAMABLE
    elif "/sse" in toolhive_url:
        return ToolHiveTransportMode.SSE
    else:
        raise ValueError(f"Unsupported ToolHive URL: {toolhive_url}")
