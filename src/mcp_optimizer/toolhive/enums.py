from enum import Enum


class ToolHiveTransportType(str, Enum):
    """
    Enum for ToolHive transport types.
    Transport types represent the MCP transport type in which ToolHive's operates for a workload.
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


def url_to_toolhive_transport_type(toolhive_url: str) -> ToolHiveTransportType:
    """Map ToolHive URL to ToolHiveTransportType enum.

    Args:
        toolhive_url: ToolHive URL

    Returns:
        Corresponding ToolHiveTransportType enum value

    Raises:
        IngestionError: If URL is not supported
    """
    if "/mcp" in toolhive_url:
        return ToolHiveTransportType.STREAMABLE
    elif "/sse" in toolhive_url:
        return ToolHiveTransportType.SSE
    else:
        raise ValueError(f"Unsupported ToolHive URL: {toolhive_url}")
