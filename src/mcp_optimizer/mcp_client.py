"""
MCP client for connecting to and listing tools from MCP servers.
"""

import asyncio
from typing import Any, Awaitable, Callable

import structlog
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ListToolsResult

from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.enums import ToolHiveTransportMode, url_to_toolhive_transport_mode

logger = structlog.get_logger(__name__)


class WorkloadConnectionError(Exception):
    """Custom exception for workload-related errors."""

    pass


def determine_transport_type(workload: Workload, runtime_mode: str) -> ToolHiveTransportMode:
    """
    Determine the transport type from workload configuration based on runtime mode.

    Depending on the runtime mode, the transport type is determined differently:
    - In docker mode: determined from proxy_mode (how the proxy connects to the container)
    - In k8s mode: determined from transport_type (the direct connection type to the pod)

    Args:
        workload: Workload configuration containing transport information
        runtime_mode: Runtime environment - "docker" or "k8s"

    Returns:
        ToolHiveProxyMode: The transport type to use (SSE or STREAMABLE)

    Raises:
        WorkloadConnectionError: If transport type cannot be determined
    """
    # Docker mode: Check proxy_mode field (proxy connection type)
    if runtime_mode == "docker":
        if workload.proxy_mode:
            transport_field_lower = workload.proxy_mode.lower()
            logger.debug(
                f"Docker mode: determining transport from proxy_mode field: "
                f"{transport_field_lower}",
                workload=workload.name,
                runtime_mode=runtime_mode,
            )
            if transport_field_lower == "streamable-http":
                return ToolHiveTransportMode.STREAMABLE
            elif transport_field_lower == "sse":
                return ToolHiveTransportMode.SSE
            else:
                logger.warning(
                    f"Unknown transport in proxy_mode: '{transport_field_lower}', "
                    "falling back to URL detection",
                    workload=workload.name,
                )

    # K8s mode: Check transport_type field (direct connection type)
    elif runtime_mode == "k8s":
        if workload.transport_type:
            transport_field_lower = workload.transport_type.lower()
            logger.debug(
                f"K8s mode: determining transport from transport_type field: "
                f"{transport_field_lower}",
                workload=workload.name,
                runtime_mode=runtime_mode,
            )
            if transport_field_lower == "streamable-http":
                return ToolHiveTransportMode.STREAMABLE
            elif transport_field_lower == "sse":
                return ToolHiveTransportMode.SSE
            else:
                logger.warning(
                    f"Unknown transport in transport_type: '{transport_field_lower}', "
                    "falling back to URL detection",
                    workload=workload.name,
                )

    # Fallback to URL-based detection for backwards compatibility
    if not workload.url:
        raise WorkloadConnectionError(
            f"No transport type or URL available. Workload: {workload.name}",
        )

    logger.debug(
        "No transport field available, falling back to URL-based detection",
        workload=workload.name,
        runtime_mode=runtime_mode,
    )
    return url_to_toolhive_transport_mode(workload.url)


class MCPServerClient:
    """Client for connecting to individual MCP servers."""

    def __init__(self, workload: Workload, timeout: float, runtime_mode: str = "docker"):
        """
        Initialize MCP client for a specific workload.

        Args:
            workload: The workload (MCP server) to connect to
            timeout: Timeout in seconds for operations (default: from config)
            runtime_mode: Runtime environment - "docker" or "k8s" (default: "docker")
        """
        self.workload = workload
        self.timeout = timeout
        self.runtime_mode = runtime_mode

    def _extract_error_from_exception_group(self, eg: ExceptionGroup) -> str:
        """
        Extract meaningful error message from ExceptionGroup.

        Args:
            eg: ExceptionGroup to extract error from

        Returns:
            Error message string
        """
        # Flatten the exception group to get all underlying exceptions
        exceptions = []

        def collect_exceptions(exc_group):
            for exc in exc_group.exceptions:
                if isinstance(exc, ExceptionGroup):
                    collect_exceptions(exc)
                else:
                    exceptions.append(exc)

        collect_exceptions(eg)

        # Look for McpError first, as it's the most specific
        for exc in exceptions:
            if isinstance(exc, McpError):
                return f"MCP protocol error: {exc}"

        # If no McpError found, return the first exception message
        if exceptions:
            first_exc = exceptions[0]
            return f"{type(first_exc).__name__}: {first_exc}"

        # Fallback to the exception group message
        return str(eg)

    async def _execute_with_session(self, operation: Callable[[ClientSession], Awaitable]) -> Any:
        """
        Execute an operation with an MCP session.

        Args:
            operation: Async function that takes a ClientSession and returns a result

        Returns:
            The result of the operation
        """
        if not self.workload.url:
            raise WorkloadConnectionError("Workload has no URL")

        logger.debug(f"Workload URL: {self.workload.url}")

        # Determine transport type based on runtime mode
        # Docker: uses proxy_mode field (how proxy connects to container)
        # K8s: uses transport_type field (direct connection to pod)
        transport_type = determine_transport_type(self.workload, self.runtime_mode)

        logger.info(
            f"Using {transport_type} client for workload '{self.workload.name}'",
            workload=self.workload.name,
            transport_type_field=self.workload.transport_type,
            proxy_mode_field=self.workload.proxy_mode,
            url=self.workload.url,
        )

        try:
            if transport_type == ToolHiveTransportMode.STREAMABLE:
                return await asyncio.wait_for(
                    self._execute_streamable_session(operation, self.workload.url),
                    timeout=self.timeout,
                )
            elif transport_type == ToolHiveTransportMode.SSE:
                return await asyncio.wait_for(
                    self._execute_sse_session(operation, self.workload.url), timeout=self.timeout
                )
            else:
                logger.error(
                    f"Unsupported transport type: {transport_type}", workload=self.workload
                )
                raise WorkloadConnectionError(f"Unsupported transport type: {transport_type}")
        except asyncio.TimeoutError as e:
            logger.error(
                f"Operation timed out after {self.timeout} seconds", workload=self.workload
            )
            raise WorkloadConnectionError(
                f"Operation timed out after {self.timeout} seconds"
            ) from e
        except ExceptionGroup as eg:
            # Python 3.13+ wraps exceptions from TaskGroups in ExceptionGroup
            # Extract the underlying error and convert to WorkloadConnectionError
            error_msg = self._extract_error_from_exception_group(eg)
            logger.error(
                "MCP session error",
                workload=self.workload,
                error=error_msg,
            )
            raise WorkloadConnectionError(error_msg) from eg
        except McpError as e:
            # Handle direct McpError exceptions
            logger.error(
                "MCP protocol error",
                workload=self.workload,
                error=str(e),
            )
            raise WorkloadConnectionError(f"MCP protocol error: {e}") from e

    async def _execute_streamable_session(
        self, operation: Callable[[ClientSession], Awaitable], url: str
    ) -> Any:
        """Execute operation with streamable HTTP session."""
        logger.debug(
            f"Establishing streamable HTTP session for workload '{self.workload.name}'",
            workload=self.workload.name,
            url=url,
        )
        async with streamable_http_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                logger.info(
                    "Initializing Streamable MCP session for workload",
                    workload=self.workload.name,
                )
                await session.initialize()
                logger.debug(
                    "Streamable MCP session initialized successfully",
                    workload=self.workload.name,
                )
                return await operation(session)

    async def _execute_sse_session(
        self, operation: Callable[[ClientSession], Awaitable], url: str
    ) -> Any:
        """Execute operation with SSE session."""
        logger.debug(
            f"Establishing SSE session for workload '{self.workload.name}'",
            workload=self.workload.name,
            url=url,
        )
        async with sse_client(url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                logger.info(
                    "Initializing SSE MCP session for workload",
                    workload=self.workload.name,
                )
                await session.initialize()
                logger.debug(
                    "SSE MCP session initialized successfully for workload",
                    workload=self.workload.name,
                )
                return await operation(session)

    async def list_tools(self) -> ListToolsResult:
        """
        List available tools from the MCP server.

        Returns:
            ListToolsResult: Available tools from the MCP server
        """
        logger.debug(
            f"Listing tools for workload '{self.workload.name}'", workload=self.workload.name
        )
        result = await self._execute_with_session(lambda session: session.list_tools())
        logger.debug(
            f"Retrieved {len(result.tools)} tools from workload '{self.workload.name}'",
            workload=self.workload.name,
            tool_count=len(result.tools),
        )
        return result

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Call a specific tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            CallToolResult: Result of the tool execution
        """
        logger.debug(
            f"Calling tool '{tool_name}' on workload '{self.workload.name}'",
            workload=self.workload.name,
            tool_name=tool_name,
        )
        result = await self._execute_with_session(
            lambda session: session.call_tool(tool_name, arguments)
        )
        logger.debug(
            f"Tool '{tool_name}' call completed on workload '{self.workload.name}'",
            workload=self.workload.name,
            tool_name=tool_name,
        )
        return result
