"""
MCP client for connecting to and listing tools from MCP servers.
"""

import asyncio
from typing import Any, Awaitable, Callable

import structlog
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ListToolsResult

from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.enums import ToolHiveProxyMode, url_to_toolhive_proxy_mode

logger = structlog.get_logger(__name__)


class WorkloadConnectionError(Exception):
    """Custom exception for workload-related errors."""

    pass


class MCPServerClient:
    """Client for connecting to individual MCP servers."""

    def __init__(self, workload: Workload, timeout: float):
        """
        Initialize MCP client for a specific workload.

        Args:
            workload: The workload (MCP server) to connect to
            timeout: Timeout in seconds for operations (default: from config)
        """
        self.workload = workload
        self.timeout = timeout

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

    def _determine_proxy_mode(self) -> ToolHiveProxyMode:
        """
        Determine the proxy mode from workload configuration.

        Returns:
            ToolHiveProxyMode: The proxy mode to use

        Raises:
            WorkloadConnectionError: If proxy mode is unknown or not set
        """
        if self.workload.proxy_mode:
            proxy_mode_lower = self.workload.proxy_mode.lower()
            logger.debug(
                f"Determining proxy mode from proxy_mode field: {proxy_mode_lower}",
                workload=self.workload.name,
            )
            if proxy_mode_lower == "streamable-http":
                return ToolHiveProxyMode.STREAMABLE
            elif proxy_mode_lower == "sse":
                return ToolHiveProxyMode.SSE
            else:
                logger.warning(
                    f"Unknown proxy_mode '{proxy_mode_lower}', falling back to URL detection",
                    workload=self.workload.name,
                )

        # Fallback to URL-based detection for backwards compatibility
        logger.debug(
            "No proxy_mode available, falling back to URL-based detection",
            workload=self.workload.name,
        )
        return url_to_toolhive_proxy_mode(self.workload.url)

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

        # Determine proxy mode and prepare URL
        proxy_mode = self._determine_proxy_mode()

        logger.info(
            f"Using {proxy_mode} client for workload",
            workload=self.workload.name,
            proxy_mode_field=self.workload.proxy_mode,
        )

        try:
            if proxy_mode == ToolHiveProxyMode.STREAMABLE:
                return await asyncio.wait_for(
                    self._execute_streamable_session(operation), timeout=self.timeout
                )
            elif proxy_mode == ToolHiveProxyMode.SSE:
                return await asyncio.wait_for(
                    self._execute_sse_session(operation), timeout=self.timeout
                )
            else:
                logger.error(f"Unsupported transport type: {proxy_mode}", workload=self.workload)
                raise WorkloadConnectionError(f"Unsupported transport type: {proxy_mode}")
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
        self, operation: Callable[[ClientSession], Awaitable]
    ) -> Any:
        """Execute operation with streamable HTTP session."""
        async with streamablehttp_client(self.workload.url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await operation(session)

    async def _execute_sse_session(self, operation: Callable[[ClientSession], Awaitable]) -> Any:
        """Execute operation with SSE session."""
        async with sse_client(self.workload.url) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                return await operation(session)

    async def list_tools(self) -> ListToolsResult:
        """
        List available tools from the MCP server.

        Returns:
            ListToolsResult: Available tools from the MCP server
        """
        return await self._execute_with_session(lambda session: session.list_tools())

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:
        """
        Call a specific tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            CallToolResult: Result of the tool execution
        """
        return await self._execute_with_session(
            lambda session: session.call_tool(tool_name, arguments)
        )
