"""
MCP client for connecting to and listing tools from MCP servers.
"""

import asyncio
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse, urlunparse

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

    def _normalize_url(self, url: str, proxy_mode: ToolHiveProxyMode) -> str:
        """
        Normalize URL for the given proxy mode.

        For streamable-http:
        - Fragments must be stripped as they're not supported
        - Path must be /mcp (not /sse) as streamable-http uses /mcp endpoint
        For SSE, fragments are preserved as they're used for container identification.

        Args:
            url: Original URL from ToolHive
            proxy_mode: The proxy mode being used

        Returns:
            Normalized URL without fragments and with correct path for streamable-http,
            original URL for SSE
        """
        if proxy_mode == ToolHiveProxyMode.STREAMABLE:
            # Strip fragments for streamable-http
            # (fragments not supported by streamable-http client)
            parsed = urlparse(url)

            # Fix path: streamable-http uses /mcp endpoint, not /sse
            path = parsed.path
            if path.endswith("/sse"):
                path = path.replace("/sse", "/mcp")
            elif not path.endswith("/mcp"):
                # If path doesn't end with /mcp or /sse, ensure it ends with /mcp
                if path.endswith("/"):
                    path = path + "mcp"
                else:
                    path = path + "/mcp"

            # Reconstruct URL without fragment and with corrected path
            normalized_tuple = (
                parsed.scheme,
                parsed.netloc,
                path,
                parsed.params,
                parsed.query,
                "",  # Empty fragment
            )
            normalized = str(urlunparse(normalized_tuple))
            if normalized != url:
                logger.debug(
                    "Normalized URL for streamable-http",
                    original_url=url,
                    normalized_url=normalized,
                    workload=self.workload.name,
                )
            return normalized
        else:
            # SSE preserves fragments (used for container identification)
            return url

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

        # Determine proxy mode and normalize URL
        proxy_mode = self._determine_proxy_mode()
        normalized_url = self._normalize_url(self.workload.url, proxy_mode)

        logger.info(
            f"Using {proxy_mode} client for workload '{self.workload.name}'",
            workload=self.workload.name,
            proxy_mode_field=self.workload.proxy_mode,
            original_url=self.workload.url,
            normalized_url=normalized_url,
        )

        try:
            if proxy_mode == ToolHiveProxyMode.STREAMABLE:
                return await asyncio.wait_for(
                    self._execute_streamable_session(operation, normalized_url),
                    timeout=self.timeout,
                )
            elif proxy_mode == ToolHiveProxyMode.SSE:
                return await asyncio.wait_for(
                    self._execute_sse_session(operation, normalized_url), timeout=self.timeout
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
        self, operation: Callable[[ClientSession], Awaitable], url: str
    ) -> Any:
        """Execute operation with streamable HTTP session."""
        logger.debug(
            f"Establishing streamable HTTP session for workload '{self.workload.name}'",
            workload=self.workload.name,
            url=url,
        )
        async with streamablehttp_client(url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                logger.info(
                    f"Initializing MCP session for workload '{self.workload.name}'",
                    workload=self.workload.name,
                )
                await session.initialize()
                logger.debug(
                    f"MCP session initialized successfully for workload '{self.workload.name}'",
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
                    f"Initializing MCP session for workload '{self.workload.name}'",
                    workload=self.workload.name,
                )
                await session.initialize()
                logger.debug(
                    f"MCP session initialized successfully for workload '{self.workload.name}'",
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
