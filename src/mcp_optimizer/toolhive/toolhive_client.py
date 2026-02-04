"""
Toolhive API client for discovering and managing MCP server workloads.
"""

import asyncio
import os
import time
from functools import wraps
from typing import Any, Awaitable, Callable, Self, TypeVar, cast
from urllib.parse import urlparse

import httpx
import structlog
from semver import Version

from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.api_models.registry import ImageMetadata, Registry, RemoteServerMetadata
from mcp_optimizer.toolhive.api_models.v1 import (
    CreateRequest,
    GetRegistryResponse,
    GetServerResponse,
    WorkloadListResponse,
)

logger = structlog.get_logger(__name__)

T = TypeVar("T")


def _is_running_in_docker() -> bool:
    """Check if we're running inside a Docker container.

    Checks the RUNNING_IN_DOCKER environment variable (set in Dockerfile).

    Returns:
        True if running in Docker, False otherwise
    """
    return os.getenv("RUNNING_IN_DOCKER") == "1"


class ToolhiveConnectionError(Exception):
    """Exception raised when unable to connect to ToolHive after all retries."""

    pass


class ToolhiveScanError(Exception):
    """Exception raised when unable to find ToolHive in the specified port range."""

    pass


class ToolhiveClient:
    """Client for interacting with the Toolhive API."""

    def __init__(
        self,
        host: str,
        port: int | None,
        scan_port_start: int,
        scan_port_end: int,
        timeout: int,
        max_retries: int,
        initial_backoff: float,
        max_backoff: float,
        skip_port_discovery: bool = False,
        skip_backoff: bool = False,
    ):
        """
        Initialize the Toolhive client.

        Args:
            host: Toolhive server host
            port: Toolhive server port
            scan_port_start: Start of port range to scan for Toolhive
            scan_port_end: End of port range to scan for Toolhive
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts on connection failure
            initial_backoff: Initial backoff delay in seconds
            max_backoff: Maximum backoff delay in seconds
            skip_port_discovery: Skip port scanning
                (useful when ToolHive is not needed, e.g., K8s mode)
            skip_backoff: Skip backoff period between discovery retries
                (useful for tests/CI where faster retries are needed)
        """
        self.thv_host = host
        self.timeout = timeout
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.scan_port_start = scan_port_start
        self.scan_port_end = scan_port_end
        self.skip_port_discovery = skip_port_discovery
        self.skip_backoff = skip_backoff
        self._client: httpx.AsyncClient | None = None
        self.thv_port = None
        self._initial_port = port  # Store the initial port for rediscovery
        self._rediscovery_lock: asyncio.Lock | None = None  # Lazy initialization for async context
        self._discovery_lock: asyncio.Lock | None = None  # Lazy initialization for async context
        self._discovery_attempted = False  # Track if we've attempted discovery
        self._discovery_failed = False  # Track if discovery failed
        # Timestamp of last discovery attempt
        self._last_discovery_attempt_time: float | None = None
        # 3 minutes backoff between discovery attempts
        self._discovery_backoff_seconds = 180

        # Skip port discovery if requested (e.g., when running in K8s mode)
        if skip_port_discovery:
            logger.info(
                "Skipping ToolHive port discovery (not needed in current runtime mode)",
                host=host,
            )
            self.base_url = None
            self._discovery_attempted = True
            return

        # Don't discover port at initialization - defer until needed
        # base_url will be set when connection is established
        self.base_url = None
        logger.info(
            "ToolhiveClient initialized (port discovery deferred)",
            host=host,
            port=port,
        )

    async def _discover_port_async(self, port: int | None = None) -> None:
        """
        Async version: Discover the ToolHive port.

        Sets discovery state tracking (_last_discovery_attempt_time, _discovery_failed)
        and handles logging for success/failure cases.

        Args:
            port: Optional specific port to try first

        Raises:
            ToolhiveScanError: If port discovery fails
            ConnectionError: If ToolHive is not found after scanning
        """
        # Set attempt timestamp
        self._last_discovery_attempt_time = time.time()

        try:
            if port is not None:
                for attempt in range(3):
                    try:
                        _, port = await self._is_toolhive_available(self.thv_host, port)
                        self.thv_port = port
                        break
                    except ToolhiveScanError:
                        logger.warning(
                            "ToolHive not available at specified host/port, retrying...",
                            host=self.thv_host,
                            port=port,
                            attempt=attempt + 1,
                        )
                        await asyncio.sleep(1)
            # If port is not found yet (either not specified or retries failed),
            # try scanning the port range
            if self.thv_port is None:
                # Scan for ToolHive in the port range
                self.thv_port = await self._scan_for_toolhive(
                    self.thv_host, self.scan_port_start, self.scan_port_end
                )

            # Success: set base_url and update state
            self.base_url = f"http://{self.thv_host}:{self.thv_port}"
            self._discovery_failed = False
            self._last_discovery_attempt_time = None  # Reset on success
            logger.info(
                "Successfully connected to ToolHive",
                host=self.thv_host,
                port=self.thv_port,
            )
        except Exception as e:
            # Failure: update state and log
            self._discovery_failed = True
            logger.warning(
                "Failed to discover ToolHive port",
                host=self.thv_host,
                port=port,
                error=str(e),
                error_type=type(e).__name__,
                next_retry_in_seconds=self._discovery_backoff_seconds,
            )
            raise

    def _discover_port(self, port: int | None = None) -> None:
        """
        Discover the ToolHive port synchronously.
        Detects if there's a running event loop and executes appropriately.

        Args:
            port: Optional specific port to try first
        """
        try:
            # Try to get the running event loop
            asyncio.get_running_loop()
            # If we get here, there's a running loop - we need to create a task
            # This shouldn't happen in __init__, but we'll handle it gracefully
            raise RuntimeError(
                "_discover_port called from async context. Use _discover_port_async instead."
            )
        except RuntimeError:
            # No running loop - safe to use asyncio.run()
            asyncio.run(self._discover_port_async(port))

    def is_connected(self) -> bool:
        """Check if the client has discovered a port and is ready to connect.

        Returns:
            True if port has been discovered, False otherwise
        """
        return self.thv_port is not None and self.base_url is not None

    async def ensure_connected(self) -> None:
        """
        Ensure the client has discovered the ToolHive port.
        This method performs lazy port discovery if not already done.
        Will wait for backoff period if a previous discovery attempt failed,
        then retry discovery to avoid rapid retries.

        Raises:
            ToolhiveScanError: If port discovery fails after waiting for backoff
            ConnectionError: If ToolHive is not found after scanning
        """
        # Skip if port discovery is disabled
        if self.skip_port_discovery:
            return

        # If already connected, return early
        if self.is_connected():
            return

        # Lazy initialize the lock
        if self._discovery_lock is None:
            self._discovery_lock = asyncio.Lock()

        # Use lock to prevent concurrent discovery attempts
        async with self._discovery_lock:
            # Check again after acquiring lock (another coroutine might have discovered it)
            if self.is_connected():
                return

            # Check if we're in backoff period after a failed discovery attempt
            # If so, wait for the backoff period to expire before attempting again
            # Skip backoff if skip_backoff flag is set (useful for tests/CI)
            if (
                not self.skip_backoff
                and self._discovery_failed
                and self._last_discovery_attempt_time is not None
            ):
                time_since_last_attempt = time.time() - self._last_discovery_attempt_time
                if time_since_last_attempt < self._discovery_backoff_seconds:
                    remaining_backoff = self._discovery_backoff_seconds - time_since_last_attempt
                    logger.debug(
                        "Waiting for backoff period before retrying discovery",
                        remaining_seconds=remaining_backoff,
                        backoff_period=self._discovery_backoff_seconds,
                        host=self.thv_host,
                    )
                    await asyncio.sleep(remaining_backoff)

            # Attempt discovery
            self._discovery_attempted = True
            await self._discover_port_async(self._initial_port)

    def _parse_toolhive_version(self, version_str: str) -> Version:
        """Parse ToolHive version string into a Version object.

        For development/build versions that don't follow SemVer, returns a default version.
        """
        try:
            version = Version.parse(version_str.replace("v", ""))
            return version
        except (ValueError, TypeError):
            # For development builds like "build-7c3a3077", use a default version
            # This allows the connection to succeed while still logging the issue
            logger.info(
                "Using default version for non-semver ToolHive build",
                version=version_str,
                default_version="0.0.0-dev",
            )
            return Version.parse("0.0.0-dev")

    async def _is_toolhive_available(self, host: str, port: int) -> tuple[Version, int]:
        """
        Check if ToolHive is available at the given host and port.

        Returns:
            Tuple of (Version object, port) if available, raises ToolhiveScanError otherwise
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"http://{host}:{port}/api/v1beta/version")
                response.raise_for_status()

                # Validate that the response is actually from ToolHive
                try:
                    data = response.json()
                    if not isinstance(data, dict) or "version" not in data:
                        logger.debug(
                            "Port responded but not with ToolHive format",
                            host=host,
                            port=port,
                            response=data,
                        )
                        raise ToolhiveScanError(
                            f"Port {port} on host {host} did not respond with ToolHive format"
                        )
                    parsed_version = self._parse_toolhive_version(data["version"])
                    logger.info(
                        "Found ToolHive instance", host=host, port=port, version=str(parsed_version)
                    )
                    return parsed_version, port
                except (ValueError, KeyError, ToolhiveScanError) as e:
                    logger.debug("Port responded but could not parse JSON", host=host, port=port)
                    raise ToolhiveScanError(
                        f"Port {port} on host {host} did not respond with valid JSON"
                    ) from e
        except (httpx.HTTPError, OSError) as e:
            logger.debug("Error checking ToolHive availability", host=host, port=port, error=str(e))
            raise ToolhiveScanError(f"Error checking ToolHive availability on {host}:{port}") from e

    async def _scan_for_toolhive(self, host: str, scan_port_start: int, scan_port_end: int) -> int:
        """Async version: Scan for ToolHive in the specified port range."""
        logger.info(
            "Scanning for ToolHive", host=host, port_range=f"{scan_port_start}-{scan_port_end}"
        )

        # Calculate total timeout: timeout per port * number of ports, with a max of 30 seconds
        num_ports = scan_port_end - scan_port_start + 1
        total_timeout = min(self.timeout * num_ports, 30.0)

        try:
            task_outcomes = await asyncio.wait_for(
                asyncio.gather(
                    *[
                        self._is_toolhive_available(host, port)
                        for port in range(scan_port_start, scan_port_end + 1)
                    ],
                    return_exceptions=True,
                ),
                timeout=total_timeout,
            )
        except asyncio.TimeoutError as e:
            logger.warning(
                "Port scan timed out",
                host=host,
                port_range=f"{scan_port_start}-{scan_port_end}",
                timeout=total_timeout,
            )
            raise ConnectionError(
                f"ToolHive port scan timed out after {total_timeout}s "
                f"on {host} in port range {scan_port_start}-{scan_port_end}"
            ) from e

        thv_version_port = [
            version_port
            for version_port in task_outcomes
            if not isinstance(version_port, BaseException)
        ]

        thv_version_port.sort(key=lambda x: x[0], reverse=True)

        if thv_version_port:
            return thv_version_port[0][1]

        # If no port found, raise an error
        raise ConnectionError(
            f"ToolHive not found on {host} in port range {scan_port_start}-{scan_port_end}"
        )

    async def _rediscover_port(self) -> bool:
        """
        Attempt to rediscover the ToolHive port after a connection failure.
        Uses a lock to prevent race conditions from concurrent rediscovery attempts.

        Returns:
            True if port was successfully rediscovered, False otherwise
        """
        # Lazy initialize the lock
        if self._rediscovery_lock is None:
            self._rediscovery_lock = asyncio.Lock()

        async with self._rediscovery_lock:
            logger.warning(
                "Attempting to rediscover ToolHive port",
                previous_port=self.thv_port,
                host=self.thv_host,
            )

            old_port = self.thv_port
            self.thv_port = None

            try:
                # Try the initial port first using async version
                await self._discover_port_async(self._initial_port)

                if self.thv_port and self.thv_port != old_port:
                    logger.info(
                        "Successfully rediscovered ToolHive on new port",
                        old_port=old_port,
                        new_port=self.thv_port,
                    )
                    return True
                elif self.thv_port:
                    logger.info(
                        "ToolHive still available on same port",
                        port=self.thv_port,
                    )
                    return True
                else:
                    logger.error("Failed to rediscover ToolHive port")
                    return False
            except Exception as e:
                logger.error("Error during port rediscovery", error=str(e))
                # Restore old port
                self.thv_port = old_port
                if self.thv_port:
                    self.base_url = f"http://{self.thv_host}:{self.thv_port}"
                return False

    def _with_retry(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """
        Decorator to add retry logic with exponential backoff to async functions.

        Args:
            func: Async function to wrap with retry logic

        Returns:
            Wrapped async function with retry logic
        """

        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Exception | None = None
            backoff = self.initial_backoff

            for attempt in range(self.max_retries):
                try:
                    return await func(*args, **kwargs)
                except (
                    httpx.ConnectError,
                    httpx.ConnectTimeout,
                    httpx.RemoteProtocolError,
                    httpx.ReadTimeout,
                    ConnectionError,
                ) as e:
                    last_exception = e
                    is_final_attempt = attempt == self.max_retries - 1

                    logger.warning(
                        "ToolHive connection failed",
                        attempt=attempt + 1,
                        max_retries=self.max_retries,
                        error=str(e),
                        error_type=type(e).__name__,
                        backoff_seconds=backoff if not is_final_attempt else 0,
                        will_retry=not is_final_attempt,
                    )

                    if is_final_attempt:
                        # This is the last attempt - don't retry
                        logger.error(
                            "All retry attempts exhausted. ToolHive is unavailable.",
                            total_attempts=self.max_retries,
                            host=self.thv_host,
                            port=self.thv_port,
                        )
                        # All retries exhausted - raise immediately
                        error_msg = (
                            f"Failed to connect to ToolHive after {self.max_retries} attempts. "
                            f"Last error: {last_exception}"
                        )
                        logger.critical(
                            "ToolHive connection failure - exiting",
                            max_retries=self.max_retries,
                            host=self.thv_host,
                            last_error=str(last_exception),
                        )
                        raise ToolhiveConnectionError(error_msg) from last_exception

                    # Try to rediscover the port
                    logger.info(
                        "Initiating port rediscovery after connection failure",
                        attempt=attempt + 1,
                    )
                    if await self._rediscover_port():
                        logger.info(
                            "Port rediscovery successful, retrying immediately",
                            new_port=self.thv_port,
                        )
                        # Reset backoff on successful rediscovery
                        backoff = self.initial_backoff
                        continue
                    else:
                        logger.warning(
                            "Port rediscovery failed, backing off before retry",
                            backoff_seconds=backoff,
                        )

                    # Exponential backoff
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self.max_backoff)

            # Should never reach here - all paths should either return or raise
            raise RuntimeError("Unexpected code path in _with_retry. Async wrapper failed.")

        return cast(Callable[..., Awaitable[T]], wrapper)

    async def __aenter__(self) -> Self:
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client, creating it if necessary."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def list_workloads(self, all_workloads: bool = False) -> WorkloadListResponse:
        """
        Get a list of workloads from Toolhive.

        Args:
            all_workloads: If True, include stopped workloads. If False, only running workloads.

        Returns:
            WorkloadListResponse containing the list of workloads

        Raises:
            httpx.RequestError: If the request fails
            httpx.HTTPStatusError: If the response status indicates an error
            ToolhiveConnectionError: If all retry attempts are exhausted
        """

        async def _list_workloads_impl() -> WorkloadListResponse:
            await self.ensure_connected()
            url = f"{self.base_url}/api/v1beta/workloads"
            params = {"all": "true"} if all_workloads else {}

            logger.debug("Fetching workloads from Toolhive", url=url, all_workloads=all_workloads)

            response = await self.client.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            workload_list = WorkloadListResponse.model_validate(data)
            if workload_list is None or workload_list.workloads is None:
                logger.warning("No workloads found", all_workloads=all_workloads)
                return WorkloadListResponse(workloads=[])

            # Replace the localhost/127.0.0.1 host with the toolhive host
            # This is required since in docker/podman container, we cannot use
            # localhost/127.0.0.1
            # Only replace when actually running in Docker to avoid breaking
            # local runs when TOOLHIVE_HOST is set to host.docker.internal
            if _is_running_in_docker() and self.thv_host not in ("localhost", "127.0.0.1"):
                for workload in workload_list.workloads:
                    if workload.url:
                        parsed_url = urlparse(workload.url)
                        workload_host = parsed_url.hostname
                        if workload_host in ("localhost", "127.0.0.1"):
                            workload.url = workload.url.replace(workload_host, self.thv_host)

            logger.info(
                "Successfully fetched workloads",
                count=len(workload_list.workloads),
                all_workloads=all_workloads,
            )

            return workload_list

        return await self._with_retry(_list_workloads_impl)()

    async def get_workload_details(self, workload_name: str) -> Workload:
        """
        Fetch detailed workload information including URL.

        Args:
            workload_name: Name of the workload to fetch

        Returns:
            Workload object with url field populated

        Raises:
            httpx.HTTPStatusError: If request fails (404, 500, etc.)
            httpx.TimeoutException: If request times out
            httpx.RequestError: If network error occurs
            ToolhiveConnectionError: If all retry attempts are exhausted
        """

        async def _get_workload_details_impl() -> Workload:
            await self.ensure_connected()
            url = f"{self.base_url}/api/v1beta/workloads/{workload_name}"

            logger.debug("Fetching workload details", workload_name=workload_name, url=url)

            response = await self.client.get(url)
            response.raise_for_status()

            data = response.json()
            workload = Workload.model_validate(data)

            # Replace localhost/127.0.0.1 with the toolhive host (same as list_workloads)
            # Only replace when actually running in Docker to avoid breaking
            # local runs when TOOLHIVE_HOST is set to host.docker.internal
            if _is_running_in_docker() and self.thv_host not in ("localhost", "127.0.0.1"):
                if workload.url:
                    parsed_url = urlparse(workload.url)
                    workload_host = parsed_url.hostname
                    if workload_host in ("localhost", "127.0.0.1"):
                        workload.url = workload.url.replace(workload_host, self.thv_host)

            logger.info(
                "Successfully fetched workload details",
                workload_name=workload_name,
                url=workload.url,
                remote=workload.remote,
            )

            return workload

        return await self._with_retry(_get_workload_details_impl)()

    async def get_running_mcp_workloads(self) -> list[Workload]:
        """
        Get only the running MCP server workloads.

        Returns:
            List of running MCP workloads
        """
        workload_list = await self.list_workloads(all_workloads=False)

        # Handle case where workloads might be None
        if not workload_list.workloads:
            logger.info("No workloads found")
            return []

        running_mcp_workloads = [
            workload for workload in workload_list.workloads if workload.status == "running"
        ]

        logger.info(
            "Filtered running MCP workloads",
            total_workloads=len(workload_list.workloads),
            running_mcp_count=len(running_mcp_workloads),
        )

        return running_mcp_workloads

    async def get_registry(self) -> Registry:
        """
        Get the registry from Toolhive containing server metadata.

        Returns:
            Registry containing server definitions and metadata

        Raises:
            httpx.RequestError: If the request fails
            httpx.HTTPStatusError: If the response status indicates an error
            ToolhiveConnectionError: If all retry attempts are exhausted
        """

        async def _get_registry_impl() -> Registry:
            await self.ensure_connected()
            url = f"{self.base_url}/api/v1beta/registry/default"

            logger.debug("Fetching registry from Toolhive", url=url)

            response = await self.client.get(url)
            response.raise_for_status()

            registry_response = GetRegistryResponse.model_validate(response.json())
            registry = Registry.model_validate(registry_response.registry)
            # Re-map the registry.servers so that is mapping of images to image metadata mapping
            # instead of server names to image metadata mapping
            if registry.servers is not None:
                image_to_metadata: dict[str, ImageMetadata] = {
                    image_metadata.image: image_metadata
                    for server_name, image_metadata in registry.servers.items()
                    if image_metadata.image is not None
                }
                registry.servers = image_to_metadata

            logger.info(
                "Successfully fetched registry",
                servers_count=len(registry.servers or {}),
                remote_servers_count=len(registry.remote_servers or {}),
            )

            return registry

        return await self._with_retry(_get_registry_impl)()

    async def get_server_from_registry(
        self, server_name: str
    ) -> ImageMetadata | RemoteServerMetadata | None:
        """
        Get a specific server from the registry.

        Args:
            server_name: Name of the server to fetch

        Returns:
            Server metadata or None if not found

        Raises:
            ToolhiveConnectionError: If all retry attempts are exhausted
        """

        async def _get_server_from_registry_impl() -> ImageMetadata | RemoteServerMetadata | None:
            await self.ensure_connected()
            url = f"{self.base_url}/api/v1beta/registry/default/servers/{server_name}"

            logger.info("Fetching server from Toolhive registry", url=url)

            response = await self.client.get(url)
            response.raise_for_status()

            server_registry_response = GetServerResponse.model_validate(response.json())
            if server_registry_response.server is not None:
                return server_registry_response.server

            if server_registry_response.remote_server is not None:
                return server_registry_response.remote_server

            return None

        return await self._with_retry(_get_server_from_registry_impl)()

    async def install_server(self, create_request: CreateRequest) -> dict:
        """
        Install/start an MCP server workload in ToolHive.

        Based on toolhive-studio implementation which uses POST /api/v1beta/workloads

        Args:
            create_request: CreateRequest object. Check api_models.v1.CreateRequest for details.

        Returns:
            Response dict with:
                - name: Created workload name
                - port: Port where the workload is running

        Raises:
            httpx.HTTPStatusError: If request fails (400 Bad Request, 409 Conflict, etc.)
            ToolhiveConnectionError: If all retry attempts are exhausted
        """

        async def _install_server_impl() -> dict:
            await self.ensure_connected()
            url = f"{self.base_url}/api/v1beta/workloads"

            logger.info("Installing MCP server workload", server_name=create_request.name)

            response = await self.client.post(
                url, json=create_request.model_dump(), headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

            result = response.json()
            logger.info(
                "Successfully installed MCP server workload",
                name=result.get("name"),
                port=result.get("port"),
            )

            return result

        result = await self._with_retry(_install_server_impl)()
        return cast(dict, result)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
