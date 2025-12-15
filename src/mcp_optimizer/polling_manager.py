"""
Polling manager for periodic MCP server discovery and synchronization.

This module provides functionality to periodically poll ToolHive for workload changes,
detect server status changes, and synchronize the MCP Optimizer database accordingly.
"""

import asyncio
import threading
import time

import structlog

from mcp_optimizer.config import MCPOptimizerConfig
from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import McpStatus, RegistryServer
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionError, IngestionService, ToolHiveUnavailable
from mcp_optimizer.mcp_client import WorkloadConnectionError
from mcp_optimizer.toolhive.toolhive_client import (
    ToolhiveClient,
    ToolhiveConnectionError,
    ToolhiveScanError,
)

logger = structlog.get_logger(__name__)


# Thread-safe polling manager state
class PollingState:
    """Thread-safe container for polling manager state."""

    def __init__(self):
        self._lock = threading.RLock()
        self._polling_manager: "PollingManager | None" = None
        self._polling_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()

    @property
    def polling_manager(self) -> "PollingManager | None":
        with self._lock:
            return self._polling_manager

    @polling_manager.setter
    def polling_manager(self, value: "PollingManager | None") -> None:
        with self._lock:
            self._polling_manager = value

    @property
    def polling_thread(self) -> threading.Thread | None:
        with self._lock:
            return self._polling_thread

    @polling_thread.setter
    def polling_thread(self, value: threading.Thread | None) -> None:
        with self._lock:
            self._polling_thread = value

    @property
    def shutdown_event(self) -> threading.Event:
        with self._lock:
            return self._shutdown_event

    def is_shutdown_requested(self) -> bool:
        """Thread-safe check for shutdown request."""
        return self._shutdown_event.is_set()

    def request_shutdown(self) -> None:
        """Thread-safe shutdown request."""
        with self._lock:
            self._shutdown_event.set()


# Global polling state instance
_polling_state = PollingState()


def _wait_for_startup() -> bool:
    """Wait for server startup with shutdown check."""
    for _ in range(30):  # 3 seconds total, check every 0.1 seconds
        if _polling_state.is_shutdown_requested():
            logger.info("Shutdown requested before polling start")
            return False
        time.sleep(0.1)
    return True


def _run_polling_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Run the polling loop until shutdown is requested."""
    while not _polling_state.is_shutdown_requested():
        try:
            loop.run_until_complete(asyncio.sleep(1.0))
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.exception("Error in polling loop", error=str(e))
            break


def _cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Clean up the event loop and cancel remaining tasks."""
    pending = asyncio.all_tasks(loop)
    if pending:
        logger.info("Cancelling remaining tasks", count=len(pending))
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()
    logger.info("Polling thread event loop closed")


def _start_polling_after_delay() -> None:
    """Start the polling manager in a background thread after startup delay."""
    polling_manager = _polling_state.polling_manager

    if polling_manager is None:
        logger.warning("Polling manager is not configured; exiting polling thread")
        return

    if not _wait_for_startup():
        return

    logger.info("Starting polling manager in background thread")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(polling_manager.start_polling())
        logger.info("Polling manager started successfully")

        _run_polling_loop(loop)
        logger.info("Polling thread received shutdown signal")

        loop.run_until_complete(polling_manager.stop_polling())
        logger.info("Polling manager stopped in background thread")
    except Exception as e:
        logger.exception("Error starting polling manager", error=str(e))
    finally:
        _cleanup_event_loop(loop)


def configure_polling(toolhive_client: ToolhiveClient, config: MCPOptimizerConfig) -> None:
    """Configure polling manager settings for the server."""
    workload_interval = config.workload_polling_interval
    registry_interval = config.registry_polling_interval

    logger.info(
        "configure_polling called",
        workload_interval=workload_interval,
        registry_interval=registry_interval,
        host=toolhive_client.thv_host,
        port=toolhive_client.thv_port,
    )

    # Create database and embedding manager with config values
    db_config = DatabaseConfig(database_url=config.async_db_url)
    embedding_manager_local = EmbeddingManager(
        model_name=config.embedding_model_name,
        enable_cache=config.enable_embedding_cache,
        threads=config.embedding_threads,
        fastembed_cache_path=config.fastembed_cache_path,
    )

    _polling_state.polling_manager = PollingManager(
        db_config=db_config,
        embedding_manager=embedding_manager_local,
        toolhive_client=toolhive_client,
        workload_polling_interval=workload_interval,
        registry_polling_interval=registry_interval,
        startup_polling_delay=config.startup_polling_delay,
        mcp_timeout=config.mcp_timeout,
        registry_ingestion_batch_size=config.registry_ingestion_batch_size,
        workload_ingestion_batch_size=config.workload_ingestion_batch_size,
        encoding=config.encoding,
        targeted_polling_max_attempts=config.targeted_polling_max_attempts,
        targeted_polling_interval=config.targeted_polling_interval,
        skipped_workloads=config.skipped_workloads,
        runtime_mode=config.runtime_mode,
        k8s_api_server_url=config.k8s_api_server_url,
        k8s_namespace=config.k8s_namespace,
        k8s_all_namespaces=config.k8s_all_namespaces,
    )
    logger.info(
        "Polling manager configured successfully",
        workload_interval_seconds=workload_interval,
        registry_interval_seconds=registry_interval,
    )

    thread = threading.Thread(target=_start_polling_after_delay, daemon=True)
    _polling_state.polling_thread = thread
    thread.start()
    logger.info("Background polling thread started")


def shutdown_polling() -> None:
    """Signal the background polling thread to shut down gracefully."""
    logger.info("Shutting down polling manager")

    # Signal shutdown to the background thread
    _polling_state.request_shutdown()

    # Wait for the thread to finish (with timeout)
    polling_thread = _polling_state.polling_thread
    if polling_thread and polling_thread.is_alive():
        logger.info("Waiting for polling thread to finish")
        polling_thread.join(timeout=2.0)  # Wait up to 2 seconds
        if polling_thread.is_alive():
            logger.warning("Polling thread did not finish within timeout - forcing exit")
        else:
            logger.info("Polling thread finished successfully")


class PollingManager:
    """Manages periodic polling of ToolHive for MCP server status changes."""

    def __init__(
        self,
        db_config: DatabaseConfig,
        embedding_manager: EmbeddingManager,
        toolhive_client: ToolhiveClient,
        workload_polling_interval: int,  # seconds
        registry_polling_interval: int,  # seconds
        startup_polling_delay: int,  # seconds
        mcp_timeout: int,
        registry_ingestion_batch_size: int,
        workload_ingestion_batch_size: int,
        encoding: str,
        targeted_polling_max_attempts: int,
        targeted_polling_interval: int,  # seconds
        skipped_workloads: list[str] | None = None,
        runtime_mode: str = "docker",
        k8s_api_server_url: str = "http://127.0.0.1:8001",
        k8s_namespace: str | None = None,
        k8s_all_namespaces: bool = True,
    ):
        """Initialize the polling manager.

        Args:
            db_config: Database configuration.
            embedding_manager: Embedding manager for generating embeddings.
            toolhive_client: ToolhiveClient instance for communication.
            workload_polling_interval: Interval in seconds between workload polls.
            registry_polling_interval: Interval in seconds between registry polls.
            startup_polling_delay: Delay in seconds before initial polling at startup.
            mcp_timeout: Timeout for MCP operations in seconds.
            registry_ingestion_batch_size: Batch size for parallel registry server ingestion.
            workload_ingestion_batch_size: Batch size for parallel workload ingestion.
            encoding: Tiktoken encoding to use for token counting.
            targeted_polling_max_attempts: Maximum number of attempts for targeted polling.
            targeted_polling_interval: Interval in seconds between targeted polling attempts.
            skipped_workloads: List of workload names to skip during ingestion.
        """
        self.db_config = db_config
        self.embedding_manager = embedding_manager
        self.workload_polling_interval = workload_polling_interval
        self.registry_polling_interval = registry_polling_interval
        self.startup_polling_delay = startup_polling_delay
        self.toolhive_client = toolhive_client
        self.targeted_polling_max_attempts = targeted_polling_max_attempts
        self.targeted_polling_interval = targeted_polling_interval

        self.ingestion_service = IngestionService(
            db_config,
            embedding_manager,
            mcp_timeout=mcp_timeout,
            registry_ingestion_batch_size=registry_ingestion_batch_size,
            workload_ingestion_batch_size=workload_ingestion_batch_size,
            encoding=encoding,
            skipped_workloads=skipped_workloads,
            runtime_mode=runtime_mode,
            k8s_api_server_url=k8s_api_server_url,
            k8s_namespace=k8s_namespace,
            k8s_all_namespaces=k8s_all_namespaces,
        )
        self.workload_ops = WorkloadServerOps(db_config)

        self._workload_polling_task: asyncio.Task | None = None
        self._registry_polling_task: asyncio.Task | None = None
        self._shutdown_requested = False
        self._workload_polling_paused = False
        self._workload_polling_pause_lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._registry_startup_complete: asyncio.Event | None = None

    async def start_polling(self) -> None:
        """Start the periodic polling tasks."""
        logger.info(
            "=== START POLLING CALLED ===",
            workload_interval=self.workload_polling_interval,
            registry_interval=self.registry_polling_interval,
            startup_delay=self.startup_polling_delay,
        )
        if self._workload_polling_task is not None or self._registry_polling_task is not None:
            logger.warning(
                "Polling is already running",
                workload_task_done=self._workload_polling_task.done()
                if self._workload_polling_task
                else None,
                registry_task_done=self._registry_polling_task.done()
                if self._registry_polling_task
                else None,
            )
            return

        # Store the event loop for later use by targeted polling
        self._loop = asyncio.get_running_loop()

        # Create a shared event for coordinating startup sequence
        self._registry_startup_complete = asyncio.Event()

        logger.info(
            "Creating polling tasks",
            workload_interval_seconds=self.workload_polling_interval,
            registry_interval_seconds=self.registry_polling_interval,
            startup_delay_seconds=self.startup_polling_delay,
        )
        # Always create both tasks (startup run is mandatory)
        self._registry_polling_task = asyncio.create_task(self._registry_polling_loop())
        self._workload_polling_task = asyncio.create_task(self._workload_polling_loop())
        logger.info(
            "Polling tasks created successfully",
            workload_task_created=self._workload_polling_task is not None,
            registry_task_created=self._registry_polling_task is not None,
        )

    async def stop_polling(self) -> None:
        """Stop the periodic polling tasks."""
        if self._workload_polling_task is None and self._registry_polling_task is None:
            logger.info("Polling is not running")
            return

        logger.info("Stopping periodic polling")
        self._shutdown_requested = True

        # Cancel workload polling task
        if self._workload_polling_task and not self._workload_polling_task.done():
            self._workload_polling_task.cancel()
            try:
                await self._workload_polling_task
            except asyncio.CancelledError:
                logger.info("Workload polling task cancelled successfully")
            except Exception as e:
                logger.exception("Error while stopping workload polling task", error=str(e))

        # Cancel registry polling task
        if self._registry_polling_task and not self._registry_polling_task.done():
            self._registry_polling_task.cancel()
            try:
                await self._registry_polling_task
            except asyncio.CancelledError:
                logger.info("Registry polling task cancelled successfully")
            except Exception as e:
                logger.exception("Error while stopping registry polling task", error=str(e))

        self._workload_polling_task = None
        self._registry_polling_task = None
        self._shutdown_requested = False

        # Ensure we clean up any lingering resources
        logger.info("Polling manager cleanup completed")

    async def _workload_polling_loop(self) -> None:
        """Workload polling loop that runs periodically."""
        logger.info("=== WORKLOAD POLLING LOOP STARTED ===")

        # Wait for registry startup to complete (sequential startup execution)
        if self._registry_startup_complete is not None:
            logger.info("Waiting for registry startup to complete before workload polling")
            await self._registry_startup_complete.wait()

        # ALWAYS run first poll (startup run is mandatory)
        logger.info("Running initial workload polling (startup)")
        try:
            await self._poll_workloads()
            logger.info("Initial workload polling completed successfully")
        except Exception as e:
            logger.exception("Error during initial workload polling", error=str(e))

        # If interval <= 0, stop after initial run
        if self.workload_polling_interval <= 0:
            logger.info("Workload polling disabled (interval <= 0), stopping after startup run")
            return

        # Continue with periodic polling
        cycle_count = 0
        while not self._shutdown_requested:
            # Wait for next polling interval
            logger.debug(
                "Sleeping until next workload poll", seconds=self.workload_polling_interval
            )
            try:
                await asyncio.sleep(self.workload_polling_interval)
            except asyncio.CancelledError:
                logger.info("Workload polling loop interrupted by cancellation")
                break

            # Check if polling is paused (during targeted polling) - thread-safe access
            async with self._workload_polling_pause_lock:
                is_paused = self._workload_polling_paused

            if is_paused:
                logger.debug("Workload polling is paused, skipping cycle")
                try:
                    await asyncio.sleep(1.0)  # Check every second if we can resume
                except asyncio.CancelledError:
                    logger.info("Workload polling loop interrupted by cancellation")
                    break
                continue

            # Increment cycle count for this poll
            cycle_count += 1

            # Run periodic poll
            logger.info(
                "Starting workload polling cycle",
                cycle=cycle_count,
                interval=self.workload_polling_interval,
            )
            try:
                await self._poll_workloads()
                logger.info("Workload polling cycle completed successfully", cycle=cycle_count)
            except Exception as e:
                logger.exception(
                    "Error during workload polling cycle", error=str(e), cycle=cycle_count
                )
                # Continue polling even if one cycle fails

        logger.info("Workload polling loop ended", total_cycles=cycle_count)

    async def _registry_polling_loop(self) -> None:
        """Registry polling loop that runs periodically."""
        logger.info("=== REGISTRY POLLING LOOP STARTED ===")

        # Wait for configured startup delay before first poll
        logger.info(
            "Waiting for startup delay before initial registry poll",
            seconds=self.startup_polling_delay,
        )
        await asyncio.sleep(self.startup_polling_delay)

        # ALWAYS run first poll (startup run is mandatory)
        logger.info("Running initial registry polling (startup)")
        try:
            await self._poll_registry()
            logger.info("Initial registry polling completed successfully")
        except Exception as e:
            logger.exception("Error during initial registry polling", error=str(e))

        # Signal workload polling that registry startup is complete
        if self._registry_startup_complete is not None:
            self._registry_startup_complete.set()

        # If interval <= 0, stop after initial run
        if self.registry_polling_interval <= 0:
            logger.info("Registry polling disabled (interval <= 0), stopping after startup run")
            return

        # Continue with periodic polling
        cycle_count = 0
        while not self._shutdown_requested:
            # Wait for next polling interval
            logger.debug(
                "Sleeping until next registry poll", seconds=self.registry_polling_interval
            )
            try:
                await asyncio.sleep(self.registry_polling_interval)
            except asyncio.CancelledError:
                logger.info("Registry polling loop interrupted by cancellation")
                break

            # Increment cycle count for this poll
            cycle_count += 1

            # Run periodic poll
            logger.info(
                "Starting registry polling cycle",
                cycle=cycle_count,
                interval=self.registry_polling_interval,
            )
            try:
                await self._poll_registry()
                logger.info("Registry polling cycle completed successfully", cycle=cycle_count)
            except Exception as e:
                logger.exception(
                    "Error during registry polling cycle", error=str(e), cycle=cycle_count
                )
                # Continue polling even if one cycle fails

        logger.info("Registry polling loop ended", total_cycles=cycle_count)

    async def _poll_workloads(self) -> None:
        """Poll ToolHive and synchronize the database with current workload state."""
        logger.debug("Starting workload polling cycle")

        try:
            # Use the ingestion service to sync all workloads
            await self.ingestion_service.ingest_workloads(self.toolhive_client)
            logger.debug("Workload polling cycle completed successfully")
        except Exception as e:
            # Handle ToolHive unavailability gracefully - log and continue polling
            # This allows the system to reconnect when ToolHive becomes available
            if isinstance(
                e,
                (ToolhiveConnectionError, ToolhiveScanError, ConnectionError, ToolHiveUnavailable),
            ):
                logger.warning(
                    "ToolHive unavailable during polling. Will retry on next polling cycle.",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Don't re-raise - continue polling so we can reconnect later
                return
            logger.exception("Error during workload synchronization", error=str(e))
            # Re-raise unexpected errors
            raise

    async def _poll_registry(self) -> None:
        """Poll ToolHive and synchronize the database with registry servers."""
        logger.debug("Starting registry polling cycle")

        try:
            # Use the ingestion service to sync registry servers
            await self.ingestion_service.ingest_registry(self.toolhive_client)
            logger.debug("Registry polling cycle completed successfully")
        except Exception as e:
            # Handle ToolHive unavailability gracefully - log and continue polling
            # This allows the system to reconnect when ToolHive becomes available
            if isinstance(
                e,
                (ToolhiveConnectionError, ToolhiveScanError, ConnectionError, ToolHiveUnavailable),
            ):
                logger.warning(
                    "ToolHive unavailable during polling. Will retry on next polling cycle.",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                # Don't re-raise - continue polling so we can reconnect later
                return
            logger.exception("Error during registry synchronization", error=str(e))
            # Re-raise unexpected errors
            raise

    def is_polling(self) -> bool:
        """Check if polling is currently active."""
        workload_active = (
            self._workload_polling_task is not None and not self._workload_polling_task.done()
        )
        registry_active = (
            self._registry_polling_task is not None and not self._registry_polling_task.done()
        )
        return workload_active or registry_active

    async def _targeted_workload_polling(self, registry_server: RegistryServer) -> None:
        """Poll for a specific newly installed workload with aggressive retry.

        This method polls every second for up to 2 minutes looking for a specific
        workload to become available after installation. While running, it pauses
        the global workload polling to avoid redundant work.

        Args:
            server_name: Name of the MCP server to poll for
        """
        logger.info(
            "Starting targeted polling for new workload",
            server_name=registry_server.name,
            poll_interval=self.targeted_polling_interval,
            max_duration=self.targeted_polling_max_attempts,
        )

        # Pause global workload polling while we do targeted polling - thread-safe
        async with self._workload_polling_pause_lock:
            was_paused = self._workload_polling_paused
            self._workload_polling_paused = True
        logger.info(
            "Paused global workload polling for targeted polling", server_name=registry_server.name
        )

        poll_count = 0

        try:
            while poll_count < self.targeted_polling_max_attempts:
                poll_count += 1
                logger.debug(
                    "Targeted polling attempt",
                    server_name=registry_server.name,
                    attempt=poll_count,
                    max_attempts=self.targeted_polling_max_attempts,
                )

                try:
                    # Run full workload ingestion
                    await self.ingestion_service.ingest_workloads(self.toolhive_client)

                    # Check if the server is now available and running
                    workloads_matching_registry = await self.workload_ops.list_servers_by_registry(
                        registry_server_id=registry_server.id
                    )
                    running_workloads = [
                        w for w in workloads_matching_registry if w.status == McpStatus.RUNNING
                    ]
                    if running_workloads:
                        logger.info(
                            "Targeted polling succeeded - workload found",
                            server_name=running_workloads[0].name,
                            attempts=poll_count,
                            elapsed_seconds=poll_count,
                        )
                        return
                except (IngestionError, WorkloadConnectionError) as e:
                    # Expected errors during polling - log at debug level and continue
                    logger.debug(
                        "Expected error during targeted polling attempt",
                        server_name=registry_server.name,
                        attempt=poll_count,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue polling - server may not be ready yet
                except DbNotFoundError:
                    # Server not found in DB yet - this is expected, continue polling
                    logger.debug(
                        "Server not found in database yet",
                        server_name=registry_server.name,
                        attempt=poll_count,
                    )
                except Exception as e:
                    # Unexpected errors - log at warning level but continue
                    logger.warning(
                        "Unexpected error during targeted polling attempt",
                        server_name=registry_server.name,
                        attempt=poll_count,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    # Continue polling even if one attempt fails

                # Wait before next poll
                await asyncio.sleep(self.targeted_polling_interval)

            # Timeout reached
            logger.warning(
                "Targeted polling timeout - workload not found",
                server_name=registry_server.name,
                total_attempts=poll_count,
                elapsed_seconds=poll_count,
            )
        except asyncio.CancelledError:
            logger.info(
                "Targeted polling cancelled", server_name=registry_server.name, attempts=poll_count
            )
            raise
        except Exception as e:
            logger.exception(
                "Unexpected error in targeted polling",
                server_name=registry_server.name,
                error=str(e),
            )
        finally:
            # Resume global workload polling (unless it was already paused) - thread-safe
            if not was_paused:
                async with self._workload_polling_pause_lock:
                    self._workload_polling_paused = False
                logger.info(
                    "Resumed global workload polling after targeted polling",
                    server_name=registry_server.name,
                )

    def _handle_targeted_polling_error(
        self, future: asyncio.Future, registry_server: RegistryServer
    ) -> None:
        """Handle errors from targeted polling task.

        Args:
            future: The Future returned by run_coroutine_threadsafe
            registry_server: The registry server being polled
        """
        try:
            # This will raise if the task failed
            future.result()
        except asyncio.CancelledError:
            # Task was cancelled, this is normal
            logger.debug("Targeted polling task was cancelled", server_name=registry_server.name)
        except Exception as e:
            # Log unexpected errors
            logger.error(
                "Targeted polling task failed with error",
                server_name=registry_server.name,
                error=str(e),
                error_type=type(e).__name__,
            )

    def start_targeted_workload_polling(self, registry_server: RegistryServer) -> None:
        """Launch background task to poll for a newly installed workload.

        This is a fire-and-forget operation that starts aggressive polling
        (every second) for up to 2 minutes after a server installation.
        The task runs in the background and does not block the caller.

        Args:
            registry_server: Registry server to poll for
        """
        if self._loop is None:
            logger.warning(
                "Polling manager event loop not available, cannot start targeted polling",
                server_name=registry_server.name,
            )
            return

        logger.info("Launching targeted polling task", server_name=registry_server.name)
        # Schedule the task in the polling manager's event loop (not the caller's loop)
        # This is thread-safe and avoids event loop mismatch errors
        future = asyncio.run_coroutine_threadsafe(
            self._targeted_workload_polling(registry_server), self._loop
        )
        # Add error callback to handle exceptions
        future.add_done_callback(lambda f: self._handle_targeted_polling_error(f, registry_server))
        logger.debug("Targeted polling task launched", server_name=registry_server.name)


def get_polling_manager() -> PollingManager | None:
    """Get the global polling manager instance if configured.

    Returns:
        The polling manager instance if configured, None otherwise
    """
    return _polling_state.polling_manager
