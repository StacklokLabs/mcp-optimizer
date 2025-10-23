import asyncio
import os
import signal
from pathlib import Path
from typing import Any

import click
import structlog
import uvicorn

from mcp_optimizer.config import ConfigurationError, MCPOptimizerConfig, get_config
from mcp_optimizer.configure_logging import configure_logging
from mcp_optimizer.db.config import DatabaseConfig, run_migrations
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.ingestion import IngestionService
from mcp_optimizer.polling_manager import configure_polling, shutdown_polling
from mcp_optimizer.server import initialize_server_components
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient

logger = structlog.get_logger(__name__)

# Environment variable name overrides for fields that don't follow the simple pattern
# Most fields auto-generate from FIELD_NAME -> FIELD_NAME.upper()
ENV_VAR_OVERRIDES = {}

# CLI option overrides for special cases
CLI_OPTION_OVERRIDES = {
    "runtime_mode": "runtime-mode",  # Special case for choice type
}

# Type overrides for special field types
TYPE_OVERRIDES = {
    "runtime_mode": click.Choice(["docker", "k8s"], case_sensitive=False),
}

# Fields that should not be exposed via CLI or environment variables
# These fields can only be set via configuration files
CLI_EXCLUDED_FIELDS = {
    "skipped_workloads",  # Config-only field for workload filtering
}


def _generate_config_params():
    """Generate CONFIG_PARAMS from MCPOptimizerConfig model fields.

    This ensures we only maintain field definitions in one place (the Pydantic model).
    Automatically extracts field names, types, and descriptions.
    """
    params = []

    for field_name, field_info in MCPOptimizerConfig.model_fields.items():
        # Skip fields that are excluded from CLI/env var access
        if field_name in CLI_EXCLUDED_FIELDS:
            continue

        # Convert field name to environment variable name (e.g., mcp_port -> MCP_PORT)
        env_var = ENV_VAR_OVERRIDES.get(field_name, field_name.upper())

        # Convert field name to CLI option (e.g., mcp_port -> mcp-port)
        cli_option = CLI_OPTION_OVERRIDES.get(field_name, field_name.replace("_", "-"))

        # Determine the Click type from the field annotation
        if field_name in TYPE_OVERRIDES:
            param_type = TYPE_OVERRIDES[field_name]
        else:
            # Map Python types to Click types
            annotation = field_info.annotation
            # Handle Optional types
            if hasattr(annotation, "__origin__"):
                # For Union types (e.g., int | None), get the non-None type
                args = [arg for arg in annotation.__args__ if arg is not type(None)]
                if args:
                    annotation = args[0]

            # Map to Click type
            if annotation is bool:
                param_type = bool
            elif annotation is int:
                param_type = int
            elif annotation is float:
                param_type = float
            else:
                param_type = str

        # Use the field's description from Pydantic
        help_text = field_info.description or f"Configuration for {field_name}"

        params.append((env_var, cli_option, param_type, help_text))

    return params


# Generate CONFIG_PARAMS from the Pydantic model
# This ensures single source of truth for all configuration fields
CONFIG_PARAMS = _generate_config_params()


def _get_field_default(field_name: str) -> Any:
    """Extract the default value from MCPOptimizerConfig model field.

    Args:
        field_name: Name of the field in MCPOptimizerConfig

    Returns:
        The default value from the Pydantic Field, or None if no default
    """
    field_info = MCPOptimizerConfig.model_fields.get(field_name)
    if field_info and field_info.default is not None:
        return field_info.default
    return None


def add_config_options(func):
    """Decorator to dynamically add all config options to the CLI command.

    Extracts defaults from the Pydantic model to ensure help text accuracy.
    CLI options intentionally use default=None to preserve priority chain:
    CLI option > Environment variable > Pydantic Field default
    """
    for env_var, cli_option, param_type, help_text_base in reversed(CONFIG_PARAMS):
        # Convert env var name to a valid Python parameter name
        param_name = env_var.lower()

        # Extract the actual default from the Pydantic model
        default_value = _get_field_default(param_name)

        # Format default for display in help text
        if default_value is None:
            default_str = "none"
        elif isinstance(default_value, str):
            default_str = default_value if default_value else "none"
        else:
            default_str = (
                str(default_value).lower()
                if isinstance(default_value, bool)
                else str(default_value)
            )

        # Build comprehensive help text with actual default
        full_help = (
            f"{help_text_base} (default: {default_str}). "
            f"Can also be set via {env_var} environment variable."
        )

        # Add the option decorator
        # Note: default=None preserves priority chain (CLI > env var > Pydantic default)
        func = click.option(
            f"--{cli_option}",
            param_name,
            type=param_type,
            help=full_help,
            default=None,
        )(func)

    return func


@click.command()
@add_config_options
def main(**kwargs: Any) -> None:
    """MCP Optimizer CLI tool."""
    # Set environment variables from CLI options if provided
    # CLI options take precedence over existing env vars
    for env_var, _, _, _ in CONFIG_PARAMS:
        param_name = env_var.lower()
        value = kwargs.get(param_name)
        if value is not None:
            # Convert boolean values to strings for environment variables
            if isinstance(value, bool):
                os.environ[env_var] = str(value).lower()
            else:
                os.environ[env_var] = str(value)

    # Validate configuration at startup
    try:
        config = get_config()

        # Determine if any CLI options were provided
        cli_options_used = [k for k, v in kwargs.items() if v is not None]

        logger.info(
            "Configuration validation successful",
            runtime_mode=config.runtime_mode,
            config_source="cli" if cli_options_used else "env/default",
            cli_options_provided=cli_options_used if cli_options_used else None,
            mcp_port=config.mcp_port,
            toolhive_port_range=f"{config.toolhive_start_port_scan}-{config.toolhive_end_port_scan}",
        )
    except ConfigurationError as e:
        logger.error("Configuration validation failed", error=str(e))
        raise click.ClickException(f"Configuration error: {e}") from e

    logging_dict = configure_logging(
        log_level=config.log_level,
        rich_tracebacks=config.rich_tracebacks,
        colored_logs=config.colored_logs,
    )

    toolhive_client = ToolhiveClient(
        host=config.toolhive_host,
        port=config.toolhive_port,
        scan_port_start=config.toolhive_start_port_scan,
        scan_port_end=config.toolhive_end_port_scan,
        timeout=config.toolhive_timeout,
        max_retries=config.toolhive_max_retries,
        initial_backoff=config.toolhive_initial_backoff,
        max_backoff=config.toolhive_max_backoff,
        skip_port_discovery=(config.runtime_mode == "k8s"),
    )

    # Database setup and testing is now handled in config.py's _setup_secure_database_path
    # during configuration loading, so no additional testing needed here

    # Run the migrations with any command we ran.
    run_migrations()

    # Initialize server components with config values
    initialize_server_components(config)

    try:
        # Pass config values to components instead of using get_config()
        db_config = DatabaseConfig(database_url=config.async_db_url)
        embedding_manager = EmbeddingManager(
            model_name=config.embedding_model_name,
            enable_cache=config.enable_embedding_cache,
            threads=config.embedding_threads,
        )
        ingestion_service = IngestionService(
            db_config,
            embedding_manager,
            mcp_timeout=config.mcp_timeout,
            registry_ingestion_batch_size=config.registry_ingestion_batch_size,
            workload_ingestion_batch_size=config.workload_ingestion_batch_size,
            encoding=config.encoding,
            skipped_workloads=config.skipped_workloads,
            runtime_mode=config.runtime_mode,
            k8s_api_server_url=config.k8s_api_server_url,
            k8s_namespace=config.k8s_namespace,
            k8s_all_namespaces=config.k8s_all_namespaces,
        )

        async def run_ingestion():
            """Run ingestion tasks in a single event loop."""
            await ingestion_service.ingest_registry(toolhive_client=toolhive_client)
            await ingestion_service.ingest_workloads(toolhive_client=toolhive_client)
            # Dispose the database engine to prevent event loop conflicts
            # The server will create a new engine in its own event loop
            await db_config.close()

        logger.info("Starting initial ingestion process")
        try:
            asyncio.run(run_ingestion())
            logger.info("Initial ingestion process completed")
        except Exception as e:
            # Import here to avoid circular dependency
            from mcp_optimizer.toolhive.toolhive_client import ToolhiveConnectionError

            if isinstance(e, ToolhiveConnectionError):
                logger.critical(
                    "Unable to connect to ToolHive after exhausting all retries. "
                    "Please ensure ToolHive is running and accessible.",
                    error=str(e),
                )
                raise click.ClickException(f"Failed to connect to ToolHive: {e}. Exiting.") from e
            # Re-raise other exceptions
            raise

        # Configure polling manager for the server
        logger.info(
            "Configuring polling manager",
            workload_interval=config.workload_polling_interval,
            registry_interval=config.registry_polling_interval,
            host=toolhive_client.thv_host,
            port=toolhive_client.thv_port,
        )
        configure_polling(
            toolhive_client=toolhive_client,
            config=config,
        )

        # Set up signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info("Received signal, initiating graceful shutdown", signal=signum)
            # Shutdown polling manager first
            shutdown_polling()
            # The signal will cause uvicorn to shutdown gracefully

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("Starting server")
        uvicorn.run(
            "mcp_optimizer.server:starlette_app",
            host="0.0.0.0",
            port=config.mcp_port,
            log_config=logging_dict,
            reload=config.reload_server,
            reload_dirs=[str(Path(__file__).parent.resolve())],
            timeout_graceful_shutdown=5,  # Increased timeout for better cleanup
        )
    finally:
        # Cleanup polling manager
        logger.info("Stopping polling manager")
        shutdown_polling()


if __name__ == "__main__":
    main()
