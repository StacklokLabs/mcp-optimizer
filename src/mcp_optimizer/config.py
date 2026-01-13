"""Configuration validation and management for MCP Optimizer."""

import os
import stat
from pathlib import Path
from typing import Any, Literal, Optional

import structlog
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

logger = structlog.get_logger(__name__)


class ConfigurationError(Exception):
    """Exception raised for configuration validation errors."""

    pass


class MCPOptimizerConfig(BaseModel):
    """Configuration schema for MCP Optimizer with validation."""

    # Runtime configuration
    runtime_mode: Literal["docker", "k8s"] = Field(
        default="docker",
        description="Runtime mode for MCP servers (docker or k8s)",
    )

    @field_validator("runtime_mode", mode="before")
    @classmethod
    def normalize_runtime_mode(cls, v) -> str:
        """Normalize runtime mode to lowercase for case-insensitive matching.

        Args:
            v: Runtime mode value (can be any case)

        Returns:
            Lowercase runtime mode value
        """
        if isinstance(v, str):
            return v.lower()
        return v

    # Server configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    colored_logs: bool = Field(default=True, description="Whether to enable colored logs")
    rich_tracebacks: bool = Field(default=False, description="Whether to enable rich tracebacks")
    mcp_port: int = Field(
        default=9900, ge=1024, le=65535, description="Port for the MCP server (1024-65535)"
    )
    reload_server: bool = Field(
        default=False, description="Whether to enable auto-reload for development"
    )

    # ToolHive configuration
    toolhive_host: str = Field(
        default="localhost", min_length=1, description="Host for ToolHive API"
    )
    toolhive_port: int | None = Field(
        default=None, ge=1024, le=65535, description="Port for ToolHive API (1024-65535)"
    )
    toolhive_start_port_scan: int = Field(
        default=50000,
        ge=1024,
        le=65535,
        description="Start port for ToolHive scanning (1024-65535)",
    )
    toolhive_end_port_scan: int = Field(
        default=50100, ge=1024, le=65535, description="End port for ToolHive scanning (1024-65535)"
    )

    # Kubernetes configuration (used when runtime_mode is k8s)
    k8s_api_server_url: str = Field(
        default="http://127.0.0.1:8001",
        min_length=1,
        description="Kubernetes API server URL (default: kubectl proxy URL)",
    )
    k8s_namespace: str | None = Field(
        default=None,
        description="Kubernetes namespace to query for MCPServers. If None, queries all namespaces",
    )
    k8s_all_namespaces: bool = Field(
        default=True,
        description="If True, list MCPServers across all namespaces when in k8s mode",
    )
    workload_polling_interval: int = Field(
        default=60,
        ge=0,
        le=300,
        description="Workload polling interval in seconds (0-300). 0 means no polling",
    )
    registry_polling_interval: int = Field(
        default=86400,
        ge=0,
        le=86400,
        description=(
            "Registry polling interval in seconds (0-86400, 24 hours). "
            "0 means no polling. Default is 24 hours since the registry should be fairly static."
        ),
    )
    startup_polling_delay: int = Field(
        default=3,
        ge=0,
        le=300,
        description="Delay in seconds before initial polling at startup (0-300)",
    )
    targeted_polling_max_attempts: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Maximum polling attempts for targeted polling (1-600)",
    )
    targeted_polling_interval: int = Field(
        default=1,
        ge=1,
        le=60,
        description="Interval between targeted polling attempts in seconds (1-60)",
    )
    toolhive_timeout: int = Field(
        default=10, ge=1, le=60, description="ToolHive connection timeout in seconds (1-60)"
    )
    toolhive_max_retries: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum number of retry attempts when ToolHive connection fails (1-500)",
    )
    toolhive_initial_backoff: float = Field(
        default=1.0,
        ge=0.1,
        le=10.0,
        description="Initial backoff delay in seconds for ToolHive retry logic (0.1-10.0)",
    )
    toolhive_max_backoff: float = Field(
        default=60.0,
        ge=1.0,
        le=300.0,
        description="Maximum backoff delay in seconds for ToolHive retry logic (1.0-300.0)",
    )
    toolhive_skip_backoff: bool = Field(
        default=False,
        description="Skip backoff period between discovery retries (hidden/test-only flag)",
    )

    # Timeout configuration
    mcp_timeout: int = Field(
        default=20, ge=1, le=300, description="MCP operation timeout in seconds (1-300)"
    )

    # Database configuration
    async_db_url: str = Field(default="", description="Async database URL")

    db_url: str = Field(default="", description="Sync database URL")

    # Embedding configuration
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="Name of the embedding model to use",
    )

    embedding_threads: int | None = Field(
        default=2,
        ge=1,
        le=16,
        description="Number of threads for embedding generation (1-16). "
        "Lower values reduce CPU usage. Set to None to use all CPU cores. ",
    )

    # Token counting configuration
    encoding: Literal["o200k_base", "cl100k_base", "p50k_base", "r50k_base"] = Field(
        default="cl100k_base",
        description="Tiktoken encoding to use for token counting",
    )

    # Tool search configuration
    max_tools_to_return: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Maximum number of tools to return in search results (1-50)",
    )

    tool_distance_threshold: float = Field(
        default=1.0, ge=0.0, le=2.0, description="Distance threshold for tool similarity (0.0-2.0)"
    )

    max_servers_to_return: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of servers to return in search results (1-20)",
    )

    # Hybrid search configuration
    hybrid_search_semantic_ratio: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ratio for semantic vs BM25 search (0.0=all BM25, 1.0=all semantic)",
    )

    enable_embedding_cache: bool = Field(
        default=True, description="Whether to enable embedding caching"
    )

    registry_ingestion_batch_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Batch size for parallel registry server ingestion (1-50)",
    )

    workload_ingestion_batch_size: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Batch size for parallel workload ingestion (1-50)",
    )

    # Tool response limiting configuration
    max_tool_response_tokens: int | None = Field(
        default=None,
        ge=100,
        le=100000,
        description="Maximum number of tokens to return from tool calls (100-100000). "
        "Set to None to disable token limiting. "
        "Responses exceeding this limit will be truncated or sampled.",
    )

    # Group filtering configuration
    allowed_groups: list[str] | None = Field(
        default=None,
        description="List of THV groups to filter tool lookups. "
        "Set to None to allow all groups. "
        "Can be provided as a comma-separated string which will be parsed into a list.",
    )

    # Workload filtering configuration
    skipped_workloads: list[str] = Field(
        default=["inspector", "mcp-optimizer"],
        description="List of workload names to skip during ingestion "
        "(auxiliary/management workloads). "
        "Can be provided as a comma-separated string which will be parsed into a list.",
    )

    # Feature flags
    enable_dynamic_install: bool = Field(
        default=False,
        description=(
            "Enable dynamic installation feature "
            "(search_registry and install_server tools). "
            "When disabled, only find_tool, call_tool, and list_tools are available."
        ),
    )
    fastembed_cache_path: str | None = Field(
        default=None, description="Path to FastEmbed cache directory"
    )
    tiktoken_cache_dir: str | None = Field(
        default=None, description="Path to Tiktoken cache directory"
    )

    @field_validator("skipped_workloads", mode="before")
    @classmethod
    def parse_skipped_workloads(cls, v) -> list[str]:
        """Parse skipped_workloads from string or list into a list of workload names.

        Args:
            v: Either a comma-separated string or a list of workload names

        Returns:
            List of workload names to skip (never None, defaults to empty list)
        """
        # If None or empty, return default list
        if v is None:
            return ["inspector", "mcp-optimizer"]

        # If already a list, validate and clean it
        if isinstance(v, list):
            workloads = [w.strip() for w in v if isinstance(w, str) and w.strip()]
            return workloads if workloads else ["inspector", "mcp-optimizer"]

        # If string, parse as comma-separated values
        if isinstance(v, str):
            # Handle empty string
            if not v.strip():
                return ["inspector", "mcp-optimizer"]

            # Split by comma and clean up
            workloads = [w.strip() for w in v.split(",") if w.strip()]
            return workloads if workloads else ["inspector", "mcp-optimizer"]

        # If invalid type, return default
        logger.warning(
            "Invalid type for skipped_workloads, using default",
            type=type(v).__name__,
        )
        return ["inspector", "mcp-optimizer"]

    @field_validator("allowed_groups", mode="before")
    @classmethod
    def parse_allowed_groups(cls, v) -> list[str] | None:
        """Parse allowed_groups from string or list into a list of group names.

        Args:
            v: Either a comma-separated string or a list of group names

        Returns:
            List of group names, or None if no filtering should be applied
        """
        # If None, return None
        if v is None:
            return None

        # If already a list, validate and clean it
        if isinstance(v, list):
            groups = [g.strip() for g in v if isinstance(g, str) and g.strip()]
            return groups if groups else None

        # If string, parse comma-separated values
        if isinstance(v, str):
            if not v or not v.strip():
                return None
            groups = [g.strip() for g in v.split(",") if g.strip()]
            return groups if groups else None

        # Invalid type
        raise ValueError(f"allowed_groups must be a string or list, got {type(v)}")

    @model_validator(mode="after")
    def validate_port_range(self):
        """Ensure end port is greater than start port."""
        if self.toolhive_end_port_scan <= self.toolhive_start_port_scan:
            raise ValueError(
                f"End port ({self.toolhive_end_port_scan}) must be greater than "
                f"start port ({self.toolhive_start_port_scan})"
            )
        return self

    @model_validator(mode="after")
    def validate_db_urls(self):
        """Validate that both database URLs are consistent."""
        # Validate database URL format
        for url, field_name in [(self.async_db_url, "async_db_url"), (self.db_url, "db_url")]:
            if url and not url.startswith(
                ("sqlite://", "sqlite+aiosqlite://", "postgresql://", "postgresql+asyncpg://")
            ):
                raise ValueError(
                    f"{field_name} must start with supported scheme "
                    "(sqlite://, sqlite+aiosqlite://, postgresql://, postgresql+asyncpg://)"
                )

        # Ensure both URLs point to the same database
        if self.async_db_url and self.db_url:
            # Extract the database path/name from both URLs
            async_path = self.async_db_url.replace("sqlite+aiosqlite://", "").replace(
                "postgresql+asyncpg://", ""
            )
            sync_path = self.db_url.replace("sqlite://", "").replace("postgresql://", "")

            if async_path != sync_path:
                raise ValueError(
                    f"Database URLs must point to the same database. "
                    f"async_db_url points to '{async_path}' but db_url points to '{sync_path}'"
                )

        return self

    @field_validator("async_db_url", "db_url")
    @classmethod
    def validate_db_url(cls, v):
        """Validate database URL format."""
        if v and not v.startswith(
            ("sqlite://", "sqlite+aiosqlite://", "postgresql://", "postgresql+asyncpg://")
        ):
            raise ValueError(
                "Database URL must start with supported scheme "
                "(sqlite://, sqlite+aiosqlite://, postgresql://, postgresql+asyncpg://)"
            )
        return v


def _setup_secure_database_path(db_path: Path) -> None:
    """Set up and test secure database file with comprehensive validation.

    This function:
    1. Creates parent directories if needed
    2. Tests write access to the directory
    3. Tests SQLite database creation
    4. Pre-creates the database file if it doesn't exist
    5. Sets secure permissions (read/write for owner only)

    Args:
        db_path: Path to the database file

    Raises:
        ConfigurationError: If any setup or testing step fails
    """
    import sqlite3

    try:
        db_dir = db_path.parent
        logger.info(f"Testing write access to database directory: {db_dir}")

        # Create parent directories if they don't exist
        db_dir.mkdir(parents=True, exist_ok=True)

        # Test write access with a temporary file
        test_file = db_dir / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        logger.info(f"Write test successful in {db_dir}")

        # Test SQLite database creation with a temporary database
        test_db = db_dir / ".sqlite_test.db"
        logger.info(f"Testing SQLite database creation at {test_db}")
        conn = sqlite3.connect(str(test_db))
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.close()
        test_db.unlink()
        logger.info("SQLite test successful")

        # Pre-create the actual database file if it doesn't exist
        if not db_path.exists():
            logger.info(f"Pre-creating database file at {db_path}")
            conn = sqlite3.connect(str(db_path))
            conn.close()
            logger.info("Database file created successfully")

        # Set secure permissions (read/write for owner only)
        db_path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        logger.info(f"Set secure permissions for database file: {db_path.name} at {db_path}")

    except (OSError, PermissionError) as e:
        error_msg = f"Failed to set up database file {db_path}: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e
    except Exception as e:
        error_msg = f"Database test failed for {db_path}: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


def _get_default_database_urls() -> tuple[str, str]:
    """Get default database URLs, creating data directory if needed."""
    async_db_url_env = os.getenv("ASYNC_DB_URL")
    db_url_env = os.getenv("DB_URL")

    # Only create data directory if we need to use default database URLs
    if async_db_url_env is None or db_url_env is None:
        # Try to use a writable location for the database
        # First try to create data directory in package root
        try:
            root_dir = Path(__file__).parent.parent.parent.resolve()
            data_dir = root_dir / "data"
            data_dir.mkdir(exist_ok=True)
            default_db_file = data_dir / "mcp_optimizer.db"
        except (OSError, PermissionError):
            # Fallback to /data if package directory is read-only (e.g., in containers)
            logger.warning("Cannot create data directory in package location, using /data instead")
            default_db_file = Path("/data/mcp_optimizer.db")

        # Set up database URLs with secure paths
        default_db_url = f"sqlite:///{default_db_file}"
        default_async_db_url = f"sqlite+aiosqlite:///{default_db_file}"
    else:
        # Use environment-provided URLs as defaults
        default_db_url = db_url_env
        default_async_db_url = async_db_url_env

    return default_db_url, default_async_db_url


def _populate_config_from_env() -> dict[str, Any]:
    """Populate configuration dictionary from environment variables."""
    config_data = {}

    # Only add environment variables to config_data if they are set
    env_mappings = {
        "RUNTIME_MODE": "runtime_mode",
        "LOG_LEVEL": "log_level",
        "MCP_PORT": "mcp_port",
        "RELOAD_SERVER": "reload_server",
        "TOOLHIVE_HOST": "toolhive_host",
        "TOOLHIVE_PORT": "toolhive_port",
        "TOOLHIVE_START_PORT_SCAN": "toolhive_start_port_scan",
        "TOOLHIVE_END_PORT_SCAN": "toolhive_end_port_scan",
        "WORKLOAD_POLLING_INTERVAL": "workload_polling_interval",
        "REGISTRY_POLLING_INTERVAL": "registry_polling_interval",
        "STARTUP_POLLING_DELAY": "startup_polling_delay",
        "MCP_TIMEOUT": "mcp_timeout",
        "EMBEDDING_MODEL_NAME": "embedding_model_name",
        "EMBEDDING_THREADS": "embedding_threads",
        "ENCODING": "encoding",
        "MAX_TOOLS_TO_RETURN": "max_tools_to_return",
        "TOOL_DISTANCE_THRESHOLD": "tool_distance_threshold",
        "MAX_SERVERS_TO_RETURN": "max_servers_to_return",
        "HYBRID_SEARCH_SEMANTIC_RATIO": "hybrid_search_semantic_ratio",
        "TOOLHIVE_TIMEOUT": "toolhive_timeout",
        "TOOLHIVE_MAX_RETRIES": "toolhive_max_retries",
        "TOOLHIVE_INITIAL_BACKOFF": "toolhive_initial_backoff",
        "TOOLHIVE_MAX_BACKOFF": "toolhive_max_backoff",
        "TOOLHIVE_SKIP_BACKOFF": "toolhive_skip_backoff",
        "REGISTRY_INGESTION_BATCH_SIZE": "registry_ingestion_batch_size",
        "WORKLOAD_INGESTION_BATCH_SIZE": "workload_ingestion_batch_size",
        "MAX_TOOL_RESPONSE_TOKENS": "max_tool_response_tokens",
        "ALLOWED_GROUPS": "allowed_groups",
        "SKIPPED_WORKLOADS": "skipped_workloads",
        "RICH_TRACEBACKS": "rich_tracebacks",
        "COLORED_LOGS": "colored_logs",
        "K8S_API_SERVER_URL": "k8s_api_server_url",
        "K8S_NAMESPACE": "k8s_namespace",
        "K8S_ALL_NAMESPACES": "k8s_all_namespaces",
        "ENABLE_DYNAMIC_INSTALL": "enable_dynamic_install",
        "FASTEMBED_CACHE_PATH": "fastembed_cache_path",
        "TIKTOKEN_CACHE_DIR": "tiktoken_cache_dir",
    }

    for env_var, field_name in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            config_data[field_name] = value

    return config_data


def _setup_database_urls(
    config_data: dict[str, Any], default_db_url: str, default_async_db_url: str
) -> None:
    """Set up database URLs in config data."""
    async_db_url = os.getenv("ASYNC_DB_URL")
    if async_db_url is not None:
        config_data["async_db_url"] = async_db_url
    else:
        config_data["async_db_url"] = default_async_db_url

    db_url = os.getenv("DB_URL")
    if db_url is not None:
        config_data["db_url"] = db_url
    else:
        config_data["db_url"] = default_db_url


def _auto_detect_k8s_api_url(config_data: dict[str, Any]) -> None:
    """Auto-detect K8s API server URL when in k8s mode."""
    from mcp_optimizer.toolhive.k8s_client import get_k8s_api_server_url

    runtime_mode = config_data.get("runtime_mode", "docker")
    if isinstance(runtime_mode, str):
        runtime_mode = runtime_mode.lower()

    if runtime_mode == "k8s" and "k8s_api_server_url" not in config_data:
        # Use smart detection: checks KUBERNETES_SERVICE_HOST/PORT_HTTPS,
        # falls back to kubectl proxy URL
        config_data["k8s_api_server_url"] = get_k8s_api_server_url()
        logger.info(
            "Auto-detected K8s API server URL",
            url=config_data["k8s_api_server_url"],
        )


def load_config() -> MCPOptimizerConfig:
    """Load and validate configuration from environment variables."""
    default_db_url, default_async_db_url = _get_default_database_urls()

    try:
        # Load configuration from environment - only include values that are actually set
        # This allows Pydantic Field defaults to be the single source of truth
        config_data = _populate_config_from_env()

        # Handle database URLs specially since they need dynamic default construction
        _setup_database_urls(config_data, default_db_url, default_async_db_url)

        # Auto-detect K8s API server URL when in k8s mode
        _auto_detect_k8s_api_url(config_data)

        # Validate configuration using Pydantic
        config = MCPOptimizerConfig(**config_data)

        # Set up secure database file permissions for SQLite databases
        # Skip for /tmp and /data - let SQLite handle file creation in writable locations
        if config.db_url.startswith("sqlite://"):
            db_path = Path(config.db_url.replace("sqlite://", ""))
            if not str(db_path).startswith(("/tmp", "/data")):  # nosec B108 - Checking paths to skip secure setup, not insecure usage
                _setup_secure_database_path(db_path)
            else:
                logger.info(f"Using writable database location: {db_path}")

        logger.info("Configuration loaded and validated successfully", config=config.model_dump())
        return config

    except ValidationError as e:
        error_msg = f"Configuration validation failed: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e
    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg) from e


# Global configuration instance
_config: Optional[MCPOptimizerConfig] = None


def get_config() -> MCPOptimizerConfig:
    """Get the global configuration instance, loading it if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> MCPOptimizerConfig:
    """Reload configuration from environment variables."""
    global _config
    _config = load_config()
    return _config
