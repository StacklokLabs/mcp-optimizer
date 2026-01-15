import time
from contextlib import asynccontextmanager

import structlog
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as McpTool
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from mcp_optimizer.config import MCPOptimizerConfig
from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.db.models import (
    McpStatus,
    RegistryServer,
    RegistryToolWithMetadata,
    TokenMetrics,
    WorkloadToolWithMetadata,
)
from mcp_optimizer.db.registry_server_ops import RegistryServerOps
from mcp_optimizer.db.registry_tool_ops import RegistryToolOps
from mcp_optimizer.db.tool_response_ops import ToolResponseOps
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.install import McpServerInstaller
from mcp_optimizer.mcp_client import MCPServerClient
from mcp_optimizer.response_optimizer import QueryExecutionError, ResponseOptimizer, execute_query
from mcp_optimizer.token_limiter import limit_tool_response
from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.toolhive_client import ToolhiveClient

logger = structlog.get_logger(__name__)


class ToolWithServer(McpTool):
    """Extended Tool model that includes the server name for routing calls."""

    mcp_server_name: str


class McpOptimizerError(Exception):
    """Base exception class for MCP Optimizer errors."""

    pass


class ToolConversionError(McpOptimizerError):
    """Exception raised when tool conversion fails."""

    def __init__(self, tool_id: str, original_error: Exception):
        self.tool_id = tool_id
        self.original_error = original_error
        super().__init__(f"Failed to convert tool {tool_id}: {original_error}")


class ToolDiscoveryError(McpOptimizerError):
    """Exception raised when tool discovery fails."""

    pass


class ServerConnectionError(McpOptimizerError):
    """Exception raised when server connection fails."""

    def __init__(self, server_name: str, original_error: Exception):
        self.server_name = server_name
        self.original_error = original_error
        super().__init__(f"Failed to connect to server '{server_name}': {original_error}")


class ToolExecutionError(McpOptimizerError):
    """Exception raised when tool execution fails."""

    def __init__(self, tool_name: str, server_name: str, original_error: Exception):
        self.tool_name = tool_name
        self.server_name = server_name
        self.original_error = original_error
        super().__init__(
            f"Failed to execute tool '{tool_name}' on server '{server_name}': {original_error}"
        )


class McpInstallationError(McpOptimizerError):
    """Exception raised when MCP server installation fails."""

    def __init__(self, server_name: str, original_error: Exception):
        self.server_name = server_name
        self.original_error = original_error
        super().__init__(f"Failed to install MCP server '{server_name}': {original_error}")


class ResponseQueryError(McpOptimizerError):
    """Exception raised when querying a stored response fails."""

    def __init__(self, response_id: str, query: str, reason: str):
        self.response_id = response_id
        self.query = query
        self.reason = reason
        super().__init__(f"Failed to query response '{response_id}' with '{query}': {reason}")


# Initialize FastMCP - port will be overridden during startup
mcp = FastMCP(name="mcp-optimizer", host="0.0.0.0", port=9900)  # nosec B104 - Intentionally bind to all interfaces for server accessibility

# Global instances - will be initialized with proper config values
embedding_manager: EmbeddingManager | None = None
_config: MCPOptimizerConfig | None = None
# Separated ops classes for registry and workload tables
workload_tool_ops: WorkloadToolOps | None = None
registry_tool_ops: RegistryToolOps | None = None
workload_server_ops: WorkloadServerOps | None = None
registry_server_ops: RegistryServerOps | None = None
mcp_installer: McpServerInstaller | None = None
response_optimizer: ResponseOptimizer | None = None
tool_response_ops: ToolResponseOps | None = None


def _register_tools(config: MCPOptimizerConfig) -> None:
    """Register MCP tools based on configuration.

    Core tools (always registered):
    - find_tool: Find tools from running servers
    - call_tool: Execute tools on servers
    - list_tools: List all available tools

    Response optimization tools (when response_optimizer_enabled is True):
    - search_in_tool_response: Search/query stored tool responses

    Dynamic installation tools (only when enable_dynamic_install is True):
    - search_registry: Search for tools in the registry
    - install_server: Install MCP servers from the registry
    """
    # Always register core tools
    mcp.tool()(find_tool)
    mcp.tool()(call_tool)
    mcp.tool()(list_tools)

    registered_tools = ["find_tool", "call_tool", "list_tools"]

    # Register response optimization tools if enabled
    if config.response_optimizer_enabled:
        mcp.tool()(search_in_tool_response)
        registered_tools.append("search_in_tool_response")

    # Register dynamic installation tools if feature flag is enabled
    # Dynamic installation is not implemented for k8s
    if config.enable_dynamic_install and config.runtime_mode != "k8s":
        mcp.tool()(search_registry)
        mcp.tool()(install_server)
        registered_tools.extend(["search_registry", "install_server"])
        logger.info(
            "Registered tools with dynamic installation feature enabled",
            tools=registered_tools,
            runtime_mode=config.runtime_mode,
        )
    else:
        logger.info(
            "Registered core tools only (dynamic installation disabled)",
            tools=registered_tools,
            runtime_mode=config.runtime_mode,
        )


def initialize_server_components(config: MCPOptimizerConfig) -> None:
    """Initialize server components with configuration values."""
    global embedding_manager, _config, workload_tool_ops, registry_tool_ops
    global workload_server_ops, registry_server_ops, mcp_installer
    global response_optimizer, tool_response_ops
    _config = config
    db = DatabaseConfig(database_url=config.async_db_url)
    # Initialize separated ops classes
    workload_tool_ops = WorkloadToolOps(db)
    registry_tool_ops = RegistryToolOps(db)
    workload_server_ops = WorkloadServerOps(db)
    registry_server_ops = RegistryServerOps(db)
    embedding_manager = EmbeddingManager(
        model_name=config.embedding_model_name,
        enable_cache=config.enable_embedding_cache,
        threads=config.embedding_threads,
        fastembed_cache_path=config.fastembed_cache_path,
    )
    mcp.settings.port = config.mcp_port
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
        skip_backoff=config.toolhive_skip_backoff,
    )
    mcp_installer = McpServerInstaller(
        toolhive_client=toolhive_client, workload_server_ops=workload_server_ops
    )

    # Initialize response optimizer if enabled
    if config.response_optimizer_enabled:
        tool_response_ops = ToolResponseOps(db)
        response_optimizer = ResponseOptimizer(
            token_threshold=config.response_optimizer_threshold,
            head_lines=config.response_head_lines,
            tail_lines=config.response_tail_lines,
        )
        logger.info(
            "Response optimizer enabled",
            threshold=config.response_optimizer_threshold,
            kv_ttl=config.response_kv_ttl,
        )

    # Register tools based on runtime mode
    _register_tools(config)


# Health check endpoint
# Note: Both routes (/ and /health) point to the same function using double decorator pattern.
# This provides flexibility for clients - they can use either the root path or the explicit
# /health endpoint.
@mcp.custom_route("/", methods=["GET"])
@mcp.custom_route("/health", methods=["GET"])
def health_check(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


@asynccontextmanager
async def _performance_timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"{operation_name} completed", duration_seconds=duration)


def _tool_conversion(
    db_tools: list[WorkloadToolWithMetadata] | list[RegistryToolWithMetadata],
) -> list[ToolWithServer]:
    if embedding_manager is None:
        raise RuntimeError("Server components not initialized")

    matching_tools = []
    conversion_failures = []

    for tool_with_metadata in db_tools:
        try:
            # Convert tool details to dict and add server name
            tool_dict = tool_with_metadata.tool.details.model_dump()
            tool_dict["mcp_server_name"] = tool_with_metadata.server_name
            # Create ToolWithServer which includes both tool details and server name
            tool_with_server = ToolWithServer.model_validate(tool_dict)
            matching_tools.append(tool_with_server)
        except Exception as e:
            # Log conversion failure but don't fail the entire operation
            conversion_error = ToolConversionError(tool_with_metadata.tool.id, e)
            conversion_failures.append(conversion_error)
            logger.warning(
                "Tool conversion failed",
                tool_id=tool_with_metadata.tool.id,
                server_name=tool_with_metadata.server_name,
                error=str(e),
                exc_info=False,
            )
            continue

    # Log summary of conversion issues if any occurred
    if conversion_failures:
        logger.warning(
            f"Failed to convert {len(conversion_failures)} out of "
            f"{len(db_tools)} tools. Returning {len(matching_tools)} "
            "valid tools."
        )

    logger.info(
        f"Tool discovery completed: {len(matching_tools)} tools found, "
        f"{len(conversion_failures)} conversion errors",
        cache_stats=embedding_manager.get_cache_stats(),
    )
    return matching_tools


async def find_tool(tool_description: str, tool_keywords: str) -> dict:
    """
    Find and return tools from RUNNING servers that can help accomplish the user's request.

    This searches only currently running MCP servers. If no relevant tools are found,
    use search_registry() to discover tools from servers available in the registry.

    Use this function when you need to:
    - Discover what tools are available for a specific task
    - Find the right tool(s) before attempting to solve a problem
    - Check if required functionality exists in the current environment

    Args:
        tool_description: Description of the task or capability needed
                   (e.g., "web search", "analyze CSV file", "send an email")
        tool_keywords: Space-separated keywords of the task or capability needed.
            These will be used for BM25 text search on available tools.
            (e.g. "list issues github", "SQL query postgres", "Grafana requests slow").

    Returns:
        dict: A dictionary containing:
            - tools: List of available tools matching the query, including:
                * Tool names and descriptions
                * Server names (in the mcp_server_name field)
                * Required parameters and schemas
                * Usage examples where applicable
            - token_metrics: Token efficiency metrics showing:
                * baseline_tokens: Total tokens for all running server tools
                * returned_tokens: Total tokens for returned/filtered tools
                * tokens_saved: Number of tokens saved by filtering
                * savings_percentage: Percentage of tokens saved (0-100)

    Example:
    1) User query: "Find good restaurants in San Jose, California"
    This query requires web search. Call find_tool with tool_description="search the web".

    2) User query: "Get details of an issue in stacklok/toolhive github repository"
    This query requires fetching issue details from github. Call find_tool with
    tool_description="Get issue details from GitHub".
    """
    if embedding_manager is None or _config is None or workload_tool_ops is None:
        raise RuntimeError("Server components not initialized")
    try:
        async with _performance_timer("tool discovery"):
            # Get allowed groups from config
            allowed_groups = _config.allowed_groups

            # Embed the user tool description using optimized caching
            async with _performance_timer("embedding generation"):
                logger.debug(
                    f"Generating embedding for tool description: {tool_description[:100]}..."
                )
                tool_desc_embedding = embedding_manager.generate_embedding([tool_description])[0]

            # Find similar tools using hybrid search (semantic + BM25)
            # Only search RUNNING/STOPPED workload servers
            # Use search_registry() to find registry servers
            async with _performance_timer("hybrid tool search"):
                logger.debug("Searching for similar tools using hybrid search")
                similar_db_tools = await workload_tool_ops.find_similar_tools(
                    query_embedding=tool_desc_embedding,
                    limit=_config.max_tools_to_return,
                    distance_threshold=_config.tool_distance_threshold,
                    query_text=tool_keywords,
                    hybrid_search_semantic_ratio=_config.hybrid_search_semantic_ratio,
                    server_statuses=[McpStatus.RUNNING, McpStatus.STOPPED],
                    allowed_groups=allowed_groups,
                )

            # Convert ToolWithMetadata models to response Tool models with performance tracking
            async with _performance_timer("tool conversion"):
                matching_tools = _tool_conversion(similar_db_tools)

            # Calculate token metrics for efficiency tracking
            async with _performance_timer("token metrics calculation"):
                # T022: Calculate baseline_tokens from all running servers
                baseline_tokens = await workload_tool_ops.sum_token_counts_for_running_servers(
                    allowed_groups=allowed_groups
                )

                # T023: Calculate returned_tokens by summing token_count from filtered tools
                returned_tokens = sum(
                    tool.tool.token_count
                    for tool in similar_db_tools
                    if isinstance(tool, WorkloadToolWithMetadata)
                )

                # T024: Calculate tokens_saved and savings_percentage
                tokens_saved = baseline_tokens - returned_tokens

                # T026: Handle edge case - zero baseline_tokens (no running servers)
                if baseline_tokens > 0:
                    savings_percentage = (tokens_saved / baseline_tokens) * 100.0
                else:
                    savings_percentage = 0.0

                # Create TokenMetrics object with validation
                token_metrics = TokenMetrics(
                    baseline_tokens=baseline_tokens,
                    returned_tokens=returned_tokens,
                    tokens_saved=tokens_saved,
                    savings_percentage=savings_percentage,
                )

                logger.debug(
                    f"Token metrics calculated: baseline={baseline_tokens}, "
                    f"returned={returned_tokens}, saved={tokens_saved}, "
                    f"savings={savings_percentage:.2f}%"
                )

        # T025: Return response with both tools and token_metrics
        return {
            "tools": matching_tools,
            "token_metrics": token_metrics.model_dump(),
        }
    except ValueError as e:
        # Handle validation errors specifically
        logger.error(f"Invalid tool description or embedding error: {e}")
        raise ToolDiscoveryError(f"Invalid tool discovery request: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error during tool discovery: {e}")
        raise ToolDiscoveryError(f"Tool discovery failed: {e}") from e


async def list_tools() -> list[ToolWithServer]:
    """
    List all available tools across all MCP servers.

    Use this function when you need to:
    - See all tools available in the current environment
    - Browse the complete catalog of available tools
    - Get an overview of all capabilities without filtering

    Returns:
        list[ToolWithServer]: All available tools, including:
                        - Tool names and descriptions
                        - Server names (in the mcp_server_name field)
                        - Required parameters and schemas
                        - Usage examples where applicable
    """
    if embedding_manager is None or _config is None or workload_tool_ops is None:
        logger.error("Server components not initialized - returning empty tool list")
        return []

    try:
        async with _performance_timer("list all tools"):
            # Get allowed groups from config
            allowed_groups = _config.allowed_groups

            # Get all tools from workload servers (RUNNING/STOPPED)
            async with _performance_timer("fetch all tools"):
                logger.debug("Fetching all tools from workload servers")
                all_db_tools = await workload_tool_ops.get_all_tools(
                    server_statuses=[McpStatus.RUNNING], allowed_groups=allowed_groups
                )

            # Convert ToolWithMetadata models to response Tool models
            async with _performance_timer("tool conversion"):
                all_tools = _tool_conversion(all_db_tools)

        return all_tools
    except Exception as e:
        # Log the error but return empty list instead of raising
        # This ensures the client always gets a valid response
        logger.exception(
            "Unexpected error during tool listing - returning empty list",
            error=str(e),
            error_type=type(e).__name__,
        )
        return []


async def search_registry(tool_description: str, tool_keywords: str) -> list[ToolWithServer]:
    """
    Search for tools in the ToolHive registry when find_tool returns no results
    or irrelevant results.

    This searches MCP servers available in the ToolHive registry but not currently running.
    Use this when:
    - find_tool() returns no results or irrelevant results
    - You want to discover what servers are available to install
    - You need functionality not currently available in running servers

    Note: Registry tools have limited information (name only, no detailed descriptions).
    After finding a needed tool, use install_server() to make it available.

    Args:
        tool_description: Description of the capability you need
                         (e.g., "analyze code", "manage cloud resources")
        tool_keywords: Space-separated keywords for BM25 search
                      (e.g., "code analysis", "aws cloud management")

    Returns:
        list[ToolWithServer]: Tools from registry servers (status=registry).
        Each tool's mcp_server_name field contains the server to install.

    Example workflow:
    1. find_tool("github integration") returns no results
    2. search_registry("github integration", "github repository") finds github server tools
    3. install_server("github") installs the GitHub MCP server
    4. find_tool("github integration") now returns GitHub tools
    """
    if embedding_manager is None or _config is None or registry_tool_ops is None:
        raise RuntimeError("Server components not initialized")
    try:
        async with _performance_timer("registry tool discovery"):
            # Get allowed groups from config
            allowed_groups = _config.allowed_groups

            # Embed the user tool description
            async with _performance_timer("embedding generation"):
                logger.debug(
                    f"Generating embedding for registry search: {tool_description[:100]}..."
                )
                tool_desc_embedding = embedding_manager.generate_embedding([tool_description])[0]

            # Find similar tools from registry servers using hybrid search
            async with _performance_timer("hybrid registry search"):
                logger.debug("Searching for similar tools in registry using hybrid search")
                similar_db_tools = await registry_tool_ops.find_similar_tools(
                    query_embedding=tool_desc_embedding,
                    limit=_config.max_tools_to_return,
                    distance_threshold=_config.tool_distance_threshold,
                    query_text=tool_keywords,
                    hybrid_search_semantic_ratio=_config.hybrid_search_semantic_ratio,
                    allowed_groups=allowed_groups,
                )

            # Convert ToolWithMetadata models to response Tool models
            async with _performance_timer("tool conversion"):
                matching_tools = _tool_conversion(similar_db_tools)

        return matching_tools
    except ValueError as e:
        logger.error(f"Invalid registry search request or embedding error: {e}")
        raise ToolDiscoveryError(f"Invalid registry search request: {e}") from e
    except Exception as e:
        logger.exception(f"Unexpected error during registry search: {e}")
        raise ToolDiscoveryError(f"Registry search failed: {e}") from e


async def _apply_response_optimization(
    tool_result: CallToolResult,
    tool_name: str,
    server_name: str,
) -> CallToolResult:
    """Apply response optimization or token limiting to a tool result.

    Only TextContent items are optimized. Other content types (images, etc.)
    are left untouched and returned as-is.
    """
    if _config is None:
        return tool_result

    # Apply response optimization if enabled (takes precedence over simple limiting)
    if _config.response_optimizer_enabled and response_optimizer is not None:
        # Extract text content from the result - only TextContent is optimized
        text_contents = [c for c in tool_result.content if isinstance(c, TextContent)]
        # Keep non-text content unchanged (images, etc.)
        non_text_contents = [c for c in tool_result.content if not isinstance(c, TextContent)]

        if text_contents:
            # Combine all text content for optimization
            combined_text = "\n".join(c.text for c in text_contents)

            # Optimize the response
            optimized = await response_optimizer.optimize(
                content=combined_text,
                tool_name=tool_name,
                max_tokens=_config.response_optimizer_threshold,
            )

            if optimized.was_optimized:
                logger.info(
                    "Tool response was optimized",
                    tool_name=tool_name,
                    server_name=server_name,
                    original_tokens=optimized.token_metrics.baseline_tokens,
                    final_tokens=optimized.token_metrics.returned_tokens,
                    savings_percentage=f"{optimized.token_metrics.savings_percentage:.1f}%",
                    content_type=optimized.content_type.value,
                )

                # Store original in KV store if tool_response_ops is available
                if tool_response_ops is not None:
                    await tool_response_ops.create_tool_response(
                        tool_name=tool_name,
                        original_content=combined_text,
                        content_type=optimized.content_type,
                        session_key=optimized.session_key,
                        ttl_seconds=_config.response_kv_ttl,
                        metadata={
                            "server_name": server_name,
                            "response_id": optimized.response_id,
                        },
                    )

            # Always return structured response for text content (optimized or not)
            # Serialize the full OptimizedResponse so LLM has access to metadata
            optimized_content = TextContent(type="text", text=optimized.model_dump_json(indent=2))
            # Combine: structured text content first, then non-text content unchanged
            tool_result.content = [optimized_content] + non_text_contents

        return tool_result

    # Fall back to simple token limiting if configured (legacy behavior)
    if _config.max_tool_response_tokens is not None:
        limited = limit_tool_response(tool_result, _config.max_tool_response_tokens)

        if limited.was_truncated:
            logger.warning(
                "Tool response was truncated due to token limit",
                tool_name=tool_name,
                server_name=server_name,
                original_tokens=limited.original_tokens,
                final_tokens=limited.final_tokens,
                max_tokens=_config.max_tool_response_tokens,
            )

            # Prepend truncation message to the response content
            truncation_notice = TextContent(type="text", text=limited.truncation_message or "")
            limited.result.content.insert(0, truncation_notice)

        return limited.result

    # No optimization or token limiting configured, return result as-is
    return tool_result


async def call_tool(server_name: str, tool_name: str, parameters: dict) -> CallToolResult:
    """
    Execute a specific tool with the provided parameters.

    Use this function to:
    - Run a tool after identifying it with find_tool()
    - Execute operations that require specific MCP server functionality
    - Perform actions that go beyond your built-in capabilities

    Args:
        server_name: The name of the MCP server that provides the tool
                    (obtain this from find_tool() results - it's the mcp_server_name field)
        tool_name: The name of the tool to execute
                  (obtain this from find_tool() results - it's the tool's name field)
        parameters: Dictionary of arguments required by the tool
                   (structure must match the tool's schema from find_tool())

    Returns:
        CallToolResult: The output from the tool execution. When response optimization
                       is enabled, TextContent responses are returned as structured JSON with:
                       - response_id: UUID for retrieving original content from KV store
                       - content: The optimized or original content
                       - was_optimized: Boolean flag indicating if optimization was applied
                       - hints: Query hints for retrieving specific parts of original content
                         (only present when was_optimized is true)
                       - token_metrics: Token efficiency metrics with fields:
                         - baseline_tokens: Original token count
                         - returned_tokens: Final token count after optimization
                         - tokens_saved: Number of tokens saved
                         - savings_percentage: Percentage of tokens saved (0-100)

                       Non-text content (images, etc.) is returned unchanged.

    Important: Always use find_tool() first to get the correct server_name and tool_name
              and parameter schema before calling this function.
    """
    if (
        workload_tool_ops is None
        or workload_server_ops is None
        or embedding_manager is None
        or _config is None
    ):
        raise RuntimeError("Server components not initialized")
    try:
        # Verify tool exists in workload database using server name and tool name
        logger.info(f"Verifying tool '{tool_name}' exists in server '{server_name}'")
        try:
            await workload_tool_ops.get_tool_by_server_and_name(server_name, tool_name)
        except Exception as e:
            logger.error(f"Tool verification failed: {e}")
            raise ToolExecutionError(tool_name, server_name, e) from e

        # Get the workload server details for the tool
        logger.info(f"Fetching server details for server: {server_name}")
        try:
            server = await workload_server_ops.get_server_by_name(server_name)
        except Exception as e:
            logger.error(f"Server lookup failed: {e}")
            raise ServerConnectionError(server_name, e) from e

        # Create a workload object for the MCP client
        # Map the transport type from DB to proxy_mode for the MCP client
        proxy_mode = server.transport.value  # "sse" or "streamable-http"
        workload = Workload(name=server.name, url=server.url, proxy_mode=proxy_mode)

        # Create MCP client and call the tool
        logger.info(
            f"Calling tool '{tool_name}' on server '{server_name}' with parameters: {parameters}"
        )
        mcp_client = MCPServerClient(
            workload, timeout=_config.mcp_timeout, runtime_mode=_config.runtime_mode
        )

        # Call the tool using the MCP client
        try:
            tool_result = await mcp_client.call_tool(tool_name, parameters)

            # Apply response optimization or token limiting
            return await _apply_response_optimization(tool_result, tool_name, server_name)
        except Exception as e:
            logger.exception("Tool execution failed")
            raise ToolExecutionError(tool_name, server_name, e) from e

    except (ToolExecutionError, ServerConnectionError):
        # Re-raise our custom exceptions without wrapping
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during tool execution: {e}")
        raise ToolExecutionError(tool_name, server_name, e) from e


async def install_server(server_name: str) -> str:
    """
    Install and start an MCP server from the ToolHive registry.

    Use this after search_registry() identifies a needed server.
    This creates a workload in ToolHive with the server's default configuration,
    making its tools available through find_tool() and call_tool().

    Args:
        server_name: Name of the server from search_registry() results
                    (found in the mcp_server_name field)

    Returns:
        Success message with workload details or error message

    Example workflow:
        1. search_registry() returns tools from "github" server
        2. install_server("github") starts the GitHub MCP server
        3. Wait a few moments for the server to start
        4. find_tool() can now discover GitHub tools
        5. call_tool() can execute GitHub tools

    Note: After installation, the server will be available in the next polling cycle.
          You may need to wait 30-60 seconds before the tools appear in find_tool().
    """
    if registry_server_ops is None or mcp_installer is None:
        raise RuntimeError("Server components not initialized")

    try:
        # Verify server exists in registry
        logger.info(f"Attempting to install server from registry: {server_name}")
        try:
            server = await registry_server_ops.get_server_by_name(server_name)
            if not isinstance(server, RegistryServer):
                raise ValueError(f"Server '{server_name}' is not a registry server")
        except Exception as e:
            error_msg = (
                f"Server '{server_name}' not found in database. "
                f"Use search_registry() to find available servers."
            )
            logger.error(error_msg, error=str(e))
            raise McpInstallationError(server_name, e) from e

        msg = await mcp_installer.install(server)
        return msg
    except Exception as e:
        error_msg = f"Unexpected error installing server '{server_name}': {str(e)}"
        logger.exception("Unexpected error during server installation", error=str(e))
        raise McpInstallationError(server_name, e) from e


async def search_in_tool_response(response_id: str, query: str) -> str:
    """
    Search or extract specific content from a previously optimized tool response.

    Use this function when:
    - A previous call_tool() response was optimized and you need more details
    - The optimized response includes hints suggesting you can query for more data
    - You need to retrieve the full original content or specific parts of it

    The query format depends on the original content type:
    - JSON content: Use JQ syntax (e.g., ".results", ".[0].name", ".data | length")
    - Markdown content: Specify section headers (e.g., "## Installation", "Getting Started")
    - Unstructured text: Use shell-like commands:
        * "head -n 50" - First 50 lines
        * "tail -n 20" - Last 20 lines
        * "lines 10-50" - Lines 10 through 50
        * "grep error" - Lines containing "error"
        * "grep -i warning" - Case-insensitive search for "warning"
    - Any content type: Use "full" to retrieve the complete original response

    Note: If the query result exceeds the configured token threshold, it will be
    automatically optimized to fit within budget while preserving key information.

    Args:
        response_id: The UUID from a previous optimized response
                    (found in the response_id field of call_tool() output)
        query: The search query appropriate for the content type:
              - For JSON: JQ expression (e.g., ".items[0]", ".data.users")
              - For Markdown: Section header (e.g., "## API Reference")
              - For text: Shell command (e.g., "grep ERROR", "head -n 100")
              - For any type: "full" returns the complete original content

    Returns:
        str: The extracted content matching your query. If the result exceeds
             the token threshold, it will be returned as an optimized response
             with the same structured format as call_tool() output.

    Examples:
        # Get specific field from JSON response
        search_in_tool_response("abc-123", ".results[0].title")

        # Get a markdown section
        search_in_tool_response("def-456", "## Installation")

        # Search for errors in log output
        search_in_tool_response("ghi-789", "grep -i error")

        # Get the complete original response
        search_in_tool_response("jkl-012", "full")
    """
    if tool_response_ops is None or _config is None:
        raise RuntimeError("Server components not initialized")

    # Retrieve the stored response from KV store
    try:
        stored_response = await tool_response_ops.get_tool_response(response_id)
    except DbNotFoundError as e:
        raise ResponseQueryError(
            response_id=response_id,
            query=query,
            reason="Response not found or has expired. Responses are stored temporarily.",
        ) from e

    original_content = stored_response.original_content
    content_type = stored_response.content_type

    logger.info(
        "Searching in stored tool response",
        response_id=response_id,
        query=query,
        content_type=content_type.value,
        content_length=len(original_content),
    )

    if query.strip().lower() == "full":
        logger.info(
            "Query requests full original content",
            response_id=response_id,
        )
        return original_content

    # Execute the query
    try:
        result = execute_query(original_content, content_type, query)
    except QueryExecutionError as e:
        raise ResponseQueryError(
            response_id=response_id,
            query=query,
            reason=e.reason,
        ) from e
    except Exception as e:
        raise ResponseQueryError(
            response_id=response_id,
            query=query,
            reason=f"Query execution failed: {str(e)}",
        ) from e

    # Check if result exceeds threshold and re-optimize if needed
    if response_optimizer is not None and _config.response_optimizer_enabled:
        threshold = _config.response_optimizer_threshold
        # Use the optimizer's token estimation
        estimated_tokens = len(result) // 4  # Rough estimate: ~4 chars per token

        if estimated_tokens > threshold:
            logger.info(
                "Query result exceeds threshold, applying optimization",
                response_id=response_id,
                estimated_tokens=estimated_tokens,
                threshold=threshold,
            )

            # Re-optimize the query result
            optimized = await response_optimizer.optimize(
                content=result,
                tool_name=f"search_in_tool_response:{stored_response.tool_name}",
                max_tokens=threshold,
            )

            return optimized.content

    return result


# Create the starlette app first
starlette_app = mcp.streamable_http_app()
