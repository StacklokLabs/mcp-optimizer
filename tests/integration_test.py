# /// script
# dependencies = [
#   "mcp[cli]>=1.12.2",
#   "structlog>=25.4.0",
# ]
# ///

"""
Integration tests for mcp-optimizer server with ToolHive.

This script tests the mcp-optimizer server running in ToolHive by:
- Connecting to the mcp-optimizer server via streamable-http
- Testing ListTools to get available tools
- Testing find_tool from mcp-optimizer to search for time server tools
- Testing search_registry from mcp-optimizer to find servers in the registry
- Testing list_tools from mcp-optimizer to get available tools from installed MCP servers
- Testing call_tool from mcp-optimizer to execute a tool from the time server
"""

import asyncio
import json
import os
import sys
from typing import NamedTuple

import structlog
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult
from mcp.types import Tool as McpTool
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)


class ToolWithServer(McpTool):
    """Extended Tool model that includes the server name for routing calls."""

    model_config = ConfigDict(extra="allow")
    mcp_server_name: str


class ListToolsWithServerResult(BaseModel):
    """ListToolsResult that uses ToolWithServer instead of Tool."""

    tools: list[ToolWithServer]


class McpToolTuple(NamedTuple):
    name: str
    mcp_server_name: str


def _validate_tool_result(tool_call_result: CallToolResult) -> str | None:
    """Parse the tool result text into a dictionary."""
    if not tool_call_result.content:
        logger.error(f"Empty content in tool result: {tool_call_result}")
        return None

    try:
        return tool_call_result.content[0].text
    except Exception:
        logger.error(f"Failed to parse tool result: {tool_call_result}", exc_info=True)
        return None


def _get_list_tools(tool_call_result: CallToolResult) -> ListToolsWithServerResult | None:
    """Parse the tool result text into a ListToolsWithServerResult."""
    result = _validate_tool_result(tool_call_result)
    if result is None:
        return None

    try:
        return ListToolsWithServerResult.model_validate_json(result)
    except Exception:
        logger.error(f"Failed to parse list_tools result: {result}", exc_info=True)
        return None


def _check_expected_tools(
    list_tools: ListToolsWithServerResult, expected_tools: set[McpToolTuple]
) -> bool:
    """
    Check if the expected tools are present in the ListToolsWithServerResult.
    """
    found_tools = {McpToolTuple(tool.name, tool.mcp_server_name) for tool in list_tools.tools}

    if not expected_tools.issubset(found_tools):
        logger.error(f"Expected tools not found. Expected: {expected_tools}, Found: {found_tools}")
        return False

    return True


def _is_find_tool_result_valid(tool_call_result: CallToolResult) -> bool:
    """Check if the result text is a valid find_tool result."""
    list_tools = _get_list_tools(tool_call_result)
    if list_tools is None:
        return False

    expected_tools = {
        McpToolTuple("get_current_time", "time"),
        McpToolTuple("convert_time", "time"),
    }
    return _check_expected_tools(list_tools, expected_tools)


def _is_search_registry_result_valid(tool_call_result: CallToolResult) -> bool:
    """Check if the result text is a valid search_registry result."""
    list_tools = _get_list_tools(tool_call_result)
    if list_tools is None:
        return False

    expected_tools = {McpToolTuple("fetch", "fetch")}
    return _check_expected_tools(list_tools, expected_tools)


def _is_list_tools_result_valid(
    tool_call_result: CallToolResult, expect_fetch_absent: bool
) -> bool:
    """Check if the result text is a valid list_tools result."""
    list_tools = _get_list_tools(tool_call_result)
    if list_tools is None:
        return False

    # Always expect time server tools
    expected_tools = {
        McpToolTuple("get_current_time", "time"),
        McpToolTuple("convert_time", "time"),
    }

    # Conditionally expect fetch tool based on expect_fetch_absent variable
    if not expect_fetch_absent:
        expected_tools.add(McpToolTuple("fetch", "fetch"))

    return _check_expected_tools(list_tools, expected_tools)


def _is_call_tool_result_valid(tool_call_result: CallToolResult) -> bool:
    """Check if the result text is a valid call_tool result."""
    result_text = _validate_tool_result(tool_call_result)
    if result_text is None:
        return False

    try:
        # The result_text is the actual tool response JSON, not a CallToolResult
        time_call_tool = json.loads(result_text)
        if "timezone" not in time_call_tool or "datetime" not in time_call_tool:
            logger.error(
                f"'timezone' or 'datetime' key not found in call_tool result: {time_call_tool}"
            )
            return False
    except Exception:
        logger.error(f"Failed to parse call_tool content: {result_text}", exc_info=True)
        return False

    return True


async def test_mcp_optimizer_integration():
    """Run integration tests against mcp-optimizer server in ToolHive."""
    # Get mcp-optimizer URL from environment or use default
    mcp_optimizer_url = os.getenv("MCP_OPTIMIZER_URL", "http://127.0.0.1:8080/mcp")
    logger.info(f"Connecting to mcp-optimizer server at {mcp_optimizer_url}")

    try:
        async with streamablehttp_client(mcp_optimizer_url) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                # Connection test
                assert session is not None, "Failed to create MCP client session"
                logger.info("Connected to mcp-optimizer server successfully")

                # ListTools - Get all available tools
                logger.info("Listing all tools...")
                list_result = await session.list_tools()
                assert list_result.tools, "No tools returned from list_tools"
                logger.info(f"✓ list_tools returned {len(list_result.tools)} tools")

                # Verify mcp-optimizer tools exist
                tool_names = {tool.name for tool in list_result.tools}
                expected_tools = {
                    "find_tool",
                    "list_tools",
                    "call_tool",
                }

                if os.getenv("ENABLE_DYNAMIC_INSTALL", False):
                    expected_tools.add("search_registry")
                    expected_tools.add("install_server")

                assert expected_tools.issubset(tool_names), (
                    f"Expected tools not found: {expected_tools - tool_names}"
                )
                logger.info(f"✓ All expected mcp-optimizer tools found: {expected_tools}")

                # find_tool - Search for time server tools
                logger.info("Finding tools from time MCP server...")
                find_result = await session.call_tool(
                    "find_tool",
                    {"tool_description": "get current time", "tool_keywords": "time current now"},
                )
                assert _is_find_tool_result_valid(find_result), (
                    "find_tool did not return expected time server tools"
                )
                logger.info("✓ find_tool returned expected time server tools")

                # list_tools - Get tools from installed MCP servers
                logger.info("Listing tools from installed MCP servers...")
                search_result = await session.call_tool("list_tools")
                # Verify fetch tool presence based on environment variable (set on GHA)
                expect_fetch_absent = os.getenv("EXPECT_FETCH_ABSENT", "false").lower() == "true"
                assert _is_list_tools_result_valid(search_result, expect_fetch_absent), (
                    "list_tools did not return expected tools"
                )
                logger.info("✓ list_tools returned expected tools from installed MCP servers")

                # call_tool - Execute a tool from time server
                logger.info("Calling a tool from time server...")
                call_result = await session.call_tool(
                    "call_tool",
                    {
                        "server_name": "time",
                        "tool_name": "get_current_time",
                        "parameters": {"timezone": "UTC"},
                    },
                )
                assert _is_call_tool_result_valid(call_result), (
                    "call_tool did not return valid time data"
                )
                logger.info("✓ call_tool returned valid time data")

                # search_registry - Search for fetch server in registry
                if os.getenv("ENABLE_DYNAMIC_INSTALL", False):
                    logger.info("Searching registry for fetch server...")
                    search_result = await session.call_tool(
                        "search_registry",
                        {
                            "tool_description": "fetch web content",
                            "tool_keywords": "fetch web http",
                        },
                    )
                    assert _is_search_registry_result_valid(search_result), (
                        "search_registry did not return expected fetch server tools"
                    )
                    logger.info("✓ search_registry returned expected fetch server tools")

                return 0

    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(exception_formatter=structlog.dev.plain_traceback),
        ]
    )

    exit_code = asyncio.run(test_mcp_optimizer_integration())
    sys.exit(exit_code)
