"""MCP Optimizer agent-based approach for comparison."""

import json
import os
import time
from pathlib import Path

import structlog
from mcp.types import ListToolsResult
from models import ChosenMcpServerTool, McpOptimizerSearchResult, TestCase
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from mcp_optimizer.config import MCPOptimizerConfig
from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.server import find_tool

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are a tool selection agent designed to identify the most appropriate tool
for solving user queries. Your primary function is to analyze user requests and recommend
the single best tool to address their needs.

Instructions:
- Analyze the user's query to understand their specific need or problem
- Use the find_tool function exactly once to search for relevant tools
- Select the most appropriate tool from the results
- Respond with your recommendation
"""


class McpOptimizerAgentRunner:
    """Runs comparison using agent with find_tool."""

    def __init__(
        self,
        llm_model: str = "anthropic/claude-sonnet-4.5",
        test_db_path: Path | None = None,
    ):
        """Initialize agent with find_tool as available tool.

        Args:
            llm_model: OpenRouter model to use
            test_db_path: Optional path to test database
                (defaults to mcp_optimizer_test.db in current directory)
        """
        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Initialize MCP Optimizer components needed for find_tool
        # We only need: embedding_manager, _config, workload_tool_ops
        if test_db_path is None:
            # Default to test database in examples/anthropic_comparison directory
            test_db_path = (Path(__file__).parent / "mcp_optimizer_test.db").resolve()

        async_db_url = f"sqlite+aiosqlite:///{test_db_path}"

        config = MCPOptimizerConfig(
            async_db_url=async_db_url,
            db_url=f"sqlite:///{test_db_path}",
        )

        # Initialize only what find_tool needs, avoiding ToolhiveClient initialization
        import mcp_optimizer.server as server_module

        server_module._config = config
        db = DatabaseConfig(database_url=config.async_db_url)
        server_module.workload_tool_ops = WorkloadToolOps(db)
        server_module.embedding_manager = EmbeddingManager(
            model_name=config.embedding_model_name,
            enable_cache=config.enable_embedding_cache,
            threads=config.embedding_threads,
            fastembed_cache_path=config.fastembed_cache_path,
        )

        logger.info(
            "Initialized MCP Optimizer components for find_tool",
            db_path=str(test_db_path),
        )

        # Create agent with find_tool
        self.agent: Agent[None, ChosenMcpServerTool] = Agent(
            model=OpenAIChatModel(
                llm_model, provider=OpenRouterProvider(api_key=openrouter_api_key)
            ),
            system_prompt=SYSTEM_PROMPT,
            tools=[find_tool],
            retries=2,
            output_retries=2,
            output_type=ChosenMcpServerTool,
        )

    async def search_tool(self, test_case: TestCase) -> McpOptimizerSearchResult:
        """Execute search using agent with find_tool.

        Args:
            test_case: Test case with query

        Returns:
            McpOptimizerSearchResult with selected tool and metrics
        """
        start_time = time.perf_counter()

        try:
            # Run agent with test query
            result = await self.agent.run(test_case.query)

            search_time = time.perf_counter() - start_time

            # Extract tool calls and results from messages
            tools_retrieved, made_queries = self._get_tool_calls(result)

            # Parse retrieved tools
            retrieved_tools = self._parse_tools(tools_retrieved)

            # Get token usage
            usage = result.usage()

            actual_output = ChosenMcpServerTool.model_validate(result.output)

            return McpOptimizerSearchResult(
                test_case=test_case,
                selected_server_name=actual_output.server_name,
                selected_tool_name=actual_output.tool_name,
                retrieved_tools=retrieved_tools,
                search_time_s=search_time,
                request_tokens=usage.input_tokens,
                response_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
                num_tools_returned=len(retrieved_tools),
            )

        except Exception as e:
            logger.exception(
                "MCP Optimizer agent search failed", query=test_case.query, error=str(e)
            )
            return McpOptimizerSearchResult(
                test_case=test_case,
                selected_server_name=None,
                selected_tool_name=None,
                retrieved_tools=[],
                search_time_s=time.perf_counter() - start_time,
                request_tokens=0,
                response_tokens=0,
                total_tokens=0,
                num_tools_returned=0,
                error=str(e),
            )

    def _get_tool_calls(
        self, agent_response: AgentRunResult
    ) -> tuple[list[ListToolsResult], list[dict]]:
        """Extract tool calls from agent messages.

        Following agent_evaluation.py pattern:
        - Walk through agent_response.all_messages()
        - Find ModelResponse with ToolCallPart for find_tool
        - Find ModelRequest with ToolReturnPart with results

        Args:
            agent_response: Agent run result

        Returns:
            Tuple of (tools_retrieved, made_queries)
        """
        current_query = None
        tools_retrieved = []
        made_queries = []

        for message in agent_response.all_messages():
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart) and part.tool_name == "find_tool":
                        current_query = json.loads(part.args)
                        made_queries.append(current_query)
            elif isinstance(message, ModelRequest):
                for part in message.parts:
                    if isinstance(part, ToolReturnPart) and part.tool_name == "find_tool":
                        if current_query is None:
                            logger.warning("Find query in response but no find_tool call found")
                            continue
                        tools_response = ListToolsResult.model_validate(part.content)
                        tools_retrieved.append(tools_response)
                        current_query = None

        return tools_retrieved, made_queries

    def _parse_tools(self, tools_in_response: list[ListToolsResult]) -> list[tuple[str, str]]:
        """Parse tools from ListToolsResult.

        Args:
            tools_in_response: List of ListToolsResult from find_tool

        Returns:
            List of (server_name, tool_name) tuples
        """
        if not tools_in_response:
            return []
        elif len(tools_in_response) > 1:
            logger.warning(
                "Multiple find_tool responses found, using first",
                count=len(tools_in_response),
            )

        # Use first response
        tools_result = tools_in_response[0]
        return [(tool.mcp_server_name, tool.name) for tool in tools_result.tools]
