"""Pydantic AI agent for executing AppWorld tasks using MCP Optimizer tools.

This module follows the pattern from examples/anthropic_comparison/mcp_optimizer_agent.py
but extends it to include call_tool and search_in_tool_response tools.
"""

import json
import os
import time
from pathlib import Path

import structlog
from models import ExperimentConfig
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from mcp_optimizer.config import MCPOptimizerConfig
from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.tool_response_ops import ToolResponseOps
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps
from mcp_optimizer.db.workload_tool_ops import WorkloadToolOps
from mcp_optimizer.embeddings import EmbeddingManager
from mcp_optimizer.response_optimizer import ResponseOptimizer
from mcp_optimizer.server import call_tool, find_tool, search_in_tool_response

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are an AI assistant executing tasks in the AppWorld environment.
Your goal is to complete the given task by using the available tools.

Available tools:
1. find_tool - Discover tools that can help with a specific task
   Use this first to find relevant APIs/tools for the task
2. call_tool - Execute a discovered tool with parameters
   Use this to actually perform actions (API calls, etc.)
3. search_in_tool_response - Query stored tool responses for specific information
   Use this when a response was optimized and you need more details

Workflow:
1. Analyze the task instruction carefully
2. Use find_tool to discover relevant tools by describing what you need
3. Select the most appropriate tool from the results
4. Use call_tool to execute the tool with the required parameters
5. If the response was optimized (has response_id), use search_in_tool_response for details
6. Continue calling tools until the task is complete

Important:
- Always use find_tool first to discover available tools
- Use the exact server_name and tool_name from find_tool results when calling tools
- Check response structure - if it contains response_id, the response was optimized
- Follow the task instructions precisely to complete the objective
"""


class AppWorldAgentRunner:
    """Runs Pydantic AI agent for AppWorld tasks using MCP Optimizer tools."""

    def __init__(
        self,
        config: ExperimentConfig,
        db_path: Path,
    ):
        """Initialize agent with MCP Optimizer tools.

        Args:
            config: Experiment configuration
            db_path: Path to the MCP Optimizer database
        """
        self.config = config
        self.db_path = db_path

        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Initialize MCP Optimizer components needed for tools
        self._initialize_mcp_components()

        # Create agent with all three tools
        self.agent: Agent[None, str] = Agent(
            model=OpenAIChatModel(
                config.llm_model, provider=OpenRouterProvider(api_key=openrouter_api_key)
            ),
            system_prompt=SYSTEM_PROMPT,
            tools=[find_tool, call_tool, search_in_tool_response],
            retries=2,
            output_retries=2,
        )

        logger.info(
            "Initialized AppWorld agent",
            model=config.llm_model,
            db_path=str(db_path),
        )

    def _initialize_mcp_components(self) -> None:
        """Initialize MCP Optimizer components for tools.

        Sets up the server module globals needed by find_tool, call_tool,
        and search_in_tool_response.
        """
        import mcp_optimizer.server as server_module

        async_db_url = f"sqlite+aiosqlite:///{self.db_path}"

        # Create config with response optimizer settings
        mcp_config = MCPOptimizerConfig(
            async_db_url=async_db_url,
            db_url=f"sqlite:///{self.db_path}",
            response_optimizer_enabled=True,
            response_optimizer_threshold=self.config.response_optimizer_threshold,
            response_head_lines=self.config.response_head_lines,
            response_tail_lines=self.config.response_tail_lines,
        )

        # Initialize database
        db = DatabaseConfig(database_url=mcp_config.async_db_url)

        # Set server module globals
        server_module._config = mcp_config
        server_module.workload_tool_ops = WorkloadToolOps(db)
        server_module.workload_server_ops = WorkloadServerOps(db)
        server_module.embedding_manager = EmbeddingManager(
            model_name=mcp_config.embedding_model_name,
            enable_cache=mcp_config.enable_embedding_cache,
            threads=mcp_config.embedding_threads,
            fastembed_cache_path=mcp_config.fastembed_cache_path,
        )

        # Initialize response optimizer
        server_module.response_optimizer = ResponseOptimizer(
            token_threshold=mcp_config.response_optimizer_threshold,
            head_lines=mcp_config.response_head_lines,
            tail_lines=mcp_config.response_tail_lines,
        )

        # Initialize tool response ops for KV store
        server_module.tool_response_ops = ToolResponseOps(db)

        logger.info(
            "Initialized MCP Optimizer components",
            response_optimizer_threshold=mcp_config.response_optimizer_threshold,
        )

    async def execute_task(self, instruction: str) -> dict:
        """Execute a single task with the agent.

        Args:
            instruction: The AppWorld task instruction

        Returns:
            dict with:
                - messages: List of agent messages (serialized)
                - tool_calls: Count of each tool type called
                - final_response: Agent's final response
                - execution_time_s: Time taken
                - request_tokens: Request tokens used
                - response_tokens: Response tokens used
        """
        start_time = time.perf_counter()

        try:
            # Run agent with task instruction
            result = await self.agent.run(instruction)

            execution_time = time.perf_counter() - start_time

            # Extract tool call statistics
            tool_stats = self._extract_tool_stats(result)

            # Get token usage
            usage = result.usage()

            return {
                "messages": self._serialize_messages(result),
                "tool_calls": tool_stats,
                "final_response": str(result.output) if result.output else None,
                "execution_time_s": execution_time,
                "request_tokens": usage.input_tokens,
                "response_tokens": usage.output_tokens,
                "error": None,
            }

        except Exception as e:
            logger.exception("Agent execution failed", error=str(e))
            return {
                "messages": [],
                "tool_calls": {
                    "find_tool": 0,
                    "call_tool": 0,
                    "search_in_tool_response": 0,
                    "total": 0,
                },
                "final_response": None,
                "execution_time_s": time.perf_counter() - start_time,
                "request_tokens": 0,
                "response_tokens": 0,
                "error": str(e),
            }

    def _extract_tool_stats(self, result: AgentRunResult) -> dict:
        """Extract tool call statistics from agent result.

        Args:
            result: Agent run result

        Returns:
            dict with counts for each tool type
        """
        stats = {
            "find_tool": 0,
            "call_tool": 0,
            "search_in_tool_response": 0,
            "total": 0,
        }

        for message in result.all_messages():
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        tool_name = part.tool_name
                        if tool_name in stats:
                            stats[tool_name] += 1
                            stats["total"] += 1

        return stats

    def _serialize_messages(self, result: AgentRunResult) -> list[dict]:
        """Serialize agent messages for storage.

        Args:
            result: Agent run result

        Returns:
            List of serialized message dictionaries
        """
        messages = []

        for message in result.all_messages():
            if isinstance(message, ModelResponse):
                msg_data = {
                    "type": "model_response",
                    "parts": [],
                }
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        msg_data["parts"].append(
                            {
                                "type": "tool_call",
                                "tool_name": part.tool_name,
                                "args": json.loads(part.args)
                                if isinstance(part.args, str)
                                else part.args,
                            }
                        )
                    else:
                        msg_data["parts"].append(
                            {
                                "type": "text",
                                "content": str(part),
                            }
                        )
                messages.append(msg_data)

            elif isinstance(message, ModelRequest):
                msg_data = {
                    "type": "model_request",
                    "parts": [],
                }
                for part in message.parts:
                    if isinstance(part, ToolReturnPart):
                        # Truncate long tool returns for storage
                        content = str(part.content)
                        if len(content) > 1000:
                            content = content[:1000] + "..."
                        msg_data["parts"].append(
                            {
                                "type": "tool_return",
                                "tool_name": part.tool_name,
                                "content": content,
                            }
                        )
                    else:
                        msg_data["parts"].append(
                            {
                                "type": "other",
                                "content": str(part)[:500],
                            }
                        )
                messages.append(msg_data)

        return messages
