"""Baseline Pydantic AI agent for executing AppWorld tasks using direct MCP connection.

This module provides a baseline agent that connects directly to the AppWorld MCP server
without using MCP Optimizer. This allows comparison between the optimizer approach
and direct MCP usage.
"""

import os
import time

import structlog
from pydantic_ai import Agent
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.mcp import MCPServerStreamableHTTP
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import UsageLimits

from .agent_messsge_processing import serialize_agent_messages
from .models import ExperimentConfig

logger = structlog.get_logger(__name__)

BASELINE_SYSTEM_PROMPT = """You are an AI assistant executing tasks in the AppWorld environment.
Your goal is to complete the given task by using the available tools.

You have direct access to all AppWorld tools. Use them to complete the task.

Workflow:
1. Analyze the task instruction carefully
2. Use the appropriate tools to complete the task
3. Continue calling tools until the task is complete
4. Use the complete_task tool with the final answer when done

Important supervisor tools:
- show_profile: retrieves the supervisor's profile information
- show_account_passwords: retrieves the supervisor's account passwords
- show_payment_cards: retrieves the supervisor's payment methods
- show_addresses: retrieves the supervisor's saved addresses
- complete_task: marks the task as complete with the final answer

Important:
- Follow the task instructions precisely to complete the objective
- When done, always call complete_task with your final answer
"""


class BaselineAgentRunner:
    """Runs Pydantic AI agent for AppWorld tasks using direct MCP connection."""

    def __init__(self, config: ExperimentConfig):
        """Initialize agent with direct MCP connection.

        Args:
            config: Experiment configuration
        """
        self.config = config

        openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Create MCP server connection
        self.mcp_server = MCPServerStreamableHTTP(url=f"{config.appworld_mcp_url}/mcp/")

        # Create agent with direct MCP tools
        self.agent: Agent[None, str] = Agent(
            model=OpenAIChatModel(
                config.llm_model, provider=OpenRouterProvider(api_key=openrouter_api_key)
            ),
            system_prompt=BASELINE_SYSTEM_PROMPT,
            toolsets=[self.mcp_server],
            retries=2,
            output_retries=2,
        )

        logger.info(
            "Initialized Baseline agent",
            model=config.llm_model,
            mcp_url=config.appworld_mcp_url,
        )

    async def execute_task(self, instruction: str) -> dict:
        """Execute a single task with the agent.

        Args:
            instruction: The AppWorld task instruction

        Returns:
            dict with:
                - messages: List of agent messages (serialized)
                - tool_calls: Count of each tool type called
                - tool_breakdown: Breakdown of tool calls by tool name
                - final_response: Agent's final response
                - execution_time_s: Time taken
                - request_tokens: Request tokens used
                - response_tokens: Response tokens used
        """
        start_time = time.perf_counter()

        try:
            # Run agent with task instruction using MCP context manager
            result = await self.agent.run(
                instruction,
                usage_limits=UsageLimits(request_limit=self.config.max_agent_steps),
            )

            execution_time = time.perf_counter() - start_time

            # Extract tool call statistics
            tool_stats, tool_breakdown = self._extract_tool_stats(result)

            # Get token usage
            usage = result.usage()

            return {
                "messages": serialize_agent_messages(result),
                "tool_calls": tool_stats,
                "tool_breakdown": tool_breakdown,
                "final_response": str(result.output) if result.output else None,
                "execution_time_s": execution_time,
                "request_tokens": usage.input_tokens,
                "response_tokens": usage.output_tokens,
                "error": None,
            }

        except Exception as e:
            logger.exception("Baseline agent execution failed", error=str(e))
            return {
                "messages": [],
                "tool_calls": {
                    "direct_tool_calls": 0,
                    "total": 0,
                },
                "tool_breakdown": {},
                "final_response": None,
                "execution_time_s": time.perf_counter() - start_time,
                "request_tokens": 0,
                "response_tokens": 0,
                "error": str(e),
            }

    def _extract_tool_stats(self, result: AgentRunResult) -> tuple[dict, dict[str, int]]:
        """Extract tool call statistics from agent result.

        Args:
            result: Agent run result

        Returns:
            Tuple of (stats dict, tool breakdown dict)
        """
        stats = {
            "direct_tool_calls": 0,
            "total": 0,
        }
        tool_breakdown: dict[str, int] = {}

        for message in result.all_messages():
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        tool_name = part.tool_name
                        stats["direct_tool_calls"] += 1
                        stats["total"] += 1
                        tool_breakdown[tool_name] = tool_breakdown.get(tool_name, 0) + 1

        return stats, tool_breakdown
