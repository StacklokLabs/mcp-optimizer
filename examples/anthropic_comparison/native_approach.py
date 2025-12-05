"""Native Anthropic search approaches for comparison."""

import asyncio
import os
import time

import structlog
from anthropic import AsyncAnthropic
from models import NativeSearchResult, TestCase

logger = structlog.get_logger(__name__)


class NativeApproachRunner:
    """Runs comparison using Anthropic's native search approaches."""

    APPROACHES = ["bm25", "regex"]

    def __init__(self, api_key: str, all_tools: list[dict]):
        """Initialize with Anthropic API key and converted tools.

        Args:
            api_key: Anthropic API key
            all_tools: List of tools in Anthropic format (from ToolConverter)
        """
        self.client = AsyncAnthropic(api_key=api_key)
        self.all_tools = all_tools
        self.bm25_search_tool = [
            {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
        ]
        self.regex_search_tool = [
            {"type": "tool_search_tool_regex_20251119", "name": "tool_search_tool_regex"}
        ]

    async def measure_baseline_all_tools_tokens(
        self, test_query: str = "List available tools"
    ) -> int:
        """Measure total tokens when loading all tools without defer, betas, or search tools.

        This provides a baseline for comparison - the token cost when all tools are loaded
        upfront without any search mechanism.

        Args:
            test_query: Simple query to use for the test

        Returns:
            Total tokens used (input + output)
        """
        try:
            baseline_tools = [
                {
                    "name": tool.get("name"),
                    "description": tool.get("description"),
                    "input_schema": tool.get("input_schema"),
                }
                for tool in self.all_tools
            ]
            # Call Claude with all tools, no betas, no search tools
            message = await self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=2048,
                tools=baseline_tools,
                messages=[{"role": "user", "content": test_query}],
            )

            total_tokens = message.usage.input_tokens + message.usage.output_tokens
            logger.info(
                "Baseline all tools token measurement",
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                total_tokens=total_tokens,
                num_tools=len(baseline_tools),
            )
            return total_tokens

        except Exception as e:
            logger.exception("Failed to measure baseline all tools tokens", error=str(e))
            return 0

    async def search_tool(self, test_case: TestCase, approach: str = "bm25") -> NativeSearchResult:
        """Execute search using Anthropic native approach.

        Args:
            test_case: Test case with query
            approach: Search approach to use (bm25, regex, or both)

        Returns:
            NativeSearchResult with selected tool and metrics
        """
        if approach not in self.APPROACHES:
            raise ValueError(f"Invalid approach: {approach}. Must be one of {self.APPROACHES}")

        start_time = time.perf_counter()

        try:
            # Determine which search tools to include
            approach_config = {
                "bm25": (self.bm25_search_tool, {"tool_search_tool_bm25"}),
                "regex": (self.regex_search_tool, {"tool_search_tool_regex"}),
            }

            if approach not in approach_config:
                raise ValueError(f"Unsupported approach: {approach}")

            search_tool, skip_tool_names = approach_config[approach]

            # Call Claude with search tool(s) + all tools
            message = await self.client.beta.messages.create(
                model="claude-sonnet-4-5-20250929",
                betas=["advanced-tool-use-2025-11-20"],
                max_tokens=2048,
                tools=search_tool + self.all_tools,
                messages=[{"role": "user", "content": test_case.query}],
            )

            search_time = time.perf_counter() - start_time

            # Extract tool uses and search results from content
            selected_tool = None
            retrieved_tools = []

            for block in message.content:
                # Extract retrieved tools from tool_search_tool_result blocks
                if hasattr(block, "type") and block.type == "tool_search_tool_result":
                    if hasattr(block, "content") and hasattr(block.content, "tool_references"):
                        retrieved_tools = [ref.tool_name for ref in block.content.tool_references]

                # Extract selected tool from tool_use blocks
                if hasattr(block, "type") and block.type == "tool_use":
                    # Skip the search tools themselves
                    if block.name not in skip_tool_names:
                        selected_tool = block.name
                        break

            return NativeSearchResult(
                test_case=test_case,
                approach=approach,
                selected_tool_name=selected_tool,
                retrieved_tools=retrieved_tools,
                num_tools_returned=len(retrieved_tools),
                search_time_s=search_time,
                request_tokens=message.usage.input_tokens,
                response_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
            )

        except Exception as e:
            logger.exception(
                "Native search failed", query=test_case.query, approach=approach, error=str(e)
            )
            return NativeSearchResult(
                test_case=test_case,
                approach=approach,
                selected_tool_name=None,
                retrieved_tools=[],
                num_tools_returned=0,
                search_time_s=time.perf_counter() - start_time,
                request_tokens=0,
                response_tokens=0,
                total_tokens=0,
                error=str(e),
            )


async def main():
    """Debug entrypoint with hardcoded values."""
    import json
    from pathlib import Path

    from tool_converter import ToolConverter

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    # Define paths to the data files
    script_dir = Path(__file__).parent
    tools_file = script_dir / "mcp_tools_cleaned.json"
    tests_file = script_dir / "mcp_tools_cleaned_tests_claude-sonnet-4.json"

    # Load and convert tools (same logic as in comparison_orchestrator.py)
    logger.info("Loading tools from", path=str(tools_file))
    with open(tools_file) as f:
        mcp_servers = json.load(f)

    # Convert all tools using shared ToolConverter
    converter = ToolConverter()
    all_tools = []

    for server in mcp_servers:
        for tool in server["tools"]:
            anthropic_tool = converter.convert_tool(server, tool)
            all_tools.append(anthropic_tool.model_dump())

    logger.info("Converted tools", count=len(all_tools))

    # Load test cases
    logger.info("Loading test cases from", path=str(tests_file))
    with open(tests_file) as f:
        test_data = json.load(f)

    # Use first test case for debugging
    test_case_data = test_data[0]
    test_case = TestCase(
        query=test_case_data["query"],
        target_mcp_server_name=test_case_data["target_mcp_server_name"],
        target_tool_name=test_case_data["target_tool_name"],
        keywords=test_case_data["keywords"],
        predicted_server_name=test_case_data.get("predicted_server_name"),
        generated_tool_description=test_case_data.get("generated_tool_description"),
        generated_server_description=test_case_data.get("generated_server_description"),
    )

    logger.info("Using test case", query=test_case.query, target=test_case.target_tool_name)

    # Initialize runner
    runner = NativeApproachRunner(api_key=api_key, all_tools=all_tools)

    # Run search with different approaches
    print("\n" + "=" * 80)
    print(f"Query: {test_case.query}")
    print(f"Target: {test_case.target_mcp_server_name}/{test_case.target_tool_name}")
    print("=" * 80)

    print("\nTesting BM25 approach...")
    result_bm25 = await runner.search_tool(test_case, approach="bm25")
    print(f"Selected tool: {result_bm25.selected_tool_name}")
    print(f"Retrieved tools: {result_bm25.retrieved_tools}")
    expected_tool = f"{test_case.target_mcp_server_name.lower()}-{test_case.target_tool_name}"
    print(f"Correct: {result_bm25.selected_tool_name == expected_tool}")
    print(f"Time: {result_bm25.search_time_s:.2f}s, Tokens: {result_bm25.total_tokens}")

    print("\nTesting Regex approach...")
    result_regex = await runner.search_tool(test_case, approach="regex")
    print(f"Selected tool: {result_regex.selected_tool_name}")
    print(f"Retrieved tools: {result_regex.retrieved_tools}")
    print(f"Correct: {result_regex.selected_tool_name == expected_tool}")
    print(f"Time: {result_regex.search_time_s:.2f}s, Tokens: {result_regex.total_tokens}")


if __name__ == "__main__":
    asyncio.run(main())
