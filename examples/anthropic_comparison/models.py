"""Pydantic models for MCP Optimizer vs Anthropic Native comparison."""

import json
from pathlib import Path

from pydantic import BaseModel, Field


class TestCase(BaseModel):
    """Test case from dataset."""

    query: str = Field(description="User query")
    target_mcp_server_name: str = Field(description="Expected server name")
    target_tool_name: str = Field(description="Expected tool name")
    keywords: str = Field(description="Keywords for BM25 search")
    predicted_server_name: str | None = Field(default=None, description="LLM-predicted server name")
    generated_tool_description: str | None = Field(
        default=None, description="LLM-generated tool description"
    )
    generated_server_description: str | None = Field(
        default=None, description="LLM-generated server description"
    )


class TestDataset(BaseModel):
    """Collection of test cases."""

    cases: list[TestCase]

    @classmethod
    def load_from_json(cls, path: Path, limit: int | None = None) -> "TestDataset":
        """Load test cases from JSON file.

        Args:
            path: Path to test dataset JSON
            limit: Optional limit on number of cases

        Returns:
            TestDataset instance
        """
        with open(path) as f:
            data = json.load(f)

        cases = [TestCase.model_validate(item) for item in data]

        if limit:
            cases = cases[:limit]

        return cls(cases=cases)


class ChosenMcpServerTool(BaseModel):
    """Tool chosen by agent."""

    server_name: str = Field(description="MCP server name")
    tool_name: str = Field(description="Tool name")


class NativeSearchResult(BaseModel):
    """Result from Anthropic native search."""

    test_case: TestCase
    approach: str = Field(description="Search approach used: bm25, regex, or both")
    selected_tool_name: str | None = Field(
        default=None, description="Name of selected tool in format {server}-{tool}"
    )
    retrieved_tools: list[str] = Field(
        default_factory=list,
        description="Tools returned by tool search as tool names",
    )
    num_tools_returned: int = Field(
        default=0, description="Number of tools returned by tool search"
    )
    search_time_s: float = Field(default=0.0, description="Search time in seconds")
    request_tokens: int = Field(default=0, description="Request tokens used")
    response_tokens: int = Field(default=0, description="Response tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens used")
    error: str | None = Field(default=None, description="Error message if failed")


class McpOptimizerSearchResult(BaseModel):
    """Result from MCP Optimizer agent search."""

    test_case: TestCase
    selected_server_name: str | None = Field(default=None, description="Selected MCP server name")
    selected_tool_name: str | None = Field(default=None, description="Selected tool name")
    retrieved_tools: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Tools returned by find_tool as (server, tool) tuples",
    )
    search_time_s: float = Field(default=0.0, description="Search time in seconds")
    request_tokens: int = Field(default=0, description="Request tokens used")
    response_tokens: int = Field(default=0, description="Response tokens generated")
    total_tokens: int = Field(default=0, description="Total tokens used")
    num_tools_returned: int = Field(default=0, description="Number of tools returned by find_tool")
    error: str | None = Field(default=None, description="Error message if failed")


class ComparisonResult(BaseModel):
    """Combined result comparing both approaches."""

    test_case: TestCase
    native_results: dict[str, NativeSearchResult] = Field(
        description="Native results keyed by approach (bm25, regex, both)"
    )
    mcp_optimizer_result: McpOptimizerSearchResult

    # Computed metrics (now keyed by approach)
    native_is_correct: dict[str, bool] = Field(
        description="Whether native approach selected correct tool, by approach"
    )
    mcp_optimizer_is_correct: bool = Field(
        description="Whether MCP Optimizer selected correct tool"
    )
    native_precision_at_3: dict[str, float] = Field(
        description="1.0 if target tool in first 3 retrieved tools, 0.0 otherwise, by approach"
    )
    native_precision_at_5: dict[str, float] = Field(
        description="1.0 if target tool in first 5 retrieved tools, 0.0 otherwise, by approach"
    )
    native_precision_at_max: dict[str, float] = Field(
        default_factory=dict,
        description="1.0 if tool found at any position, 0.0 otherwise, by approach",
    )
    mcp_optimizer_precision_at_3: float = Field(
        description="1.0 if target tool in first 3 retrieved tools, 0.0 otherwise"
    )
    mcp_optimizer_precision_at_5: float = Field(
        description="1.0 if target tool in first 5 retrieved tools, 0.0 otherwise"
    )
    mcp_optimizer_precision_at_max: float = Field(
        default=0.0,
        description="1.0 if target tool found in retrieved tools at any position, 0.0 otherwise",
    )
    native_tool_position: dict[str, int | None] = Field(
        description=(
            "0-based position of target tool in retrieved results (None if not found), by approach"
        )
    )
    mcp_optimizer_tool_position: int | None = Field(
        default=None,
        description="0-based position of target tool in retrieved results (None if not found)",
    )


class AggregateMetrics(BaseModel):
    """Aggregate metrics across all test cases."""

    total_cases: int = Field(description="Total number of test cases")
    native_success_count: dict[str, int] = Field(
        description="Number of correct selections by native approach, by approach"
    )
    mcp_optimizer_success_count: int = Field(
        description="Number of correct selections by MCP Optimizer"
    )
    native_accuracy: dict[str, float] = Field(
        description="Accuracy rate for native approach (0.0-1.0), by approach"
    )
    mcp_optimizer_accuracy: float = Field(description="Accuracy rate for MCP Optimizer (0.0-1.0)")
    native_precision_at_3: dict[str, float] = Field(
        description="Precision@3 for native approach (0.0-1.0), by approach"
    )
    native_precision_at_5: dict[str, float] = Field(
        description="Precision@5 for native approach (0.0-1.0), by approach"
    )
    native_precision_at_max: dict[str, float] = Field(
        description="Precision@max for native approach (0.0-1.0), by approach"
    )
    mcp_optimizer_precision_at_3: float = Field(
        description="Precision@3 for MCP Optimizer (0.0-1.0)"
    )
    mcp_optimizer_precision_at_5: float = Field(
        description="Precision@5 for MCP Optimizer (0.0-1.0)"
    )
    mcp_optimizer_precision_at_max: float = Field(
        description="Precision@max for MCP Optimizer (0.0-1.0)"
    )
    native_avg_position: dict[str, float] = Field(
        description="Average position of target tool (when found), by approach"
    )
    mcp_optimizer_avg_position: float = Field(
        description="Average position of target tool (when found)"
    )
    native_avg_search_time_s: dict[str, float] = Field(
        description="Average search time in seconds, by approach"
    )
    mcp_optimizer_avg_search_time_s: float = Field(description="Average search time in seconds")
    native_avg_request_tokens: dict[str, float] = Field(
        description="Average request tokens used per query, by approach"
    )
    native_avg_response_tokens: dict[str, float] = Field(
        description="Average response tokens used per query, by approach"
    )
    native_avg_total_tokens: dict[str, float] = Field(
        description="Average total tokens used per query, by approach"
    )
    mcp_optimizer_avg_request_tokens: float = Field(
        description="Average request tokens used per query"
    )
    mcp_optimizer_avg_response_tokens: float = Field(
        description="Average response tokens used per query"
    )
    mcp_optimizer_avg_total_tokens: float = Field(description="Average total tokens used per query")
    native_error_rate: dict[str, float] = Field(
        description="Error rate for native approach (0.0-1.0), by approach"
    )
    mcp_optimizer_error_rate: float = Field(description="Error rate for MCP Optimizer (0.0-1.0)")
    native_avg_tools_retrieved: dict[str, float] = Field(
        description="Average number of tools retrieved per query, by approach"
    )
    mcp_optimizer_avg_tools_retrieved: float = Field(
        description="Average number of tools retrieved per query"
    )


class ComparisonReport(BaseModel):
    """Full comparison report."""

    aggregate_metrics: AggregateMetrics
    individual_results: list[ComparisonResult]
    dataset_path: str = Field(description="Path to test dataset")
    num_test_cases: int = Field(description="Number of test cases evaluated")
    timestamp: str = Field(description="Timestamp of evaluation (ISO format)")
    native_model: str = Field(default="claude-sonnet-4-5-20250929", description="Native model used")
    mcp_optimizer_model: str = Field(
        default="anthropic/claude-sonnet-4", description="MCP Optimizer model used"
    )
    baseline_all_tools_tokens: int | None = Field(
        default=None,
        description=(
            "Total tokens used when loading all tools without defer, betas, or search tools"
        ),
    )
