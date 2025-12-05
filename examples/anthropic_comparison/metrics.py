"""Metrics computation for comparison evaluation."""

import json
from pathlib import Path

import click
import structlog
from models import (
    AggregateMetrics,
    ComparisonReport,
    ComparisonResult,
    McpOptimizerSearchResult,
    NativeSearchResult,
    TestCase,
)
from results_exporter import ResultsExporter

logger = structlog.get_logger(__name__)


class MetricsComputer:
    """Computes metrics for comparison results."""

    @staticmethod
    def compute_single_result(
        test_case: TestCase,
        native_results: dict[str, NativeSearchResult],
        mcp_result: McpOptimizerSearchResult,
    ) -> ComparisonResult:
        """Compute metrics for a single test case.

        Args:
            test_case: Test case with expected tool
            native_results: Results from native approaches, keyed by approach
            mcp_result: Result from MCP Optimizer approach

        Returns:
            ComparisonResult with computed metrics
        """
        # Expected tool name format is {server}-{tool}, e.g. "agentql-extract-web-data"
        expected_native_name = (
            f"{test_case.target_mcp_server_name.lower()}-{test_case.target_tool_name}"
        )

        # Compute metrics for each native approach
        native_is_correct = {}
        native_precision_at_3 = {}
        native_precision_at_5 = {}
        native_precision_at_max = {}
        native_tool_position = {}

        for approach, native_result in native_results.items():
            # Check if selected tool matches target
            is_correct = (
                native_result.selected_tool_name is not None
                and native_result.selected_tool_name == expected_native_name
            )
            native_is_correct[approach] = is_correct

            # Find tool position in retrieved tools list
            tool_position = None
            for idx, tool_name in enumerate(native_result.retrieved_tools):
                if tool_name == expected_native_name:
                    tool_position = idx
                    break

            native_tool_position[approach] = tool_position

            # Precision@3: 1.0 if target tool in first 3 retrieved tools, 0.0 otherwise
            native_precision_at_3[approach] = (
                1.0 if tool_position is not None and tool_position < 3 else 0.0
            )

            # Precision@5: 1.0 if target tool in first 5 retrieved tools, 0.0 otherwise
            native_precision_at_5[approach] = (
                1.0 if tool_position is not None and tool_position < 5 else 0.0
            )

            # Precision@max: 1.0 if target tool found at any position, 0.0 otherwise
            native_precision_at_max[approach] = 1.0 if tool_position is not None else 0.0

        # MCP Optimizer: Check if selected tool matches target
        mcp_optimizer_is_correct = (
            mcp_result.selected_server_name == test_case.target_mcp_server_name
            and mcp_result.selected_tool_name == test_case.target_tool_name
        )

        # MCP Optimizer: Find tool position and compute precision
        mcp_optimizer_tool_position = None
        for idx, (server, tool) in enumerate(mcp_result.retrieved_tools):
            if server == test_case.target_mcp_server_name and tool == test_case.target_tool_name:
                mcp_optimizer_tool_position = idx
                break

        # Precision@3: 1.0 if target tool in first 3 retrieved tools, 0.0 otherwise
        mcp_optimizer_precision_at_3 = (
            1.0
            if mcp_optimizer_tool_position is not None and mcp_optimizer_tool_position < 3
            else 0.0
        )

        # Precision@5: 1.0 if target tool in first 5 retrieved tools, 0.0 otherwise
        mcp_optimizer_precision_at_5 = (
            1.0
            if mcp_optimizer_tool_position is not None and mcp_optimizer_tool_position < 5
            else 0.0
        )

        # Precision@max: 1.0 if target tool found at any position, 0.0 otherwise
        mcp_optimizer_precision_at_max = 1.0 if mcp_optimizer_tool_position is not None else 0.0

        return ComparisonResult(
            test_case=test_case,
            native_results=native_results,
            mcp_optimizer_result=mcp_result,
            native_is_correct=native_is_correct,
            mcp_optimizer_is_correct=mcp_optimizer_is_correct,
            native_precision_at_3=native_precision_at_3,
            native_precision_at_5=native_precision_at_5,
            native_precision_at_max=native_precision_at_max,
            mcp_optimizer_precision_at_3=mcp_optimizer_precision_at_3,
            mcp_optimizer_precision_at_5=mcp_optimizer_precision_at_5,
            mcp_optimizer_precision_at_max=mcp_optimizer_precision_at_max,
            native_tool_position=native_tool_position,
            mcp_optimizer_tool_position=mcp_optimizer_tool_position,
        )

    @staticmethod
    def compute_aggregate_metrics(  # noqa: C901
        results: list[ComparisonResult],
    ) -> AggregateMetrics:
        """Compute aggregate metrics across all results.

        Args:
            results: List of comparison results

        Returns:
            AggregateMetrics with summary statistics
        """
        if not results:
            raise ValueError("No results to compute metrics")

        total_cases = len(results)

        # Get all approaches from the first result
        approaches = list(results[0].native_results.keys())

        # Count successes per approach
        native_success_count = {}
        for approach in approaches:
            native_success_count[approach] = sum(
                1 for r in results if r.native_is_correct[approach]
            )

        mcp_optimizer_success_count = sum(1 for r in results if r.mcp_optimizer_is_correct)

        # Accuracy (exact match rate) per approach
        native_accuracy = {}
        for approach in approaches:
            native_accuracy[approach] = native_success_count[approach] / total_cases

        mcp_optimizer_accuracy = mcp_optimizer_success_count / total_cases

        # Precision@3 (target tool in top 3 results rate) per approach
        native_precision_at_3 = {}
        for approach in approaches:
            native_precision_at_3[approach] = (
                sum(r.native_precision_at_3[approach] for r in results) / total_cases
            )

        mcp_optimizer_precision_at_3 = (
            sum(r.mcp_optimizer_precision_at_3 for r in results) / total_cases
        )

        # Precision@5 (target tool in top 5 results rate) per approach
        native_precision_at_5 = {}
        for approach in approaches:
            native_precision_at_5[approach] = (
                sum(r.native_precision_at_5[approach] for r in results) / total_cases
            )

        mcp_optimizer_precision_at_5 = (
            sum(r.mcp_optimizer_precision_at_5 for r in results) / total_cases
        )

        # Precision@max (target tool found at any position rate) per approach
        native_precision_at_max = {}
        for approach in approaches:
            native_precision_at_max[approach] = (
                sum(r.native_precision_at_max[approach] for r in results) / total_cases
            )

        mcp_optimizer_precision_at_max = (
            sum(r.mcp_optimizer_precision_at_max for r in results) / total_cases
        )

        # Average position (only for cases where tool was found) per approach
        native_avg_position = {}
        for approach in approaches:
            positions = [
                r.native_tool_position[approach]
                for r in results
                if r.native_tool_position[approach] is not None
            ]
            native_avg_position[approach] = sum(positions) / len(positions) if positions else 0.0

        mcp_optimizer_positions = [
            r.mcp_optimizer_tool_position
            for r in results
            if r.mcp_optimizer_tool_position is not None
        ]
        mcp_optimizer_avg_position = (
            sum(mcp_optimizer_positions) / len(mcp_optimizer_positions)
            if mcp_optimizer_positions
            else 0.0
        )

        # Search time per approach
        native_avg_search_time_s = {}
        for approach in approaches:
            native_avg_search_time_s[approach] = (
                sum(r.native_results[approach].search_time_s for r in results) / total_cases
            )

        mcp_optimizer_avg_search_time_s = (
            sum(r.mcp_optimizer_result.search_time_s for r in results) / total_cases
        )

        # Token usage per approach
        native_avg_request_tokens = {}
        native_avg_response_tokens = {}
        native_avg_total_tokens = {}
        for approach in approaches:
            native_avg_request_tokens[approach] = (
                sum(r.native_results[approach].request_tokens for r in results) / total_cases
            )
            native_avg_response_tokens[approach] = (
                sum(r.native_results[approach].response_tokens for r in results) / total_cases
            )
            native_avg_total_tokens[approach] = (
                sum(r.native_results[approach].total_tokens for r in results) / total_cases
            )

        mcp_optimizer_avg_request_tokens = (
            sum(r.mcp_optimizer_result.request_tokens for r in results) / total_cases
        )
        mcp_optimizer_avg_response_tokens = (
            sum(r.mcp_optimizer_result.response_tokens for r in results) / total_cases
        )
        mcp_optimizer_avg_total_tokens = (
            sum(r.mcp_optimizer_result.total_tokens for r in results) / total_cases
        )

        # Error rates per approach
        native_error_rate = {}
        for approach in approaches:
            error_count = sum(1 for r in results if r.native_results[approach].error is not None)
            native_error_rate[approach] = error_count / total_cases

        mcp_optimizer_error_count = sum(
            1 for r in results if r.mcp_optimizer_result.error is not None
        )
        mcp_optimizer_error_rate = mcp_optimizer_error_count / total_cases

        # Average number of tools retrieved per approach
        native_avg_tools_retrieved = {}
        for approach in approaches:
            native_avg_tools_retrieved[approach] = (
                sum(r.native_results[approach].num_tools_returned for r in results) / total_cases
            )

        mcp_optimizer_avg_tools_retrieved = (
            sum(r.mcp_optimizer_result.num_tools_returned for r in results) / total_cases
        )

        logger.info(
            "Computed aggregate metrics",
            total_cases=total_cases,
            native_accuracy={k: f"{v:.2%}" for k, v in native_accuracy.items()},
            mcp_optimizer_accuracy=f"{mcp_optimizer_accuracy:.2%}",
        )

        return AggregateMetrics(
            total_cases=total_cases,
            native_success_count=native_success_count,
            mcp_optimizer_success_count=mcp_optimizer_success_count,
            native_accuracy=native_accuracy,
            mcp_optimizer_accuracy=mcp_optimizer_accuracy,
            native_precision_at_3=native_precision_at_3,
            native_precision_at_5=native_precision_at_5,
            native_precision_at_max=native_precision_at_max,
            mcp_optimizer_precision_at_3=mcp_optimizer_precision_at_3,
            mcp_optimizer_precision_at_5=mcp_optimizer_precision_at_5,
            mcp_optimizer_precision_at_max=mcp_optimizer_precision_at_max,
            native_avg_position=native_avg_position,
            mcp_optimizer_avg_position=mcp_optimizer_avg_position,
            native_avg_search_time_s=native_avg_search_time_s,
            mcp_optimizer_avg_search_time_s=mcp_optimizer_avg_search_time_s,
            native_avg_request_tokens=native_avg_request_tokens,
            native_avg_response_tokens=native_avg_response_tokens,
            native_avg_total_tokens=native_avg_total_tokens,
            mcp_optimizer_avg_request_tokens=mcp_optimizer_avg_request_tokens,
            mcp_optimizer_avg_response_tokens=mcp_optimizer_avg_response_tokens,
            mcp_optimizer_avg_total_tokens=mcp_optimizer_avg_total_tokens,
            native_error_rate=native_error_rate,
            mcp_optimizer_error_rate=mcp_optimizer_error_rate,
            native_avg_tools_retrieved=native_avg_tools_retrieved,
            mcp_optimizer_avg_tools_retrieved=mcp_optimizer_avg_tools_retrieved,
        )


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to comparison results JSON file",
)
@click.option(
    "--output-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to save updated results (defaults to same as input)",
)
def recalculate_metrics(input_file: Path, output_file: Path | None) -> None:
    """Recalculate aggregate metrics from a comparison results JSON file.

    This is useful when the metrics computation logic has been updated
    and you want to recalculate metrics for existing results.

    Args:
        input_file: Path to comparison results JSON file
        output_file: Optional path to save updated results (defaults to input_file)
    """
    # Default output to input if not specified
    if output_file is None:
        output_file = input_file

    logger.info("Loading comparison report", input_file=str(input_file))

    # Load the comparison report from JSON
    with open(input_file) as f:
        data = json.load(f)

    # Extract individual results - we'll recalculate metrics from raw data
    raw_results = data["individual_results"]

    logger.info(
        "Loaded report",
        num_cases=len(raw_results),
        dataset=data.get("dataset_path", "unknown"),
    )

    # Recalculate individual result metrics to ensure new fields are populated
    logger.info("Recalculating individual result metrics")
    individual_results = []
    for result_data in raw_results:
        # Extract the core data needed for recalculation
        test_case = TestCase.model_validate(result_data["test_case"])
        native_results = {
            approach: NativeSearchResult.model_validate(native_data)
            for approach, native_data in result_data["native_results"].items()
        }
        mcp_result = McpOptimizerSearchResult.model_validate(result_data["mcp_optimizer_result"])

        # Recalculate all metrics for this result
        recalculated_result = MetricsComputer.compute_single_result(
            test_case=test_case,
            native_results=native_results,
            mcp_result=mcp_result,
        )
        individual_results.append(recalculated_result)

    # Recalculate aggregate metrics from individual results
    logger.info("Recalculating aggregate metrics")
    new_metrics = MetricsComputer.compute_aggregate_metrics(individual_results)

    # Create updated report with new metrics
    report = ComparisonReport(
        aggregate_metrics=new_metrics,
        individual_results=individual_results,
        dataset_path=data.get("dataset_path", "unknown"),
        num_test_cases=len(individual_results),
        timestamp=data.get("timestamp", "unknown"),
        native_model=data.get("native_model", "claude-sonnet-4-5-20250929"),
        mcp_optimizer_model=data.get("mcp_optimizer_model", "anthropic/claude-sonnet-4"),
    )

    # Print summary to console
    ResultsExporter.print_console_summary(report)

    # Save updated report
    ResultsExporter.save_json_report(report, output_file)

    # Generate and save accuracy comparison plot
    plot_file = output_file.parent / f"{output_file.stem}_accuracy_comparison.png"
    ResultsExporter.save_accuracy_comparison_plot(report, plot_file)

    logger.info("Updated metrics saved", output_file=str(output_file))
    print(f"\n✓ Updated results saved to {output_file}")
    print(f"✓ Accuracy comparison plot saved to {plot_file}")


if __name__ == "__main__":
    recalculate_metrics()
