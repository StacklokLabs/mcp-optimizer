"""Utility functions for exporting comparison results."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import structlog
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .models import ComparisonReport

logger = structlog.get_logger(__name__)


class ResultsExporter:
    """Export comparison results to multiple formats."""

    @staticmethod
    def print_console_summary(report: ComparisonReport) -> None:  # noqa: C901
        """Print formatted summary to console using Rich.

        Args:
            report: Comparison report to display
        """
        console = Console()
        metrics = report.aggregate_metrics

        # Header
        console.print("\n")
        console.print(
            Panel.fit(
                "[bold blue]MCP Optimizer vs Anthropic Native Comparison[/bold blue]",
                border_style="blue",
            )
        )

        # Get all approaches
        approaches = list(metrics.native_accuracy.keys())

        # Summary table - one row per approach
        summary_table = Table(title="Summary", show_header=True)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Approach", style="yellow")
        summary_table.add_column("Native", justify="right", style="magenta")
        summary_table.add_column("MCP Optimizer", justify="right", style="green")
        summary_table.add_column("Difference", justify="right")

        # Accuracy
        for i, approach in enumerate(approaches):
            mcp_text = f"{metrics.mcp_optimizer_accuracy:.2%}" if i == 0 else ""  # Only show once
            summary_table.add_row(
                "Accuracy" if i == 0 else "",
                approach,
                f"{metrics.native_accuracy[approach]:.2%}",
                mcp_text,
                f"{(metrics.mcp_optimizer_accuracy - metrics.native_accuracy[approach]):.2%}",
            )

        # Precision@3
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_precision_at_3:.2%}" if i == 0 else ""
            )  # Only show once
            diff = metrics.mcp_optimizer_precision_at_3 - metrics.native_precision_at_3[approach]
            summary_table.add_row(
                "Precision@3" if i == 0 else "",
                approach,
                f"{metrics.native_precision_at_3[approach]:.2%}",
                mcp_text,
                f"{diff:.2%}",
            )

        # Precision@5
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_precision_at_5:.2%}" if i == 0 else ""
            )  # Only show once
            diff = metrics.mcp_optimizer_precision_at_5 - metrics.native_precision_at_5[approach]
            summary_table.add_row(
                "Precision@5" if i == 0 else "",
                approach,
                f"{metrics.native_precision_at_5[approach]:.2%}",
                mcp_text,
                f"{diff:.2%}",
            )

        # Precision@max
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_precision_at_max:.2%}" if i == 0 else ""
            )  # Only show once
            diff = (
                metrics.mcp_optimizer_precision_at_max - metrics.native_precision_at_max[approach]
            )
            summary_table.add_row(
                "Precision@max" if i == 0 else "",
                approach,
                f"{metrics.native_precision_at_max[approach]:.2%}",
                mcp_text,
                f"{diff:.2%}",
            )

        # Avg position
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_avg_position:.1f}" if i == 0 else ""
            )  # Only show once
            diff = metrics.mcp_optimizer_avg_position - metrics.native_avg_position[approach]
            summary_table.add_row(
                "Avg Position" if i == 0 else "",
                approach,
                f"{metrics.native_avg_position[approach]:.1f}",
                mcp_text,
                f"{diff:.1f}",
            )

        # Avg tools retrieved
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_avg_tools_retrieved:.1f}" if i == 0 else ""
            )  # Only show once
            diff = (
                metrics.mcp_optimizer_avg_tools_retrieved
                - metrics.native_avg_tools_retrieved[approach]
            )
            summary_table.add_row(
                "Avg Tools Retrieved" if i == 0 else "",
                approach,
                f"{metrics.native_avg_tools_retrieved[approach]:.1f}",
                mcp_text,
                f"{diff:.1f}",
            )

        # Search time
        for i, approach in enumerate(approaches):
            mcp_text = (
                f"{metrics.mcp_optimizer_avg_search_time_s:.3f}" if i == 0 else ""
            )  # Only show once
            diff = (
                metrics.mcp_optimizer_avg_search_time_s - metrics.native_avg_search_time_s[approach]
            )
            summary_table.add_row(
                "Search Time (s)" if i == 0 else "",
                approach,
                f"{metrics.native_avg_search_time_s[approach]:.3f}",
                mcp_text,
                f"{diff:.3f}",
            )

        # Error rate
        for i, approach in enumerate(approaches):
            mcp_text = f"{metrics.mcp_optimizer_error_rate:.2%}" if i == 0 else ""  # Only show once
            summary_table.add_row(
                "Error Rate" if i == 0 else "",
                approach,
                f"{metrics.native_error_rate[approach]:.2%}",
                mcp_text,
                f"{(metrics.mcp_optimizer_error_rate - metrics.native_error_rate[approach]):.2%}",
            )

        console.print(summary_table)

        # Token usage table
        token_table = Table(title="Token Usage", show_header=True)
        token_table.add_column("Approach", style="cyan")
        token_table.add_column("Request Tokens", justify="right")
        token_table.add_column("Response Tokens", justify="right")
        token_table.add_column("Total Tokens", justify="right")

        for approach in approaches:
            token_table.add_row(
                f"Native ({approach})",
                f"{metrics.native_avg_request_tokens[approach]:.0f}",
                f"{metrics.native_avg_response_tokens[approach]:.0f}",
                f"{metrics.native_avg_total_tokens[approach]:.0f}",
            )
        token_table.add_row(
            "MCP Optimizer",
            f"{metrics.mcp_optimizer_avg_request_tokens:.0f}",
            f"{metrics.mcp_optimizer_avg_response_tokens:.0f}",
            f"{metrics.mcp_optimizer_avg_total_tokens:.0f}",
        )

        console.print(token_table)

        # Success counts
        console.print("\n[bold]Success Counts:[/bold]")
        for approach in approaches:
            count = metrics.native_success_count[approach]
            total = metrics.total_cases
            console.print(f"  Native ({approach}): {count}/{total}")
        console.print(
            f"  MCP Optimizer: {metrics.mcp_optimizer_success_count}/{metrics.total_cases}"
        )

        # Precision@max counts (tool found at any position)
        console.print("\n[bold]Tool Found Counts (Precision@max):[/bold]")
        for approach in approaches:
            # Count cases where precision@max = 1.0
            count = sum(
                1
                for r in report.individual_results
                if r.native_precision_at_max.get(approach, 0.0) == 1.0
            )
            total = metrics.total_cases
            console.print(f"  Native ({approach}): {count}/{total}")
        mcp_found_count = sum(
            1 for r in report.individual_results if r.mcp_optimizer_precision_at_max == 1.0
        )
        console.print(f"  MCP Optimizer: {mcp_found_count}/{metrics.total_cases}")
        console.print("\n")

    @staticmethod
    def save_json_report(report: ComparisonReport, output_path: Path) -> None:
        """Save full report to JSON file.

        Args:
            report: Comparison report
            output_path: Path to output JSON file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2)

        logger.info("Saved JSON report", path=str(output_path))

    @staticmethod
    def save_csv_summary(report: ComparisonReport, output_path: Path) -> None:
        """Save individual results to CSV for analysis.

        Args:
            report: Comparison report
            output_path: Path to output CSV file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Get all approaches from first result
        if not report.individual_results:
            return

        approaches = list(report.individual_results[0].native_results.keys())

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Header - dynamically create columns for each approach
            header = ["query", "target_server", "target_tool"]
            for approach in approaches:
                header.extend(
                    [
                        f"native_{approach}_selected_tool",
                        f"native_{approach}_is_correct",
                        f"native_{approach}_search_time_s",
                        f"native_{approach}_tokens",
                        f"native_{approach}_error",
                    ]
                )
            header.extend(
                [
                    "mcp_selected_server",
                    "mcp_selected_tool",
                    "mcp_is_correct",
                    "mcp_precision_at_3",
                    "mcp_precision_at_5",
                    "mcp_tool_position",
                    "mcp_search_time_s",
                    "mcp_tokens",
                    "mcp_num_tools_returned",
                    "mcp_error",
                ]
            )
            writer.writerow(header)

            # Rows
            for result in report.individual_results:
                row = [
                    result.test_case.query,
                    result.test_case.target_mcp_server_name,
                    result.test_case.target_tool_name,
                ]

                # Add columns for each approach
                for approach in approaches:
                    native_result = result.native_results[approach]
                    row.extend(
                        [
                            native_result.selected_tool_name or "",
                            result.native_is_correct[approach],
                            f"{native_result.search_time_s:.3f}",
                            native_result.total_tokens,
                            native_result.error or "",
                        ]
                    )

                # Add MCP columns
                row.extend(
                    [
                        result.mcp_optimizer_result.selected_server_name or "",
                        result.mcp_optimizer_result.selected_tool_name or "",
                        result.mcp_optimizer_is_correct,
                        f"{result.mcp_optimizer_precision_at_3:.2f}",
                        f"{result.mcp_optimizer_precision_at_5:.2f}",
                        result.mcp_optimizer_tool_position
                        if result.mcp_optimizer_tool_position is not None
                        else "",
                        f"{result.mcp_optimizer_result.search_time_s:.3f}",
                        result.mcp_optimizer_result.total_tokens,
                        result.mcp_optimizer_result.num_tools_returned,
                        result.mcp_optimizer_result.error or "",
                    ]
                )

                writer.writerow(row)

        logger.info("Saved CSV report", path=str(output_path))

    @staticmethod
    def save_accuracy_comparison_plot(report: ComparisonReport, output_path: Path) -> None:
        """Save bar plot comparing Selection Accuracy vs Retrieval Accuracy.

        Args:
            report: Comparison report
            output_path: Path to output PNG file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metrics = report.aggregate_metrics
        approaches = list(metrics.native_accuracy.keys())

        # Prepare data - organize by metric type (Selection vs Retrieval)
        approach_labels = []
        approach_colors = []

        # Stacklok-inspired colors for dark theme
        native_colors = ["#4A90E2", "#5BA3F5", "#6FB6FF"]  # Blues for native approaches
        mcp_color = "#FF6B35"  # Orange accent for MCP Optimizer

        # Build approach labels and colors
        for i, approach in enumerate(approaches):
            approach_labels.append(f"Native\n({approach})")
            approach_colors.append(native_colors[i % len(native_colors)])
        approach_labels.append("MCP\nOptimizer")
        approach_colors.append(mcp_color)

        # Get accuracy values
        selection_values = [metrics.native_accuracy[app] * 100 for app in approaches]
        selection_values.append(metrics.mcp_optimizer_accuracy * 100)

        retrieval_values = [metrics.native_precision_at_max[app] * 100 for app in approaches]
        retrieval_values.append(metrics.mcp_optimizer_precision_at_max * 100)

        # Create the plot with dark background
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(12, 7))

        # Set dark background colors
        fig.patch.set_facecolor("#0A0E27")
        ax.set_facecolor("#0A0E27")

        # Plot settings
        num_approaches = len(approach_labels)
        bar_width = 0.35
        group_gap = 1.5  # Gap between Selection and Retrieval groups

        # Create positions for bars
        selection_positions = list(range(num_approaches))
        retrieval_positions = [x + num_approaches + group_gap for x in range(num_approaches)]

        # Plot Selection Accuracy bars
        for pos, val, color in zip(
            selection_positions, selection_values, approach_colors, strict=True
        ):
            ax.bar(pos, val, bar_width * 1.5, color=color, edgecolor="white", linewidth=0.5)

        # Plot Retrieval Accuracy bars
        for pos, val, color in zip(
            retrieval_positions, retrieval_values, approach_colors, strict=True
        ):
            ax.bar(pos, val, bar_width * 1.5, color=color, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        def add_value_labels(positions, values):
            for pos, val in zip(positions, values, strict=True):
                ax.text(
                    pos,
                    val + 2,
                    f"{val:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="white",
                )

        add_value_labels(selection_positions, selection_values)
        add_value_labels(retrieval_positions, retrieval_values)

        # Set x-axis labels
        all_positions = selection_positions + retrieval_positions
        all_labels = approach_labels + approach_labels
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, fontsize=10, color="white")

        # Add group labels
        selection_center = (selection_positions[0] + selection_positions[-1]) / 2
        retrieval_center = (retrieval_positions[0] + retrieval_positions[-1]) / 2

        ax.text(
            selection_center,
            -12,
            "Selection Accuracy",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color="#4A90E2",
        )
        ax.text(
            retrieval_center,
            -12,
            "Retrieval Accuracy",
            ha="center",
            fontsize=13,
            fontweight="bold",
            color="#4A90E2",
        )

        # Customize plot
        ax.set_ylabel("Accuracy (%)", fontsize=13, fontweight="bold", color="white")
        ax.set_title(
            "Tool Search Performance: Selection vs Retrieval Accuracy",
            fontsize=15,
            fontweight="bold",
            color="white",
            pad=20,
        )
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.2, linestyle="--", color="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        ax.tick_params(colors="white")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="#0A0E27")
        plt.close()

        # Reset style to default
        plt.style.use("default")

        logger.info("Saved accuracy comparison plot", path=str(output_path))
