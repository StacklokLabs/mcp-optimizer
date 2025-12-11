import asyncio
import os
from pathlib import Path

import click
from comparison_orchestrator import ComparisonOrchestrator
from results_exporter import ResultsExporter

from mcp_optimizer.configure_logging import configure_logging


@click.command()
@click.option(
    "--dataset",
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent / Path("mcp_tools_cleaned_tests_claude-sonnet-4.json"),
    help="Path to test dataset JSON",
)
@click.option(
    "--output-file",
    type=click.Path(resolve_path=True, dir_okay=False, file_okay=True, path_type=Path),
    default=Path(__file__).parent / Path("results.json"),
    help="Output file path for results JSON",
)
@click.option(
    "--limit",
    type=int,
    default=None,
    help="Limit number of test cases (for testing)",
)
@click.option(
    "--max-concurrency",
    type=int,
    default=10,
    help="Max concurrent test executions",
)
@click.option(
    "--llm-model",
    type=str,
    default="anthropic/claude-sonnet-4.5",
    help="LLM model for MCP Optimizer agent",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from existing results file (skips completed tests, retries errors)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Validate setup without running comparison",
)
def main(
    dataset: Path,
    output_file: Path,
    limit: int | None,
    max_concurrency: int,
    llm_model: str,
    resume: bool,
    dry_run: bool,
):
    """Run MCP Optimizer vs Anthropic Native comparison."""

    # Setup logging
    configure_logging("INFO", rich_tracebacks=False, colored_logs=True)

    # Check for API keys
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

    if not anthropic_api_key or not openrouter_api_key:
        raise ValueError("ANTHROPIC_API_KEY and OPENROUTER_API_KEY required")

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine resume path
    resume_from = output_file if resume and output_file.exists() else None

    if resume and resume_from:
        print(f"Resuming from existing results: {output_file}")
    elif resume and not resume_from:
        print(f"Resume flag set but no existing results found at {output_file}, starting fresh")

    # Create orchestrator
    orchestrator = ComparisonOrchestrator(
        dataset_path=dataset,
        anthropic_api_key=anthropic_api_key,
        openrouter_api_key=openrouter_api_key,
        limit=limit,
        max_concurrency=max_concurrency,
        llm_model=llm_model,
        resume_from=resume_from,
        output_file=output_file,
    )

    # Run dry-run validation if requested
    if dry_run:
        try:
            orchestrator.dry_run()
            print("\n✓ Dry-run validation passed - setup is ready")
            return
        except ValueError as e:
            print(f"\n✗ Dry-run validation failed: {e}")
            raise SystemExit(1) from e

    # Run comparison
    report = asyncio.run(orchestrator.run())

    # Export results
    ResultsExporter.print_console_summary(report)

    # Save to JSON only
    ResultsExporter.save_json_report(report, output_file)

    # Generate and save accuracy comparison plot
    plot_file = output_file.parent / f"{output_file.stem}_accuracy_comparison.png"
    ResultsExporter.save_accuracy_comparison_plot(report, plot_file)

    print(f"\n✓ Results saved to {output_file}")
    print(f"✓ Accuracy comparison plot saved to {plot_file}")


if __name__ == "__main__":
    # CLI interface using click
    main()
