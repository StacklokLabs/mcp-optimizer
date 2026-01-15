#!/usr/bin/env python3
"""
AppWorld MCP Optimizer Experiment Runner CLI.

This script runs experiments against AppWorld tasks using MCP Optimizer tools
(find_tool, call_tool, search_in_tool_response) with a Pydantic AI agent.

Prerequisites:
    1. Start AppWorld API server: task appworld-serve-api
    2. Start AppWorld MCP server: task appworld-serve-mcp
    3. Set OPENROUTER_API_KEY environment variable

Usage:
    # Run new experiment (limited to 5 tasks)
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --experiment-name test1 --dataset train --limit 5

    # Resume interrupted experiment
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --experiment-name test1 --resume

    # Run with custom settings
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --experiment-name test2 --dataset dev \\
        --model anthropic/claude-opus-4 --threshold 500 --verbose
"""

import asyncio
import sys
from pathlib import Path

import click
import structlog

from mcp_optimizer.configure_logging import configure_logging

logger = structlog.get_logger(__name__)


@click.command()
@click.option(
    "--experiment-name",
    required=True,
    help="Name for this experiment run (used for state file and database naming)",
)
@click.option(
    "--dataset",
    default="train",
    type=click.Choice(["train", "dev", "test_normal", "test_challenge"]),
    help="AppWorld dataset to use (default: train)",
)
@click.option(
    "--limit",
    default=None,
    type=int,
    help="Limit number of tasks to run (default: all tasks in dataset)",
)
@click.option(
    "--model",
    default="anthropic/claude-sonnet-4",
    help="LLM model for the agent (OpenRouter format, default: anthropic/claude-sonnet-4)",
)
@click.option(
    "--threshold",
    default=1000,
    type=int,
    help="Token threshold for response optimization (default: 1000)",
)
@click.option(
    "--head-lines",
    default=20,
    type=int,
    help="Lines to preserve from start for unstructured text (default: 20)",
)
@click.option(
    "--tail-lines",
    default=20,
    type=int,
    help="Lines to preserve from end for unstructured text (default: 20)",
)
@click.option(
    "--max-steps",
    default=50,
    type=int,
    help="Maximum agent steps per task (default: 50)",
)
@click.option(
    "--appworld-mcp-url",
    default="http://localhost:10000",
    help="AppWorld MCP server URL (default: http://localhost:10000)",
)
@click.option(
    "--state-file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to state file (default: {experiment_name}_state.json)",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to output results file (default: {experiment_name}_results.json)",
)
@click.option(
    "--db-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to database file (default: {experiment_name}.db)",
)
@click.option(
    "--resume",
    is_flag=True,
    help="Resume from existing state file if available",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Enable verbose output (debug logging)",
)
def main(
    experiment_name: str,
    dataset: str,
    limit: int | None,
    model: str,
    threshold: int,
    head_lines: int,
    tail_lines: int,
    max_steps: int,
    appworld_mcp_url: str,
    state_file: Path | None,
    output: Path | None,
    db_path: Path | None,
    resume: bool,
    verbose: bool,
) -> None:
    """Run AppWorld experiment with MCP Optimizer agent."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(log_level, rich_tracebacks=False, colored_logs=True)

    click.echo("\n" + "=" * 80)
    click.echo("APPWORLD MCP OPTIMIZER EXPERIMENT")
    click.echo("=" * 80)
    click.echo(f"\nExperiment: {experiment_name}")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Model: {model}")
    click.echo(f"Response optimizer threshold: {threshold}")
    if limit:
        click.echo(f"Task limit: {limit}")
    if resume:
        click.echo("Mode: Resume from existing state")
    click.echo("")

    # Import here to avoid circular imports and ensure logging is configured first
    from experiment_runner import AppWorldExperimentRunner
    from models import ExperimentConfig

    # Set default paths if not provided
    examples_dir = Path(__file__).parent
    if state_file is None:
        state_file = examples_dir / f"{experiment_name}_state.json"
    if output is None:
        output = examples_dir / f"{experiment_name}_results.json"

    # Create experiment config
    config = ExperimentConfig(
        experiment_name=experiment_name,
        dataset=dataset,
        llm_model=model,
        response_optimizer_threshold=threshold,
        response_head_lines=head_lines,
        response_tail_lines=tail_lines,
        max_agent_steps=max_steps,
        appworld_mcp_url=appworld_mcp_url,
        db_path=db_path,
    )

    # Create runner
    runner = AppWorldExperimentRunner(
        config=config,
        state_file=state_file,
        output_file=output,
        resume=resume,
        limit=limit,
    )

    # Run experiment
    try:
        results = asyncio.run(runner.run())

        # Print summary
        click.echo("\n" + "=" * 80)
        click.echo("EXPERIMENT RESULTS")
        click.echo("=" * 80)
        click.echo(f"\nTotal tasks: {results.total_tasks}")
        click.echo(f"Completed tasks: {results.completed_tasks}")
        click.echo(f"Successful tasks: {results.successful_tasks}")
        click.echo(f"Failed tasks: {results.failed_tasks}")
        click.echo(f"Success rate: {results.success_rate:.1%}")
        click.echo(f"Average goal progress: {results.avg_goal_progress:.1%}")
        click.echo(f"Average agent steps: {results.avg_agent_steps:.1f}")
        click.echo(f"Average execution time: {results.avg_execution_time_s:.1f}s")
        click.echo("\nTotal tool calls:")
        click.echo(f"  find_tool: {results.total_find_tool_calls}")
        click.echo(f"  call_tool: {results.total_call_tool_calls}")
        click.echo(f"  search_in_tool_response: {results.total_search_response_calls}")
        click.echo("\nTotal tokens used:")
        click.echo(f"  Request: {results.total_request_tokens}")
        click.echo(f"  Response: {results.total_response_tokens}")
        click.echo(f"\nResults saved to: {output}")
        click.echo(f"State saved to: {state_file}")
        click.echo("=" * 80 + "\n")

    except RuntimeError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nMake sure AppWorld servers are running:", err=True)
        click.echo("  Terminal 1: task appworld-serve-api", err=True)
        click.echo("  Terminal 2: task appworld-serve-mcp", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n\nExperiment interrupted. Progress has been saved.", err=True)
        click.echo(f"Resume with: --resume --state-file {state_file}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Experiment failed", error=str(e))
        click.echo(f"\nExperiment failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
