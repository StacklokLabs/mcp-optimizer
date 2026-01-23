"""
AppWorld MCP Optimizer Experiment Runner CLI.

This script runs experiments against AppWorld tasks using MCP Optimizer tools
(find_tool, call_tool, search_in_tool_response) with a Pydantic AI agent.

Prerequisites:
    1. Start AppWorld API server: task appworld-serve-api
    2. Start AppWorld MCP server: task appworld-serve-mcp
    3. Set OPENROUTER_API_KEY environment variable (or create a .env file)

Usage:
    # Run experiment (auto-discovers matching config or creates new)
    # Will auto-resume if an experiment with matching config exists
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --dataset train --limit 5

    # Force fresh start (generates new experiment, ignores existing)
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --dataset train --limit 5 --force

    # Run with explicit experiment name
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --experiment-name my_test --dataset train --limit 5

    # Run with custom settings
    uv run python examples/call_tool_optimizer/run_experiment.py \\
        --dataset dev --model anthropic/claude-opus-4 --threshold 500 --verbose

Environment Variables:
    OPENROUTER_API_KEY: Required API key for OpenRouter LLM access

    Create a .env file in the project root or examples/call_tool_optimizer/ with:
        OPENROUTER_API_KEY=your_api_key_here
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
# Searches in current directory, examples/call_tool_optimizer/, and project root
_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Try loading .env from multiple locations (first found wins)
for env_path in [
    _SCRIPT_DIR / ".env",
    _PROJECT_ROOT / ".env",
]:
    if env_path.exists():
        load_dotenv(env_path)
        break
else:
    # Load from default locations (cwd and parent dirs)
    load_dotenv()

# Add project root to path to support running as a script
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import click  # noqa: E402
import structlog  # noqa: E402

from examples.call_tool_optimizer.experiment_runner import AppWorldExperimentRunner  # noqa: E402
from examples.call_tool_optimizer.models import (  # noqa: E402
    ExperimentConfig,
    ExperimentSummaryFull,
)
from mcp_optimizer.configure_logging import configure_logging  # noqa: E402

logger = structlog.get_logger(__name__)


def _display_banner(
    experiment_name: str,
    dataset: str,
    model: str,
    max_steps: int,
    threshold: int,
    limit: int | None,
    force: bool,
    baseline: bool,
) -> None:
    """Display experiment banner with configuration."""
    click.echo("\n" + "=" * 80)
    if baseline:
        click.echo("APPWORLD BASELINE EXPERIMENT (Direct MCP)")
    else:
        click.echo("APPWORLD MCP OPTIMIZER EXPERIMENT")
    click.echo("=" * 80)
    if experiment_name:
        click.echo(f"\nExperiment: {experiment_name}")
    else:
        click.echo("\nExperiment: (auto-discover or generate)")
    click.echo(f"Dataset: {dataset}")
    click.echo(f"Model: {model}")
    click.echo(f"Max agent steps: {max_steps}")
    if not baseline:
        click.echo(f"Response optimizer threshold: {threshold}")
    if limit:
        click.echo(f"Task limit: {limit}")
    if force:
        click.echo("Run mode: Force fresh start (deleting existing state)")
    else:
        click.echo("Run mode: Auto-resume if matching config exists")
    click.echo("")


def _display_summary(runner: AppWorldExperimentRunner, summary: ExperimentSummaryFull) -> None:
    """Display experiment summary."""
    click.echo("\n" + "=" * 80)
    click.echo("EXPERIMENT SUMMARY")
    click.echo("=" * 80)
    click.echo(f"\nExperiment name: {runner.config.experiment_name}")
    click.echo(f"Mode: {summary.experiment_mode}")
    click.echo(f"Total tasks: {summary.total_tasks}")
    click.echo(f"Completed tasks: {summary.completed_tasks}")
    click.echo(f"Successful tasks: {summary.successful_tasks}")
    click.echo(f"Failed tasks: {summary.failed_tasks}")
    click.echo(f"Success rate: {summary.success_rate:.1%}")
    click.echo(f"Average agent steps: {summary.avg_agent_steps:.1f}")
    click.echo(f"Average execution time: {summary.avg_execution_time_s:.1f}s")
    click.echo("\nTotal tool calls:")
    if summary.experiment_mode == "baseline":
        click.echo(f"  direct_tool_calls: {summary.total_direct_tool_calls}")
    else:
        click.echo(f"  find_tool: {summary.total_find_tool_calls}")
        click.echo(f"  call_tool: {summary.total_call_tool_calls}")
        click.echo(f"  search_in_tool_response: {summary.total_search_response_calls}")
    click.echo("\nTotal tokens used:")
    click.echo(f"  Request: {summary.total_request_tokens}")
    click.echo(f"  Response: {summary.total_response_tokens}")
    click.echo(f"\nExperiment state: {runner.state_file}")
    click.echo(f"Conversations: {runner.conversations_dir}")
    click.echo("=" * 80 + "\n")


@click.command()
@click.option(
    "--experiment-name",
    default="",
    help="Name for experiment (auto-generated if not provided, auto-resumes matching config)",
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
    default=100,
    type=int,
    help="Maximum agent steps per task (default: 100)",
)
@click.option(
    "--appworld-mcp-url",
    default="http://localhost:10000",
    help="AppWorld MCP server URL (default: http://localhost:10000)",
)
@click.option(
    "--appworld-api-url",
    default="http://localhost:9000",
    help="AppWorld API server URL for remote_apis_url (default: http://localhost:9000)",
)
@click.option(
    "--state-file",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to state file (default: {experiment_name}_state.json)",
)
@click.option(
    "--db-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to database file (default: experiments_shared.db, shared across experiments)",
)
@click.option(
    "--force",
    is_flag=True,
    help="Delete existing state file and start fresh (does not delete shared database)",
)
@click.option(
    "--baseline",
    is_flag=True,
    help="Run baseline agent using direct MCP (ignores optimizer-specific options)",
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
    appworld_api_url: str,
    state_file: Path | None,
    db_path: Path | None,
    force: bool,
    baseline: bool,
    verbose: bool,
) -> None:
    """Run AppWorld experiment with MCP Optimizer agent."""
    # Configure logging
    log_level = "DEBUG" if verbose else "INFO"
    configure_logging(log_level, rich_tracebacks=False, colored_logs=True)

    # Determine experiment mode
    mode = "baseline" if baseline else "optimizer"

    # Display banner
    _display_banner(experiment_name, dataset, model, max_steps, threshold, limit, force, baseline)

    # Set state file path only if experiment_name is provided
    # Otherwise, the runner will determine paths after finding/generating the name
    examples_dir = Path(__file__).parent
    if state_file is None and experiment_name:
        state_file = examples_dir / f"{experiment_name}_state.json"

    # Create experiment config
    config = ExperimentConfig(
        experiment_name=experiment_name,
        mode=mode,
        dataset=dataset,
        llm_model=model,
        response_optimizer_threshold=threshold,
        response_head_lines=head_lines,
        response_tail_lines=tail_lines,
        max_agent_steps=max_steps,
        appworld_mcp_url=appworld_mcp_url,
        appworld_api_url=appworld_api_url,
        db_path=db_path if not baseline else None,  # Baseline mode doesn't need db
    )

    # Create runner
    runner = AppWorldExperimentRunner(
        config=config,
        state_file=state_file,
        force=force,
        limit=limit,
    )

    # Run experiment
    try:
        summary = asyncio.run(runner.run())
        _display_summary(runner, summary)

    except RuntimeError as e:
        click.echo(f"\nError: {e}", err=True)
        click.echo("\nMake sure AppWorld servers are running:", err=True)
        click.echo("  Terminal 1: task appworld-serve-api", err=True)
        click.echo("  Terminal 2: task appworld-serve-mcp", err=True)
        sys.exit(1)
    except KeyboardInterrupt:
        click.echo("\n\nExperiment interrupted. Progress has been saved.", err=True)
        click.echo("Run the same command again to auto-resume from saved state.", err=True)
        sys.exit(1)
    except Exception as e:
        logger.exception("Experiment failed", error=str(e))
        click.echo(f"\nExperiment failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
