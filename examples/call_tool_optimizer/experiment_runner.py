"""Orchestrator for running AppWorld experiments with MCP Optimizer.

This module coordinates the full experiment workflow including:
- State management for resume/recovery
- Tool ingestion from AppWorld MCP server
- Agent execution on tasks
- Evaluation using AppWorld's evaluate() method (via subprocess)

AppWorld operations run in an isolated environment via subprocess to avoid
dependency conflicts with mcp-optimizer.
"""

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import structlog
from appworld_agent import AppWorldAgentRunner
from appworld_tool_loader import AppWorldToolLoader
from models import (
    ExperimentConfig,
    ExperimentResults,
    ExperimentState,
    TaskResult,
    TaskState,
)

logger = structlog.get_logger(__name__)

# Path to the appworld helper script
APPWORLD_HELPER = Path(__file__).parent / "appworld_helper.py"


def _run_appworld_command(command: dict) -> dict:
    """Run a command via the appworld helper script in isolated environment.

    Args:
        command: Dictionary with action and parameters

    Returns:
        Result dictionary from the helper script

    Raises:
        RuntimeError: If the command fails
    """
    cmd = [
        "uv",
        "run",
        "--no-project",
        "--with",
        "appworld",
        "python",
        str(APPWORLD_HELPER),
    ]

    try:
        result = subprocess.run(
            cmd,
            input=json.dumps(command),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        if result.returncode != 0:
            error_msg = result.stderr or result.stdout or "Unknown error"
            raise RuntimeError(f"AppWorld command failed: {error_msg}")

        return json.loads(result.stdout)

    except subprocess.TimeoutExpired as e:
        raise RuntimeError("AppWorld command timed out") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response from AppWorld: {e}") from e


class ExperimentStateManager:
    """Manages experiment state persistence for recovery/resume."""

    def __init__(self, state_file: Path):
        """Initialize with path to state JSON file.

        Args:
            state_file: Path to the state file
        """
        self.state_file = state_file

    def load_state(self) -> ExperimentState | None:
        """Load existing state from JSON file.

        Returns:
            ExperimentState if file exists, None otherwise
        """
        if not self.state_file.exists():
            return None

        try:
            return ExperimentState.model_validate_json(self.state_file.read_text())
        except Exception as e:
            logger.warning("Failed to load state file", error=str(e))
            return None

    def save_state(self, state: ExperimentState) -> None:
        """Save state to JSON file atomically.

        Uses atomic write (write to temp file, then rename) for safety.

        Args:
            state: Experiment state to save
        """
        # Update last_updated timestamp
        state.last_updated = datetime.now(timezone.utc)

        # Ensure parent directory exists
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: write to temp file, then rename
        temp_file = self.state_file.with_suffix(".tmp")
        temp_file.write_text(state.model_dump_json(indent=2))
        temp_file.rename(self.state_file)

        logger.debug("Saved state", path=str(self.state_file))

    def create_new_state(
        self,
        config: ExperimentConfig,
        task_ids: list[str],
    ) -> ExperimentState:
        """Create a new experiment state.

        Args:
            config: Experiment configuration
            task_ids: List of AppWorld task IDs to process

        Returns:
            New ExperimentState with all tasks as pending
        """
        now = datetime.now(timezone.utc)

        # Initialize all tasks as pending
        task_states = {
            task_id: TaskState(task_id=task_id, status="pending") for task_id in task_ids
        }

        return ExperimentState(
            config=config,
            task_ids=task_ids,
            task_states=task_states,
            started_at=now,
            last_updated=now,
            ingestion_completed=False,
            tools_count=0,
        )

    def config_matches(self, state: ExperimentState, config: ExperimentConfig) -> bool:
        """Check if experiment parameters match existing state.

        Args:
            state: Existing experiment state
            config: New configuration to compare

        Returns:
            True if configurations match on key parameters
        """
        return (
            state.config.llm_model == config.llm_model
            and state.config.response_optimizer_threshold == config.response_optimizer_threshold
            and state.config.response_head_lines == config.response_head_lines
            and state.config.response_tail_lines == config.response_tail_lines
            and state.config.dataset == config.dataset
        )

    def get_pending_tasks(self, state: ExperimentState) -> list[str]:
        """Return list of task IDs that haven't completed yet.

        Args:
            state: Experiment state

        Returns:
            List of pending or in_progress task IDs
        """
        return [
            task_id
            for task_id, task_state in state.task_states.items()
            if task_state.status in ("pending", "in_progress")
        ]

    def update_task_state(
        self,
        state: ExperimentState,
        task_id: str,
        **updates,
    ) -> ExperimentState:
        """Update a task's state and save.

        Args:
            state: Current experiment state
            task_id: Task to update
            **updates: Fields to update (status, evaluation_result, etc.)

        Returns:
            Updated ExperimentState
        """
        if task_id in state.task_states:
            task_state = state.task_states[task_id]
            for key, value in updates.items():
                if hasattr(task_state, key):
                    setattr(task_state, key, value)

        self.save_state(state)
        return state


class AppWorldExperimentRunner:
    """Orchestrates the full AppWorld experiment workflow."""

    def __init__(
        self,
        config: ExperimentConfig,
        state_file: Path,
        output_file: Path | None = None,
        resume: bool = False,
        limit: int | None = None,
    ):
        """Initialize runner with configuration.

        Args:
            config: Experiment configuration
            state_file: Path to state file
            output_file: Optional path to output results file
            resume: Whether to resume from existing state
            limit: Optional limit on number of tasks
        """
        self.config = config
        self.state_file = state_file
        self.output_file = output_file
        self.resume = resume
        self.limit = limit

        self.state_manager = ExperimentStateManager(state_file)
        self.agent: AppWorldAgentRunner | None = None
        self.tool_loader: AppWorldToolLoader | None = None

        # Determine database path
        if config.db_path:
            self.db_path = config.db_path
        else:
            self.db_path = Path(__file__).parent / f"{config.experiment_name}.db"

    async def check_appworld_mcp_running(self) -> bool:
        """Check if AppWorld MCP server is accessible.

        Returns:
            True if server is reachable
        """
        try:
            async with httpx.AsyncClient() as client:
                # Try to connect to the MCP server
                await client.get(
                    self.config.appworld_mcp_url,
                    timeout=5.0,
                )
                # MCP servers may return various status codes, any response is OK
                return True
        except Exception as e:
            logger.warning(
                "AppWorld MCP server not accessible",
                url=self.config.appworld_mcp_url,
                error=str(e),
            )
            return False

    async def _ingest_tools_if_needed(self, state: ExperimentState) -> ExperimentState:
        """Ingest AppWorld tools if not already done.

        Args:
            state: Current experiment state

        Returns:
            Updated state with ingestion status
        """
        if state.ingestion_completed:
            logger.info("Tools already ingested", count=state.tools_count)
            return state

        logger.info("Ingesting tools from AppWorld MCP server")

        self.tool_loader = AppWorldToolLoader(
            appworld_mcp_url=self.config.appworld_mcp_url,
            db_path=self.db_path,
        )

        stats = await self.tool_loader.load_and_ingest()

        state.ingestion_completed = True
        state.tools_count = stats["tools_count"]
        self.state_manager.save_state(state)

        logger.info("Tool ingestion complete", tools_count=stats["tools_count"])
        return state

    async def run(self) -> ExperimentResults:
        """Run the full experiment.

        Returns:
            ExperimentResults with aggregated metrics
        """
        logger.info(
            "Starting experiment",
            experiment_name=self.config.experiment_name,
            dataset=self.config.dataset,
        )

        # Check if AppWorld MCP server is running
        if not await self.check_appworld_mcp_running():
            raise RuntimeError(
                f"AppWorld MCP server is not running at {self.config.appworld_mcp_url}. "
                "Start it with: task appworld-serve-api && task appworld-serve-mcp"
            )

        # Load or create state
        state = self._load_or_create_state()

        # Ingest tools if needed
        state = await self._ingest_tools_if_needed(state)

        # Initialize agent
        self.agent = AppWorldAgentRunner(config=self.config, db_path=self.db_path)

        # Get pending tasks
        pending_tasks = self.state_manager.get_pending_tasks(state)
        logger.info(
            "Processing tasks",
            pending=len(pending_tasks),
            total=len(state.task_ids),
        )

        # Run each task
        for i, task_id in enumerate(pending_tasks):
            logger.info(
                "Processing task",
                task_id=task_id,
                progress=f"{i + 1}/{len(pending_tasks)}",
            )

            try:
                result = await self._run_single_task(task_id, state)

                # Update task state
                state = self.state_manager.update_task_state(
                    state,
                    task_id,
                    status="completed",
                    evaluation_result=result.model_dump(),
                    completed_at=datetime.now(timezone.utc),
                )

                logger.info(
                    "Task completed",
                    task_id=task_id,
                    success=result.success,
                    goal_progress=result.goal_progress,
                )

            except Exception as e:
                logger.exception("Task failed", task_id=task_id, error=str(e))

                state = self.state_manager.update_task_state(
                    state,
                    task_id,
                    status="failed",
                    error=str(e),
                    completed_at=datetime.now(timezone.utc),
                )

        # Generate results
        results = self._compute_results(state)

        # Save results if output file specified
        if self.output_file:
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            self.output_file.write_text(results.model_dump_json(indent=2))
            logger.info("Results saved", path=str(self.output_file))

        return results

    def _load_or_create_state(self) -> ExperimentState:
        """Load existing state or create new one.

        Returns:
            ExperimentState to use for the experiment
        """
        if self.resume:
            existing_state = self.state_manager.load_state()

            if existing_state:
                # Check if config matches
                if self.state_manager.config_matches(existing_state, self.config):
                    logger.info(
                        "Resuming experiment",
                        completed=len(
                            [
                                t
                                for t in existing_state.task_states.values()
                                if t.status == "completed"
                            ]
                        ),
                        pending=len(self.state_manager.get_pending_tasks(existing_state)),
                    )
                    return existing_state
                else:
                    logger.warning(
                        "Config mismatch with existing state. Creating new experiment.",
                    )

        # Create new state - get task IDs via subprocess
        logger.info("Loading task IDs via AppWorld helper", dataset=self.config.dataset)

        result = _run_appworld_command(
            {
                "action": "list_tasks",
                "dataset": self.config.dataset,
                "limit": self.limit,
            }
        )

        if "error" in result:
            raise RuntimeError(f"Failed to load task IDs: {result['error']}")

        task_ids = result["task_ids"]
        logger.info("Loaded task IDs", count=len(task_ids))

        state = self.state_manager.create_new_state(self.config, task_ids)
        self.state_manager.save_state(state)

        return state

    async def _run_single_task(self, task_id: str, state: ExperimentState) -> TaskResult:
        """Execute a single AppWorld task.

        Args:
            task_id: AppWorld task ID
            state: Current experiment state

        Returns:
            TaskResult with execution results
        """
        start_time = time.perf_counter()

        # Update task to in_progress
        state = self.state_manager.update_task_state(
            state,
            task_id,
            status="in_progress",
            started_at=datetime.now(timezone.utc),
        )

        try:
            # Get task instruction via subprocess
            instruction_result = _run_appworld_command(
                {
                    "action": "get_instruction",
                    "task_id": task_id,
                    "experiment_name": self.config.experiment_name,
                }
            )

            if "error" in instruction_result:
                raise RuntimeError(f"Failed to get instruction: {instruction_result['error']}")

            instruction = instruction_result["instruction"]

            # Update state with instruction
            state = self.state_manager.update_task_state(state, task_id, instruction=instruction)

            # Run agent
            agent_result = await self.agent.execute_task(instruction)

            # Check for agent error
            if agent_result.get("error"):
                return TaskResult(
                    task_id=task_id,
                    success=False,
                    error=agent_result["error"],
                    execution_time_s=time.perf_counter() - start_time,
                    find_tool_calls=agent_result["tool_calls"]["find_tool"],
                    call_tool_calls=agent_result["tool_calls"]["call_tool"],
                    search_response_calls=agent_result["tool_calls"]["search_in_tool_response"],
                    agent_steps=agent_result["tool_calls"]["total"],
                    request_tokens=agent_result["request_tokens"],
                    response_tokens=agent_result["response_tokens"],
                )

            # Evaluate task completion via subprocess
            eval_result = _run_appworld_command(
                {
                    "action": "evaluate",
                    "task_id": task_id,
                    "experiment_name": self.config.experiment_name,
                }
            )

            if "error" in eval_result:
                raise RuntimeError(f"Failed to evaluate task: {eval_result['error']}")

            # Extract success and goal progress
            success = eval_result.get("success", False)
            goal_progress = eval_result.get("goal_progress", 0.0)

            return TaskResult(
                task_id=task_id,
                success=success,
                goal_progress=goal_progress,
                agent_steps=agent_result["tool_calls"]["total"],
                find_tool_calls=agent_result["tool_calls"]["find_tool"],
                call_tool_calls=agent_result["tool_calls"]["call_tool"],
                search_response_calls=agent_result["tool_calls"]["search_in_tool_response"],
                execution_time_s=time.perf_counter() - start_time,
                request_tokens=agent_result["request_tokens"],
                response_tokens=agent_result["response_tokens"],
            )

        except Exception as e:
            logger.exception("Task execution failed", task_id=task_id, error=str(e))
            return TaskResult(
                task_id=task_id,
                success=False,
                error=str(e),
                execution_time_s=time.perf_counter() - start_time,
            )

    def _compute_results(self, state: ExperimentState) -> ExperimentResults:
        """Compute aggregated results from task states.

        Args:
            state: Final experiment state

        Returns:
            ExperimentResults with aggregated metrics
        """
        task_results = []
        successful_count = 0
        failed_count = 0
        total_goal_progress = 0.0
        total_steps = 0
        total_execution_time = 0.0
        total_find_tool_calls = 0
        total_call_tool_calls = 0
        total_search_response_calls = 0
        total_request_tokens = 0
        total_response_tokens = 0

        for task_state in state.task_states.values():
            if task_state.status == "completed" and task_state.evaluation_result:
                result = TaskResult.model_validate(task_state.evaluation_result)
                task_results.append(result)

                if result.success:
                    successful_count += 1
                else:
                    failed_count += 1

                total_goal_progress += result.goal_progress
                total_steps += result.agent_steps
                total_execution_time += result.execution_time_s
                total_find_tool_calls += result.find_tool_calls
                total_call_tool_calls += result.call_tool_calls
                total_search_response_calls += result.search_response_calls
                total_request_tokens += result.request_tokens
                total_response_tokens += result.response_tokens

            elif task_state.status == "failed":
                failed_count += 1
                task_results.append(
                    TaskResult(
                        task_id=task_state.task_id,
                        success=False,
                        error=task_state.error,
                    )
                )

        completed_count = len(task_results)
        success_rate = successful_count / completed_count if completed_count > 0 else 0.0
        avg_goal_progress = total_goal_progress / completed_count if completed_count > 0 else 0.0
        avg_steps = total_steps / completed_count if completed_count > 0 else 0.0
        avg_execution_time = total_execution_time / completed_count if completed_count > 0 else 0.0

        return ExperimentResults(
            config=state.config,
            total_tasks=len(state.task_ids),
            completed_tasks=completed_count,
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            success_rate=success_rate,
            avg_goal_progress=avg_goal_progress,
            avg_agent_steps=avg_steps,
            avg_execution_time_s=avg_execution_time,
            total_find_tool_calls=total_find_tool_calls,
            total_call_tool_calls=total_call_tool_calls,
            total_search_response_calls=total_search_response_calls,
            total_request_tokens=total_request_tokens,
            total_response_tokens=total_response_tokens,
            task_results=task_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
