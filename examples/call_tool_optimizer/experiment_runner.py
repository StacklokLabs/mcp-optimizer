"""Orchestrator for running AppWorld experiments with MCP Optimizer.

This module coordinates the full experiment workflow including:
- State management for resume/recovery
- Tool ingestion from AppWorld MCP server
- Agent execution on tasks
- Evaluation using AppWorld's evaluate() method

AppWorld is directly imported since it's installed in the environment.
"""

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path to support running as a script
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import httpx  # noqa: E402
import structlog  # noqa: E402
from appworld import AppWorld, load_task_ids  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from examples.call_tool_optimizer.appworld_agent import AppWorldAgentRunner  # noqa: E402
from examples.call_tool_optimizer.appworld_tool_loader import AppWorldToolLoader  # noqa: E402
from examples.call_tool_optimizer.baseline_agent import BaselineAgentRunner  # noqa: E402
from examples.call_tool_optimizer.models import (  # noqa: E402
    ExperimentConfig,
    ExperimentState,
    ExperimentSummary,
    ExperimentSummaryFull,
    TaskResult,
)

logger = structlog.get_logger(__name__)


class ExperimentStateManager:
    """Manages experiment state persistence for recovery/resume."""

    def __init__(self, state_file: Path, conversations_dir: Path):
        """Initialize with path to state JSON file.

        Args:
            state_file: Path to the state file
            conversations_dir: Path to directory for conversation files
        """
        self.state_file = state_file
        self.conversations_dir = conversations_dir

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

    def save_conversation(
        self, experiment_name: str, task_id: str, conversation: list[dict]
    ) -> str:
        """Save conversation to a separate JSON file.

        Args:
            experiment_name: Name of the experiment
            task_id: Task ID
            conversation: List of conversation messages

        Returns:
            Relative path to the conversation file
        """
        self.conversations_dir.mkdir(parents=True, exist_ok=True)

        # Create filename: <experiment>_<task_id>.json
        filename = f"{experiment_name}_{task_id}.json"
        filepath = self.conversations_dir / filename

        # Save conversation
        filepath.write_text(json.dumps(conversation, indent=2))

        # Return relative path from experiment directory
        return f"conversations/{filename}"

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
        tasks = {task_id: TaskResult(task_id=task_id, status="pending") for task_id in task_ids}

        return ExperimentState(
            config=config,
            task_ids=task_ids,
            tasks=tasks,
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
        # Mode must always match
        if state.config.mode != config.mode:
            return False

        # Basic parameters that must match for both modes
        base_match = (
            state.config.llm_model == config.llm_model and state.config.dataset == config.dataset
        )

        if config.mode == "baseline":
            # Baseline mode only needs basic parameters to match
            return base_match
        else:
            # Optimizer mode also needs optimizer-specific parameters to match
            return (
                base_match
                and state.config.response_optimizer_threshold == config.response_optimizer_threshold
                and state.config.response_head_lines == config.response_head_lines
                and state.config.response_tail_lines == config.response_tail_lines
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
            for task_id, task in state.tasks.items()
            if task.status in ("pending", "in_progress")
        ]

    def get_tasks_to_run(self, state: ExperimentState) -> list[str]:
        """Return task IDs that need to be run (pending, in_progress, failed, or unsuccessful).

        This method extends get_pending_tasks by also including:
        - Tasks with status "failed" (exception during execution)
        - Tasks with status "completed" but success=False (task ran but evaluation failed)

        Args:
            state: Experiment state

        Returns:
            List of task IDs that should be (re)run
        """
        tasks = []
        for task_id, task in state.tasks.items():
            if task.status in ("pending", "in_progress", "failed"):
                tasks.append(task_id)
            elif task.status == "completed" and task.success is False:
                tasks.append(task_id)
        return tasks

    def reset_task_for_retry(self, state: ExperimentState, task_id: str) -> ExperimentState:
        """Reset a task's state for retry.

        Args:
            state: Current experiment state
            task_id: Task ID to reset

        Returns:
            Updated ExperimentState with reset task
        """
        state.tasks[task_id] = TaskResult(task_id=task_id, status="pending")
        self.save_state(state)
        return state

    def update_task(
        self,
        state: ExperimentState,
        task_id: str,
        task_result: TaskResult,
    ) -> ExperimentState:
        """Update a task with new result and save state.

        Args:
            state: Current experiment state
            task_id: Task to update
            task_result: New task result

        Returns:
            Updated ExperimentState
        """
        state.tasks[task_id] = task_result
        self.save_state(state)
        return state

    def find_matching_experiment(
        self,
        config: ExperimentConfig,
        search_dir: Path,
    ) -> tuple[Path, ExperimentState] | None:
        """Scan existing state files to find an experiment with matching config.

        Args:
            config: Configuration to match against
            search_dir: Directory to scan for state files

        Returns:
            Tuple of (state_file_path, ExperimentState) if match found, None otherwise
        """
        if not search_dir.exists():
            return None

        for state_file in search_dir.glob("*_state.json"):
            try:
                state = ExperimentState.model_validate_json(state_file.read_text())
                if self.config_matches(state, config):
                    logger.info(
                        "Found matching experiment",
                        experiment_name=state.config.experiment_name,
                        state_file=str(state_file),
                    )
                    return (state_file, state)
            except Exception as e:
                logger.debug(
                    "Failed to load state file during scan",
                    path=str(state_file),
                    error=str(e),
                )
                continue

        return None

    @staticmethod
    def generate_experiment_name(mode: str = "optimizer") -> str:
        """Generate a unique experiment name.

        Args:
            mode: Experiment mode ('optimizer' or 'baseline')

        Returns:
            Experiment name like "exp_a1b2c3d4" or "baseline_a1b2c3d4" using a short UUID
        """
        short_id = uuid.uuid4().hex[:8]
        prefix = "baseline" if mode == "baseline" else "exp"
        return f"{prefix}_{short_id}"


class AppWorldExperimentRunner:
    """Orchestrates the full AppWorld experiment workflow."""

    def __init__(
        self,
        config: ExperimentConfig,
        state_file: Path | None = None,
        force: bool = False,
        limit: int | None = None,
    ):
        """Initialize runner with configuration.

        Args:
            config: Experiment configuration (experiment_name can be empty for auto-discovery)
            state_file: Optional path to state file (derived from experiment_name if not provided)
            force: If True, delete existing state and start fresh; if False, auto-resume
            limit: Optional limit on number of tasks
        """
        self.config = config
        self.force = force
        self.limit = limit
        self.examples_dir = Path(__file__).parent

        # State file and experiment name will be resolved in _load_or_create_state
        self._provided_state_file = state_file
        self.state_file: Path | None = None
        self.conversations_dir: Path | None = None
        self.state_manager: ExperimentStateManager | None = None
        self.agent: AppWorldAgentRunner | BaselineAgentRunner | None = None
        self.tool_loader: AppWorldToolLoader | None = None
        self.db_path: Path | None = None

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

    def _load_task_ids(self, dataset: str, limit: int | None) -> list[str]:
        """Load task IDs directly from AppWorld.

        Args:
            dataset: Dataset name (train, dev, test_normal, test_challenge)
            limit: Optional limit on number of tasks

        Returns:
            List of task IDs
        """
        task_ids = load_task_ids(dataset)
        if limit:
            task_ids = task_ids[:limit]
        return task_ids

    def _get_instruction(self, world: AppWorld) -> dict:
        """Get task instruction from AppWorld.

        Args:
            world: Active AppWorld instance

        Returns:
            dict with instruction and supervisor info
        """
        return {
            "instruction": world.task.instruction,
            "supervisor": {
                "name": getattr(world.task.supervisor, "name", None),
                "email": getattr(world.task.supervisor, "email", None),
            },
        }

    def _save_and_evaluate(self, world: AppWorld) -> dict:
        """Save AppWorld state and evaluate task completion.

        This is required when not using world.execute() for agent execution.

        Args:
            world: Active AppWorld instance

        Returns:
            dict with success and full evaluation
        """
        world.save()  # Required when not using world.execute()
        evaluation = world.evaluate()
        eval_dict = evaluation.to_dict()
        return {
            "success": eval_dict.get("success", False),
            "evaluation": eval_dict,
        }

    async def _ingest_tools_if_needed(self, state: ExperimentState) -> ExperimentState:
        """Ingest AppWorld tools if not already done.

        Uses the shared database. Tools are ingested once and reused across experiments.

        Args:
            state: Current experiment state

        Returns:
            Updated state with ingestion status
        """
        # Check if tools already exist in the shared database
        tools_count = await self._get_existing_tools_count()
        if tools_count > 0:
            logger.info(
                "Tools already exist in shared database, skipping ingestion",
                count=tools_count,
            )
            state.ingestion_completed = True
            state.tools_count = tools_count
            self.state_manager.save_state(state)
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

    async def _get_existing_tools_count(self) -> int:
        """Check if tools already exist in the shared database.

        Returns:
            Number of tools in the database, or 0 if database doesn't exist
        """
        if not self.db_path.exists():
            return 0

        try:
            async_db_url = f"sqlite+aiosqlite:///{self.db_path}"
            engine = create_async_engine(async_db_url)

            async with engine.connect() as conn:
                result = await conn.execute(text("SELECT COUNT(*) FROM workload_tools"))
                count = result.scalar()
                return count or 0
        except Exception as e:
            logger.debug("Could not check existing tools", error=str(e))
            return 0
        finally:
            await engine.dispose()

    async def run(self) -> ExperimentSummary:
        """Run the full experiment.

        Returns:
            ExperimentSummary with aggregated metrics
        """
        logger.info("Starting experiment", dataset=self.config.dataset)

        # Check if AppWorld MCP server is running
        if not await self.check_appworld_mcp_running():
            raise RuntimeError(
                f"AppWorld MCP server is not running at {self.config.appworld_mcp_url}. "
                "Start it with: task appworld-serve-api && task appworld-serve-mcp"
            )

        # Load or create state (this resolves experiment name if not provided)
        state = self._load_or_create_state()

        logger.info(
            "Experiment initialized",
            experiment_name=self.config.experiment_name,
            mode=self.config.mode,
            state_file=str(self.state_file),
            db_path=str(self.db_path) if self.db_path else None,
        )

        # Initialize agent based on mode
        if self.config.mode == "baseline":
            # Baseline mode: skip tool ingestion, use direct MCP connection
            logger.info("Running in baseline mode (direct MCP connection)")
            self.agent = BaselineAgentRunner(config=self.config)
        else:
            # Optimizer mode: ingest tools and use MCP Optimizer
            state = await self._ingest_tools_if_needed(state)
            self.agent = AppWorldAgentRunner(config=self.config, db_path=self.db_path)

        # Get tasks to run (includes pending, failed, and unsuccessful tasks)
        tasks_to_run = self.state_manager.get_tasks_to_run(state)
        logger.info(
            "Processing tasks",
            to_run=len(tasks_to_run),
            total=len(state.task_ids),
        )

        # Run each task
        for i, task_id in enumerate(tasks_to_run):
            task = state.tasks[task_id]

            # Reset task state if retrying a failed or unsuccessful task
            if task.status in ("failed", "completed"):
                logger.info(
                    "Retrying task",
                    task_id=task_id,
                    previous_status=task.status,
                    previous_success=task.success,
                )
                state = self.state_manager.reset_task_for_retry(state, task_id)

            logger.info(
                "Processing task",
                task_id=task_id,
                progress=f"{i + 1}/{len(tasks_to_run)}",
            )

            try:
                result = await self._run_single_task(task_id, state)

                # Update task in state with full result
                state = self.state_manager.update_task(state, task_id, result)

                logger.info(
                    "Task completed",
                    task_id=task_id,
                    success=result.success,
                )

            except Exception as e:
                logger.exception("Task failed", task_id=task_id, error=str(e))

                # Create failed task result
                failed_result = TaskResult(
                    task_id=task_id,
                    status="failed",
                    success=False,
                    error=str(e),
                    completed_at=datetime.now(timezone.utc),
                )
                state = self.state_manager.update_task(state, task_id, failed_result)

        # Generate summary and save to state
        summary = self._compute_summary(state)

        # Save summary (without task_results) to state file
        state.summary = ExperimentSummary(
            experiment_mode=summary.experiment_mode,
            total_tasks=summary.total_tasks,
            completed_tasks=summary.completed_tasks,
            successful_tasks=summary.successful_tasks,
            failed_tasks=summary.failed_tasks,
            success_rate=summary.success_rate,
            avg_agent_steps=summary.avg_agent_steps,
            avg_execution_time_s=summary.avg_execution_time_s,
            total_find_tool_calls=summary.total_find_tool_calls,
            total_call_tool_calls=summary.total_call_tool_calls,
            total_search_response_calls=summary.total_search_response_calls,
            total_direct_tool_calls=summary.total_direct_tool_calls,
            total_request_tokens=summary.total_request_tokens,
            total_response_tokens=summary.total_response_tokens,
            timestamp=summary.timestamp,
        )
        self.state_manager.save_state(state)

        return summary

    def _handle_force_delete_files(self, state_file: Path) -> None:
        """Delete state file and conversations when force flag is set.

        Args:
            state_file: Path to the state file to delete
        """
        if state_file.exists():
            state_file.unlink()
            logger.debug("Deleted state file", path=str(state_file))

        # Delete conversations directory for this experiment
        experiment_name = state_file.stem.replace("_state", "")
        conversations_dir = state_file.parent / "conversations"
        if conversations_dir.exists():
            for conv_file in conversations_dir.glob(f"{experiment_name}_*.json"):
                conv_file.unlink()
                logger.debug("Deleted conversation file", path=str(conv_file))

    def _try_resume_matching_experiment(
        self, temp_manager: ExperimentStateManager
    ) -> ExperimentState | None:
        """Search for and resume an experiment with matching config.

        Args:
            temp_manager: State manager for scanning experiments

        Returns:
            ExperimentState if match found and resumed, None otherwise
        """
        match = temp_manager.find_matching_experiment(self.config, self.examples_dir)
        if not match:
            return None

        found_state_file, existing_state = match
        self.config.experiment_name = existing_state.config.experiment_name
        self.state_file = found_state_file
        self._setup_paths()
        self.state_manager = ExperimentStateManager(self.state_file, self.conversations_dir)

        # Check if we need to expand tasks to reach the new limit
        existing_state = self._expand_tasks_if_needed(existing_state)

        completed = len([t for t in existing_state.tasks.values() if t.status == "completed"])
        logger.info(
            "Auto-resuming matching experiment",
            experiment_name=self.config.experiment_name,
            completed=completed,
            pending=len(self.state_manager.get_pending_tasks(existing_state)),
        )
        return existing_state

    def _load_or_create_state(self) -> ExperimentState:
        """Load existing state or create new one.

        Behavior:
        - If force=True: delete existing matching state and start fresh
        - If experiment_name provided: use that specific state file
        - If no experiment_name: scan for experiments with matching config parameters
        - If match found: auto-resume that experiment
        - If no match: create new experiment with auto-generated name

        Returns:
            ExperimentState to use for the experiment
        """
        # Create a temporary state manager for scanning
        temp_conversations = self.examples_dir / "conversations"
        temp_manager = ExperimentStateManager(
            self.examples_dir / "temp_state.json", temp_conversations
        )

        # If force flag is set and we have a specific experiment name, delete state file
        if self.force and self.config.experiment_name:
            state_file = (
                self._provided_state_file
                or self.examples_dir / f"{self.config.experiment_name}_state.json"
            )
            logger.info("Force flag set, deleting existing state file", path=str(state_file))
            self._handle_force_delete_files(state_file)

        # If no experiment name provided, search for matching experiments
        if not self.config.experiment_name:
            if self.force:
                # Force flag WITHOUT explicit name: find matching config, reuse name, delete files
                match = temp_manager.find_matching_experiment(self.config, self.examples_dir)
                if match:
                    found_state_file, existing_state = match
                    self.config.experiment_name = existing_state.config.experiment_name
                    logger.info(
                        "Force: reusing experiment number", name=self.config.experiment_name
                    )
                    self._handle_force_delete_files(found_state_file)
            else:
                # Try to auto-resume a matching experiment
                existing_state = self._try_resume_matching_experiment(temp_manager)
                if existing_state:
                    return existing_state

            # No match found - generate a new experiment name
            if not self.config.experiment_name:
                self.config.experiment_name = temp_manager.generate_experiment_name(
                    self.config.mode
                )
                logger.info(
                    "Generated new experiment name", experiment_name=self.config.experiment_name
                )

        # Now we have an experiment name - set up paths
        self.state_file = (
            self._provided_state_file
            or self.examples_dir / f"{self.config.experiment_name}_state.json"
        )
        self._setup_paths()
        self.state_manager = ExperimentStateManager(self.state_file, self.conversations_dir)

        # If not force, try to load existing state for this specific experiment
        if not self.force:
            existing_state = self.state_manager.load_state()
            if existing_state:
                if self.state_manager.config_matches(existing_state, self.config):
                    # Check if we need to expand tasks to reach the new limit
                    existing_state = self._expand_tasks_if_needed(existing_state)

                    logger.info(
                        "Auto-resuming experiment",
                        experiment_name=self.config.experiment_name,
                        completed=len(
                            [t for t in existing_state.tasks.values() if t.status == "completed"]
                        ),
                        pending=len(self.state_manager.get_pending_tasks(existing_state)),
                    )
                    return existing_state
                else:
                    logger.warning(
                        "Config mismatch with existing state. Creating new experiment.",
                    )

        # Create new state - load task IDs directly from AppWorld
        logger.info("Loading task IDs from AppWorld", dataset=self.config.dataset)

        task_ids = self._load_task_ids(self.config.dataset, self.limit)
        logger.info("Loaded task IDs", count=len(task_ids))

        state = self.state_manager.create_new_state(self.config, task_ids)
        self.state_manager.save_state(state)

        return state

    def _setup_paths(self) -> None:
        """Set up database and conversations paths based on experiment name."""
        if self.config.db_path:
            self.db_path = self.config.db_path
        else:
            # Use a shared database for all experiments
            self.db_path = self.examples_dir / "experiments_shared.db"

        # Conversations directory is shared across experiments
        self.conversations_dir = self.examples_dir / "conversations"

    def _expand_tasks_if_needed(self, state: ExperimentState) -> ExperimentState:
        """Expand task list if the new limit is higher than current task count.

        When resuming an experiment with a higher --limit than before, this method
        adds new tasks from the dataset to reach the new limit.

        Args:
            state: Existing experiment state

        Returns:
            Updated state with additional tasks if limit was expanded
        """
        if self.limit is None:
            # No limit specified, nothing to expand
            return state

        current_task_count = len(state.task_ids)
        if current_task_count >= self.limit:
            # Already have enough tasks
            return state

        # Load full task list from AppWorld to get additional tasks
        all_task_ids = self._load_task_ids(self.config.dataset, limit=None)

        # Find the index where current tasks end in the full list
        # to ensure we add the next sequential tasks
        existing_task_set = set(state.task_ids)
        new_tasks_to_add = []

        for task_id in all_task_ids:
            if task_id not in existing_task_set:
                new_tasks_to_add.append(task_id)
                if current_task_count + len(new_tasks_to_add) >= self.limit:
                    break

        if not new_tasks_to_add:
            logger.info(
                "No additional tasks to add (all dataset tasks already in experiment)",
                current_count=current_task_count,
                limit=self.limit,
            )
            return state

        # Add new tasks to state
        for task_id in new_tasks_to_add:
            state.task_ids.append(task_id)
            state.tasks[task_id] = TaskResult(task_id=task_id, status="pending")

        self.state_manager.save_state(state)

        logger.info(
            "Expanded task list to reach new limit",
            previous_count=current_task_count,
            new_count=len(state.task_ids),
            added_tasks=len(new_tasks_to_add),
            limit=self.limit,
        )

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
        started_at = datetime.now(timezone.utc)

        # Update task to in_progress
        in_progress_task = TaskResult(
            task_id=task_id,
            status="in_progress",
            started_at=started_at,
        )
        state = self.state_manager.update_task(state, task_id, in_progress_task)

        # Create a single AppWorld instance for the entire task execution
        world = AppWorld(
            task_id=task_id,
            experiment_name=self.config.experiment_name,
            remote_apis_url=self.config.appworld_api_url,
            remote_mcp_url=self.config.appworld_mcp_url,
        )

        try:
            # Get task instruction directly from AppWorld
            instruction_result = self._get_instruction(world)
            instruction = instruction_result["instruction"]

            # Run agent
            agent_result = await self.agent.execute_task(instruction)

            # Save conversation to separate file
            conversation = agent_result.get("messages", [])
            conversation_file = self.state_manager.save_conversation(
                self.config.experiment_name, task_id, conversation
            )

            completed_at = datetime.now(timezone.utc)
            execution_time = time.perf_counter() - start_time

            # Check for agent error
            if agent_result.get("error"):
                if self.config.mode == "baseline":
                    return TaskResult(
                        task_id=task_id,
                        status="completed",
                        success=False,
                        error=agent_result["error"],
                        execution_time_s=execution_time,
                        direct_tool_calls=agent_result["tool_calls"].get("direct_tool_calls", 0),
                        tool_breakdown=agent_result.get("tool_breakdown", {}),
                        agent_steps=agent_result["tool_calls"]["total"],
                        request_tokens=agent_result["request_tokens"],
                        response_tokens=agent_result["response_tokens"],
                        conversation_file=conversation_file,
                        started_at=started_at,
                        completed_at=completed_at,
                    )
                else:
                    return TaskResult(
                        task_id=task_id,
                        status="completed",
                        success=False,
                        error=agent_result["error"],
                        execution_time_s=execution_time,
                        find_tool_calls=agent_result["tool_calls"]["find_tool"],
                        call_tool_calls=agent_result["tool_calls"]["call_tool"],
                        search_response_calls=agent_result["tool_calls"]["search_in_tool_response"],
                        agent_steps=agent_result["tool_calls"]["total"],
                        request_tokens=agent_result["request_tokens"],
                        response_tokens=agent_result["response_tokens"],
                        conversation_file=conversation_file,
                        started_at=started_at,
                        completed_at=completed_at,
                    )

            # Save and evaluate task completion
            eval_result = self._save_and_evaluate(world)

            # Extract success
            success = eval_result.get("success", False)

            if self.config.mode == "baseline":
                return TaskResult(
                    task_id=task_id,
                    status="completed",
                    success=success,
                    agent_steps=agent_result["tool_calls"]["total"],
                    direct_tool_calls=agent_result["tool_calls"].get("direct_tool_calls", 0),
                    tool_breakdown=agent_result.get("tool_breakdown", {}),
                    execution_time_s=execution_time,
                    request_tokens=agent_result["request_tokens"],
                    response_tokens=agent_result["response_tokens"],
                    conversation_file=conversation_file,
                    started_at=started_at,
                    completed_at=completed_at,
                )
            else:
                return TaskResult(
                    task_id=task_id,
                    status="completed",
                    success=success,
                    agent_steps=agent_result["tool_calls"]["total"],
                    find_tool_calls=agent_result["tool_calls"]["find_tool"],
                    call_tool_calls=agent_result["tool_calls"]["call_tool"],
                    search_response_calls=agent_result["tool_calls"]["search_in_tool_response"],
                    execution_time_s=execution_time,
                    request_tokens=agent_result["request_tokens"],
                    response_tokens=agent_result["response_tokens"],
                    conversation_file=conversation_file,
                    started_at=started_at,
                    completed_at=completed_at,
                )

        except Exception as e:
            logger.exception("Task execution failed", task_id=task_id, error=str(e))
            return TaskResult(
                task_id=task_id,
                status="failed",
                success=False,
                error=str(e),
                execution_time_s=time.perf_counter() - start_time,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )
        finally:
            # Always close the AppWorld instance to release DB connections
            world.close()

    def _compute_summary(self, state: ExperimentState) -> ExperimentSummaryFull:
        """Compute aggregated summary from state tasks.

        Args:
            state: Final experiment state

        Returns:
            ExperimentSummaryFull with aggregated metrics
        """
        task_results = []
        successful_count = 0
        failed_count = 0
        total_steps = 0
        total_execution_time = 0.0
        total_find_tool_calls = 0
        total_call_tool_calls = 0
        total_search_response_calls = 0
        total_direct_tool_calls = 0
        total_request_tokens = 0
        total_response_tokens = 0

        for task in state.tasks.values():
            if task.status in ("completed", "failed"):
                task_results.append(task)

                if task.success:
                    successful_count += 1
                else:
                    failed_count += 1

                total_steps += task.agent_steps
                total_execution_time += task.execution_time_s
                total_find_tool_calls += task.find_tool_calls
                total_call_tool_calls += task.call_tool_calls
                total_search_response_calls += task.search_response_calls
                total_direct_tool_calls += task.direct_tool_calls
                total_request_tokens += task.request_tokens
                total_response_tokens += task.response_tokens

        completed_count = len(task_results)
        success_rate = successful_count / completed_count if completed_count > 0 else 0.0
        avg_steps = total_steps / completed_count if completed_count > 0 else 0.0
        avg_execution_time = total_execution_time / completed_count if completed_count > 0 else 0.0

        return ExperimentSummaryFull(
            config=state.config,
            experiment_mode=state.config.mode,
            total_tasks=len(state.task_ids),
            completed_tasks=completed_count,
            successful_tasks=successful_count,
            failed_tasks=failed_count,
            success_rate=success_rate,
            avg_agent_steps=avg_steps,
            avg_execution_time_s=avg_execution_time,
            total_find_tool_calls=total_find_tool_calls,
            total_call_tool_calls=total_call_tool_calls,
            total_search_response_calls=total_search_response_calls,
            total_direct_tool_calls=total_direct_tool_calls,
            total_request_tokens=total_request_tokens,
            total_response_tokens=total_response_tokens,
            task_results=task_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
