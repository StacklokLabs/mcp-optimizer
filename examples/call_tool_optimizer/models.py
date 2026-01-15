"""Pydantic models for AppWorld MCP Optimizer experiments."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""

    experiment_name: str = Field(description="Name for this experiment run")
    dataset: str = Field(default="train", description="AppWorld dataset to use")
    llm_model: str = Field(
        default="anthropic/claude-sonnet-4", description="LLM model for the agent"
    )
    response_optimizer_threshold: int = Field(
        default=1000, description="Token threshold for response optimization"
    )
    response_head_lines: int = Field(
        default=20, description="Lines to preserve from start for unstructured text"
    )
    response_tail_lines: int = Field(
        default=20, description="Lines to preserve from end for unstructured text"
    )
    max_agent_steps: int = Field(default=50, description="Maximum agent steps per task")
    appworld_mcp_url: str = Field(
        default="http://localhost:10000", description="AppWorld MCP server URL"
    )
    db_path: Path | None = Field(default=None, description="Path to database file")


class TaskState(BaseModel):
    """State of a single task execution."""

    task_id: str = Field(description="AppWorld task ID")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        description="Current status of the task"
    )
    instruction: str | None = Field(default=None, description="Task instruction from AppWorld")
    evaluation_result: dict | None = Field(default=None, description="AppWorld evaluation result")
    error: str | None = Field(default=None, description="Error message if task failed")
    started_at: datetime | None = Field(default=None, description="When task execution started")
    completed_at: datetime | None = Field(default=None, description="When task execution completed")


class ExperimentState(BaseModel):
    """Full experiment state for persistence and recovery."""

    config: ExperimentConfig = Field(description="Experiment configuration")
    task_ids: list[str] = Field(description="List of task IDs to process")
    task_states: dict[str, TaskState] = Field(
        default_factory=dict, description="Task states keyed by task ID"
    )
    started_at: datetime = Field(description="When experiment started")
    last_updated: datetime = Field(description="When state was last updated")
    ingestion_completed: bool = Field(
        default=False, description="Whether tool ingestion is complete"
    )
    tools_count: int = Field(default=0, description="Number of tools ingested")


class TaskResult(BaseModel):
    """Result of a single task execution."""

    task_id: str = Field(description="AppWorld task ID")
    success: bool = Field(description="Whether task was successfully completed")
    goal_progress: float = Field(default=0.0, description="Goal progress from AppWorld (0.0-1.0)")
    agent_steps: int = Field(default=0, description="Number of agent steps taken")
    find_tool_calls: int = Field(default=0, description="Number of find_tool calls")
    call_tool_calls: int = Field(default=0, description="Number of call_tool calls")
    search_response_calls: int = Field(
        default=0, description="Number of search_in_tool_response calls"
    )
    execution_time_s: float = Field(default=0.0, description="Execution time in seconds")
    request_tokens: int = Field(default=0, description="Total request tokens used")
    response_tokens: int = Field(default=0, description="Total response tokens used")
    error: str | None = Field(default=None, description="Error message if task failed")


class ExperimentResults(BaseModel):
    """Aggregated experiment results."""

    config: ExperimentConfig = Field(description="Experiment configuration")
    total_tasks: int = Field(description="Total number of tasks in experiment")
    completed_tasks: int = Field(description="Number of tasks completed (success or failure)")
    successful_tasks: int = Field(description="Number of tasks successfully completed")
    failed_tasks: int = Field(default=0, description="Number of tasks that failed")
    success_rate: float = Field(description="Success rate (0.0-1.0)")
    avg_goal_progress: float = Field(default=0.0, description="Average goal progress")
    avg_agent_steps: float = Field(default=0.0, description="Average agent steps per task")
    avg_execution_time_s: float = Field(default=0.0, description="Average execution time")
    total_find_tool_calls: int = Field(default=0, description="Total find_tool calls")
    total_call_tool_calls: int = Field(default=0, description="Total call_tool calls")
    total_search_response_calls: int = Field(
        default=0, description="Total search_in_tool_response calls"
    )
    total_request_tokens: int = Field(default=0, description="Total request tokens")
    total_response_tokens: int = Field(default=0, description="Total response tokens")
    task_results: list[TaskResult] = Field(description="Individual task results")
    timestamp: str = Field(description="Timestamp of report generation (ISO format)")
