"""Pydantic models for AppWorld MCP Optimizer experiments."""

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""

    experiment_name: str = Field(
        default="", description="Name for this experiment run (auto-generated if not provided)"
    )
    mode: Literal["optimizer", "baseline"] = Field(
        default="optimizer",
        description="Experiment mode: 'optimizer' uses MCP Optimizer, 'baseline' uses direct MCP",
    )
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
    max_agent_steps: int = Field(default=100, description="Maximum agent steps per task")
    appworld_mcp_url: str = Field(
        default="http://localhost:10000", description="AppWorld MCP server URL"
    )
    appworld_api_url: str = Field(
        default="http://localhost:9000", description="AppWorld API server URL for remote_apis_url"
    )
    db_path: Path | None = Field(
        default=None, description="Path to database file (shared across experiments by default)"
    )


class TaskResult(BaseModel):
    """Result of a single task execution, including status tracking.

    This model serves as both the task state (for tracking progress) and the
    task result (for storing execution metrics). Conversations are stored
    in separate files and referenced by conversation_file.
    """

    task_id: str = Field(description="AppWorld task ID")
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending", description="Current status of the task"
    )
    success: bool | None = Field(
        default=None, description="Whether task was successfully completed"
    )
    agent_steps: int = Field(default=0, description="Number of agent steps taken")
    find_tool_calls: int = Field(default=0, description="Number of find_tool calls")
    call_tool_calls: int = Field(default=0, description="Number of call_tool calls")
    search_response_calls: int = Field(
        default=0, description="Number of search_in_tool_response calls"
    )
    direct_tool_calls: int = Field(default=0, description="Direct tool calls (baseline mode)")
    tool_breakdown: dict[str, int] = Field(
        default_factory=dict, description="Tool call breakdown by name"
    )
    execution_time_s: float = Field(default=0.0, description="Execution time in seconds")
    request_tokens: int = Field(default=0, description="Total request tokens used")
    response_tokens: int = Field(default=0, description="Total response tokens used")
    error: str | None = Field(default=None, description="Error message if task failed")
    conversation_file: str | None = Field(
        default=None, description="Path to conversation JSON file (relative to experiment dir)"
    )
    started_at: datetime | None = Field(default=None, description="When task execution started")
    completed_at: datetime | None = Field(default=None, description="When task execution completed")


class ExperimentSummary(BaseModel):
    """Aggregated experiment summary metrics."""

    experiment_mode: str = Field(default="optimizer", description="Mode the experiment was run in")
    total_tasks: int = Field(description="Total number of tasks in experiment")
    completed_tasks: int = Field(description="Number of tasks completed (success or failure)")
    successful_tasks: int = Field(description="Number of tasks successfully completed")
    failed_tasks: int = Field(default=0, description="Number of tasks that failed")
    success_rate: float = Field(description="Success rate (0.0-1.0)")
    avg_agent_steps: float = Field(default=0.0, description="Average agent steps per task")
    avg_execution_time_s: float = Field(default=0.0, description="Average execution time")
    total_find_tool_calls: int = Field(default=0, description="Total find_tool calls")
    total_call_tool_calls: int = Field(default=0, description="Total call_tool calls")
    total_search_response_calls: int = Field(
        default=0, description="Total search_in_tool_response calls"
    )
    total_direct_tool_calls: int = Field(
        default=0, description="Total direct tool calls (baseline)"
    )
    total_request_tokens: int = Field(default=0, description="Total request tokens")
    total_response_tokens: int = Field(default=0, description="Total response tokens")
    timestamp: str = Field(description="Timestamp of summary generation (ISO format)")


class ExperimentState(BaseModel):
    """Full experiment state - single source of truth for an experiment.

    This file contains all experiment data except conversations, which are
    stored in separate files under the conversations/ directory.
    """

    config: ExperimentConfig = Field(description="Experiment configuration")
    task_ids: list[str] = Field(description="List of task IDs to process")
    tasks: dict[str, TaskResult] = Field(
        default_factory=dict, description="Task results keyed by task ID"
    )
    started_at: datetime = Field(description="When experiment started")
    last_updated: datetime = Field(description="When state was last updated")
    ingestion_completed: bool = Field(
        default=False, description="Whether tool ingestion is complete"
    )
    tools_count: int = Field(default=0, description="Number of tools ingested")
    summary: ExperimentSummary | None = Field(
        default=None, description="Experiment summary (populated when experiment completes)"
    )


class ExperimentSummaryFull(ExperimentSummary):
    """Full experiment summary with config and task results for display/return."""

    config: ExperimentConfig = Field(description="Experiment configuration")
    task_results: list[TaskResult] = Field(description="Individual task results")
