#!/usr/bin/env python3
"""
Helper script for AppWorld operations.

This script runs in an isolated environment with appworld installed
(via `uv run --no-project --with appworld`) and provides access to:
- Task IDs for datasets
- Task instructions
- Task evaluation

It communicates via JSON on stdin/stdout for easy subprocess integration.

Usage:
    # Get task IDs
    echo '{"action": "list_tasks", "dataset": "train"}' | \
        uv run --no-project --with appworld python appworld_helper.py

    # Get task instruction
    echo '{"action": "get_instruction", "task_id": "train_001", "experiment_name": "exp1"}' | \
        uv run --no-project --with appworld python appworld_helper.py

    # Evaluate task (after agent has run)
    echo '{"action": "evaluate", "task_id": "train_001", "experiment_name": "exp1"}' | \
        uv run --no-project --with appworld python appworld_helper.py
"""

import json
import sys


def list_tasks(dataset: str, limit: int | None = None) -> dict:
    """Get task IDs for a dataset.

    Args:
        dataset: Dataset name (train, dev, test_normal, test_challenge)
        limit: Optional limit on number of tasks

    Returns:
        dict with task_ids list
    """
    from appworld import load_task_ids

    task_ids = load_task_ids(dataset)
    if limit:
        task_ids = task_ids[:limit]

    return {"task_ids": task_ids}


def get_instruction(task_id: str, experiment_name: str) -> dict:
    """Get task instruction.

    Args:
        task_id: AppWorld task ID
        experiment_name: Experiment name for AppWorld context

    Returns:
        dict with instruction and task metadata
    """
    from appworld import AppWorld

    with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
        return {
            "task_id": task_id,
            "instruction": world.task.instruction,
            "supervisor": {
                "name": getattr(world.task.supervisor, "name", None),
                "email": getattr(world.task.supervisor, "email", None),
            },
        }


def evaluate(task_id: str, experiment_name: str) -> dict:
    """Evaluate task completion.

    Args:
        task_id: AppWorld task ID
        experiment_name: Experiment name for AppWorld context

    Returns:
        dict with evaluation result
    """
    from appworld import AppWorld

    with AppWorld(task_id=task_id, experiment_name=experiment_name) as world:
        evaluation = world.evaluate()
        eval_dict = evaluation.to_dict()

        return {
            "task_id": task_id,
            "success": eval_dict.get("success", False),
            "goal_progress": eval_dict.get("goal_progress", 0.0),
            "evaluation": eval_dict,
        }


def main():
    """Process command from stdin and output result to stdout."""
    try:
        # Read JSON command from stdin
        input_data = sys.stdin.read()
        command = json.loads(input_data)

        action = command.get("action")

        if action == "list_tasks":
            result = list_tasks(
                dataset=command["dataset"],
                limit=command.get("limit"),
            )
        elif action == "get_instruction":
            result = get_instruction(
                task_id=command["task_id"],
                experiment_name=command["experiment_name"],
            )
        elif action == "evaluate":
            result = evaluate(
                task_id=command["task_id"],
                experiment_name=command["experiment_name"],
            )
        else:
            result = {"error": f"Unknown action: {action}"}

        # Output result as JSON
        print(json.dumps(result))

    except Exception as e:
        # Output error as JSON
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
