"""Utility functions shared across agent implementations."""

import json

from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart


def _safe_parse_args(args: str | dict | None) -> dict | str | None:
    """Safely parse tool call arguments.

    Args:
        args: Tool call arguments (can be str, dict, or None)

    Returns:
        Parsed arguments as dict, or original value if parsing fails
    """
    if args is None:
        return None

    if isinstance(args, dict):
        return args

    if isinstance(args, str):
        # Handle empty strings
        if not args or not args.strip():
            return {}

        try:
            return json.loads(args)
        except json.JSONDecodeError:
            # Return the original string if it's not valid JSON
            return args

    # Fallback for other types
    return str(args)


def serialize_agent_messages(result: AgentRunResult) -> list[dict]:
    """Serialize agent messages for storage.

    Args:
        result: Agent run result

    Returns:
        List of serialized message dictionaries
    """
    messages = []

    for message in result.all_messages():
        if isinstance(message, ModelResponse):
            msg_data: dict = {
                "type": "model_response",
                "parts": [],
            }
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    msg_data["parts"].append(
                        {
                            "type": "tool_call",
                            "tool_name": part.tool_name,
                            "args": _safe_parse_args(part.args),
                        }
                    )
                else:
                    msg_data["parts"].append(
                        {
                            "type": "text",
                            "content": str(part),
                        }
                    )
            messages.append(msg_data)

        elif isinstance(message, ModelRequest):
            msg_data = {
                "type": "model_request",
                "parts": [],
            }
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    # Truncate long tool returns for storage
                    content = str(part.content)
                    if len(content) > 1000:
                        content = content[:1000] + "..."
                    msg_data["parts"].append(
                        {
                            "type": "tool_return",
                            "tool_name": part.tool_name,
                            "content": content,
                        }
                    )
                else:
                    msg_data["parts"].append(
                        {
                            "type": "other",
                            "content": str(part)[:500],
                        }
                    )
            messages.append(msg_data)

    return messages
