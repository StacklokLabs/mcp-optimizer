"""Tool conversion utilities for converting MCP tools to Anthropic format."""

import re

from pydantic import BaseModel, field_validator


class ToolParameter(BaseModel):
    """Represents a tool parameter with validation."""

    name: str
    type: str
    description: str
    is_optional: bool = False
    items: dict | None = None  # For array types

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Sanitize parameter name to match ^[a-zA-Z0-9_.-]{1,64}$"""
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]", "_", v)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")[:64]
        if not sanitized:
            sanitized = "param"
        return sanitized


class AnthropicTool(BaseModel):
    """Represents an Anthropic API tool with validation."""

    name: str
    description: str
    input_schema: dict
    defer_loading: bool | None = True

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Sanitize tool name to match ^[a-zA-Z0-9_-]{1,128}$"""
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", v)
        sanitized = re.sub(r"-+", "-", sanitized).strip("-")[:128]
        if not sanitized:
            sanitized = "tool"
        return sanitized


class ToolConverter:
    """Converts MCP tools to Anthropic format with validation."""

    def __init__(self):
        self.tool_name_counts: dict[str, int] = {}

    def parse_parameter_type(self, param_desc: str) -> tuple[str, str, bool]:
        """
        Parse parameter description to extract type, description, and optionality.
        Returns: (type, description, is_optional)
        """
        match = re.match(r"\(([^)]+)\)\s*(.*)", param_desc)
        if not match:
            return "string", param_desc, False

        type_info = match.group(1).lower()
        description = match.group(2).strip()

        is_optional = (
            "optional" in type_info
            or "opt" in type_info
            or type_info.startswith("optional,")
            or type_info.endswith(", optional")
        )

        # Determine JSON Schema type (check arrays first before scalar types)
        if "array" in type_info or "[]" in type_info or "list" in type_info:
            if "string" in type_info:
                return "array_string", description, is_optional
            elif "object" in type_info:
                return "array_object", description, is_optional
            elif "number" in type_info or "int" in type_info:
                return "array_number", description, is_optional
            else:
                return "array", description, is_optional
        elif "object" in type_info or "dict" in type_info:
            return "object", description, is_optional
        elif "string" in type_info or "str" in type_info:
            return "string", description, is_optional
        elif "number" in type_info or "int" in type_info or "float" in type_info:
            return "number", description, is_optional
        elif "bool" in type_info:
            return "boolean", description, is_optional
        else:
            return "string", description, is_optional

    def convert_parameter(self, param_name: str, param_desc: str) -> tuple[ToolParameter, dict]:
        """Convert a single parameter to ToolParameter and its schema."""
        param_type, description, is_optional = self.parse_parameter_type(param_desc)

        # Create parameter
        items = None
        json_type = param_type

        if param_type == "array_string":
            json_type = "array"
            items = {"type": "string"}
        elif param_type == "array_object":
            json_type = "array"
            items = {"type": "object"}
        elif param_type == "array_number":
            json_type = "array"
            items = {"type": "number"}
        elif param_type == "array":
            json_type = "array"
            items = {"type": "string"}  # Default to string array

        param = ToolParameter(
            name=param_name,
            type=json_type,
            description=description,
            is_optional=is_optional,
            items=items,
        )

        # Build schema dict
        schema = {"type": json_type, "description": description}
        if items:
            schema["items"] = items

        return param, schema

    def make_unique_name(self, base_name: str) -> str:
        """Make tool name unique by appending a counter if needed."""
        if base_name not in self.tool_name_counts:
            self.tool_name_counts[base_name] = 1
            return base_name

        self.tool_name_counts[base_name] += 1
        return f"{base_name}-{self.tool_name_counts[base_name]}"

    def convert_tool(self, server: dict, tool: dict) -> AnthropicTool:
        """Convert an MCP tool to Anthropic format."""
        properties = {}
        required = []

        if "parameter" in tool and tool["parameter"]:
            for param_name, param_desc in tool["parameter"].items():
                param, schema = self.convert_parameter(param_name, param_desc)
                properties[param.name] = schema

                if not param.is_optional:
                    required.append(param.name)

        # Create base tool name
        server_prefix = server["name"].lower()
        tool_name = tool["name"]
        base_name = f"{server_prefix}-{tool_name}"

        # Make name unique
        unique_name = self.make_unique_name(base_name)

        return AnthropicTool(
            name=unique_name,
            description=tool["description"],
            input_schema={
                "type": "object",
                "properties": properties,
                "required": required,
            },
        )
