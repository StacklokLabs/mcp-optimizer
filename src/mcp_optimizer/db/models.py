import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Self

import numpy as np
from mcp.types import Tool as McpTool
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class TransportType(str, Enum):
    """
    Enum for transport types.
    There is 1:1 relation between ToolHive transport modes to database transport types.
    """

    SSE = "sse"
    STREAMABLE = "streamable-http"


class McpStatus(str, Enum):
    """
    Enum for MCP server status.
    - RUNNING: ToolHive workload is running
    - STOPPED: ToolHive workload is stopped
    """

    RUNNING = "running"
    STOPPED = "stopped"


class TokenMetrics(BaseModel):
    """Token efficiency metrics for tool filtering."""

    baseline_tokens: int = Field(ge=0, description="Total tokens for all running server tools")
    returned_tokens: int = Field(ge=0, description="Total tokens for returned/filtered tools")
    tokens_saved: int = Field(ge=0, description="Number of tokens saved by filtering")
    savings_percentage: float = Field(ge=0.0, le=100.0, description="Percentage of tokens saved")

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        """Validate token metrics consistency."""
        if self.tokens_saved != self.baseline_tokens - self.returned_tokens:
            raise ValueError("tokens_saved must equal baseline_tokens - returned_tokens")

        if self.baseline_tokens > 0:
            expected_pct = (self.tokens_saved / self.baseline_tokens) * 100
            if abs(self.savings_percentage - expected_pct) > 0.01:
                raise ValueError("savings_percentage does not match calculated value")
        else:
            if self.savings_percentage != 0.0:
                raise ValueError("savings_percentage must be 0 when baseline_tokens is 0")

        return self


class BaseMcpServer(BaseModel):
    """Base class for MCP servers (registry and workload)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # UUID4
    name: str
    remote: bool
    transport: TransportType
    description: str | None = None
    server_embedding: np.ndarray | None = Field(default=None, exclude=True)  # 1D numpy array
    group: str = Field(default="default")
    last_updated: datetime
    created_at: datetime

    @field_validator("server_embedding", mode="after")
    @classmethod
    def validate_server_embedding_is_1d_array(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("server_embedding must be a numpy array")
            if v.ndim != 1:
                raise ValueError(f"server_embedding must be 1D array, got {v.ndim}D")
        return v

    @field_serializer("last_updated", "created_at")
    def serialize_datetime(self, value: datetime) -> str:
        """Convert datetime to ISO string for JSON serialization."""
        return value.isoformat()


class RegistryServer(BaseMcpServer):
    """Registry MCP server from catalog."""

    url: str | None = None
    package: str | None = None

    @model_validator(mode="after")
    def validate_identifier(self) -> Self:
        """Ensure remote servers have URL, container servers have package."""
        if self.remote and not self.url:
            raise ValueError("Remote servers must have URL")
        if not self.remote and not self.package:
            raise ValueError("Container servers must have package")
        return self


class WorkloadServer(BaseMcpServer):
    """Workload MCP server from running instance."""

    url: str
    workload_identifier: str
    status: McpStatus
    registry_server_id: str | None = None  # NULL if autonomous
    registry_server_name: str | None = None  # Cached for tool embedding context


class BaseTool(BaseModel):
    """Base class for tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str  # UUID4
    mcpserver_id: str  # References mcpservers_registry.id
    details: McpTool  # MCP tool definition with name, description, inputSchema
    details_embedding: np.ndarray | None = Field(default=None, exclude=True)  # 1D numpy array
    last_updated: datetime
    created_at: datetime

    @field_validator("details_embedding", mode="after")
    @classmethod
    def validate_embedding_is_1d_array(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is not None:
            if not isinstance(v, np.ndarray):
                raise ValueError("details_embedding must be a numpy array")
            if v.ndim != 1:
                raise ValueError(f"details_embedding must be 1D array, got {v.ndim}D")
        return v

    @field_serializer("last_updated", "created_at")
    def serialize_datetime(self, value: datetime) -> str:
        """Convert datetime to ISO string for JSON serialization."""
        return value.isoformat()


class RegistryTool(BaseTool):
    """Tool from registry MCP server (matches current Tool schema)."""

    pass


class WorkloadTool(BaseTool):
    """Tool from workload MCP server (matches current Tool schema)."""

    token_count: int = Field(default=0, ge=0)  # Token count for LLM consumption


class BaseUpdateDetails(BaseModel):
    """Base class for update details."""

    def needs_update(self) -> bool:
        """Check if any field is set for update."""
        return any(value is not None for value in self.model_dump().values())

    def get_update_fields(self) -> dict[str, Any]:
        """Build the SET clause for SQL update."""
        update_fields = {}
        for field, value in self.model_dump().items():
            if value is not None:
                if isinstance(value, dict):
                    update_fields[field] = json.dumps(value)
                elif isinstance(value, np.ndarray):
                    update_fields[field] = value.tobytes()
                else:
                    update_fields[field] = value
        update_fields["last_updated"] = datetime.now(timezone.utc)  # Always update last_updated
        return update_fields


class BaseServerUpdateDetails(BaseUpdateDetails):
    """Base class for server update details."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    description: str | None = None
    server_embedding: np.ndarray | None = Field(default=None, exclude=True)  # 1D numpy array
    transport: TransportType | None = None
    group: str | None = None

    @field_validator("server_embedding", mode="after")
    @classmethod
    def validate_server_embedding_is_1d_array(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return v
        if not isinstance(v, np.ndarray):
            raise ValueError("server_embedding must be a numpy array")
        if v.ndim != 1:
            raise ValueError(f"server_embedding must be 1D array, got {v.ndim}D")
        return v


class RegistryServerUpdateDetails(BaseServerUpdateDetails):
    """Details for updating a registry server."""

    url: str | None = None
    package: str | None = None


class WorkloadServerUpdateDetails(BaseServerUpdateDetails):
    """Details for updating a workload server."""

    url: str | None = None
    workload_identifier: str | None = None
    status: McpStatus | None = None
    registry_server_id: str | None = None
    registry_server_name: str | None = None


class BaseToolUpdateDetails(BaseUpdateDetails):
    """Base class for tool update details."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mcpserver_id: str | None = None
    details: McpTool | None = None
    details_embedding: np.ndarray | None = None

    @field_validator("details_embedding", mode="after")
    @classmethod
    def validate_embedding_is_1d_array(cls, v: np.ndarray | None) -> np.ndarray | None:
        if v is None:
            return v
        if not isinstance(v, np.ndarray):
            raise ValueError("details_embedding must be a numpy array")
        if v.ndim != 1:
            raise ValueError(f"details_embedding must be 1D array, got {v.ndim}D")
        return v

    @model_validator(mode="after")
    def check_details(self) -> Self:
        if self.details is None and self.details_embedding is not None:
            raise ValueError("Cannot update details_embedding without providing details.")
        if self.details is not None and self.details_embedding is None:
            raise ValueError("Cannot update details without providing details_embedding.")
        return self


class RegistryToolUpdateDetails(BaseToolUpdateDetails):
    """Details for updating a registry tool."""

    pass


class WorkloadToolUpdateDetails(BaseToolUpdateDetails):
    """Details for updating a workload tool."""

    token_count: int | None = Field(default=None, ge=0)  # Token count for LLM consumption


class BaseToolWithMetadata(BaseModel):
    """Base tool with server information and similarity distance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    server_name: str
    server_description: str | None = Field(default=None)
    distance: float = Field(description="Cosine distance from query embedding")


class RegistryToolWithMetadata(BaseToolWithMetadata):
    """Registry tool with server information and similarity distance."""

    tool: RegistryTool


class WorkloadToolWithMetadata(BaseToolWithMetadata):
    """Workload tool with server information and similarity distance."""

    tool: WorkloadTool


class WorkloadWithRegistry(BaseModel):
    """Workload server with resolved registry relationship."""

    workload: WorkloadServer
    registry: RegistryServer | None = None  # None if autonomous

    @property
    def effective_description(self) -> str | None:
        """Get description (inherited from registry or own)."""
        if self.registry:
            return self.registry.description
        return self.workload.description

    @property
    def effective_embedding(self) -> np.ndarray | None:
        """Get embedding (inherited from registry or own)."""
        if self.registry:
            return self.registry.server_embedding
        return self.workload.server_embedding

    @property
    def server_name_for_tools(self) -> str:
        """Get server name to use as context for tool embeddings."""
        if self.registry:
            return self.registry.name
        return self.workload.name
