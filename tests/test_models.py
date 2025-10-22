from datetime import datetime, timezone

import numpy as np
import pytest
from mcp.types import Tool as McpTool
from pydantic import ValidationError

from mcp_optimizer.db.models import WorkloadTool, WorkloadToolUpdateDetails


class TestToolModelValidation:
    """Test cases for WorkloadTool model numpy array validation."""

    def test_tool_with_valid_1d_array(self):
        """Test that WorkloadTool accepts valid 1D numpy arrays."""
        valid_embedding = np.random.rand(384).astype(np.float32)
        tool = WorkloadTool(
            id="test-id",
            mcpserver_id="test-server",
            details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
            details_embedding=valid_embedding,
            token_count=42,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        assert isinstance(tool.details_embedding, np.ndarray)
        assert tool.details_embedding.ndim == 1

    def test_tool_with_2d_array_fails(self):
        """Test that WorkloadTool rejects 2D numpy arrays."""
        invalid_embedding = np.random.rand(10, 38).astype(np.float32)
        with pytest.raises(ValueError, match="details_embedding must be 1D array, got 2D"):
            WorkloadTool(
                id="test-id",
                mcpserver_id="test-server",
                details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
                details_embedding=invalid_embedding,
                token_count=42,
                last_updated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )

    def test_tool_with_non_numpy_array_fails(self):
        """Test that WorkloadTool rejects non-numpy array inputs."""
        invalid_embedding = [1, 2, 3, 4]  # list instead of numpy array
        with pytest.raises(ValidationError, match="Input should be an instance of ndarray"):
            WorkloadTool(
                id="test-id",
                mcpserver_id="test-server",
                details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
                details_embedding=invalid_embedding,
                token_count=42,
                last_updated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )

    def test_tool_update_details_with_valid_1d_array(self):
        """Test that ToolUpdateDetails accepts valid 1D numpy arrays."""
        valid_embedding = np.random.rand(384).astype(np.float32)
        # Need to provide details when providing details_embedding due to model validator
        update_details = WorkloadToolUpdateDetails(
            details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
            details_embedding=valid_embedding,
        )
        assert isinstance(update_details.details_embedding, np.ndarray)
        assert update_details.details_embedding.ndim == 1

    def test_tool_update_details_with_2d_array_fails(self):
        """Test that ToolUpdateDetails rejects 2D numpy arrays."""
        invalid_embedding = np.random.rand(10, 38).astype(np.float32)
        with pytest.raises(ValueError, match="details_embedding must be 1D array, got 2D"):
            WorkloadToolUpdateDetails(
                details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
                details_embedding=invalid_embedding,
            )

    def test_tool_update_details_with_none_allowed(self):
        """Test that ToolUpdateDetails allows None for optional field."""
        update_details = WorkloadToolUpdateDetails(details_embedding=None)
        assert update_details.details_embedding is None

    def test_tool_update_details_with_non_numpy_array_fails(self):
        """Test that ToolUpdateDetails rejects non-numpy array inputs."""
        invalid_embedding = [1, 2, 3, 4]  # list instead of numpy array
        with pytest.raises(ValidationError, match="Input should be an instance of ndarray"):
            WorkloadToolUpdateDetails(
                details=McpTool(name="test", description="test", inputSchema={"type": "object"}),
                details_embedding=invalid_embedding,
            )
