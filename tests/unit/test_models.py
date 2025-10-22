"""Tests for database models."""

import pytest
from mcp.types import Tool as McpTool

from mcp_optimizer.db.models import TokenMetrics, WorkloadTool


class TestTokenMetrics:
    """Test TokenMetrics model."""

    def test_valid_metrics_with_savings(self):
        """Test TokenMetrics with valid metrics showing savings."""
        metrics = TokenMetrics(
            baseline_tokens=1000,
            returned_tokens=400,
            tokens_saved=600,
            savings_percentage=60.0,
        )
        assert metrics.baseline_tokens == 1000
        assert metrics.returned_tokens == 400
        assert metrics.tokens_saved == 600
        assert metrics.savings_percentage == 60.0

    def test_valid_metrics_no_savings(self):
        """Test TokenMetrics when no tokens are saved (all tools returned)."""
        metrics = TokenMetrics(
            baseline_tokens=1000,
            returned_tokens=1000,
            tokens_saved=0,
            savings_percentage=0.0,
        )
        assert metrics.baseline_tokens == 1000
        assert metrics.returned_tokens == 1000
        assert metrics.tokens_saved == 0
        assert metrics.savings_percentage == 0.0

    def test_zero_baseline_tokens(self):
        """Test TokenMetrics handles zero baseline (no running servers)."""
        metrics = TokenMetrics(
            baseline_tokens=0,
            returned_tokens=0,
            tokens_saved=0,
            savings_percentage=0.0,
        )
        assert metrics.baseline_tokens == 0
        assert metrics.returned_tokens == 0
        assert metrics.tokens_saved == 0
        assert metrics.savings_percentage == 0.0

    def test_tokens_saved_calculation_validation(self):
        """Test that tokens_saved must equal baseline_tokens - returned_tokens."""
        with pytest.raises(
            ValueError, match="tokens_saved must equal baseline_tokens - returned_tokens"
        ):
            TokenMetrics(
                baseline_tokens=1000,
                returned_tokens=400,
                tokens_saved=500,  # Wrong! Should be 600
                savings_percentage=50.0,
            )

    def test_tokens_saved_calculation_validation_negative(self):
        """Test that tokens_saved calculation rejects negative values."""
        with pytest.raises(
            ValueError, match="tokens_saved must equal baseline_tokens - returned_tokens"
        ):
            TokenMetrics(
                baseline_tokens=1000,
                returned_tokens=400,
                tokens_saved=700,  # Wrong! Should be 600
                savings_percentage=70.0,
            )

    def test_savings_percentage_calculation_validation(self):
        """Test that savings_percentage matches calculated value."""
        with pytest.raises(ValueError, match="savings_percentage does not match calculated value"):
            TokenMetrics(
                baseline_tokens=1000,
                returned_tokens=400,
                tokens_saved=600,
                savings_percentage=50.0,  # Wrong! Should be 60.0
            )

    def test_savings_percentage_must_be_zero_when_baseline_is_zero(self):
        """Test that savings_percentage must be 0 when baseline_tokens is 0."""
        with pytest.raises(
            ValueError, match="savings_percentage must be 0 when baseline_tokens is 0"
        ):
            TokenMetrics(
                baseline_tokens=0,
                returned_tokens=0,
                tokens_saved=0,
                savings_percentage=50.0,  # Wrong! Should be 0.0
            )

    def test_high_savings_percentage(self):
        """Test TokenMetrics with high savings percentage."""
        metrics = TokenMetrics(
            baseline_tokens=1000,
            returned_tokens=50,
            tokens_saved=950,
            savings_percentage=95.0,
        )
        assert metrics.savings_percentage == 95.0

    def test_savings_percentage_precision(self):
        """Test TokenMetrics handles floating point precision correctly."""
        # Test with a percentage that requires precision: 1/3 = 33.333...%
        metrics = TokenMetrics(
            baseline_tokens=300,
            returned_tokens=200,
            tokens_saved=100,
            savings_percentage=33.33,  # Rounded to 2 decimal places
        )
        assert abs(metrics.savings_percentage - 33.33) < 0.01


class TestWorkloadToolModel:
    """Test WorkloadTool model."""

    def test_tool_with_default_token_count(self):
        """Test WorkloadTool model has default token_count of 0."""
        from datetime import datetime, timezone

        tool = WorkloadTool(
            id="test-id",
            mcpserver_id="server-id",
            details=McpTool(name="test_tool", inputSchema={}),
            details_embedding=None,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        assert tool.token_count == 0

    def test_tool_with_explicit_token_count(self):
        """Test WorkloadTool model with explicit token_count value."""
        from datetime import datetime, timezone

        tool = WorkloadTool(
            id="test-id",
            mcpserver_id="server-id",
            details=McpTool(name="test_tool", inputSchema={}),
            details_embedding=None,
            token_count=150,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        assert tool.token_count == 150

    def test_tool_rejects_negative_token_count(self):
        """Test WorkloadTool model rejects negative token_count."""
        from datetime import datetime, timezone

        with pytest.raises(ValueError):
            WorkloadTool(
                id="test-id",
                mcpserver_id="server-id",
                details=McpTool(name="test_tool", inputSchema={}),
                details_embedding=None,
                token_count=-10,  # Invalid: negative token count
                last_updated=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
            )

    def test_tool_with_zero_token_count(self):
        """Test WorkloadTool model accepts zero token_count."""
        from datetime import datetime, timezone

        tool = WorkloadTool(
            id="test-id",
            mcpserver_id="server-id",
            details=McpTool(name="test_tool", inputSchema={}),
            details_embedding=None,
            token_count=0,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        assert tool.token_count == 0

    def test_tool_with_large_token_count(self):
        """Test WorkloadTool model with large token_count value."""
        from datetime import datetime, timezone

        tool = WorkloadTool(
            id="test-id",
            mcpserver_id="server-id",
            details=McpTool(name="test_tool", inputSchema={}),
            details_embedding=None,
            token_count=100000,
            last_updated=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc),
        )
        assert tool.token_count == 100000
