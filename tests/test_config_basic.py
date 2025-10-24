"""Basic tests for configuration validation."""

import pytest
from pydantic import ValidationError

from mcp_optimizer.config import MCPOptimizerConfig


def test_config_validation():
    """Test basic configuration validation."""
    config = MCPOptimizerConfig()
    assert config.mcp_port == 9900

    # Test invalid port
    with pytest.raises(ValidationError):
        MCPOptimizerConfig(mcp_port=1023)


def test_port_range_validation():
    """Test port range validation with model validator."""
    # Test invalid range: start port > end port
    with pytest.raises(ValidationError, match="End port.*must be greater than.*start port"):
        MCPOptimizerConfig(toolhive_start_port_scan=50100, toolhive_end_port_scan=50000)

    # Test invalid range: start port = end port
    with pytest.raises(ValidationError, match="End port.*must be greater than.*start port"):
        MCPOptimizerConfig(toolhive_start_port_scan=50000, toolhive_end_port_scan=50000)

    # Test valid range
    config = MCPOptimizerConfig(toolhive_start_port_scan=50000, toolhive_end_port_scan=50100)
    assert config.toolhive_start_port_scan == 50000
    assert config.toolhive_end_port_scan == 50100


def test_port_range_validation_with_dict():
    """Test that model validator catches invalid ranges regardless of field assignment order."""
    # Test with dictionary (field order doesn't matter with model validator)
    invalid_config_data = {
        "toolhive_end_port_scan": 50000,  # Set end port first
        "toolhive_start_port_scan": 50100,  # Then set start port higher
    }

    with pytest.raises(ValidationError, match="End port.*must be greater than.*start port"):
        MCPOptimizerConfig(**invalid_config_data)


def test_database_url_validation():
    """Test database URL validation."""
    # Test valid URLs
    config = MCPOptimizerConfig(
        async_db_url="sqlite+aiosqlite:///test.db", db_url="sqlite:///test.db"
    )
    assert config.async_db_url == "sqlite+aiosqlite:///test.db"
    assert config.db_url == "sqlite:///test.db"

    # Test invalid URL scheme
    with pytest.raises(ValidationError, match="Database URL must start with supported scheme"):
        MCPOptimizerConfig(db_url="mysql://localhost/test")

    # Test inconsistent database paths
    with pytest.raises(ValidationError, match="Database URLs must point to the same database"):
        MCPOptimizerConfig(async_db_url="sqlite+aiosqlite:///test1.db", db_url="sqlite:///test2.db")


def test_pydantic_type_conversion():
    """Test that Pydantic handles type conversion automatically."""
    # Test string to int conversion
    config = MCPOptimizerConfig(mcp_port="9901")
    assert config.mcp_port == 9901
    assert isinstance(config.mcp_port, int)

    # Test string to float conversion
    config = MCPOptimizerConfig(tool_distance_threshold="1.5")
    assert config.tool_distance_threshold == 1.5
    assert isinstance(config.tool_distance_threshold, float)

    # Test invalid conversion
    with pytest.raises(ValidationError):
        MCPOptimizerConfig(mcp_port="invalid")


def test_runtime_mode_default():
    """Test that runtime mode defaults to docker."""
    config = MCPOptimizerConfig()
    assert config.runtime_mode == "docker"


def test_runtime_mode_docker():
    """Test setting runtime mode to docker."""
    config = MCPOptimizerConfig(runtime_mode="docker")
    assert config.runtime_mode == "docker"


def test_runtime_mode_k8s():
    """Test setting runtime mode to k8s."""
    config = MCPOptimizerConfig(runtime_mode="k8s")
    assert config.runtime_mode == "k8s"


def test_runtime_mode_invalid():
    """Test that invalid runtime mode values are rejected."""
    with pytest.raises(ValidationError, match="Input should be 'docker' or 'k8s'"):
        MCPOptimizerConfig(runtime_mode="invalid")

    with pytest.raises(ValidationError, match="Input should be 'docker' or 'k8s'"):
        MCPOptimizerConfig(runtime_mode="compose")

    with pytest.raises(ValidationError, match="Input should be 'docker' or 'k8s'"):
        MCPOptimizerConfig(runtime_mode="kubernetes")


def test_embedding_threads_default():
    """Test that embedding_threads defaults to 2."""
    config = MCPOptimizerConfig()
    assert config.embedding_threads == 2


def test_embedding_threads_none():
    """Test that embedding_threads can be set to None (use all CPU cores)."""
    config = MCPOptimizerConfig(embedding_threads=None)
    assert config.embedding_threads is None


def test_embedding_threads_boundaries():
    """Test embedding_threads validation with boundary values (1, 16)."""
    # Test lower boundary (1)
    config_min = MCPOptimizerConfig(embedding_threads=1)
    assert config_min.embedding_threads == 1

    # Test upper boundary (16)
    config_max = MCPOptimizerConfig(embedding_threads=16)
    assert config_max.embedding_threads == 16

    # Test valid middle value
    config_mid = MCPOptimizerConfig(embedding_threads=8)
    assert config_mid.embedding_threads == 8


def test_embedding_threads_invalid_values():
    """Test that invalid embedding_threads values are rejected."""
    # Test below lower boundary (0)
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        MCPOptimizerConfig(embedding_threads=0)

    # Test negative value
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        MCPOptimizerConfig(embedding_threads=-1)

    # Test above upper boundary (17)
    with pytest.raises(ValidationError, match="less than or equal to 16"):
        MCPOptimizerConfig(embedding_threads=17)

    # Test far above upper boundary
    with pytest.raises(ValidationError, match="less than or equal to 16"):
        MCPOptimizerConfig(embedding_threads=100)


def test_embedding_threads_string_conversion():
    """Test that embedding_threads handles string-to-int conversion."""
    # Test string to int conversion
    config = MCPOptimizerConfig(embedding_threads="4")
    assert config.embedding_threads == 4
    assert isinstance(config.embedding_threads, int)

    # Test invalid string conversion
    with pytest.raises(ValidationError):
        MCPOptimizerConfig(embedding_threads="invalid")
