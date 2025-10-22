"""Tests for runtime mode configuration via environment variables and CLI."""

import os
from unittest.mock import patch

import pytest

from mcp_optimizer.config import MCPOptimizerConfig, load_config


@pytest.fixture(autouse=True)
def clean_environment():
    """Clean RUNTIME_MODE environment variable before and after each test."""
    # Save original value
    original_value = os.environ.get("RUNTIME_MODE")

    # Clean before test
    if "RUNTIME_MODE" in os.environ:
        del os.environ["RUNTIME_MODE"]

    yield

    # Restore original value after test
    if "RUNTIME_MODE" in os.environ:
        del os.environ["RUNTIME_MODE"]
    if original_value is not None:
        os.environ["RUNTIME_MODE"] = original_value


def test_runtime_mode_env_docker(clean_environment, monkeypatch):
    """Test setting runtime mode to docker via environment variable."""
    monkeypatch.setenv("RUNTIME_MODE", "docker")

    # Set required DB URLs for load_config - use /tmp for writable location
    monkeypatch.setenv("ASYNC_DB_URL", "sqlite+aiosqlite:///tmp/test.db")
    monkeypatch.setenv("DB_URL", "sqlite:///tmp/test.db")

    config = load_config()
    assert config.runtime_mode == "docker"


def test_runtime_mode_env_k8s(clean_environment, monkeypatch):
    """Test setting runtime mode to k8s via environment variable."""
    monkeypatch.setenv("RUNTIME_MODE", "k8s")

    # Set required DB URLs for load_config - use /tmp for writable location
    monkeypatch.setenv("ASYNC_DB_URL", "sqlite+aiosqlite:///tmp/test.db")
    monkeypatch.setenv("DB_URL", "sqlite:///tmp/test.db")

    # Mock k8s API server URL detection
    with patch(
        "mcp_optimizer.toolhive.k8s_client.get_k8s_api_server_url",
        return_value="http://127.0.0.1:8001",
    ):
        config = load_config()
    assert config.runtime_mode == "k8s"


def test_runtime_mode_env_case_insensitive(clean_environment, monkeypatch):
    """Test that runtime mode environment variable accepts uppercase values via field validator."""
    # Field validator normalizes to lowercase before Pydantic Literal validation
    monkeypatch.setenv("RUNTIME_MODE", "K8S")

    # Set required DB URLs for load_config - use /tmp for writable location
    monkeypatch.setenv("ASYNC_DB_URL", "sqlite+aiosqlite:///tmp/test.db")
    monkeypatch.setenv("DB_URL", "sqlite:///tmp/test.db")

    # Should succeed - uppercase is normalized to lowercase
    # Mock k8s API server URL detection
    with patch(
        "mcp_optimizer.toolhive.k8s_client.get_k8s_api_server_url",
        return_value="http://127.0.0.1:8001",
    ):
        config = load_config()
    assert config.runtime_mode == "k8s"


def test_runtime_mode_env_mixed_case(clean_environment, monkeypatch):
    """Test that runtime mode environment variable accepts mixed case values."""
    # Test various case combinations
    test_cases = [
        ("Docker", "docker"),
        ("DOCKER", "docker"),
        ("K8s", "k8s"),
        ("k8S", "k8s"),
    ]

    for input_value, expected in test_cases:
        monkeypatch.setenv("RUNTIME_MODE", input_value)
        monkeypatch.setenv("ASYNC_DB_URL", "sqlite+aiosqlite:///tmp/test.db")
        monkeypatch.setenv("DB_URL", "sqlite:///tmp/test.db")

        # Mock k8s API server URL detection for k8s modes
        with patch(
            "mcp_optimizer.toolhive.k8s_client.get_k8s_api_server_url",
            return_value="http://127.0.0.1:8001",
        ):
            config = load_config()
        assert config.runtime_mode == expected, f"Failed for input: {input_value}"


def test_runtime_mode_default_without_env(clean_environment, monkeypatch):
    """Test that runtime mode defaults to docker when environment variable is not set."""
    # Set required DB URLs for load_config - use /tmp for writable location
    monkeypatch.setenv("ASYNC_DB_URL", "sqlite+aiosqlite:///tmp/test.db")
    monkeypatch.setenv("DB_URL", "sqlite:///tmp/test.db")

    config = load_config()
    assert config.runtime_mode == "docker"


def test_direct_config_creation():
    """Test creating config directly with runtime mode."""
    # Test docker mode
    config = MCPOptimizerConfig(
        runtime_mode="docker",
        async_db_url="sqlite+aiosqlite:///test.db",
        db_url="sqlite:///test.db",
    )
    assert config.runtime_mode == "docker"

    # Test k8s mode
    config = MCPOptimizerConfig(
        runtime_mode="k8s", async_db_url="sqlite+aiosqlite:///test.db", db_url="sqlite:///test.db"
    )
    assert config.runtime_mode == "k8s"
