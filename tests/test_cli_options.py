"""Tests for CLI option handling."""

import os

from click.testing import CliRunner

from mcp_optimizer.cli import main


def test_cli_options_set_env_vars():
    """Test that CLI options properly set environment variables."""
    runner = CliRunner()

    # Clear any existing env vars
    for env_var in ["RUNTIME_MODE", "MCP_PORT", "LOG_LEVEL"]:
        if env_var in os.environ:
            del os.environ[env_var]

    # Mock the main function to just check env vars are set
    from mcp_optimizer import cli

    original_get_config = cli.get_config

    env_vars_set = {}

    def mock_get_config():
        # Capture environment variables that were set
        env_vars_set["RUNTIME_MODE"] = os.environ.get("RUNTIME_MODE")
        env_vars_set["MCP_PORT"] = os.environ.get("MCP_PORT")
        env_vars_set["LOG_LEVEL"] = os.environ.get("LOG_LEVEL")
        # Raise an exception to stop execution early
        raise SystemExit(0)

    # Monkey patch temporarily
    cli.get_config = mock_get_config

    try:
        runner = CliRunner()
        runner.invoke(
            main,
            ["--runtime-mode", "k8s", "--mcp-port", "9999", "--log-level", "DEBUG"],
            catch_exceptions=True,
        )

        # Verify env vars were set
        assert env_vars_set.get("RUNTIME_MODE") == "k8s"
        assert env_vars_set.get("MCP_PORT") == "9999"
        assert env_vars_set.get("LOG_LEVEL") == "DEBUG"
    finally:
        # Restore original function
        cli.get_config = original_get_config


def test_cli_help():
    """Test that help shows all CLI options."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "--runtime-mode" in result.output
    assert "--mcp-port" in result.output
    assert "--log-level" in result.output
    assert "--toolhive-host" in result.output
    assert "RUNTIME_MODE" in result.output  # Env var name appears in help
    assert "MCP_PORT" in result.output
    assert "LOG_LEVEL" in result.output
