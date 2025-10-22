# MCP-Optimizer

An intelligent intermediary MCP server that provides semantic tool discovery, caching, and unified access to multiple MCP servers through a single endpoint.

## Features

- **Semantic Tool Discovery**: Intelligently discover and route requests to appropriate MCP tools
- **Unified Access**: Single endpoint to access multiple MCP servers
- **Tool Management**: Manage large numbers of MCP tools seamlessly
- **Group Filtering**: Filter tool discovery by ToolHive groups for multi-environment support

## Requirements for Development

- Python 3.13+
- uv package manager

## Usage

MCP-Optimizer server is meant to be run with ToolHive. It will automatically discover the MCP workloads running in ToolHive and their tools.

### Prerequisites

- [ToolHive UI](https://docs.stacklok.com/toolhive/tutorials/quickstart-ui#step-1-install-the-toolhive-ui) (version >= 0.6.0)
- [ToolHive CLI](https://docs.stacklok.com/toolhive/tutorials/quickstart-cli#step-1-install-toolhive) (version >= 0.3.1)

### Setup

ToolHive UI must be running for the setup.

#### 1. Run MCP-Optimizer server in a dedicated group in ToolHive.

We need to run MCP-Optimizer in a dedicated group to configure AI clients (Cursor, Claude Desktop, etc.) for this group only. This ensures that clients are configured with only the MCP-Optimizer server, while other MCP servers remain installed but not configured in the AI clients.
```bash
# 1. Create the group
thv group create optim
# 2. Run MCP-Optimizer in the dedicated group
thv run --group optim mcp-optimizer
```

#### 2. Configure MCP-Optimizer server with your favorite AI client.

```bash
# thv client register <client_name> --group optim. Example:
thv client register cursor --group optim
# Check the configuration
thv client list-registered
```

#### 3. Add MCP servers to the default group in ToolHive

```bash
# thv run <mcp_server>. Example
thv run time
# Check the config. mcp-optimizer should be in group `optim` and the rest in `default`
thv ls
```

### Run

Now you should be able to use MCP-Optimizer in the chat of the configured client. Examples:
- With the Github MCP server installed, get a GitHub issue details
```markdown
Get the details of GitHub issue 1911 from stacklok/toolhive repo
```

## Configuration

### Runtime Mode

MCP-Optimizer supports two runtime modes for deploying and managing MCP servers:

- **docker** (default): Run MCP servers as Docker containers
- **k8s**: Run MCP servers as Kubernetes workloads

Configuration is case-insensitive (e.g., `K8S`, `Docker`, `k8s` are all valid).

**Quick Start:**
```bash
# Run in Kubernetes mode via environment variable
export RUNTIME_MODE=k8s
mcpo

# Run in Kubernetes mode via command line option
mcpo --runtime-mode k8s

# Run in Docker mode (default)
mcpo --runtime-mode docker
```

For detailed documentation on runtime modes, including code examples, see [docs/runtime-modes.md](docs/runtime-modes.md).

### Group Filtering

MCP-Optimizer supports filtering tool discovery by ToolHive groups. This is useful when running multiple MCP servers across different environments (production, staging, development, etc.) and you want to limit tool discovery to specific groups.

**Quick Start:**
```bash
# Only discover tools from production and staging groups
export ALLOWED_GROUPS="production,staging"
mcp-optimizer
```

For detailed documentation on group filtering, including usage examples, see [docs/group-filtering.md](docs/group-filtering.md).

### Connection Resilience

MCP-Optimizer automatically handles ToolHive connection failures with intelligent retry logic. If ToolHive (`thv serve`) restarts on a different port, MCP-Optimizer will:

1. **Detect the connection failure** across all ToolHive API operations
2. **Automatically rescan** for ToolHive on the new port
3. **Retry with exponential backoff** (1s → 2s → 4s → ... up to 60s)
4. **Gracefully exit** after exhausting all retries (default: 100 attempts over ~100 minutes)

**Configuration:**
```bash
# Customize retry behavior via environment variables
export TOOLHIVE_MAX_RETRIES=150             # Max retry attempts (default: 100)
export TOOLHIVE_INITIAL_BACKOFF=2.0         # Initial delay in seconds (default: 1.0)
export TOOLHIVE_MAX_BACKOFF=120.0           # Maximum delay in seconds (default: 60.0)

# Or via CLI options
mcp-optimizer --toolhive-max-retries 15 --toolhive-initial-backoff 2.0
```

This ensures MCP-Optimizer remains operational even when ToolHive restarts, minimizing service interruptions. For detailed information on configuration options, testing scenarios, and troubleshooting, see [docs/connection-resilience.md](docs/connection-resilience.md).

### Environment Variables

- `RUNTIME_MODE`: Runtime mode for MCP servers (`docker` or `k8s`, default: `docker`)
- `ALLOWED_GROUPS`: Comma-separated list of ToolHive group names to filter tool lookups (default: no filtering)
- `TOOLHIVE_MAX_RETRIES`: Maximum retry attempts on connection failure (default: `100`, range: 1-500)
- `TOOLHIVE_INITIAL_BACKOFF`: Initial retry backoff delay in seconds (default: `1.0`, range: 0.1-10.0)
- `TOOLHIVE_MAX_BACKOFF`: Maximum retry backoff delay in seconds (default: `60.0`, range: 1.0-300.0)
- Additional configuration options can be found in `src/mcp_optimizer/config.py`
