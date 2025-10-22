# Runtime Modes

MCP-Optimizer supports two runtime modes for deploying and managing MCP servers:

## Modes

### Docker Mode (default)
In Docker mode, MCP servers are deployed and managed as Docker containers.

### K8s Mode
In Kubernetes mode, MCP servers are deployed and managed as Kubernetes workloads (pods, deployments, etc.).

## Configuration

The runtime mode can be configured in three ways:

### 1. Command Line Option
```bash
# Set to k8s mode
mtm --runtime-mode k8s

# Set to docker mode (default)
mtm --runtime-mode docker
```

### 2. Environment Variable
```bash
# Set to k8s mode
export RUNTIME_MODE=k8s
mtm

# Set to docker mode
export RUNTIME_MODE=docker
mtm
```

### 3. Default Behavior
If neither the command line option nor environment variable is set, the system defaults to `docker` mode.

## Priority
When both command line and environment variable are set, the command line option takes precedence.

## Accessing Runtime Mode in Code

The runtime mode is available throughout the application via the configuration object:

```python
from mcp_optimizer.config import get_config

config = get_config()
runtime_mode = config.runtime_mode

if runtime_mode == "k8s":
    # K8s-specific logic
    pass
elif runtime_mode == "docker":
    # Docker-specific logic
    pass
```

## Examples

### Running with K8s mode
```bash
# Using CLI option
mtm --runtime-mode k8s

# Using environment variable
RUNTIME_MODE=k8s mtm

# In Docker with environment variable
docker run -e RUNTIME_MODE=k8s mcp-optimizer
```

### Running with Docker mode (default)
```bash
# Default behavior
mtm

# Explicit CLI option
mtm --runtime-mode docker

# Using environment variable
RUNTIME_MODE=docker mtm
```

## Validation

The runtime mode is validated at startup and must be either `docker` or `k8s`. The validation is **case-insensitive**, so values like `DOCKER`, `Docker`, `K8S`, or `k8s` are all valid and will be normalized to lowercase. Any value other than a case variation of `docker` or `k8s` will cause a configuration error and the application will fail to start.

