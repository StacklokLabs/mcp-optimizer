# MCP-Optimizer Helm Charts

This directory contains Helm charts for deploying MCP-Optimizer in Kubernetes.

## Available Charts

### mcp-optimizer

The main chart for deploying MCP-Optimizer as an MCPServer resource in a Kubernetes cluster with ToolHive operator.

- **Location**: `./mcp-optimizer/`
- **Documentation**: See [mcp-optimizer/README.md](./mcp-optimizer/README.md)

## Quick Start

### Prerequisites

Before installing any charts, ensure you have:

1. **Kubernetes cluster** (v1.19+)
2. **Helm 3** (v3.2.0+)
3. **ToolHive Operator** installed
4. **ToolHive CRDs** installed

### Installing ToolHive Operator

If you haven't already installed the ToolHive operator:

```bash
# Install ToolHive CRDs
helm upgrade -i toolhive-operator-crds \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator-crds \
  --version latest

# Install ToolHive Operator
helm upgrade -i toolhive-operator \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator \
  -n toolhive-system \
  --create-namespace \
  --version latest
```

### Installing MCP-Optimizer

```bash
# From the mcp-optimizer repository root
cd /path/to/mcp-optimizer

# Install with default values
helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --create-namespace

# Or install with custom values
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-with-persistence.yaml \
  -n toolhive-system
```

### Verifying Installation

```bash
# Check the MCPServer resource
kubectl get mcpserver mcp-optimizer -n toolhive-system

# Check the deployment created by the operator
kubectl get pods -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system

# View logs
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system -f
```

## Example Configurations

The `mcp-optimizer/examples/` directory contains several example values files:

- **values-with-persistence.yaml** - Production configuration with persistent storage
- **values-with-group-filtering.yaml** - Configuration with group-based filtering
- **values-single-namespace.yaml** - Configuration to query only a single namespace
- **values-development.yaml** - Development/testing configuration

## Documentation

For detailed information about the mcp-optimizer chart, including:
- Configuration parameters
- Advanced usage
- Troubleshooting
- Production recommendations

See the [mcp-optimizer Chart README](./mcp-optimizer/README.md).

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/mcp-optimizer/issues
- Documentation: https://github.com/your-org/mcp-optimizer/docs

