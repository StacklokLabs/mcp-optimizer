# MCP Server Examples

This directory contains example MCPServer manifests for deploying MCP servers with ToolHive.

**📖 For complete installation instructions, start with the [Kubernetes Integration and Installation Guide](../../docs/kubernetes-integration.md).**

## Overview

These examples demonstrate how to deploy MCP servers that work with MCP Optimizer. MCP Optimizer automatically discovers these servers and aggregates their tools into a unified interface.

## Prerequisites

Before deploying these example servers, you must:

1. ✅ Have a Kubernetes cluster with the ToolHive operator installed
2. ✅ Have `kubectl` configured to access your cluster
3. ✅ **Have MCP Optimizer deployed** (see [Deploying MCP Optimizer](../../docs/kubernetes-integration.md#deploying-mcp-optimizer))

**Important:** MCP Optimizer must be running to aggregate and expose the tools from these servers. If you haven't deployed MCP Optimizer yet, follow the [complete installation guide](../../docs/kubernetes-integration.md) first.

## Quick Start

### 0. Create GitHub Secrets (Required for pulling images and GitHub API access)

Before deploying any servers, create the required GitHub secrets. You can use the same GitHub Personal Access Token for both secrets.

**Option 1: Use the convenience script (Recommended)**

```bash
# Set your GitHub token and username
export GITHUB_TOKEN=your_token_here
export GITHUB_USERNAME=your_username  # Optional, will prompt if not set

# Run the script to create both secrets
./examples/mcp-servers/create-github-secrets.sh
```

**Option 2: Create secrets manually**

```bash
# Set your token and username
GITHUB_TOKEN=your_token_here
GITHUB_USERNAME=your_username

# Create the pull secret for ghcr.io
kubectl create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_TOKEN \
  -n toolhive-system

# Create the GitHub API token secret
kubectl create secret generic github-token -n toolhive-system \
  --from-literal=token=$GITHUB_TOKEN
```

**Note:** You need a GitHub Personal Access Token with:
- `read:packages` scope for pulling images from ghcr.io
- GitHub API scopes (repo, read:org, etc.) for MCP server access

The `shared-serviceaccount.yaml` will automatically reference the pull secret, making it available to all MCP servers that use the shared service account.

### 1. Install Fetch Server

```bash
kubectl apply -f examples/mcp-servers/mcpserver_fetch.yaml
kubectl get mcpserver fetch -n toolhive-system
```

### 2. Install GitHub Server

```bash
# Note: If you used the create-github-secrets.sh script in step 0, 
# the github-token secret already exists. You can skip creating it again.

# Deploy GitHub server
kubectl apply -f examples/mcp-servers/mcpserver_github.yaml
kubectl get mcpserver github -n toolhive-system
```

**Alternative:** If you didn't use the script, create the github-token secret manually:
```bash
kubectl create secret generic github-token -n toolhive-system \
  --from-literal=token=YOUR_GITHUB_TOKEN_HERE
```

### 3. Install ToolHive Doc MCP Server

The ToolHive Doc MCP server provides documentation search and retrieval capabilities.

```bash
# Note: This server uses the same github-token secret as the GitHub server
# If you've already created github-token secret in step 2, you can skip creating it again

# Deploy ToolHive Doc MCP server
kubectl apply -f examples/mcp-servers/mcpserver_toolhive-doc-mcp.yaml
kubectl get mcpserver toolhive-doc-mcp -n toolhive-system
```

### 4. Install MCP Optimizer

MCP Optimizer aggregates tools from all MCP servers in the cluster and provides unified tool discovery.

```bash
# Deploy MCP Optimizer (includes ServiceAccount and RBAC)
kubectl apply -f examples/mcp-servers/mcpserver_mcp-optimizer.yaml

# Verify deployment
kubectl get mcpserver mcp-optimizer -n toolhive-system
kubectl get pods -n toolhive-system | grep mcp-optimizer

# Check logs to see tool discovery
kubectl logs -n toolhive-system -l app.kubernetes.io/name=mcp-optimizer --tail=50
```

**Note:** MCP Optimizer requires RBAC permissions to discover MCPServer resources in the cluster. The example includes the necessary ServiceAccount, ClusterRole, and ClusterRoleBinding.

### 5. Verify Deployment

Check that MCP Optimizer discovers the deployed servers:

```bash
# Check that all MCPServers are running
kubectl get mcpserver -n toolhive-system

# Check MCP Optimizer logs for server discovery
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=50 | grep -E "fetch|github|total_tools"
```

Expected output:

```text
Using streamable-http client for workload workload=fetch
Using sse client for workload workload=github
Workload ingestion with cleanup completed failed=0 successful=2 total_tools=50
```

This confirms that MCP Optimizer has successfully discovered and ingested the tools from both servers.

### 4. Test the Connection

To test MCP Optimizer and the example servers from your local machine:

```bash
# Port forward MCP Optimizer service
kubectl port-forward -n toolhive-system svc/mcp-optimizer-proxy 9900:9900
```

Keep this running and test the connection:

```bash
# In another terminal, test the endpoint
curl -s http://localhost:9900/mcp \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}'
```

You should see a successful initialization response from MCP Optimizer.

For client configuration (Cursor, VSCode, Claude Desktop), see [Connecting Clients](../../docs/kubernetes-integration.md#connecting-clients).

## Files

- **`create-github-secrets.sh`** - Convenience script to create both GitHub secrets from GITHUB_TOKEN environment variable
- **`shared-serviceaccount.yaml`** - Shared ServiceAccount with cluster-wide imagePullSecrets for ghcr.io (applied automatically)
- **`mcpserver_fetch.yaml`** - Fetch server for web scraping and URL fetching (uses shared ServiceAccount)
- **`mcpserver_github.yaml`** - GitHub API integration server (uses shared ServiceAccount)
- **`mcpserver_toolhive-doc-mcp.yaml`** - ToolHive documentation search and retrieval server (uses shared ServiceAccount, shares github-token secret)
- **`mcpserver_mcp-optimizer.yaml`** - MCP Optimizer server that aggregates tools from all MCP servers (includes its own ServiceAccount with imagePullSecrets and RBAC)

## Complete Documentation

For the complete installation and configuration guide, see the [Kubernetes Integration and Installation Guide](../../docs/kubernetes-integration.md), which covers:

- **Getting Started**: Complete installation flow from scratch
- **Building Images**: How to build or use MCP Optimizer images
- **Deploying MCP Optimizer**: Helm installation with local or remote images
- **Transport Configuration**: streamable-http, SSE, and stdio transports
- **Client Connections**: Cursor, VSCode, and Claude Desktop setup
- **RBAC Configuration**: Service accounts and permissions
- **Troubleshooting**: Common issues and solutions
- **Advanced Topics**: Custom configurations and best practices
