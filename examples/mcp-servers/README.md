# MCP Server Examples

This directory contains example MCPServer manifests for deploying MCP servers with ToolHive.

**📖 For complete installation instructions, start with the [Kubernetes Integration and Installation Guide](../../docs/kubernetes-integration.md).**

## Overview

These examples demonstrate how to deploy MCP servers that work with MCP-Optimizer. MCP-Optimizer automatically discovers these servers and aggregates their tools into a unified interface.

## Prerequisites

Before deploying these example servers, you must:

1. ✅ Have a Kubernetes cluster with the ToolHive operator installed
2. ✅ Have `kubectl` configured to access your cluster
3. ✅ **Have MCP-Optimizer deployed** (see [Deploying MCP-Optimizer](../../docs/kubernetes-integration.md#deploying-mcp-optimizer))

**Important:** MCP-Optimizer must be running to aggregate and expose the tools from these servers. If you haven't deployed MCP-Optimizer yet, follow the [complete installation guide](../../docs/kubernetes-integration.md) first.

## Quick Start

### 1. Install Fetch Server

```bash
kubectl apply -f examples/mcp-servers/mcpserver_fetch.yaml
kubectl get mcpserver fetch -n toolhive-system
```

### 2. Install GitHub Server

```bash
# Create GitHub token secret first
kubectl create secret generic github-token -n toolhive-system \
  --from-literal=token=YOUR_GITHUB_TOKEN_HERE

# Deploy GitHub server
kubectl apply -f examples/mcp-servers/mcpserver_github.yaml
kubectl get mcpserver github -n toolhive-system
```

### 3. Verify Deployment

Check that MCP-Optimizer discovers the deployed servers:

```bash
# Check that all MCPServers are running
kubectl get mcpserver -n toolhive-system

# Check MCP-Optimizer logs for server discovery
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=50 | grep -E "fetch|github|total_tools"
```

Expected output:

```text
Using streamable-http client for workload workload=fetch
Using sse client for workload workload=github
Workload ingestion with cleanup completed failed=0 successful=2 total_tools=50
```

This confirms that MCP-Optimizer has successfully discovered and ingested the tools from both servers.

### 4. Test the Connection

To test MCP-Optimizer and the example servers from your local machine:

```bash
# Port forward MCP-Optimizer service
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

You should see a successful initialization response from MCP-Optimizer.

For client configuration (Cursor, VSCode, Claude Desktop), see [Connecting Clients](../../docs/kubernetes-integration.md#connecting-clients).

## Files

- **`mcpserver_fetch.yaml`** - Fetch server for web scraping and URL fetching
- **`mcpserver_github.yaml`** - GitHub API integration server

## Complete Documentation

For the complete installation and configuration guide, see the [Kubernetes Integration and Installation Guide](../../docs/kubernetes-integration.md), which covers:

- **Getting Started**: Complete installation flow from scratch
- **Building Images**: How to build or use MCP-Optimizer images
- **Deploying MCP-Optimizer**: Helm installation with local or remote images
- **Transport Configuration**: streamable-http, SSE, and stdio transports
- **Client Connections**: Cursor, VSCode, and Claude Desktop setup
- **RBAC Configuration**: Service accounts and permissions
- **Troubleshooting**: Common issues and solutions
- **Advanced Topics**: Custom configurations and best practices
