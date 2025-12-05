# Kubernetes Integration and Installation Guide

This guide covers deploying MCP Optimizer and MCP servers in Kubernetes using the ToolHive operator, including configuration, installation, and client integration.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [MCP Optimizer Kubernetes Mode](#mcp-optimizer-kubernetes-mode)
- [Installing ToolHive Operator](#installing-toolhive-operator)
- [Building MCP Optimizer Image](#building-mcp-optimizer-image)
- [Installing MCP Servers](#installing-mcp-servers)
- [Deploying MCP Optimizer](#deploying-mcp-optimizer)
- [Connecting Clients](#connecting-clients)
- [RBAC Configuration](#rbac-configuration)
- [MCPServer CRD Mapping](#mcpserver-crd-mapping)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

MCP Optimizer supports running in Kubernetes mode, where it queries MCPServer Custom Resource Definitions (CRDs) instead of the Docker-based workloads API. This enables:

- Native Kubernetes integration with MCPServer CRDs
- Automatic service discovery and tool aggregation
- RBAC-based access control
- Both in-cluster and out-of-cluster configurations
- Seamless integration with the ToolHive operator

## Quick Start

**TL;DR** - Complete installation in 5 steps:

```bash
# 1. Install ToolHive operator
helm upgrade -i toolhive-operator-crds oci://ghcr.io/stacklok/toolhive/toolhive-operator-crds
helm upgrade -i toolhive-operator oci://ghcr.io/stacklok/toolhive/toolhive-operator \
  -n toolhive-system --create-namespace

# 2. Deploy MCP Optimizer
# Option A: From OCI registry (recommended)
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer -n toolhive-system

# Option B: From local chart source
helm install mcp-optimizer ./helm/mcp-optimizer -n toolhive-system

# 3. (Optional) For local image development, build and override image settings
docker build -t mcp-optimizer:local .
kind load docker-image mcp-optimizer:local  # If using kind
helm install mcp-optimizer ./helm/mcp-optimizer -n toolhive-system \
  --set mcpserver.image.repository=mcp-optimizer \
  --set mcpserver.image.tag=local \
  --set mcpserver.podTemplateSpec.spec.containers[0].imagePullPolicy=Never

# 4. Deploy example MCP servers
kubectl apply -f examples/mcp-servers/mcpserver_fetch.yaml
kubectl create secret generic github-token -n toolhive-system --from-literal=token=YOUR_TOKEN
kubectl apply -f examples/mcp-servers/mcpserver_github.yaml

# 5. Verify deployment
kubectl get mcpserver -n toolhive-system
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=20 | grep total_tools
```

**📋 See [MCP Server Examples](../examples/mcp-servers/README.md) for a simplified quick reference guide.**

Read the sections below for detailed explanations, configuration options, and troubleshooting.

## Prerequisites

- Kubernetes cluster (v1.19+)
- `kubectl` configured to access your cluster
- Helm 3.x (for operator and MCP Optimizer installation)
- ToolHive operator (installation steps below)
- Docker (for building local MCP Optimizer image)

## MCP Optimizer Kubernetes Mode

### Configuration

MCP Optimizer uses environment variables for Kubernetes mode configuration:

```bash
# Required: Set runtime mode to k8s
export RUNTIME_MODE=k8s

# Optional: Kubernetes API server URL (auto-detected if not set)
# - Auto-detects from KUBERNETES_SERVICE_HOST + KUBERNETES_SERVICE_PORT_HTTPS (in-cluster)
# - Falls back to http://127.0.0.1:8001 (kubectl proxy)
# export K8S_API_SERVER_URL=http://127.0.0.1:8001

# Optional: Kubernetes namespace to query (defaults to all namespaces)
# export K8S_NAMESPACE=toolhive-system

# Optional: Whether to query all namespaces (default: true)
# export K8S_ALL_NAMESPACES=true
```

**Note**: In most cases, you only need to set `RUNTIME_MODE=k8s`. The API server URL is automatically detected from your environment.

### Running Modes

#### Mode 1: Using kubectl proxy (Development/Testing)

Easiest way to test Kubernetes integration locally:

1. Start kubectl proxy:

   ```bash
   kubectl proxy
   ```

   This starts a proxy server at `http://127.0.0.1:8001`

2. Run mcp-optimizer in K8s mode:

   ```bash
   export RUNTIME_MODE=k8s
   mtm
   ```

   The API server URL is automatically detected!

#### Mode 2: Running In-Cluster (Production)

When MCP Optimizer is deployed as a pod in Kubernetes, everything is automatically configured:

- **Runtime mode**: Set to `k8s` via environment variable
- **API server URL**: Automatically constructed from `KUBERNETES_SERVICE_HOST` and `KUBERNETES_SERVICE_PORT_HTTPS`
- **Authentication**: Service account token from `/var/run/secrets/kubernetes.io/serviceaccount/token`
- **CA certificate**: Read from `/var/run/secrets/kubernetes.io/serviceaccount/ca.crt`
- **Namespace**: Read from `/var/run/secrets/kubernetes.io/serviceaccount/namespace`

**You don't need to set additional environment variables!** Kubernetes provides everything automatically.

#### Mode 3: Remote Access with kubeconfig

For running MCP Optimizer outside the cluster with cluster access:

```bash
export RUNTIME_MODE=k8s
export K8S_API_SERVER_URL=https://your-k8s-api-server:6443

# Requires proper authentication configured in your kubeconfig
mtm
```

## Installing ToolHive Operator

Before deploying MCP servers, install the ToolHive operator:

```bash
# Install CRDs
helm upgrade -i toolhive-operator-crds \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator-crds

# Install operator
helm upgrade -i toolhive-operator \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator \
  -n toolhive-system --create-namespace
```

Verify installation:
```bash
kubectl get crd mcpservers.toolhive.stacklok.dev
kubectl get pods -n toolhive-system
```

## Building MCP Optimizer Image (Optional for Local Development)

By default, the Helm chart uses published container images from `ghcr.io/stackloklabs/mcp-optimizer`. For most users, you can skip this section and proceed directly to deployment.

### For Local Development

If you need to test local changes to MCP Optimizer, you can build and use a local image:

```bash
# From the mcp-optimizer repository root
docker build -t mcp-optimizer:local .
```

If you're using a local Kubernetes cluster (kind, minikube, etc.), load the image into the cluster:

**For kind:**
```bash
kind load docker-image mcp-optimizer:local
```

**For minikube:**
```bash
minikube image load mcp-optimizer:local
```

Then override the image settings during Helm installation:

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --set mcpserver.image.repository=mcp-optimizer \
  --set mcpserver.image.tag=local \
  --set mcpserver.podTemplateSpec.spec.containers[0].imagePullPolicy=Never
```

## Installing MCP Servers

### 1. Create GitHub Secret

Before installing the GitHub MCP server, create a secret with your GitHub Personal Access Token:

```bash
kubectl create secret generic github-token -n toolhive-system \
  --from-literal=token=YOUR_GITHUB_TOKEN_HERE
```

Replace `YOUR_GITHUB_TOKEN_HERE` with your actual GitHub Personal Access Token. The token should have appropriate permissions (e.g., `repo`, `user:email`).

### 2. Install Fetch MCP Server

The Fetch server provides web scraping and URL fetching capabilities.

```bash
kubectl apply -f - <<'EOF'
apiVersion: toolhive.stacklok.dev/v1alpha1
kind: MCPServer
metadata:
  name: fetch
  namespace: toolhive-system
spec:
  image: ghcr.io/stackloklabs/gofetch/server
  transport: streamable-http
  proxyMode: streamable-http
  port: 8080
  targetPort: 8080
  permissionProfile:
    name: network
    type: builtin
  resources:
    limits:
      cpu: "100m"
      memory: "128Mi"
    requests:
      cpu: "50m"
      memory: "64Mi"
EOF
```

**Verify:**
```bash
kubectl get mcpserver fetch -n toolhive-system
kubectl get pods -n toolhive-system | grep fetch
```

### 3. Install GitHub MCP Server

The GitHub server provides integration with GitHub APIs for repository management, issues, pull requests, and more.

```bash
kubectl apply -f - <<'EOF'
apiVersion: toolhive.stacklok.dev/v1alpha1
kind: MCPServer
metadata:
  name: github
  namespace: toolhive-system
spec:
  image: ghcr.io/github/github-mcp-server:v0.18.0
  transport: stdio
  proxyMode: streamable-http
  port: 8080
  targetPort: 8080
  permissionProfile:
    name: network
    type: builtin
  secrets:
    - name: github-token
      key: token
      targetEnvName: GITHUB_PERSONAL_ACCESS_TOKEN
  resources:
    limits:
      cpu: 200m
      memory: 256Mi
    requests:
      cpu: 100m
      memory: 128Mi
EOF
```

**Verify:**
```bash
kubectl get mcpserver github -n toolhive-system
kubectl get pods -n toolhive-system | grep github
```

**Note on Readiness Probes:**
When using `proxyMode: streamable-http` with `transport: stdio`, the default readiness probe may fail because the streamable-http proxy doesn't expose a `/health` endpoint. If you experience pod restarts, you may need to remove or adjust the readiness probe on the deployment.

### Expected Results

After successful installation:

- **Fetch Server**:
  - Status: Running
  - Tools: 1 tool (fetch)
  - Transport: streamable-http

- **GitHub Server**:
  - Status: Running
  - Tools: 50+ tools (various GitHub API operations)
  - Transport: stdio
  - Requires: GITHUB_PERSONAL_ACCESS_TOKEN secret

## Deploying MCP Optimizer

MCP Optimizer aggregates all MCP servers in the cluster and provides unified tool discovery.

### Using OCI Registry (Recommended)

Install MCP Optimizer from the published Helm chart in the OCI registry:

```bash
# Deploy MCP Optimizer from OCI registry
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer -n toolhive-system
```

This is the recommended approach as it:
- Uses published, versioned releases
- Automatically pulls the correct container image
- Simplifies deployment and upgrades

To install a specific version:

```bash
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  --version 0.1.0 \
  -n toolhive-system
```

### Using Local Chart Source

You can also install from the source repository:

```bash
# Deploy MCP Optimizer from local chart directory
helm install mcp-optimizer ./helm/mcp-optimizer -n toolhive-system
```

This uses the default values:
- Image: `ghcr.io/stackloklabs/mcp-optimizer` (published image)
- Tag: Uses chart `appVersion`
- Pull Policy: `IfNotPresent`

### Custom Image Configuration

To override image settings (e.g., for local development or custom registries):

```bash
# For local development with a custom-built image
helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --set mcpserver.image.repository=mcp-optimizer \
  --set mcpserver.image.tag=local \
  --set mcpserver.podTemplateSpec.spec.containers[0].imagePullPolicy=Never

# For a custom registry
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  -n toolhive-system \
  --set mcpserver.image.repository=ghcr.io/myorg/mcp-optimizer \
  --set mcpserver.image.tag=v1.0.0
```

### Verify MCP Optimizer Installation

```bash
# Check MCP Optimizer MCPServer resource
kubectl get mcpserver mcp-optimizer -n toolhive-system

# Check MCP Optimizer pod
kubectl get pods -n toolhive-system -l app.kubernetes.io/name=mcp-optimizer

# Check MCP Optimizer logs to see discovered tools
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=50 | grep -E "fetch|github|successful|total_tools"
```

Expected log output:

```text
Using streamable-http client for workload workload=fetch
Using streamable-http client for workload workload=github
Workload ingestion with cleanup completed failed=0 successful=2 total_tools=51
```

### Verify Proxy Mode

MCP Optimizer must use `streamable-http` proxy mode for proper Cursor integration:

```bash
kubectl get mcpserver mcp-optimizer -n toolhive-system -o jsonpath='{.spec.proxyMode}' && echo
```

If it's not set to `streamable-http`, update it:

```bash
kubectl patch mcpserver mcp-optimizer -n toolhive-system --type=merge -p '{"spec":{"proxyMode":"streamable-http"}}'
```

## Connecting Clients

### Connecting Cursor

To use MCP Optimizer from Cursor, expose the service locally and configure Cursor's MCP settings.

#### Step 1: Port Forward MCP Optimizer Service

In a terminal, run and keep this command running:

```bash
kubectl port-forward -n toolhive-system svc/mcp-mcp-optimizer-proxy 9900:9900
```

This forwards the Kubernetes service to `localhost:9900`.

#### Step 2: Configure Cursor MCP Settings

Edit your Cursor MCP configuration file at `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "mcp-optimizer": {
      "type": "http",
      "url": "http://localhost:9900/mcp"
    }
  }
}
```

**Important Notes:**
- Use `"type": "http"` for streamable-http transport
- The endpoint path is `/mcp` (not `/sse`)
- Ensure the port-forward is running before starting Cursor

#### Step 3: Restart Cursor

1. Quit Cursor completely
2. Restart Cursor
3. Open the MCP panel to verify `mcp-optimizer` is connected

#### Step 4: Test the Connection

Try asking Cursor:
- "What MCP tools are available?"
- "Fetch the content from https://example.com"
- "Get GitHub issue #123 from stacklok/toolhive"

MCP Optimizer will automatically discover and route requests to the appropriate MCP servers.

### Alternative: Direct Connection to GitHub

If you want to connect directly to the GitHub MCP server without MCP Optimizer:

#### Port Forward GitHub Service

```bash
kubectl port-forward -n toolhive-system svc/mcp-github-proxy 9901:8080
```

#### Configure Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "github": {
      "type": "http",
      "url": "http://localhost:9901/mcp"
    }
  }
}
```

### Other Clients

#### VSCode/VSCode Insiders

Configuration file: `~/.config/Code/User/mcp.json` (Linux) or `~/Library/Application Support/Code/User/mcp.json` (macOS)

```json
{
  "servers": {
    "mcp-optimizer": {
      "type": "http",
      "url": "http://localhost:9900/mcp"
    }
  }
}
```

#### Claude Desktop

Configuration file: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)

```json
{
  "mcpServers": {
    "mcp-optimizer": {
      "url": "http://localhost:9900/sse"
    }
  }
}
```

**Note**: Claude Desktop may use SSE transport, so verify the appropriate endpoint for your client.

## RBAC Configuration

When running in-cluster, MCP Optimizer requires appropriate RBAC permissions to read MCPServer resources:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: mcp-optimizer
  namespace: toolhive-system
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: mcp-optimizer-reader
rules:
- apiGroups: ["toolhive.stacklok.dev"]
  resources: ["mcpservers"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: mcp-optimizer-reader-binding
subjects:
- kind: ServiceAccount
  name: mcp-optimizer
  namespace: toolhive-system
roleRef:
  kind: ClusterRole
  name: mcp-optimizer-reader
  apiGroup: rbac.authorization.k8s.io
```

The MCP Optimizer Helm chart automatically creates these resources when `rbac.create: true` (default).

## MCPServer CRD Mapping

MCP Optimizer converts MCPServer CRDs to internal Workload models with the following field mappings:

| MCPServer Field | Workload Field | Notes |
|----------------|---------------|-------|
| `metadata.name` | `name` | Server name |
| `metadata.namespace` | N/A | Used for scoping |
| `metadata.creationTimestamp` | `created_at` | Creation timestamp |
| `metadata.labels` | `labels` | All labels |
| `spec.groupRef` (preferred) or `metadata.labels["toolhive.stacklok.dev/group"]` | `group` | Server group (prefers `spec.groupRef`, falls back to label, defaults to "default") |
| `spec.image` | `package` | Container image |
| `spec.transport` | `transport_type` | Transport type (stdio, sse, streamable-http) |
| `spec.port` | `port` | Server port |
| `spec.proxyMode` | `proxy_mode` | Proxy mode (sse, streamable-http) |
| `spec.remote` | `remote` | Whether this is a remote server |
| `status.phase` | `status` | Mapped: Running → running, other → stopped |
| `status.url` | `url` | Server URL |
| `status.tools` | `tools` | List of available tool names |
| `status.message` | `status_context` | Status context message |

## Troubleshooting

### Check Server Logs

```bash
# Fetch server
kubectl logs -n toolhive-system fetch-0

# GitHub server
kubectl logs -n toolhive-system github-0

# MCP Optimizer
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=100
```

### Common Issues

#### Connection Refused

If you see connection refused errors:
- Ensure kubectl proxy is running: `kubectl proxy`
- Verify the proxy URL matches `K8S_API_SERVER_URL`
- Check firewall settings

#### No MCPServers Found

If no MCPServers are listed:
- Verify MCPServer CRDs are installed: `kubectl get crd mcpservers.toolhive.stacklok.dev`
- Check if any MCPServers exist: `kubectl get mcpserver --all-namespaces`
- Deploy a sample server from the examples directory

#### GitHub Server Crashes

- Ensure the `github-token` secret exists and contains a valid token
- Verify the token has appropriate permissions

#### Pods in CrashLoopBackOff

- Check logs for missing environment variables or permissions
- Verify resource limits are sufficient

#### RBAC Permission Errors

If you see permission denied errors:
- Verify the service account has the correct RBAC permissions
- Check ClusterRole and ClusterRoleBinding are created
- Test permissions: `kubectl auth can-i list mcpservers.toolhive.stacklok.dev --as=system:serviceaccount:toolhive-system:mcp-optimizer`

#### Tool Discovery Issues

If tools are not being discovered:
- Check the MCPServer status.tools field: `kubectl get mcpserver <name> -o jsonpath='{.status.tools}'`
- Ensure the MCP server pod is running: `kubectl get pods -l toolhive=true`
- Check MCP server logs for errors
- Verify MCP Optimizer is polling correctly: Check MCP Optimizer logs for ingestion messages

### Verification Commands

```bash
# Check all MCP servers
kubectl get mcpserver -n toolhive-system

# Check all MCP server pods
kubectl get pods -n toolhive-system | grep -E "fetch|github|mcp-optimizer"

# Check MCP Optimizer ingestion status
kubectl logs -n toolhive-system mcp-optimizer-0 --tail=100 | grep "Workload ingestion with cleanup completed"
```

## Advanced Topics

### Testing Kubernetes Integration

Use the provided test script to verify the Kubernetes integration:

```bash
# Start kubectl proxy in one terminal
kubectl proxy

# In another terminal, run the test script
cd /path/to/mcp-optimizer
python test_k8s_integration.py
```

The test script will:
1. Connect to the Kubernetes API
2. List all MCPServer resources
3. Display details about each server
4. Filter for running MCP workloads
5. Test namespace-specific queries

### Differences from Docker Mode

| Aspect | Docker Mode | Kubernetes Mode |
|--------|-------------|----------------|
| Workload Source | ToolHive workloads API | Kubernetes MCPServer CRDs |
| Discovery | Docker containers | Kubernetes custom resources |
| Registry | Always fetched from ToolHive | Optional, falls back to mean pooling |
| Namespace Support | N/A | Full namespace support |
| RBAC | N/A | Required for in-cluster deployment |
| Authentication | No auth needed | Service account token (in-cluster) or kubeconfig (remote) |
| SSL/TLS | N/A | Automatic CA cert validation (in-cluster) |

### Transport and ProxyMode Configuration

It's critical that `transport` and `proxyMode` are configured correctly:

- **`transport: streamable-http`** → **`proxyMode: streamable-http`**
- **`transport: stdio`** → **`proxyMode: streamable-http`** (recommended) or **`proxyMode: sse`** (deprecated)
- **`transport: sse`** → **`proxyMode: sse`** (deprecated)

**Note**: For stdio transport servers, prefer `proxyMode: streamable-http` over SSE as SSE is deprecated. However, be aware that the default readiness probe may fail with streamable-http + stdio combinations and may need to be removed or adjusted.

Mismatches between these fields will cause connection failures.

### Group Filtering

MCP Optimizer supports filtering MCPServers by group. This is useful for isolating servers by environment or team:

```yaml
# In MCP Optimizer deployment, set environment variable:
- name: ALLOWED_GROUPS
  value: "development,production"
```

Servers can specify their group using either `spec.groupRef` (preferred) or the `toolhive.stacklok.dev/group` label. The `spec.groupRef` field takes precedence if both are present:

```yaml
apiVersion: toolhive.stacklok.dev/v1alpha1
kind: MCPServer
metadata:
  name: my-server
spec:
  # Preferred method: use spec.groupRef
  groupRef: development
  # ... other spec fields
```

Or using labels (legacy method, still supported):

```yaml
apiVersion: toolhive.stacklok.dev/v1alpha1
kind: MCPServer
metadata:
  name: my-server
  labels:
    toolhive.stacklok.dev/group: development
spec:
  # ... spec fields
```

If neither `spec.groupRef` nor the label is specified, the server will be assigned to the "default" group.

See [Group Filtering documentation](group-filtering.md) for more details.

### In-Cluster Authentication

For detailed information about in-cluster authentication mechanisms, service account tokens, and CA certificates, see [In-Cluster Authentication documentation](in-cluster-authentication.md).

## Next Steps

- Review the [ToolHive documentation](https://github.com/stacklok/toolhive)
- Learn about [MCPServer CRD configuration options](https://github.com/stacklok/toolhive/tree/main/deploy/operator)
- Explore additional MCP servers in the [examples directory](../examples/mcp-servers/)
- Deploy MCP Optimizer from [OCI registry (releases)](https://github.com/StacklokLabs/mcp-optimizer/releases) or [local Helm chart](../helm/mcp-optimizer/)

## See Also

- [Runtime Modes documentation](runtime-modes.md)
- [Helm Deployment Guide](helm-deployment.md)
- [Example MCP Servers](../examples/mcp-servers/README.md)
