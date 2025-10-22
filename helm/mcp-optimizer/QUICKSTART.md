# MCP-Optimizer Helm Chart - Quick Start

A quick reference guide for deploying MCP-Optimizer with Helm.

## Prerequisites Check

```bash
# Check Kubernetes
kubectl cluster-info
kubectl get nodes

# Check Helm
helm version

# Check ToolHive operator
kubectl get pods -n toolhive-system
kubectl get crd mcpservers.toolhive.stacklok.dev
```

## Install ToolHive (if not already installed)

```bash
# Install CRDs
helm upgrade -i toolhive-operator-crds \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator-crds

# Install Operator
helm upgrade -i toolhive-operator \
  oci://ghcr.io/stacklok/toolhive/toolhive-operator \
  -n toolhive-system \
  --create-namespace
```

## Basic Installation

```bash
# Default installation
helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --create-namespace
```

## Common Installation Patterns

### Production with Persistence

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-with-persistence.yaml \
  -n toolhive-system
```

### Development

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-development.yaml \
  -n default
```

### Custom Image

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.image.repository=myregistry.com/mcp-optimizer \
  --set mcpserver.image.tag=v0.2.0 \
  -n toolhive-system
```

### With Group Filtering

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set groupFiltering.allowedGroups="{prod,staging}" \
  -n toolhive-system
```

## Quick Verification

```bash
# Check MCPServer
kubectl get mcpserver mcp-optimizer -n toolhive-system

# Check Pod
kubectl get pods -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system

# View Logs
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system -f
```

## Common Commands

### Status

```bash
# Helm status
helm status mcp-optimizer -n toolhive-system

# Get all resources
helm get all mcp-optimizer -n toolhive-system
```

### Upgrade

```bash
# Upgrade release
helm upgrade mcp-optimizer ./helm/mcp-optimizer -n toolhive-system

# Upgrade with new values
helm upgrade mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.image.tag=v0.3.0 \
  -n toolhive-system
```

### Uninstall

```bash
# Uninstall release
helm uninstall mcp-optimizer -n toolhive-system

# Delete PVC if persistence was enabled
kubectl delete pvc mcp-optimizer-data -n toolhive-system
```

## Troubleshooting Quick Checks

```bash
# Describe MCPServer
kubectl describe mcpserver mcp-optimizer -n toolhive-system

# Check pod events
kubectl describe pod -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system

# Operator logs
kubectl logs -n toolhive-system -l app.kubernetes.io/name=toolhive-operator

# Check RBAC
kubectl auth can-i list mcpservers.toolhive.stacklok.dev \
  --as=system:serviceaccount:toolhive-system:mcp-optimizer
```

## Port Forwarding

```bash
# Forward MCP port to localhost
kubectl port-forward -n toolhive-system \
  svc/mcp-optimizer 9900:9900
```

## View Configuration

```bash
# Get current values
helm get values mcp-optimizer -n toolhive-system

# See all computed values
helm get values mcp-optimizer -n toolhive-system --all
```

## Template Validation

```bash
# Lint chart
helm lint ./helm/mcp-optimizer

# Dry run
helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --dry-run \
  --debug

# Template output
helm template mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  > rendered.yaml
```

## Useful kubectl Commands

```bash
# Get all MCPServers
kubectl get mcpserver --all-namespaces

# Watch pod status
kubectl get pods -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system -w

# Get pod logs with timestamps
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer \
  -n toolhive-system \
  --timestamps

# Shell into pod
kubectl exec -it -n toolhive-system \
  $(kubectl get pods -n toolhive-system \
    -l toolhive.stacklok.dev/mcpserver=mcp-optimizer \
    -o name | head -1) \
  -- /bin/bash
```

## Values Override Patterns

### Via Command Line

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.image.tag=v0.2.0 \
  --set persistence.enabled=true \
  --set persistence.size=10Gi \
  --set mcpserver.resources.limits.memory=2Gi \
  -n toolhive-system
```

### Via Values File

```yaml
# custom-values.yaml
mcpserver:
  image:
    tag: v0.2.0
  resources:
    limits:
      memory: 2Gi

persistence:
  enabled: true
  size: 10Gi
```

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f custom-values.yaml \
  -n toolhive-system
```

### Multiple Values Files

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f base-values.yaml \
  -f prod-overrides.yaml \
  -n toolhive-system
```

## Environment-Specific Deployments

### Development

```bash
helm install mcp-optimizer-dev ./helm/mcp-optimizer \
  --set mcpserver.name=mcp-optimizer-dev \
  --set mcpserver.env[0].value=DEBUG \
  -n default
```

### Staging

```bash
helm install mcp-optimizer-staging ./helm/mcp-optimizer \
  --set mcpserver.name=mcp-optimizer-staging \
  --set groupFiltering.allowedGroups="{staging}" \
  -n toolhive-staging
```

### Production

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-with-persistence.yaml \
  --set mcpserver.image.tag=v0.1.0 \
  --set groupFiltering.allowedGroups="{production}" \
  -n toolhive-system
```

## Quick Configuration Reference

| What | Command Flag | Example |
|------|--------------|---------|
| Image | `--set mcpserver.image.repository` | `myregistry.com/mcp-optimizer` |
| Tag | `--set mcpserver.image.tag` | `v0.2.0` |
| CPU Limit | `--set mcpserver.resources.limits.cpu` | `2000m` |
| Memory Limit | `--set mcpserver.resources.limits.memory` | `2Gi` |
| Enable Persistence | `--set persistence.enabled` | `true` |
| PVC Size | `--set persistence.size` | `10Gi` |
| Storage Class | `--set persistence.storageClass` | `fast-ssd` |
| Log Level | `--set mcpserver.env[3].value` | `DEBUG` |
| Allowed Groups | `--set groupFiltering.allowedGroups` | `"{prod,staging}"` |

## Getting Help

```bash
# Chart README
cat ./helm/mcp-optimizer/README.md

# Default values
cat ./helm/mcp-optimizer/values.yaml

# Example values
ls -l ./helm/mcp-optimizer/examples/

# Helm help
helm help
helm install --help
```

## Next Steps

After successful installation:
1. ✅ Verify the MCPServer is running
2. ✅ Check logs for any errors
3. ✅ Test MCP server discovery (create a test MCPServer)
4. ✅ Configure monitoring and alerts
5. ✅ Set up backup for the database
6. ✅ Review security settings

For detailed documentation, see:
- [README.md](./README.md) - Full chart documentation
- [helm-deployment.md](../../docs/helm-deployment.md) - Deployment guide
- [kubernetes-integration.md](../../docs/kubernetes-integration.md) - K8s integration details

