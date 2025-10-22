# Deploying MCP-Optimizer with Helm

This guide walks you through deploying MCP-Optimizer in Kubernetes using the Helm chart.

## Overview

The MCP-Optimizer Helm chart deploys MCP-Optimizer as an MCPServer Custom Resource Definition (CRD) in your Kubernetes cluster. The ToolHive operator then manages the actual deployment, creating a pod that runs the MCP-Optimizer container.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                    │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │          toolhive-system namespace                 │ │
│  │                                                    │ │
│  │  ┌──────────────┐         ┌──────────────┐       │ │
│  │  │   MCP-Optimizer   │         │  ToolHive    │       │ │
│  │  │  MCPServer   │────────>│  Operator    │       │ │
│  │  │   (CRD)      │         │              │       │ │
│  │  └──────────────┘         └──────┬───────┘       │ │
│  │                                   │               │ │
│  │                                   │ creates       │ │
│  │                                   v               │ │
│  │                          ┌──────────────┐        │ │
│  │                          │   MCP-Optimizer   │        │ │
│  │                          │     Pod      │        │ │
│  │                          │              │        │ │
│  │                          │  queries K8s │        │ │
│  │                          │  API for     │        │ │
│  │                          │  MCPServers  │        │ │
│  │                          └──────────────┘        │ │
│  │                                                   │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Prerequisites

### 1. Kubernetes Cluster

You need a running Kubernetes cluster (v1.19 or later). You can use:
- Local: kind, minikube, k3s, Docker Desktop
- Cloud: GKE, EKS, AKS, etc.

Verify your cluster is accessible:
```bash
kubectl cluster-info
kubectl get nodes
```

### 2. Helm 3

Install Helm 3 (v3.2.0 or later):
```bash
# macOS
brew install helm

# Linux (using script)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Windows (using Chocolatey)
choco install kubernetes-helm
```

Verify installation:
```bash
helm version
```

### 3. ToolHive Operator

The ToolHive operator must be installed in your cluster:

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

Verify the operator is running:
```bash
kubectl get pods -n toolhive-system
kubectl get crd mcpservers.toolhive.stacklok.dev
```

## Installation

### Basic Installation

Install MCP-Optimizer with default values:

```bash
cd /path/to/mcp-optimizer

helm install mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system \
  --create-namespace
```

### Installation with Custom Values

Create a custom values file or use one of the examples:

```bash
# Using an example values file
helm install mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-with-persistence.yaml \
  -n toolhive-system
```

### Installation with Command-line Overrides

You can override specific values on the command line:

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.image.tag=v0.2.0 \
  --set persistence.enabled=true \
  --set persistence.size=5Gi \
  -n toolhive-system
```

## Configuration

### Key Configuration Options

#### Image Configuration

```yaml
mcpserver:
  image:
    repository: ghcr.io/your-org/mcp-optimizer
    tag: "0.1.0"
    pullPolicy: IfNotPresent
```

#### Resource Limits

```yaml
mcpserver:
  resources:
    limits:
      cpu: "1000m"
      memory: "1Gi"
    requests:
      cpu: "250m"
      memory: "256Mi"
```

#### Persistence

```yaml
persistence:
  enabled: true
  size: 5Gi
  storageClass: "standard"
```

#### Group Filtering

Limit which MCP server groups mcp-optimizer can discover:

```yaml
groupFiltering:
  allowedGroups:
    - "development"
    - "production"
```

#### Namespace Scope

Query all namespaces (default):
```yaml
mcpserver:
  env:
    - name: K8S_ALL_NAMESPACES
      value: "true"
```

Or limit to a single namespace:
```yaml
mcpserver:
  env:
    - name: K8S_ALL_NAMESPACES
      value: "false"
    - name: K8S_NAMESPACE
      value: "my-namespace"
```

See the [values.yaml](../helm/mcp-optimizer/values.yaml) file for all available options.

## Verification

### Check MCPServer Resource

```bash
kubectl get mcpserver mcp-optimizer -n toolhive-system

# Get detailed information
kubectl describe mcpserver mcp-optimizer -n toolhive-system
```

### Check Pod Status

The ToolHive operator creates a pod for the MCPServer:

```bash
kubectl get pods -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

### View Logs

```bash
# Follow logs
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system -f

# View recent logs
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system --tail=100
```

### Test MCP-Optimizer Functionality

1. **Check that mcp-optimizer discovers MCPServers:**

```bash
# Create a test MCPServer
kubectl apply -f - <<EOF
apiVersion: toolhive.stacklok.dev/v1alpha1
kind: MCPServer
metadata:
  name: test-fetch
  namespace: default
spec:
  image: docker.io/mcp/fetch
  transport: stdio
  port: 8080
EOF

# Wait a minute for polling, then check mcp-optimizer logs
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system | grep test-fetch
```

2. **Access mcp-optimizer via port-forward:**

```bash
# Forward the MCP port
kubectl port-forward -n toolhive-system \
  svc/mcp-optimizer 9900:9900

# In another terminal, connect with an MCP client
# (This depends on your MCP client implementation)
```

## Upgrading

### Upgrade the Release

```bash
# Upgrade to a new version
helm upgrade mcp-optimizer ./helm/mcp-optimizer \
  -n toolhive-system

# Upgrade with new values
helm upgrade mcp-optimizer ./helm/mcp-optimizer \
  -f ./helm/mcp-optimizer/examples/values-with-persistence.yaml \
  -n toolhive-system
```

### View Release History

```bash
helm history mcp-optimizer -n toolhive-system
```

### Rollback a Release

```bash
# Rollback to previous version
helm rollback mcp-optimizer -n toolhive-system

# Rollback to specific revision
helm rollback mcp-optimizer 2 -n toolhive-system
```

## Uninstallation

To remove MCP-Optimizer:

```bash
helm uninstall mcp-optimizer -n toolhive-system
```

**Note**: If persistence is enabled, the PersistentVolumeClaim will not be automatically deleted. To delete it:

```bash
kubectl delete pvc mcp-optimizer-data -n toolhive-system
```

## Troubleshooting

### MCPServer Not Creating Pod

Check the MCPServer status:
```bash
kubectl describe mcpserver mcp-optimizer -n toolhive-system
```

Check operator logs:
```bash
kubectl logs -n toolhive-system -l app.kubernetes.io/name=toolhive-operator
```

### RBAC Permission Errors

Verify the service account has the correct permissions:
```bash
kubectl auth can-i list mcpservers.toolhive.stacklok.dev \
  --as=system:serviceaccount:toolhive-system:mcp-optimizer
```

### Pod CrashLoopBackOff

View pod logs:
```bash
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

Common issues:
- Database connection errors (check volume mounts)
- Missing environment variables
- Invalid configuration

### Image Pull Errors

If using a private registry:
```bash
# Create image pull secret
kubectl create secret docker-registry regcred \
  --docker-server=<your-registry> \
  --docker-username=<username> \
  --docker-password=<password> \
  -n toolhive-system

# Update values.yaml
mcpserver:
  imagePullSecrets:
    - name: regcred
```

## Production Recommendations

### 1. Enable Persistence

Always enable persistence for production:
```yaml
persistence:
  enabled: true
  size: 10Gi
  storageClass: "fast-ssd"
```

### 2. Set Resource Limits

Configure appropriate resource limits:
```yaml
mcpserver:
  resources:
    limits:
      cpu: "2000m"
      memory: "2Gi"
    requests:
      cpu: "500m"
      memory: "512Mi"
```

### 3. Use Specific Image Tags

Avoid using `latest` tag in production:
```yaml
mcpserver:
  image:
    tag: "v0.1.0"
    pullPolicy: IfNotPresent
```

### 4. Configure Monitoring

Integrate with your monitoring solution (Prometheus, Datadog, etc.):
```yaml
mcpserver:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "9900"
```

### 5. Network Policies

Enable network policies to restrict traffic:
```yaml
networkPolicy:
  enabled: true
  ingress:
    - from:
      - podSelector:
          matchLabels:
            app: allowed-client
  egress:
    - to:
      - namespaceSelector: {}
```

### 6. High Availability

For production, consider running multiple replicas (requires shared storage or external database):
```yaml
# Note: This requires PostgreSQL instead of SQLite
database:
  type: postgresql
  postgresql:
    host: postgres.default.svc.cluster.local
    # ... other config
```

## Advanced Topics

### Using PostgreSQL

For production deployments with high availability:

1. Deploy PostgreSQL:
```bash
helm install postgres bitnami/postgresql \
  --set auth.username=mcp_optimizer \
  --set auth.password=changeme \
  --set auth.database=mcp_optimizer \
  -n default
```

2. Configure MCP-Optimizer to use PostgreSQL:
```yaml
database:
  type: postgresql
  postgresql:
    host: postgres-postgresql.default.svc.cluster.local
    port: 5432
    database: mcp_optimizer
    username: mcp_optimizer
    passwordSecretName: postgres-postgresql
    passwordSecretKey: password
```

### Custom RBAC

To use a custom Role instead of ClusterRole (namespace-scoped):

1. Disable built-in RBAC:
```yaml
rbac:
  create: false
```

2. Create your own RBAC resources manually.

### Integration with GitOps

To use with ArgoCD or Flux:

```yaml
# argocd-application.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: mcp-optimizer
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/your-org/mcp-optimizer
    targetRevision: main
    path: helm/mcp-optimizer
    helm:
      valueFiles:
        - values.yaml
  destination:
    server: https://kubernetes.default.svc
    namespace: toolhive-system
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
```

## Next Steps

- Read the [Kubernetes Integration Guide](./kubernetes-integration.md)
- Explore the [example values files](../helm/mcp-optimizer/examples/)
- Review the [ToolHive documentation](https://github.com/stacklok/toolhive)
- Set up monitoring and observability
- Configure backup and disaster recovery

## Support

For help and support:
- GitHub Issues: https://github.com/your-org/mcp-optimizer/issues
- Documentation: https://github.com/your-org/mcp-optimizer/docs
- ToolHive Support: https://github.com/stacklok/toolhive/issues

