# MCP Optimizer Helm Chart

This Helm chart deploys MCP Optimizer as an MCPServer resource in a Kubernetes cluster with ToolHive operator installed.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.2.0+
- ToolHive Operator installed (see [ToolHive Installation](https://github.com/stacklok/toolhive))
- ToolHive CRDs installed (`mcpservers.toolhive.stacklok.dev`)

## Installing the Chart

### From OCI Registry (Recommended)

To install the chart from the OCI registry:

```bash
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  -n toolhive-system --create-namespace
```

To install a specific version:

```bash
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  --version 0.1.0 \
  -n toolhive-system --create-namespace
```

### From Local Chart Directory

For development or customization:

```bash
cd /path/to/mcp-optimizer
helm install mcp-optimizer ./helm/mcp-optimizer -n toolhive-system --create-namespace
```

## Uninstalling the Chart

To uninstall/delete the `mcp-optimizer` release:

```bash
helm uninstall mcp-optimizer -n toolhive-system
```

## Configuration

The following table lists the configurable parameters of the MCP Optimizer chart and their default values.

### MCPServer Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mcpserver.name` | Name of the MCPServer resource | `mcp-optimizer` |
| `mcpserver.namespace` | Namespace for deployment | `toolhive-system` |
| `mcpserver.image.repository` | Container image repository | `ghcr.io/stacklok/mcp-optimizer` |
| `mcpserver.image.tag` | Container image tag | `""` (uses chart appVersion) |
| `mcpserver.transport` | MCP transport protocol | `streamable-http` |
| `mcpserver.port` | MCP server port | `9900` |
| `mcpserver.resources.limits.cpu` | CPU limit | `1000m` |
| `mcpserver.resources.limits.memory` | Memory limit | `1Gi` |
| `mcpserver.resources.requests.cpu` | CPU request | `250m` |
| `mcpserver.resources.requests.memory` | Memory request | `256Mi` |

### Service Account Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `serviceAccount.create` | Create a service account | `true` |
| `serviceAccount.name` | Service account name | `""` (generated from fullname) |
| `serviceAccount.annotations` | Service account annotations | `{}` |

### RBAC Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `rbac.create` | Create RBAC resources | `true` |
| `rbac.rules` | ClusterRole rules | See values.yaml |

### Database Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `database.type` | Database type (sqlite or postgresql) | `sqlite` |
| `database.sqlite.path` | SQLite database path | `/data/mcp_optimizer.db` |

### Persistence Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `persistence.enabled` | Enable persistent storage | `false` |
| `persistence.accessMode` | PVC access mode | `ReadWriteOnce` |
| `persistence.size` | PVC size | `1Gi` |
| `persistence.storageClass` | Storage class | `""` |
| `persistence.existingClaim` | Use existing PVC | `""` |

### Environment Variables

| Parameter | Description | Default |
|-----------|-------------|---------|
| `mcpserver.env` | Additional environment variables to merge with defaults | `[]` (empty) |
| `mcpserver.podTemplateSpec` | Full pod template specification | See values.yaml |

Default environment variables (defined in `podTemplateSpec`):

- `SQLITE_TMPDIR=/tmp` - Temporary directory for SQLite
- `RUNTIME_MODE=k8s` - Run in Kubernetes mode
- `K8S_ALL_NAMESPACES=true` - Query all namespaces
- `LOG_LEVEL=INFO` - Logging level
- `WORKLOAD_POLLING_INTERVAL=60` - Polling interval in seconds
- `REGISTRY_POLLING_INTERVAL=300` - Registry polling interval in seconds
- `MAX_TOOLS_TO_RETURN=8` - Maximum tools in search results
- `MAX_SERVERS_TO_RETURN=5` - Maximum servers in search results
- `HYBRID_SEARCH_SEMANTIC_RATIO=0.5` - Semantic search ratio

**Automatically Added:**
- `ASYNC_DB_URL` - Generated from `database` configuration
- `DB_URL` - Generated from `database` configuration
- `ALLOWED_GROUPS` - Set from `groupFiltering.allowedGroups` if configured
- `MCP_PORT` - Set by the ToolHive operator from `spec.port`

**Adding Custom Environment Variables:**
Use `mcpserver.env` to add additional env vars that will be merged with the defaults:

```yaml
mcpserver:
  env:
    - name: CUSTOM_VAR
      value: "custom-value"
```

**Environment Variable Precedence:**
Environment variables are merged with deduplication (last value wins) in this order:
1. Base env vars from `podTemplateSpec.spec.containers[0].env` (lowest priority)
2. Template-generated env vars (`ASYNC_DB_URL`, `DB_URL`, `ALLOWED_GROUPS`) (middle priority)
3. Custom env vars from `mcpserver.env` (highest priority)

If the same variable name appears multiple times, the last occurrence takes precedence. This allows you to override defaults using `mcpserver.env`.

## Examples

### Basic Installation

```bash
# From OCI registry (recommended)
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  -n toolhive-system --create-namespace

# From local chart
helm install mcp-optimizer ./helm/mcp-optimizer -n toolhive-system --create-namespace
```

### Installation with Local Development Image

```bash
# Build and load local image
docker build -t mcp-optimizer:local .
kind load docker-image mcp-optimizer:local

# Install with local image
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.image.repository=mcp-optimizer \
  --set mcpserver.image.tag=local \
  -n toolhive-system
```

Note: For advanced pod customization (like setting `imagePullPolicy`), use `podTemplateSpec` - see the "Advanced Pod Customization" section below.

### Installation with Custom Registry

```bash
helm install mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer \
  --set mcpserver.image.repository=myregistry.com/mcp-optimizer \
  --set mcpserver.image.tag=v0.2.0 \
  -n toolhive-system
```

### Installation with Persistence Enabled

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set persistence.enabled=true \
  --set persistence.size=5Gi \
  --set persistence.storageClass=fast-ssd \
  -n toolhive-system
```

### Installation with Resource Limits

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set mcpserver.resources.limits.cpu=2000m \
  --set mcpserver.resources.limits.memory=2Gi \
  --set mcpserver.resources.requests.cpu=500m \
  --set mcpserver.resources.requests.memory=512Mi \
  -n toolhive-system
```

### Installation with Group Filtering

To restrict mcp-optimizer to only certain MCP server groups:

```bash
helm install mcp-optimizer ./helm/mcp-optimizer \
  --set groupFiltering.allowedGroups="{development,production}" \
  -n toolhive-system
```

Or create a custom values file:

```yaml
# custom-values.yaml
groupFiltering:
  allowedGroups:
    - development
    - production
```

```bash
helm install mcp-optimizer ./helm/mcp-optimizer -f custom-values.yaml -n toolhive-system
```

### Installation with Single Namespace Scope

To limit mcp-optimizer to query only a specific namespace, you can override the default environment variables:

```yaml
# custom-values.yaml
mcpserver:
  env:
    - name: K8S_ALL_NAMESPACES
      value: "false"
    - name: K8S_NAMESPACE
      value: "my-namespace"
```

```bash
helm install mcp-optimizer ./helm/mcp-optimizer -f custom-values.yaml -n toolhive-system
```

Note: Custom env vars in `mcpserver.env` have the highest precedence and will override any defaults with the same name. To override a default env var (like `K8S_ALL_NAMESPACES`), simply add it to `mcpserver.env` with your desired value.

### Advanced Pod Customization with podTemplateSpec

The default configuration uses `podTemplateSpec` with emptyDir volumes for `/data` and `/tmp`. The helm chart automatically:
- Adds `ASYNC_DB_URL` and `DB_URL` based on database configuration
- Adds `ALLOWED_GROUPS` if `groupFiltering.allowedGroups` is set
- Appends custom env vars from `mcpserver.env`

You can customize `podTemplateSpec` for advanced use cases (different volumes, security contexts, init containers, etc.):

```yaml
# custom-values.yaml
mcpserver:
  podTemplateSpec:
    spec:
      securityContext:
        fsGroup: 2000  # Custom security context
      volumes:
        - name: data
          persistentVolumeClaim:  # Use PVC instead of emptyDir
            claimName: mcp-data
        - name: tmp
          emptyDir: {}
      containers:
        - name: mcp
          imagePullPolicy: Always
          volumeMounts:
            - name: data
              mountPath: /data
            - name: tmp
              mountPath: /tmp
          env:
            # Define your base env vars
            - name: LOG_LEVEL
              value: "DEBUG"
            - name: RUNTIME_MODE
              value: "k8s"
            # ... other env vars ...
            # NOTE: ASYNC_DB_URL, DB_URL, and ALLOWED_GROUPS will be automatically added
```

```bash
helm install mcp-optimizer ./helm/mcp-optimizer -f custom-values.yaml -n toolhive-system
```

**Note:** When customizing `podTemplateSpec`:
- Database URLs (`ASYNC_DB_URL`, `DB_URL`) are automatically added by the template
- `ALLOWED_GROUPS` is automatically added if you set `groupFiltering.allowedGroups`
- Additional env vars from `mcpserver.env` are automatically appended
- See `examples/mcp-servers/mcpserver_mcp-optimizer.yaml` for a complete working example

## Verifying the Installation

After installation, verify that the MCPServer resource was created:

```bash
kubectl get mcpserver mcp-optimizer -n toolhive-system
```

Check the status of the deployment:

```bash
kubectl get pods -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

View the logs:

```bash
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

## Upgrading

To upgrade an existing release:

```bash
# From OCI registry
helm upgrade mcp-optimizer oci://ghcr.io/stackloklabs/mcp-optimizer/mcp-optimizer -n toolhive-system

# From local chart
helm upgrade mcp-optimizer ./helm/mcp-optimizer -n toolhive-system
```

## Troubleshooting

### Pod Not Starting

Check the MCPServer status:

```bash
kubectl describe mcpserver mcp-optimizer -n toolhive-system
```

Check pod events:

```bash
kubectl describe pod -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

### RBAC Permission Errors

Verify the service account has the correct permissions:

```bash
kubectl auth can-i list mcpservers.toolhive.stacklok.dev \
  --as=system:serviceaccount:toolhive-system:mcp-optimizer-mcp-optimizer
```

### No MCPServers Discovered

Ensure other MCPServer resources exist:

```bash
kubectl get mcpserver --all-namespaces
```

Check mcp-optimizer logs for errors:

```bash
kubectl logs -l toolhive.stacklok.dev/mcpserver=mcp-optimizer -n toolhive-system
```

## Production Recommendations

For production deployments, consider:

1. **Enable Persistence**: Set `persistence.enabled=true` to persist the SQLite database
2. **Use PostgreSQL**: For better performance and reliability in multi-replica scenarios
3. **Resource Limits**: Set appropriate CPU and memory limits based on your workload
4. **Network Policies**: Enable `networkPolicy.enabled=true` and configure ingress/egress rules
5. **Image Pull Secrets**: Configure image pull secrets if using a private registry
6. **Monitoring**: Integrate with your monitoring solution (Prometheus, etc.)

## Additional Resources

- [MCP Optimizer Documentation](https://github.com/StacklokLabs/mcp-optimizer)
- [ToolHive Documentation](https://github.com/stacklok/toolhive)
- [Kubernetes Integration Guide](/docs/kubernetes-integration.md)

## License

This Helm chart is licensed under the same license as the MCP Optimizer project.
