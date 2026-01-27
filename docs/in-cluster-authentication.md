# In-Cluster Authentication

MCP Optimizer automatically handles authentication when running inside a Kubernetes cluster by using the Kubernetes service account system.

## How It Works

When MCP Optimizer runs as a Pod in Kubernetes:

### 1. Service Account Token

Kubernetes automatically mounts a service account token into every Pod at:
```
/var/run/secrets/kubernetes.io/serviceaccount/token
```

MCP Optimizer detects this file and uses it for authentication:
```python
# Automatically loaded
token = Path("/var/run/secrets/kubernetes.io/serviceaccount/token").read_text()
headers = {"Authorization": f"Bearer {token}"}
```

### 2. CA Certificate

The cluster's CA certificate is mounted at:
```
/var/run/secrets/kubernetes.io/serviceaccount/ca.crt
```

MCP Optimizer uses this for SSL verification when communicating with the Kubernetes API server:
```python
# Automatically configured
verify_ssl = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
```

### 3. API Server URL

Kubernetes injects environment variables into every Pod:
- `KUBERNETES_SERVICE_HOST` - The API server hostname
- `KUBERNETES_SERVICE_PORT_HTTPS` - The HTTPS port (typically 443)

MCP Optimizer automatically constructs the API server URL from these:
```python
# Automatically done when RUNTIME_MODE=k8s and in-cluster
api_url = f"https://{KUBERNETES_SERVICE_HOST}:{KUBERNETES_SERVICE_PORT_HTTPS}"
```

No manual configuration needed!

### 4. Namespace Detection

The current namespace is available at:
```
/var/run/secrets/kubernetes.io/serviceaccount/namespace
```

## Automatic Detection

The `K8sClient` automatically detects when it's running in-cluster:

```python
def _is_in_cluster(self) -> bool:
    """Check if running inside a Kubernetes cluster."""
    token_path = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    return token_path.exists()
```

When in-cluster:
1. ✅ Service account token is automatically loaded
2. ✅ CA certificate is automatically configured for SSL verification
3. ✅ API server URL is constructed from environment variables
4. ✅ Namespace is read from mounted secret

When NOT in-cluster (kubectl proxy):
1. ❌ No service account token needed (proxy handles auth)
2. ❌ No SSL verification needed (proxy uses HTTP)
3. ℹ️  API server URL defaults to `http://127.0.0.1:8001`

## Required RBAC Permissions

For the service account to access MCPServer and VirtualMCPServer CRDs, you need appropriate RBAC:

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
  resources: ["mcpservers", "virtualmcpservers"]
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

## Environment Variables

Standard Kubernetes environment variables (automatically injected):

| Variable | Description | Example |
|----------|-------------|---------|
| `KUBERNETES_SERVICE_HOST` | API server hostname | `10.96.0.1` or `kubernetes.default.svc` |
| `KUBERNETES_SERVICE_PORT_HTTPS` | HTTPS port | `443` |
| `KUBERNETES_SERVICE_PORT` | Fallback port | `443` |

MCP Optimizer configuration (optional overrides):

| Variable | Description | Default | Auto-detected? |
|----------|-------------|---------|----------------|
| `K8S_API_SERVER_URL` | Override API server URL | See below | ✅ Yes |
| `K8S_NAMESPACE` | Specific namespace to query | All namespaces | ✅ Yes (in-cluster only) |
| `K8S_ALL_NAMESPACES` | Query all namespaces | `true` | No |

**K8S_API_SERVER_URL Auto-detection**:
- In-cluster: Constructed from `KUBERNETES_SERVICE_HOST` + `KUBERNETES_SERVICE_PORT_HTTPS`
- kubectl proxy: Defaults to `http://127.0.0.1:8001`
- Can be overridden by setting the environment variable explicitly

## Testing In-Cluster Authentication

### Step 1: Create Service Account

```bash
kubectl apply -f - <<EOF
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
  resources: ["mcpservers", "virtualmcpservers"]
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
EOF
```

### Step 2: Deploy MCP Optimizer

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-optimizer
  namespace: toolhive-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mcp-optimizer
  template:
    metadata:
      labels:
        app: mcp-optimizer
    spec:
      serviceAccountName: mcp-optimizer
      containers:
      - name: mcp-optimizer
        image: your-registry/mcp-optimizer:latest
        env:
        - name: RUNTIME_MODE
          value: "k8s"
        - name: K8S_ALL_NAMESPACES
          value: "true"
        ports:
        - containerPort: 9900
          name: mcp
```

### Step 3: Verify

```bash
# Check the pod logs
kubectl logs -f deployment/mcp-optimizer -n toolhive-system

# You should see:
# [info] Initialized K8s client authenticated=True in_cluster=True
# [info] Loaded service account token for authentication
```

## Troubleshooting

### Permission Denied

```
Error: mcpservers.toolhive.stacklok.dev is forbidden
```

**Solution:** Check RBAC permissions:
```bash
kubectl auth can-i list mcpservers.toolhive.stacklok.dev \
  --as=system:serviceaccount:toolhive-system:mcp-optimizer
```

Should return `yes`. If not, verify ClusterRole and ClusterRoleBinding are created.

### SSL Certificate Verification Failed

```
Error: SSL certificate verify failed
```

**Solution:** Ensure CA certificate is mounted:
```bash
kubectl exec -n toolhive-system deployment/mcp-optimizer -- \
  ls -la /var/run/secrets/kubernetes.io/serviceaccount/
```

Should show `ca.crt` file.

### Service Account Token Not Found

```
Warning: Failed to read service account token
```

**Solution:** Verify service account is configured:
```bash
kubectl get pod -n toolhive-system -l app=mcp-optimizer -o yaml | grep serviceAccountName
```

Should show `serviceAccountName: mcp-optimizer`.

## Security Best Practices

1. **Least Privilege**: Only grant the minimum required permissions
   - Use `get`, `list`, `watch` - avoid `create`, `update`, `delete` if not needed

2. **Namespace Scoping**: Use Role instead of ClusterRole when possible
   - If mcp-optimizer only needs to monitor one namespace

3. **Service Account Isolation**: Create dedicated service account
   - Don't use the default service account

4. **Token Rotation**: Kubernetes automatically rotates service account tokens
   - No manual intervention needed

5. **Audit Logging**: Enable audit logging for MCPServer access
   - Track who/what is accessing the resources

## Reference

- [Kubernetes Service Accounts](https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/)
- [Accessing the API from a Pod](https://kubernetes.io/docs/tasks/run-application/access-api-from-pod/)
- [RBAC Authorization](https://kubernetes.io/docs/reference/access-authn-authz/rbac/)

