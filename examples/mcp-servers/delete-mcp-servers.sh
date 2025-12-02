#!/bin/bash
# Delete all MCP server examples from Kubernetes cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}"

echo "Deleting MCP server examples..."
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Delete in reverse order (dependencies first)
kubectl delete -f "${EXAMPLES_DIR}/mcpserver_mcp-optimizer.yaml" --ignore-not-found=true
kubectl delete -f "${EXAMPLES_DIR}/mcpserver_toolhive-doc-mcp.yaml" --ignore-not-found=true
kubectl delete -f "${EXAMPLES_DIR}/mcpserver_github.yaml" --ignore-not-found=true
kubectl delete -f "${EXAMPLES_DIR}/mcpserver_fetch.yaml" --ignore-not-found=true
kubectl delete -f "${EXAMPLES_DIR}/shared-serviceaccount.yaml" --ignore-not-found=true

# Note: GitHub secrets (ghcr-pull-secret and github-token) are not deleted
# as they may be used by other resources. Delete them manually if needed:
# kubectl delete secret ghcr-pull-secret github-token -n toolhive-system --ignore-not-found=true

echo "✓ Deleted all MCP server examples!"

