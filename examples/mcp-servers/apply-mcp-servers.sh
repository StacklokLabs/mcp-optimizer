#!/bin/bash
# Apply all MCP server examples to Kubernetes cluster

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXAMPLES_DIR="${SCRIPT_DIR}"

echo "Applying MCP server examples..."
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace toolhive-system &> /dev/null; then
    echo "Creating toolhive-system namespace..."
    kubectl create namespace toolhive-system
fi

# Check if GitHub secrets exist, prompt to create if not
echo "Checking for GitHub secrets..."
if ! kubectl get secret ghcr-pull-secret -n toolhive-system &> /dev/null; then
    echo "  Warning: ghcr-pull-secret does not exist"
    echo "  Images from ghcr.io may fail to pull without this secret"
    echo "  Create it with:"
    echo "    export GITHUB_TOKEN=your_token_here"
    echo "    export GITHUB_USERNAME=your_username"
    echo "    ./examples/mcp-servers/create-github-secrets.sh"
    echo "  Continuing anyway..."
else
    echo "  ✓ ghcr-pull-secret found"
fi

if ! kubectl get secret github-token -n toolhive-system &> /dev/null; then
    echo "  Warning: github-token secret does not exist"
    echo "  GitHub MCP servers may fail without this secret"
    echo "  Create it with:"
    echo "    export GITHUB_TOKEN=your_token_here"
    echo "    export GITHUB_USERNAME=your_username"
    echo "    ./examples/mcp-servers/create-github-secrets.sh"
    echo "  Continuing anyway..."
else
    echo "  ✓ github-token found"
fi

# Apply shared ServiceAccount with imagePullSecrets
echo ""
echo "Applying shared-serviceaccount.yaml..."
kubectl apply -f "${EXAMPLES_DIR}/shared-serviceaccount.yaml"

# Apply MCP servers
echo ""
echo "Applying MCP servers..."
kubectl apply -f "${EXAMPLES_DIR}/mcpserver_fetch.yaml"
kubectl apply -f "${EXAMPLES_DIR}/mcpserver_github.yaml"
kubectl apply -f "${EXAMPLES_DIR}/mcpserver_toolhive-doc-mcp.yaml"
kubectl apply -f "${EXAMPLES_DIR}/mcpserver_mcp-optimizer.yaml"

echo ""
echo "✓ Applied all MCP server examples!"
echo ""
echo "Check status with: kubectl get mcpservers -n toolhive-system"
echo "Check pods with: kubectl get pods -n toolhive-system"

