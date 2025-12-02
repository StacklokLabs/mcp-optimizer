#!/bin/bash
# Check status of all MCP server examples

set -e

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

echo "MCP Server Status:"
echo "=================="
kubectl get mcpservers -n toolhive-system 2>/dev/null || echo "No MCPServer resources found or namespace doesn't exist"

echo ""
echo "Pods Status:"
echo "============"
kubectl get pods -n toolhive-system -l app.kubernetes.io/managed-by=toolhive-operator 2>/dev/null || echo "No pods found"

echo ""
echo "Services Status:"
echo "================"
kubectl get svc -n toolhive-system -l app.kubernetes.io/managed-by=toolhive-operator 2>/dev/null || echo "No services found"

