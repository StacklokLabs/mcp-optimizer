#!/bin/bash
# Create GitHub secrets from GITHUB_TOKEN environment variable
#
# This script creates both:
# 1. ghcr-pull-secret (docker-registry type) for pulling images from ghcr.io
# 2. github-token (Opaque type) for GitHub API access by MCP servers
#
# Usage:
#   export GITHUB_TOKEN=your_token_here
#   export GITHUB_USERNAME=your_username  # Optional, will prompt if not set
#   ./create-github-secrets.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="toolhive-system"

echo "Creating GitHub secrets..."
echo ""

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "Error: kubectl is not installed or not in PATH"
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace "${NAMESPACE}" &> /dev/null; then
    echo "Creating ${NAMESPACE} namespace..."
    kubectl create namespace "${NAMESPACE}"
fi

# Check for GITHUB_TOKEN environment variable
if [ -z "${GITHUB_TOKEN}" ]; then
    echo "Error: GITHUB_TOKEN environment variable is not set"
    echo ""
    echo "Please set it with:"
    echo "  export GITHUB_TOKEN=your_token_here"
    echo ""
    echo "The token needs:"
    echo "  - 'read:packages' scope for pulling images from ghcr.io"
    echo "  - GitHub API scopes (repo, read:org, etc.) for MCP server access"
    exit 1
fi

# Check for GITHUB_USERNAME, prompt if not set
if [ -z "${GITHUB_USERNAME}" ]; then
    echo "GITHUB_USERNAME not set. Please enter your GitHub username:"
    read -r GITHUB_USERNAME
    if [ -z "${GITHUB_USERNAME}" ]; then
        echo "Error: GitHub username is required"
        exit 1
    fi
fi

echo "Using GitHub username: ${GITHUB_USERNAME}"
echo ""

# Create or update ghcr-pull-secret
echo "Creating/updating ghcr-pull-secret..."
if kubectl get secret ghcr-pull-secret -n "${NAMESPACE}" &> /dev/null; then
    echo "  Secret already exists, deleting it first..."
    kubectl delete secret ghcr-pull-secret -n "${NAMESPACE}" --ignore-not-found=true
fi

kubectl create secret docker-registry ghcr-pull-secret \
  --docker-server=ghcr.io \
  --docker-username="${GITHUB_USERNAME}" \
  --docker-password="${GITHUB_TOKEN}" \
  -n "${NAMESPACE}"

echo "  ✓ Created ghcr-pull-secret"
echo ""

# Create or update github-token secret
echo "Creating/updating github-token secret..."
if kubectl get secret github-token -n "${NAMESPACE}" &> /dev/null; then
    echo "  Secret already exists, deleting it first..."
    kubectl delete secret github-token -n "${NAMESPACE}" --ignore-not-found=true
fi

kubectl create secret generic github-token \
  --from-literal=token="${GITHUB_TOKEN}" \
  -n "${NAMESPACE}"

echo "  ✓ Created github-token secret"
echo ""

echo "✓ Successfully created both GitHub secrets!"
echo ""
echo "Both secrets use the same token value but serve different purposes:"
echo "  - ghcr-pull-secret: Used by ServiceAccount imagePullSecrets to pull images"
echo "  - github-token: Used by MCP servers as environment variable for API access"
echo ""
echo "Verify secrets:"
echo "  kubectl get secrets -n ${NAMESPACE} | grep -E 'ghcr-pull-secret|github-token'"

