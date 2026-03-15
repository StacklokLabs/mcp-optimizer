"""Tests for the Kubernetes client."""

from mcp_optimizer.toolhive.k8s_client import K8sClient


def test_mcpserver_to_workload_unknown_transport_type():
    """Test that _mcpserver_to_workload handles unknown transport types gracefully."""
    k8s_client = K8sClient()

    mcpserver = {
        "metadata": {
            "name": "test-server",
            "namespace": "default",
            "creationTimestamp": "2025-01-01T00:00:00Z",
        },
        "spec": {
            "image": "test/server:latest",
            "transport": "grpc",
            "port": 8080,
        },
        "status": {
            "phase": "Running",
            "url": "http://127.0.0.1:8080",
        },
    }

    # Should not raise — unknown transport defaults to None
    workload = k8s_client._mcpserver_to_workload(mcpserver)

    assert workload.name == "test-server"
    assert workload.transport_type is None
    assert workload.status == "running"


def test_mcpserver_to_workload_empty_transport_type():
    """Test that _mcpserver_to_workload handles empty transport string."""
    k8s_client = K8sClient()

    mcpserver = {
        "metadata": {
            "name": "empty-transport",
            "namespace": "default",
            "creationTimestamp": "2025-01-01T00:00:00Z",
        },
        "spec": {
            "image": "test/server:latest",
            "transport": "",
            "port": 8080,
        },
        "status": {
            "phase": "Running",
            "url": "http://127.0.0.1:8080",
        },
    }

    # Empty string is falsy, so takes the `else None` path without hitting TransportType()
    workload = k8s_client._mcpserver_to_workload(mcpserver)

    assert workload.name == "empty-transport"
    assert workload.transport_type is None
    assert workload.status == "running"
