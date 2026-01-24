"""
Kubernetes client for interacting with MCPServer CRDs in k8s mode.

This module provides functionality to list and query MCPServer custom resources
when mcp-optimizer is running in Kubernetes mode.
"""

import os
from pathlib import Path
from typing import Any

import httpx
import structlog

from mcp_optimizer.toolhive.api_models.core import Workload
from mcp_optimizer.toolhive.api_models.runtime import WorkloadStatus
from mcp_optimizer.toolhive.api_models.types import TransportType

logger = structlog.get_logger(__name__)

# In-cluster service account paths
SERVICE_ACCOUNT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"  # nosec B105 - Standard Kubernetes service account path, not a hardcoded password
SERVICE_ACCOUNT_CA_CERT_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
SERVICE_ACCOUNT_NAMESPACE_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


class K8sClientError(Exception):
    """Exception raised for Kubernetes client errors."""

    pass


class K8sClient:
    """Client for interacting with Kubernetes MCPServer CRDs."""

    def __init__(
        self,
        api_server_url: str = "http://127.0.0.1:8001",
        namespace: str | None = None,
        timeout: int = 10,
        token: str | None = None,
        verify_ssl: bool | str = True,
    ):
        """Initialize the Kubernetes client.

        Args:
            api_server_url: URL of the Kubernetes API server (default: kubectl proxy URL)
            namespace: Kubernetes namespace to query. If None, uses all namespaces
            timeout: Request timeout in seconds
            token: Service account token for authentication (auto-detected if None)
            verify_ssl: SSL verification setting. Can be True, False, or path to CA cert
        """
        self.api_server_url = api_server_url.rstrip("/")
        self.namespace = namespace
        self.timeout = timeout

        # Auto-detect in-cluster configuration
        self.token = token
        self.verify_ssl = verify_ssl

        if self.token is None and self._is_in_cluster():
            self.token = self._read_service_account_token()

        if self.verify_ssl is True and self._is_in_cluster():
            ca_cert_path = Path(SERVICE_ACCOUNT_CA_CERT_PATH)
            if ca_cert_path.exists():
                self.verify_ssl = str(ca_cert_path)

        # Set up HTTP client with authentication
        headers = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self.client = httpx.AsyncClient(
            timeout=timeout,
            headers=headers,
            verify=self.verify_ssl,
        )

        logger.info(
            "Initialized K8s client",
            api_server_url=self.api_server_url,
            namespace=namespace or "all",
            authenticated=bool(self.token),
            in_cluster=self._is_in_cluster(),
        )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    def _is_in_cluster(self) -> bool:
        """Check if running inside a Kubernetes cluster.

        Returns:
            True if running in-cluster, False otherwise
        """
        token_path = Path(SERVICE_ACCOUNT_TOKEN_PATH)
        return token_path.exists()

    def _read_service_account_token(self) -> str | None:
        """Read the service account token from the mounted secret.

        Returns:
            Token string or None if not found
        """
        token_path = Path(SERVICE_ACCOUNT_TOKEN_PATH)
        try:
            if token_path.exists():
                token = token_path.read_text().strip()
                logger.info("Loaded service account token for authentication")
                return token
        except Exception as e:
            logger.warning(
                "Failed to read service account token",
                error=str(e),
                path=str(token_path),
            )
        return None

    def _mcpserver_to_workload(
        self, mcpserver: dict[str, Any], is_virtual: bool = False
    ) -> Workload:
        """Convert an MCPServer CRD to a Workload model.

        Args:
            mcpserver: MCPServer custom resource dictionary
            is_virtual: Whether this is a VirtualMCPServer resource

        Returns:
            Workload model instance
        """
        metadata = mcpserver.get("metadata", {})
        spec = mcpserver.get("spec", {})
        status = mcpserver.get("status", {})

        # Extract basic info
        name = metadata.get("name", "")
        namespace = metadata.get("namespace", "default")
        created_at = metadata.get("creationTimestamp", "")

        # Extract spec fields
        image = spec.get("image", "")
        port = spec.get("port", 8080)

        # Extract status fields
        phase = status.get("phase", "Unknown")
        url = status.get("url", "")
        tools = status.get("tools", [])

        # VirtualMCPServers always use streamable-http transport and serve on /mcp path
        # The K8s status URL doesn't include the path, so we need to append it
        if is_virtual:
            if url and not url.endswith("/mcp"):
                url = f"{url}/mcp"
            transport_type: TransportType | None = TransportType.transport_type_streamable_http
        else:
            # Regular MCPServer - get transport from spec
            transport_str = spec.get("transport", "stdio")
            transport_type = TransportType(transport_str) if transport_str else None

        # Map k8s phase to workload status
        # Phases: Pending, Running, Failed, Unknown (MCPServer)
        # Phases: Pending, Ready, Failed, Unknown (VirtualMCPServer)
        workload_status = WorkloadStatus.workload_status_stopped
        if phase == "Running" or phase == "Ready":
            workload_status = WorkloadStatus.workload_status_running

        # Determine if this is a remote server or not
        # In k8s, we don't have direct "remote" field, so we check if it's defined as remote in spec
        is_remote = spec.get("remote", False)

        # Get proxy mode from spec
        # VirtualMCPServers always use streamable-http proxy mode
        if is_virtual:
            proxy_mode = "streamable-http"
        else:
            proxy_mode = spec.get("proxyMode", "sse")

        # Get labels and group
        labels = metadata.get("labels", {})
        # Prefer spec.groupRef over label
        # MCPServer CRD uses spec.groupRef, VirtualMCPServer uses spec.config.groupRef
        if is_virtual:
            config = spec.get("config", {})
            group = config.get("groupRef") or labels.get("toolhive.stacklok.dev/group", "default")
        else:
            group = spec.get("groupRef") or labels.get("toolhive.stacklok.dev/group", "default")

        # Package is typically the image name, but for VirtualMCPServers use a unique identifier
        if is_virtual:
            # Use vmcp:<name> as package identifier for VirtualMCPServers
            package = f"vmcp:{name}"
        else:
            package = image

        workload = Workload(
            name=name,
            url=url,
            status=workload_status,
            transport_type=transport_type,
            port=port,
            proxy_mode=proxy_mode,
            remote=is_remote,
            package=package,
            group=group,
            labels=labels,
            tools=tools,
            created_at=created_at,
            status_context=status.get("message", ""),
            virtual_mcp=is_virtual,
        )

        logger.debug(
            "Converted MCPServer to Workload",
            name=name,
            namespace=namespace,
            status=workload_status,
        )

        return workload

    async def list_mcpservers(
        self, namespace: str | None = None, all_namespaces: bool = False
    ) -> list[Workload]:
        """List MCPServer and VirtualMCPServer resources from Kubernetes.

        Args:
            namespace: Specific namespace to query (overrides instance namespace)
            all_namespaces: If True, list across all namespaces

        Returns:
            List of Workload objects converted from MCPServer and VirtualMCPServer CRDs

        Raises:
            K8sClientError: If the request fails
        """
        # Determine which namespace to use
        target_namespace = namespace or self.namespace

        if all_namespaces or target_namespace is None:
            # List across all namespaces
            mcpserver_url = f"{self.api_server_url}/apis/toolhive.stacklok.dev/v1alpha1/mcpservers"
            vmcpserver_url = (
                f"{self.api_server_url}/apis/toolhive.stacklok.dev/v1alpha1/virtualmcpservers"
            )
            scope = "all namespaces"
        else:
            # List in specific namespace
            mcpserver_url = (
                f"{self.api_server_url}/apis/toolhive.stacklok.dev/v1alpha1/"
                f"namespaces/{target_namespace}/mcpservers"
            )
            vmcpserver_url = (
                f"{self.api_server_url}/apis/toolhive.stacklok.dev/v1alpha1/"
                f"namespaces/{target_namespace}/virtualmcpservers"
            )
            scope = f"namespace {target_namespace}"

        logger.info("Listing MCPServers and VirtualMCPServers from Kubernetes", scope=scope)

        workloads = []

        # Fetch regular MCPServers
        try:
            response = await self.client.get(mcpserver_url)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            logger.info(
                "Successfully listed MCPServers",
                count=len(items),
                scope=scope,
            )

            # Convert MCPServer CRDs to Workload models
            workloads.extend(
                [self._mcpserver_to_workload(item, is_virtual=False) for item in items]
            )

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error listing MCPServers: {e.response.status_code}"
            logger.error(
                error_msg,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise K8sClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Request error listing MCPServers: {e}"
            logger.error(error_msg, error=str(e))
            raise K8sClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error listing MCPServers: {e}"
            logger.exception(error_msg)
            raise K8sClientError(error_msg) from e

        # Fetch VirtualMCPServers
        try:
            response = await self.client.get(vmcpserver_url)
            response.raise_for_status()

            data = response.json()
            items = data.get("items", [])

            logger.info(
                "Successfully listed VirtualMCPServers",
                count=len(items),
                scope=scope,
            )

            # Convert VirtualMCPServer CRDs to Workload models
            workloads.extend([self._mcpserver_to_workload(item, is_virtual=True) for item in items])

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error listing VirtualMCPServers: {e.response.status_code}"
            logger.error(
                error_msg,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise K8sClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Request error listing VirtualMCPServers: {e}"
            logger.error(error_msg, error=str(e))
            raise K8sClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error listing VirtualMCPServers: {e}"
            logger.exception(error_msg)
            raise K8sClientError(error_msg) from e

        logger.info(
            "Successfully listed all resources",
            total_count=len(workloads),
            scope=scope,
        )

        return workloads

    async def get_mcpserver(self, name: str, namespace: str) -> Workload | None:
        """Get a specific MCPServer resource.

        Args:
            name: Name of the MCPServer
            namespace: Namespace of the MCPServer

        Returns:
            Workload object or None if not found

        Raises:
            K8sClientError: If the request fails
        """
        url = (
            f"{self.api_server_url}/apis/toolhive.stacklok.dev/v1alpha1/"
            f"namespaces/{namespace}/mcpservers/{name}"
        )

        logger.info("Getting MCPServer from Kubernetes", name=name, namespace=namespace)

        try:
            response = await self.client.get(url)

            if response.status_code == 404:
                logger.info("MCPServer not found", name=name, namespace=namespace)
                return None

            response.raise_for_status()

            mcpserver = response.json()
            workload = self._mcpserver_to_workload(mcpserver)

            logger.info("Successfully retrieved MCPServer", name=name, namespace=namespace)
            return workload

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            error_msg = f"HTTP error getting MCPServer: {e.response.status_code}"
            logger.error(
                error_msg,
                status_code=e.response.status_code,
                response_text=e.response.text,
            )
            raise K8sClientError(error_msg) from e
        except httpx.RequestError as e:
            error_msg = f"Request error getting MCPServer: {e}"
            logger.error(error_msg, error=str(e))
            raise K8sClientError(error_msg) from e
        except Exception as e:
            error_msg = f"Unexpected error getting MCPServer: {e}"
            logger.exception(error_msg)
            raise K8sClientError(error_msg) from e

    async def get_running_mcp_workloads(self) -> list[Workload]:
        """Get only the running MCP workloads from Kubernetes.

        Returns:
            List of running MCP workloads
        """
        all_workloads = await self.list_mcpservers(all_namespaces=True)

        running_mcp_workloads = [
            workload for workload in all_workloads if workload.status == "running"
        ]

        logger.info(
            "Filtered running MCP workloads",
            total_workloads=len(all_workloads),
            running_mcp_count=len(running_mcp_workloads),
        )

        return running_mcp_workloads


def get_k8s_namespace() -> str:
    """Get the current Kubernetes namespace.

    Returns the namespace from the environment variable or the service account
    namespace file if running in-cluster.

    Returns:
        Namespace name (defaults to 'default' if not found)
    """
    # First try environment variable
    namespace = os.getenv("KUBERNETES_NAMESPACE")
    if namespace:
        return namespace

    # Try reading from service account namespace file (in-cluster)
    namespace_path = Path(SERVICE_ACCOUNT_NAMESPACE_PATH)
    try:
        if namespace_path.exists():
            namespace = namespace_path.read_text().strip()
            if namespace:
                return namespace
    except Exception as e:  # nosec B110 - Intentional fallback to default namespace
        logger.debug("Failed to read namespace from service account, using default", error=str(e))

    # Default to 'default' namespace
    return "default"


def get_k8s_api_server_url() -> str:
    """Get the Kubernetes API server URL.

    Returns the API server URL, checking environment variables and in-cluster config.
    Uses KUBERNETES_SERVICE_PORT_HTTPS for secure in-cluster communication.

    Returns:
        API server URL
    """
    # Check for explicit environment variable (for testing with kubectl proxy)
    k8s_api_url = os.getenv("K8S_API_SERVER_URL")
    if k8s_api_url:
        return k8s_api_url

    # Check if running in-cluster
    k8s_host = os.getenv("KUBERNETES_SERVICE_HOST")
    k8s_port_https = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS")

    if k8s_host and k8s_port_https:
        # Running in-cluster, use the HTTPS service URL
        return f"https://{k8s_host}:{k8s_port_https}"

    # Fallback to KUBERNETES_SERVICE_PORT if HTTPS port not available
    k8s_port = os.getenv("KUBERNETES_SERVICE_PORT")
    if k8s_host and k8s_port:
        return f"https://{k8s_host}:{k8s_port}"

    # Default to kubectl proxy URL for local development
    return "http://127.0.0.1:8001"
