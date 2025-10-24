"""Database exceptions for MCP Optimizer."""

from mcp_optimizer.db.models import RegistryServer


class DbNotFoundError(Exception):
    """Exception raised when a database record is not found."""

    pass


class DuplicateRegistryServersError(Exception):
    """Raised when multiple registry servers match the same URL or package."""

    def __init__(self, servers: list[RegistryServer]):
        self.servers = servers
        super().__init__(
            f"Found {len(servers)} registry servers with matching identifiers. "
            "Registry must have unique URLs and packages."
        )
