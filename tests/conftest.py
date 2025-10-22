"""Shared test fixtures for mcp-optimizer tests."""

import tempfile
from pathlib import Path

import pytest_asyncio
from alembic import command
from alembic.config import Config

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps


@pytest_asyncio.fixture
async def test_db():
    """Create a temporary SQLite database and run migrations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = Path(tmp_file.name)

    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    project_root = Path(__file__).parent.parent
    migrations_dir = project_root / "migrations"

    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(migrations_dir))
    alembic_config.set_main_option("db_url", f"sqlite:///{test_db_path}")

    command.upgrade(alembic_config, "head")

    # Create DatabaseConfig instance with test database
    db_config = DatabaseConfig(test_db_url)
    yield db_config

    # Cleanup
    try:
        await db_config.close()
    except Exception:
        # Ignore cleanup errors
        pass
    test_db_path.unlink(missing_ok=True)


@pytest_asyncio.fixture
async def mcp_ops(test_db) -> WorkloadServerOps:
    """Create WorkloadServerOps instance with test database."""
    return WorkloadServerOps(test_db)
