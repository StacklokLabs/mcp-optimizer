"""Shared test fixtures for mcp-optimizer tests."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.workload_server_ops import WorkloadServerOps


@pytest.fixture(scope="session", autouse=True)
def ensure_llmlingua_model():
    """Verify LLMLingua model is available from pre-downloaded models directory.

    Models should be downloaded by the download-models workflow in CI or locally
    via 'task download-models'. The default path is 'models/llmlingua' relative
    to the project root, as configured in config.py.
    """
    from mcp_optimizer.config import _get_default_model_paths
    from mcp_optimizer.response_optimizer.summarizers.llmlingua import LLMLINGUA_MODEL_FOLDER

    _, _, llmlingua_default = _get_default_model_paths()
    model_path = Path(llmlingua_default) / LLMLINGUA_MODEL_FOLDER
    model_file = model_path / "model.onnx"

    if not model_file.exists():
        pytest.fail(
            f"LLMLingua model not found at {model_file}. "
            "Run 'task download-models' to download models locally, "
            "or ensure the download-models workflow runs before tests in CI."
        )

    yield


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
