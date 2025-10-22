"""Tests for database configuration and sqlite-vec extension loading."""

import tempfile
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config

from mcp_optimizer.db.config import DatabaseConfig


@pytest_asyncio.fixture
async def test_db_config():
    """Create a temporary database configuration for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp_file:
        test_db_path = Path(tmp_file.name)

    test_db_url = f"sqlite+aiosqlite:///{test_db_path}"

    # Run migrations first
    project_root = Path(__file__).parent.parent
    migrations_dir = project_root / "migrations"

    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(migrations_dir))
    alembic_config.set_main_option("db_url", f"sqlite:///{test_db_path}")

    command.upgrade(alembic_config, "head")

    # Create DatabaseConfig instance
    db_config = DatabaseConfig(test_db_url)

    yield db_config

    # Cleanup
    try:
        await db_config.close()
    except Exception:
        # Ignore cleanup errors
        pass
    test_db_path.unlink(missing_ok=True)


class TestDatabaseConfiguration:
    """Test database configuration functionality."""

    @pytest.mark.asyncio
    async def test_foreign_keys_enabled(self, test_db_config):
        """Test that foreign keys are enabled in SQLite."""
        result = await test_db_config.execute_query("PRAGMA foreign_keys")
        assert len(result) == 1
        assert result[0][0] == 1, "Foreign keys should be enabled (value should be 1)"

    @pytest.mark.asyncio
    async def test_sqlite_vec_extension_loaded(self, test_db_config):
        """Test that sqlite-vec extension is properly loaded."""
        # Test that vec_version() function is available
        result = await test_db_config.execute_query("SELECT vec_version()")
        assert len(result) == 1
        version = result[0][0]
        assert version is not None
        assert isinstance(version, str)
        assert version.startswith("v"), f"Version should start with 'v', got: {version}"

    @pytest.mark.asyncio
    async def test_vec0_virtual_table_creation(self, test_db_config):
        """Test that vec0 virtual tables can be created."""
        # Create a test virtual table
        await test_db_config.execute_non_query("""
            CREATE VIRTUAL TABLE test_vectors USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[3]
            )
        """)

        # Verify table was created by checking schema
        result = await test_db_config.execute_query("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='test_vectors'
        """)
        assert len(result) == 1
        assert result[0][0] == "test_vectors"

        # Clean up
        await test_db_config.execute_non_query("DROP TABLE test_vectors")

    @pytest.mark.asyncio
    async def test_vector_operations(self, test_db_config):
        """Test basic vector operations with sqlite-vec."""
        # Create virtual table
        await test_db_config.execute_non_query("""
            CREATE VIRTUAL TABLE test_vectors USING vec0(
                id TEXT PRIMARY KEY,
                embedding float[3]
            )
        """)

        # Insert a vector using vec_f32 function
        await test_db_config.execute_non_query("""
            INSERT INTO test_vectors(id, embedding)
            VALUES ('test1', vec_f32('[0.1, 0.2, 0.3]'))
        """)

        # Insert another vector
        await test_db_config.execute_non_query("""
            INSERT INTO test_vectors(id, embedding)
            VALUES ('test2', vec_f32('[0.4, 0.5, 0.6]'))
        """)

        # Query vectors
        result = await test_db_config.execute_query("""
            SELECT id FROM test_vectors ORDER BY id
        """)
        assert len(result) == 2
        assert result[0][0] == "test1"
        assert result[1][0] == "test2"

        # Test distance calculation
        result = await test_db_config.execute_query("""
            SELECT id, vec_distance_cosine(embedding, vec_f32('[0.1, 0.2, 0.3]')) as distance
            FROM test_vectors
            ORDER BY distance
        """)
        assert len(result) == 2
        # First result should be test1 with distance close to 0
        assert result[0][0] == "test1"
        assert abs(result[0][1]) < 0.001, f"Distance should be close to 0, got: {result[0][1]}"

        # Clean up
        await test_db_config.execute_non_query("DROP TABLE test_vectors")

    @pytest.mark.asyncio
    async def test_multiple_connections_maintain_extensions(self, test_db_config):
        """Test that sqlite-vec extensions are maintained across multiple connections."""
        # Test with first connection
        result1 = await test_db_config.execute_query("SELECT vec_version()")
        version1 = result1[0][0]

        # Create another query (which may use a different connection from pool)
        result2 = await test_db_config.execute_query("SELECT vec_version()")
        version2 = result2[0][0]

        # Both should work and return the same version
        assert version1 == version2
        assert version1.startswith("v")

    @pytest.mark.asyncio
    async def test_foreign_key_constraint_enforcement(self, test_db_config):
        """Test that foreign key constraints are actually enforced."""
        # Create parent table
        await test_db_config.execute_non_query("""
            CREATE TABLE test_parent (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)

        # Create child table with foreign key
        await test_db_config.execute_non_query("""
            CREATE TABLE test_child (
                id INTEGER PRIMARY KEY,
                parent_id INTEGER,
                value TEXT,
                FOREIGN KEY (parent_id) REFERENCES test_parent(id)
            )
        """)

        # Insert parent record
        await test_db_config.execute_non_query("""
            INSERT INTO test_parent (id, name) VALUES (1, 'test_parent')
        """)

        # Insert child record with valid foreign key - should succeed
        await test_db_config.execute_non_query("""
            INSERT INTO test_child (id, parent_id, value) VALUES (1, 1, 'test_child')
        """)

        # Try to insert child record with invalid foreign key - should fail
        with pytest.raises(Exception) as exc_info:
            await test_db_config.execute_non_query("""
                INSERT INTO test_child (id, parent_id, value) VALUES (2, 99, 'invalid_child')
            """)

        # Verify it's a foreign key constraint error
        error_message = str(exc_info.value).lower()
        assert any(
            keyword in error_message for keyword in ["foreign key", "constraint", "violat"]
        ), f"Expected foreign key constraint error, got: {exc_info.value}"

        # Clean up
        await test_db_config.execute_non_query("DROP TABLE test_child")
        await test_db_config.execute_non_query("DROP TABLE test_parent")

    @pytest.mark.asyncio
    async def test_database_connection_initialization(self, test_db_config):
        """Test that database connections are properly initialized."""
        # Test that we can perform basic database operations
        result = await test_db_config.execute_query("SELECT 1 as test_value")
        assert len(result) == 1
        assert result[0][0] == 1

        # Test that pragma settings are applied
        result = await test_db_config.execute_query("PRAGMA foreign_keys")
        assert result[0][0] == 1

        # Test that sqlite-vec functions are available
        result = await test_db_config.execute_query("SELECT vec_version()")
        assert result[0][0].startswith("v")
