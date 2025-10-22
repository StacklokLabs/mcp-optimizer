from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import sqlite_vec
import structlog
from alembic import command
from alembic.config import Config
from sqlalchemy import Row, Sequence, text
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

logger = structlog.get_logger(__name__)


def _normalize_sqlite_url(url: str) -> str:
    """Normalize SQLite URL to use absolute paths."""
    if url.startswith("sqlite://") or url.startswith("sqlite+aiosqlite://"):
        prefix = "sqlite+aiosqlite:///" if "aiosqlite" in url else "sqlite:///"
        if url.startswith(prefix):
            path = url.replace(prefix, "")
            if not path.startswith("/"):
                path = f"/{path}"
                url = f"{prefix}{path}"
                logger.info(f"Normalized SQLite URL path to absolute: {path}")
    return url


class DatabaseConfig:
    """Async database configuration and connection management."""

    def __init__(self, database_url: str):
        """Initialize database configuration.

        Args:
            database_url: The database URL to connect to.
        """
        self.database_url = _normalize_sqlite_url(database_url)
        # Configure connection pool for better concurrency handling
        # SQLite doesn't support connection pooling parameters
        engine_kwargs = {"echo": False}
        if not self.database_url.startswith("sqlite"):
            engine_kwargs.update(
                {
                    # Connection pool settings for better performance and reliability
                    "pool_size": 10,  # Number of connections to maintain in the pool
                    "max_overflow": 20,  # Additional connections beyond pool_size
                    "pool_timeout": 30,  # Seconds to wait for connection from pool
                    "pool_recycle": 3600,  # Recycle connections after 1 hour
                    "pool_pre_ping": True,  # Validate connections before use
                }
            )

        self.engine: AsyncEngine = create_async_engine(self.database_url, **engine_kwargs)

    async def _ensure_sqlite_vec_loaded(self, connection: AsyncConnection):
        """Ensure sqlite-vec is loaded for an aiosqlite connection."""
        if not self.database_url.startswith("sqlite+aiosqlite://"):
            return

        try:
            # Check if sqlite-vec is already loaded
            await connection.execute(text("SELECT vec_version()"))
        except Exception as exc:
            # sqlite-vec not loaded, need to load it
            logger.debug("Loading sqlite-vec extension")
            try:
                # Get the raw aiosqlite connection
                raw_conn = await connection.get_raw_connection()
                # aiosqlite connection has async methods
                aio_conn = raw_conn.driver_connection
                if aio_conn is None:
                    raise RuntimeError("Failed to get raw aiosqlite connection") from exc

                # Load sqlite-vec using aiosqlite async methods
                extension_path = sqlite_vec.loadable_path()
                logger.debug(f"Loading sqlite-vec from path: {extension_path}")
                await aio_conn.enable_load_extension(True)
                await aio_conn.load_extension(extension_path)
                await aio_conn.enable_load_extension(False)
                logger.debug("sqlite-vec extension loaded successfully")
            except Exception as load_exc:
                logger.error(f"Failed to load sqlite-vec extension: {load_exc}")
                raise RuntimeError("Failed to load sqlite-vec extension") from load_exc

        # Set PRAGMA foreign_keys
        await connection.execute(text("PRAGMA foreign_keys=ON"))

    @asynccontextmanager
    async def _get_connection(self) -> AsyncGenerator[AsyncConnection, None]:
        """Get a database connection context manager."""
        async with self.engine.connect() as connection:
            await self._ensure_sqlite_vec_loaded(connection)
            yield connection

    @asynccontextmanager
    async def begin_transaction(self) -> AsyncGenerator[AsyncConnection, None]:
        """Begin a database transaction and return connection context manager.

        Uses IMMEDIATE transaction to prevent read-write conflicts and ensure
        SELECT queries wait for transaction completion.
        """
        async with self.engine.connect() as connection:
            await self._ensure_sqlite_vec_loaded(connection)

            # Set busy timeout to handle waiting for locks
            await connection.execute(text("PRAGMA busy_timeout = 30000"))

            # Begin IMMEDIATE transaction to acquire write lock immediately
            await connection.execute(text("BEGIN IMMEDIATE"))

            try:
                yield connection
                # Commit transaction if no exception occurred
                await connection.execute(text("COMMIT"))
            except Exception:
                # Rollback transaction on any exception
                await connection.execute(text("ROLLBACK"))
                raise

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        conn: AsyncConnection | None = None,
    ) -> Sequence[Row]:
        """Execute a SQL query and return results.

        Args:
            query: SQL query to execute
            params: Query parameters
            conn: Optional existing connection (for transactions)
        """
        if conn is not None:
            # Use provided connection (within transaction)
            result = await conn.execute(text(query), params or {})
            return result.fetchall()
        else:
            # Create new connection
            async with self._get_connection() as connection:
                result = await connection.execute(text(query), params or {})
                return result.fetchall()

    async def execute_non_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Execute a SQL query without returning results.

        Args:
            query: SQL query to execute
            params: Query parameters
            conn: Optional existing connection (for transactions)
        """
        if conn is not None:
            # Use provided connection (within transaction, no commit needed)
            await conn.execute(text(query), params or {})
        else:
            # Create new connection and commit
            async with self._get_connection() as connection:
                await connection.execute(text(query), params or {})
                await connection.commit()

    def get_database_path(self) -> Path | None:
        """Get the database file path for SQLite databases."""
        if self.database_url.startswith("sqlite+aiosqlite:///"):
            db_path = self.database_url.replace("sqlite+aiosqlite:///", "")
            return Path(db_path)
        return None

    async def close(self) -> None:
        """Close the database engine and clean up resources."""
        try:
            await self.engine.dispose()
            logger.info("Database engine closed successfully")
        except Exception as e:
            logger.error(f"Error closing database engine: {e}")


def run_migrations() -> None:
    """Run database migrations using Alembic."""
    # Try Docker path first, then fall back to development path
    docker_migrations = Path("/app/migrations")
    if docker_migrations.exists():
        migrations_dir = docker_migrations
    else:
        project_root = Path(__file__).parent.parent.parent.parent
        migrations_dir = project_root / "migrations"

    alembic_config = Config()
    alembic_config.set_main_option("script_location", str(migrations_dir.resolve()))
    command.upgrade(alembic_config, "head")
    logger.info("Database migrations completed successfully.")
