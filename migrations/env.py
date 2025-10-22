from alembic import context
from sqlalchemy import create_engine, event, pool

from mcp_optimizer.config import get_config

# this is the Alembic Config object, which provides to the values in pyproject.toml
alembic_config = context.config

# Needs to be `get_alembic_option` to get the config from pyproject.toml
# `get_main_option` is used solely for alembic.ini files
if alembic_config.get_alembic_option("db_url", default=None) is None:
    alembic_config.set_main_option("db_url", get_config().db_url)

# We're not using SQLAlchemy ORM or autogenerate
# All migrations will be written manually using raw SQL
target_metadata = None


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    context.configure(
        url=alembic_config.get_main_option("db_url"),
        # url=DB_URL,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    import sqlite3

    import sqlite_vec

    db_url = alembic_config.get_main_option("db_url")
    print(f"[DEBUG] Alembic using db_url: {db_url}")

    if db_url is None:
        raise ValueError("db_url is required for migrations")

    # Add SQLite-specific connection args for read-only root filesystem environments
    engine_kwargs = {"poolclass": pool.NullPool}
    if db_url.startswith("sqlite://"):
        # Ensure absolute path
        if db_url.startswith("sqlite:///"):
            path = db_url.replace("sqlite:///", "")
            if not path.startswith("/"):
                path = f"/{path}"
                db_url = f"sqlite:///{path}"
            print(f"[DEBUG] Using SQLite path: {path}")

        engine_kwargs["connect_args"] = {
            "check_same_thread": False,
            "timeout": 30.0,
        }

    connectable = create_engine(db_url, **engine_kwargs)

    # Enable loading extensions for SQLite
    if db_url.startswith("sqlite://"):

        @event.listens_for(connectable, "connect")
        def _on_connect(dbapi_conn, connection_record):
            """Load sqlite-vec extension on connection."""
            if isinstance(dbapi_conn, sqlite3.Connection):
                dbapi_conn.enable_load_extension(True)
                dbapi_conn.load_extension(sqlite_vec.loadable_path())
                dbapi_conn.enable_load_extension(False)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
