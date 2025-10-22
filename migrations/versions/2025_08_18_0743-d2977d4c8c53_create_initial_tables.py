"""create_initial_tables

Revision ID: d2977d4c8c53
Revises:
Create Date: 2025-08-18 07:43:42.898598+00:00

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d2977d4c8c53"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema - Drop old tables and create new separated structure."""
    op.execute("BEGIN TRANSACTION;")

    # Drop old tables and virtual tables if they exist
    op.execute("DROP TABLE IF EXISTS tool_fts")
    op.execute("DROP TABLE IF EXISTS tool_vectors")
    op.execute("DROP TABLE IF EXISTS server_vector")
    op.execute("DROP TABLE IF EXISTS tool")
    op.execute("DROP TABLE IF EXISTS mcpserver")

    # Create mcpservers_registry table
    op.execute("""
        CREATE TABLE mcpservers_registry (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            url TEXT,
            package TEXT,
            remote INTEGER NOT NULL,
            transport TEXT NOT NULL,
            description TEXT,
            server_embedding BLOB,
            "group" TEXT NOT NULL DEFAULT 'default',
            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            CHECK ((remote = 1 AND url IS NOT NULL) OR (remote = 0 AND package IS NOT NULL))
        )
    """)

    # Create unique partial indexes for registry servers
    op.execute("CREATE UNIQUE INDEX idx_registry_url ON mcpservers_registry(url) WHERE remote = 1")
    op.execute(
        "CREATE UNIQUE INDEX idx_registry_package ON mcpservers_registry(package) WHERE remote = 0"
    )
    op.execute("CREATE INDEX idx_registry_remote ON mcpservers_registry(remote)")

    # Create mcpservers_workload table
    op.execute("""
        CREATE TABLE mcpservers_workload (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            url TEXT NOT NULL,
            workload_identifier TEXT NOT NULL,
            remote INTEGER NOT NULL,
            transport TEXT NOT NULL,
            status TEXT NOT NULL,
            registry_server_id TEXT,
            registry_server_name TEXT,
            description TEXT,
            server_embedding BLOB,
            "group" TEXT NOT NULL DEFAULT 'default',
            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (registry_server_id)
                REFERENCES mcpservers_registry(id) ON DELETE SET NULL
        )
    """)

    # Create indexes for workload servers
    op.execute("CREATE INDEX idx_workload_registry_id ON mcpservers_workload(registry_server_id)")
    op.execute("CREATE INDEX idx_workload_remote ON mcpservers_workload(remote)")
    op.execute("CREATE INDEX idx_workload_status ON mcpservers_workload(status)")

    # Create tools_registry table
    op.execute("""
        CREATE TABLE tools_registry (
            id TEXT PRIMARY KEY,
            mcpserver_id TEXT NOT NULL,
            details TEXT NOT NULL,
            details_embedding BLOB,
            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mcpserver_id) REFERENCES mcpservers_registry(id) ON DELETE CASCADE
        )
    """)

    # Create index for registry tools
    op.execute("CREATE INDEX idx_tools_registry_server ON tools_registry(mcpserver_id)")

    # Create tools_workload table
    op.execute("""
        CREATE TABLE tools_workload (
            id TEXT PRIMARY KEY,
            mcpserver_id TEXT NOT NULL,
            details TEXT NOT NULL,
            details_embedding BLOB,
            token_count INTEGER NOT NULL DEFAULT 0,    -- Token count for LLM consumption
            last_updated TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (mcpserver_id) REFERENCES mcpservers_workload(id) ON DELETE CASCADE
        )
    """)

    # Create index for workload tools
    op.execute("CREATE INDEX idx_tools_workload_server ON tools_workload(mcpserver_id)")

    # Create virtual tables for registry (sqlite-vec and FTS5)
    # Note: vec0 uses cosine distance by default
    #
    # IMPORTANT - Embedding Dimension: The dimension is set to 384 to match the
    # FastEmbed model BAAI/bge-small-en-v1.5. If you change the embedding model
    # in EmbeddingManager to one with a different dimension, you MUST:
    # 1. Create a new migration to alter this dimension
    # 2. Re-embed all existing data with the new model
    # 3. Update corresponding dimension in all vector tables below
    op.execute("""
        CREATE VIRTUAL TABLE registry_server_vector
        USING vec0(
            server_id TEXT PRIMARY KEY,
            embedding FLOAT[384] distance_metric=cosine
        )
    """)

    op.execute("""
        CREATE VIRTUAL TABLE registry_tool_vectors
        USING vec0(
            tool_id TEXT PRIMARY KEY,
            embedding FLOAT[384] distance_metric=cosine
        )
    """)

    op.execute("""
        CREATE VIRTUAL TABLE registry_tool_fts
        USING fts5(
            tool_id UNINDEXED,
            mcp_server_name,
            tool_name,
            tool_description,
            tokenize='porter'
        )
    """)

    # Create virtual tables for workload (sqlite-vec and FTS5)
    op.execute("""
        CREATE VIRTUAL TABLE workload_server_vector
        USING vec0(
            server_id TEXT PRIMARY KEY,
            embedding FLOAT[384] distance_metric=cosine
        )
    """)

    op.execute("""
        CREATE VIRTUAL TABLE workload_tool_vectors
        USING vec0(
            tool_id TEXT PRIMARY KEY,
            embedding FLOAT[384] distance_metric=cosine
        )
    """)

    op.execute("""
        CREATE VIRTUAL TABLE workload_tool_fts
        USING fts5(
            tool_id UNINDEXED,
            mcp_server_name,
            tool_name,
            tool_description,
            tokenize='porter'
        )
    """)

    op.execute("COMMIT;")  # Commit the transaction


def downgrade() -> None:
    """Downgrade schema - Drop new tables."""
    op.execute("BEGIN TRANSACTION;")

    # Drop virtual tables first
    op.execute("DROP TABLE IF EXISTS workload_tool_fts")
    op.execute("DROP TABLE IF EXISTS workload_tool_vectors")
    op.execute("DROP TABLE IF EXISTS workload_server_vector")
    op.execute("DROP TABLE IF EXISTS registry_tool_fts")
    op.execute("DROP TABLE IF EXISTS registry_tool_vectors")
    op.execute("DROP TABLE IF EXISTS registry_server_vector")

    # Drop indexes
    op.execute("DROP INDEX IF EXISTS idx_tools_workload_server")
    op.execute("DROP INDEX IF EXISTS idx_tools_registry_server")
    op.execute("DROP INDEX IF EXISTS idx_workload_status")
    op.execute("DROP INDEX IF EXISTS idx_workload_remote")
    op.execute("DROP INDEX IF EXISTS idx_workload_registry_id")
    op.execute("DROP INDEX IF EXISTS idx_registry_remote")
    op.execute("DROP INDEX IF EXISTS idx_registry_package")
    op.execute("DROP INDEX IF EXISTS idx_registry_url")

    # Drop tables (order matters due to foreign keys)
    op.execute("DROP TABLE IF EXISTS tools_workload")
    op.execute("DROP TABLE IF EXISTS tools_registry")
    op.execute("DROP TABLE IF EXISTS mcpservers_workload")
    op.execute("DROP TABLE IF EXISTS mcpservers_registry")

    op.execute("COMMIT;")  # Commit the transaction
