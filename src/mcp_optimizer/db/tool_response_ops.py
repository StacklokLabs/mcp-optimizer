"""CRUD operations for tool responses KV store."""

import json
import uuid
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncConnection

from mcp_optimizer.db.config import DatabaseConfig
from mcp_optimizer.db.exceptions import DbNotFoundError
from mcp_optimizer.response_optimizer.models import ContentType, StoredToolResponse

logger = structlog.get_logger(__name__)


class ToolResponseOps:
    """Operations for the tool_responses KV store table."""

    TABLE_NAME = "tool_responses"

    def __init__(self, db: DatabaseConfig):
        """Initialize with database configuration."""
        self.db = db

    async def create_tool_response(
        self,
        tool_name: str,
        original_content: str,
        content_type: ContentType,
        response_id: str | None = None,
        session_key: str | None = None,
        ttl_seconds: int = 300,
        metadata: dict | None = None,
        conn: AsyncConnection | None = None,
    ) -> StoredToolResponse:
        """
        Store a tool response in the KV store.

        Args:
            tool_name: Name of the tool that generated the response
            original_content: The original unmodified content
            content_type: The detected content type
            response_id: Optional response ID. If not provided, a new UUID is generated.
            session_key: Optional session key for grouping related responses.
                        If not provided, defaults to the response_id.
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
            metadata: Optional additional metadata
            conn: Optional existing connection (for transactions)

        Returns:
            The stored tool response with generated ID
        """
        if response_id is None:
            response_id = str(uuid.uuid4())
        # Default session_key to response_id if not provided
        actual_session_key = session_key if session_key is not None else response_id
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl_seconds)

        query = f"""
            INSERT INTO {self.TABLE_NAME}
            (id, session_key, tool_name, original_content, content_type,
             created_at, expires_at, metadata)
            VALUES (:id, :session_key, :tool_name, :original_content, :content_type,
                    :created_at, :expires_at, :metadata)
        """  # nosec B608 - TABLE_NAME is a code-controlled constant, not user input

        params = {
            "id": response_id,
            "session_key": actual_session_key,
            "tool_name": tool_name,
            "original_content": original_content,
            "content_type": content_type.value,
            "created_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "metadata": json.dumps(metadata or {}),
        }

        await self.db.execute_non_query(query, params, conn=conn)

        logger.debug(
            "Stored tool response",
            response_id=response_id,
            session_key=actual_session_key,
            tool_name=tool_name,
            content_type=content_type.value,
            expires_at=expires_at.isoformat(),
        )

        return StoredToolResponse(
            id=response_id,
            session_key=actual_session_key,
            tool_name=tool_name,
            original_content=original_content,
            content_type=content_type,
            created_at=now,
            expires_at=expires_at,
            metadata=metadata or {},
        )

    async def get_tool_response(
        self,
        response_id: str,
        conn: AsyncConnection | None = None,
    ) -> StoredToolResponse:
        """
        Retrieve a tool response by ID.

        Args:
            response_id: The response UUID
            conn: Optional existing connection (for transactions)

        Returns:
            The stored tool response

        Raises:
            DbNotFoundError: If the response is not found or has expired
        """
        query = f"""
            SELECT id, session_key, tool_name, original_content, content_type,
                   created_at, expires_at, metadata
            FROM {self.TABLE_NAME}
            WHERE id = :id
        """  # nosec B608 - TABLE_NAME is a code-controlled constant, not user input

        results = await self.db.execute_query(query, {"id": response_id}, conn=conn)

        if not results:
            raise DbNotFoundError(f"Tool response with ID {response_id} not found")

        row = results[0]._mapping

        # Check if expired
        expires_at = datetime.fromisoformat(row["expires_at"])
        if expires_at < datetime.now(timezone.utc):
            # Clean up expired entry
            await self._delete_response(response_id, conn=conn)
            raise DbNotFoundError(f"Tool response with ID {response_id} has expired")

        return StoredToolResponse(
            id=row["id"],
            session_key=row["session_key"],
            tool_name=row["tool_name"],
            original_content=row["original_content"],
            content_type=ContentType(row["content_type"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            expires_at=expires_at,
            metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        )

    async def get_responses_by_session(
        self,
        session_key: str,
        conn: AsyncConnection | None = None,
    ) -> list[StoredToolResponse]:
        """
        Retrieve all non-expired responses for a session.

        Args:
            session_key: The session key
            conn: Optional existing connection (for transactions)

        Returns:
            List of stored tool responses
        """
        now = datetime.now(timezone.utc).isoformat()
        query = f"""
            SELECT id, session_key, tool_name, original_content, content_type,
                   created_at, expires_at, metadata
            FROM {self.TABLE_NAME}
            WHERE session_key = :session_key AND expires_at > :now
            ORDER BY created_at DESC
        """  # nosec B608 - TABLE_NAME is a code-controlled constant, not user input

        results = await self.db.execute_query(
            query, {"session_key": session_key, "now": now}, conn=conn
        )

        return [
            StoredToolResponse(
                id=row._mapping["id"],
                session_key=row._mapping["session_key"],
                tool_name=row._mapping["tool_name"],
                original_content=row._mapping["original_content"],
                content_type=ContentType(row._mapping["content_type"]),
                created_at=datetime.fromisoformat(row._mapping["created_at"]),
                expires_at=datetime.fromisoformat(row._mapping["expires_at"]),
                metadata=json.loads(row._mapping["metadata"]) if row._mapping["metadata"] else {},
            )
            for row in results
        ]

    async def cleanup_expired(
        self,
        conn: AsyncConnection | None = None,
    ) -> int:
        """
        Delete all expired entries from the KV store.

        Args:
            conn: Optional existing connection (for transactions)

        Returns:
            Number of entries deleted
        """
        now = datetime.now(timezone.utc).isoformat()

        # First count how many will be deleted
        count_query = f"""
            SELECT COUNT(*) FROM {self.TABLE_NAME} WHERE expires_at <= :now
        """  # nosec B608 - TABLE_NAME is a code-controlled constant, not user input
        result = await self.db.execute_query(count_query, {"now": now}, conn=conn)
        count = result[0][0] if result else 0

        # Then delete
        delete_query = f"""
            DELETE FROM {self.TABLE_NAME} WHERE expires_at <= :now
        """  # nosec B608 - TABLE_NAME is a code-controlled constant, not user input
        await self.db.execute_non_query(delete_query, {"now": now}, conn=conn)

        if count > 0:
            logger.info("Cleaned up expired tool responses", count=count)

        return count

    async def _delete_response(
        self,
        response_id: str,
        conn: AsyncConnection | None = None,
    ) -> None:
        """Delete a single response by ID."""
        query = f"DELETE FROM {self.TABLE_NAME} WHERE id = :id"  # nosec B608
        await self.db.execute_non_query(query, {"id": response_id}, conn=conn)
