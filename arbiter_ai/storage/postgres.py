"""PostgreSQL storage backend for evaluation results.

Uses asyncpg for high-performance async PostgreSQL connections.
Creates 'arbiter' schema to isolate from other applications in shared databases.
Reads DATABASE_URL from environment for database connection.
"""

import json
import logging
import os
import uuid
from typing import Any, Optional

import asyncpg

from arbiter_ai.core.models import BatchEvaluationResult, EvaluationResult
from arbiter_ai.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
)

logger = logging.getLogger(__name__)


class PostgresStorage(StorageBackend):
    """PostgreSQL storage backend with connection pooling.

    Uses 'arbiter' schema for table isolation from other applications.
    Compatible with any PostgreSQL provider (self-hosted, AWS RDS, Supabase, etc.).

    Tables created in arbiter schema:
        - evaluation_results: Single evaluation results
        - batch_evaluation_results: Batch evaluation results

    Args:
        database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        schema: Schema name (default: 'arbiter')
        min_pool_size: Minimum connection pool size (default: 2)
        max_pool_size: Maximum connection pool size (default: 10)

    Example:
        >>> storage = PostgresStorage()
        >>> async with storage:
        >>>     result_id = await storage.save_result(eval_result)
        >>>     retrieved = await storage.get_result(result_id)
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        schema: str = "arbiter",
        min_pool_size: int = 2,
        max_pool_size: int = 10,
    ):
        self.database_url = database_url or os.getenv("DATABASE_URL")
        if not self.database_url:
            raise ValueError(
                "DATABASE_URL must be provided or set in environment variables"
            )

        self.schema = schema
        self.min_pool_size = min_pool_size
        self.max_pool_size = max_pool_size
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool.

        Note: Schema and tables should be created via Alembic migrations.
        Run 'alembic upgrade head' before using this storage backend.
        """
        try:
            self.pool = await asyncpg.create_pool(
                self.database_url,
                min_size=self.min_pool_size,
                max_size=self.max_pool_size,
                command_timeout=60,
            )
            logger.info(
                f"PostgreSQL connection pool created (min={self.min_pool_size}, max={self.max_pool_size})"
            )

            # Verify schema exists
            await self._verify_schema()

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise ConnectionError(f"PostgreSQL connection failed: {e}") from e

    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("PostgreSQL connection pool closed")

    async def _verify_schema(self) -> None:
        """Verify arbiter schema exists.

        Raises ConnectionError if schema doesn't exist.
        Users should run 'alembic upgrade head' to create schema.
        """
        if not self.pool:
            raise ConnectionError("Not connected to database")

        async with self.pool.acquire() as conn:
            # Check if schema exists
            result = await conn.fetchval(
                """
                SELECT EXISTS(
                    SELECT 1 FROM information_schema.schemata
                    WHERE schema_name = $1
                )
                """,
                self.schema,
            )

            if not result:
                raise ConnectionError(
                    f"Schema '{self.schema}' does not exist. "
                    f"Please run 'alembic upgrade head' to create database schema."
                )

            logger.info(f"Verified schema '{self.schema}' exists")

    async def save_result(
        self,
        result: EvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save evaluation result to PostgreSQL.

        Args:
            result: EvaluationResult to save
            metadata: Optional metadata (e.g., router_decision_id, user_id)

        Returns:
            UUID string of saved result
        """
        if not self.pool:
            raise ConnectionError("Not connected to database")

        try:
            # Serialize result to JSON
            result_data = result.model_dump(mode="json")

            # Extract fields for indexing
            overall_score = result.overall_score
            evaluators = [score.name for score in result.scores]
            model = result.interactions[0].model if result.interactions else None

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {self.schema}.evaluation_results
                    (result_data, metadata, overall_score, evaluators, model)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING id
                    """,
                    json.dumps(result_data),
                    json.dumps(metadata) if metadata else None,
                    overall_score,
                    evaluators,
                    model,
                )

                result_id = str(row["id"])
                logger.debug(f"Saved evaluation result: {result_id}")
                return result_id

        except Exception as e:
            logger.error(f"Failed to save result: {e}")
            raise SaveError(f"Failed to save result: {e}") from e

    async def save_batch_result(
        self,
        result: BatchEvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save batch evaluation result to PostgreSQL.

        Args:
            result: BatchEvaluationResult to save
            metadata: Optional metadata

        Returns:
            UUID string of saved batch
        """
        if not self.pool:
            raise ConnectionError("Not connected to database")

        try:
            # Serialize result to JSON
            result_data = result.model_dump(mode="json")

            # Calculate statistics
            avg_score = (
                sum(r.overall_score for r in result.results if r)
                / result.successful_items
                if result.successful_items > 0
                else None
            )

            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    INSERT INTO {self.schema}.batch_evaluation_results
                    (result_data, metadata, total_items, successful_items, failed_items, avg_score)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    RETURNING id
                    """,
                    json.dumps(result_data),
                    json.dumps(metadata) if metadata else None,
                    result.total_items,
                    result.successful_items,
                    result.failed_items,
                    avg_score,
                )

                batch_id = str(row["id"])
                logger.debug(f"Saved batch result: {batch_id}")
                return batch_id

        except Exception as e:
            logger.error(f"Failed to save batch result: {e}")
            raise SaveError(f"Failed to save batch result: {e}") from e

    async def get_result(self, result_id: str) -> Optional[EvaluationResult]:
        """Retrieve evaluation result by ID.

        Args:
            result_id: UUID string

        Returns:
            EvaluationResult if found, None otherwise
        """
        if not self.pool:
            raise ConnectionError("Not connected to database")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT result_data FROM {self.schema}.evaluation_results
                    WHERE id = $1
                    """,
                    uuid.UUID(result_id),
                )

                if not row:
                    return None

                result_data = json.loads(row["result_data"])
                return EvaluationResult.model_validate(result_data)

        except Exception as e:
            logger.error(f"Failed to retrieve result {result_id}: {e}")
            raise RetrievalError(f"Failed to retrieve result: {e}") from e

    async def get_batch_result(self, batch_id: str) -> Optional[BatchEvaluationResult]:
        """Retrieve batch evaluation result by ID.

        Args:
            batch_id: UUID string

        Returns:
            BatchEvaluationResult if found, None otherwise
        """
        if not self.pool:
            raise ConnectionError("Not connected to database")

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT result_data FROM {self.schema}.batch_evaluation_results
                    WHERE id = $1
                    """,
                    uuid.UUID(batch_id),
                )

                if not row:
                    return None

                result_data = json.loads(row["result_data"])
                return BatchEvaluationResult.model_validate(result_data)

        except Exception as e:
            logger.error(f"Failed to retrieve batch {batch_id}: {e}")
            raise RetrievalError(f"Failed to retrieve batch: {e}") from e
