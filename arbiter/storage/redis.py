"""Redis storage backend for fast evaluation result caching.

Uses Redis for high-performance caching of evaluation results.
Reuses REDIS_URL from environment (shares Redis instance with conduit).
"""

import json
import logging
import os
from typing import Any, Optional

import redis.asyncio as redis

from arbiter.core.models import BatchEvaluationResult, EvaluationResult
from arbiter.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
)

logger = logging.getLogger(__name__)


class RedisStorage(StorageBackend):
    """Redis storage backend for fast evaluation result caching.

    Uses Redis for temporary storage with TTL (Time To Live).
    Ideal for caching frequently accessed evaluation results.

    Args:
        redis_url: Redis connection string (defaults to REDIS_URL env var)
        ttl: Time to live in seconds (default: 86400 = 24 hours)
        key_prefix: Prefix for all Redis keys (default: 'arbiter:')

    Example:
        >>> storage = RedisStorage(ttl=3600)  # 1 hour cache
        >>> async with storage:
        >>>     result_id = await storage.save_result(eval_result)
        >>>     cached = await storage.get_result(result_id)  # Fast retrieval
    """

    def __init__(
        self,
        redis_url: Optional[str] = None,
        ttl: int = 86400,  # 24 hours default
        key_prefix: str = "arbiter:",
    ):
        self.redis_url = redis_url or os.getenv("REDIS_URL")
        if not self.redis_url:
            raise ValueError(
                "REDIS_URL must be provided or set in environment variables"
            )

        self.ttl = ttl
        self.key_prefix = key_prefix
        self.client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Establish Redis connection."""
        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )

            # Test connection
            await self.client.ping()
            logger.info(f"Redis connection established (TTL={self.ttl}s)")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionError(f"Redis connection failed: {e}") from e

    async def close(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("Redis connection closed")

    def _make_key(self, result_id: str, result_type: str = "result") -> str:
        """Generate Redis key with prefix.

        Args:
            result_id: Unique result identifier
            result_type: Type of result ('result' or 'batch')

        Returns:
            Redis key with prefix
        """
        return f"{self.key_prefix}{result_type}:{result_id}"

    async def save_result(
        self,
        result: EvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save evaluation result to Redis with TTL.

        Args:
            result: EvaluationResult to cache
            metadata: Optional metadata

        Returns:
            Result ID (generated from hash of result)
        """
        if not self.client:
            raise ConnectionError("Not connected to Redis")

        try:
            # Generate ID from result hash (deterministic)
            result_dict = result.model_dump(mode="json")
            result_id = str(hash(json.dumps(result_dict, sort_keys=True)))

            # Store result with metadata
            data = {"result": result_dict, "metadata": metadata}
            key = self._make_key(result_id, "result")

            await self.client.setex(
                key, self.ttl, json.dumps(data)
            )

            logger.debug(f"Cached evaluation result: {result_id} (TTL={self.ttl}s)")
            return result_id

        except Exception as e:
            logger.error(f"Failed to save result to Redis: {e}")
            raise SaveError(f"Failed to cache result: {e}") from e

    async def save_batch_result(
        self,
        result: BatchEvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save batch evaluation result to Redis with TTL.

        Args:
            result: BatchEvaluationResult to cache
            metadata: Optional metadata

        Returns:
            Batch ID (generated from hash)
        """
        if not self.client:
            raise ConnectionError("Not connected to Redis")

        try:
            # Generate ID from result hash
            result_dict = result.model_dump(mode="json")
            batch_id = str(hash(json.dumps(result_dict, sort_keys=True)))

            # Store result with metadata
            data = {"result": result_dict, "metadata": metadata}
            key = self._make_key(batch_id, "batch")

            await self.client.setex(
                key, self.ttl, json.dumps(data)
            )

            logger.debug(f"Cached batch result: {batch_id} (TTL={self.ttl}s)")
            return batch_id

        except Exception as e:
            logger.error(f"Failed to save batch to Redis: {e}")
            raise SaveError(f"Failed to cache batch: {e}") from e

    async def get_result(self, result_id: str) -> Optional[EvaluationResult]:
        """Retrieve evaluation result from Redis cache.

        Args:
            result_id: Result identifier

        Returns:
            EvaluationResult if cached, None if expired or not found
        """
        if not self.client:
            raise ConnectionError("Not connected to Redis")

        try:
            key = self._make_key(result_id, "result")
            data_str = await self.client.get(key)

            if not data_str:
                return None

            data = json.loads(data_str)
            return EvaluationResult.model_validate(data["result"])

        except Exception as e:
            logger.error(f"Failed to retrieve result from Redis: {e}")
            raise RetrievalError(f"Failed to retrieve cached result: {e}") from e

    async def get_batch_result(
        self, batch_id: str
    ) -> Optional[BatchEvaluationResult]:
        """Retrieve batch evaluation result from Redis cache.

        Args:
            batch_id: Batch identifier

        Returns:
            BatchEvaluationResult if cached, None if expired or not found
        """
        if not self.client:
            raise ConnectionError("Not connected to Redis")

        try:
            key = self._make_key(batch_id, "batch")
            data_str = await self.client.get(key)

            if not data_str:
                return None

            data = json.loads(data_str)
            return BatchEvaluationResult.model_validate(data["result"])

        except Exception as e:
            logger.error(f"Failed to retrieve batch from Redis: {e}")
            raise RetrievalError(f"Failed to retrieve cached batch: {e}") from e
