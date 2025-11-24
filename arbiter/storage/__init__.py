"""Storage backends for evaluation results.

Available backends:
- PostgreSQL: Persistent storage with arbiter schema (requires DATABASE_URL)
- Redis: Fast caching with TTL (requires REDIS_URL)

Setup:
    1. Set DATABASE_URL and/or REDIS_URL in .env
    2. Run migrations: alembic upgrade head
    3. Use storage backends in evaluate() calls

Example:
    >>> from arbiter import evaluate
    >>> from arbiter.storage import PostgresStorage
    >>>
    >>> storage = PostgresStorage()
    >>> async with storage:
    >>>     result = await evaluate(
    >>>         output="...",
    >>>         reference="...",
    >>>         evaluators=["semantic"],
    >>>         storage=storage
    >>>     )
"""

from arbiter.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
    StorageError,
)
from arbiter.storage.postgres import PostgresStorage
from arbiter.storage.redis import RedisStorage

__all__ = [
    "StorageBackend",
    "StorageError",
    "ConnectionError",
    "SaveError",
    "RetrievalError",
    "PostgresStorage",
    "RedisStorage",
]
