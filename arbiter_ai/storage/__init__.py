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
    >>> from arbiter_ai.storage import PostgresStorage
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

from arbiter_ai.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
    StorageError,
)

__all__ = [
    "StorageBackend",
    "StorageError",
    "ConnectionError",
    "SaveError",
    "RetrievalError",
    "PostgresStorage",
    "RedisStorage",
]


def __getattr__(name: str):
    """Lazy import for optional storage backends.

    This allows importing from arbiter_ai.storage without requiring
    optional dependencies (asyncpg, redis) to be installed.
    """
    if name == "PostgresStorage":
        try:
            from arbiter_ai.storage.postgres import PostgresStorage

            return PostgresStorage
        except ImportError as e:
            raise ImportError(
                "PostgresStorage requires asyncpg. Install with: pip install arbiter[postgres]"
            ) from e
    elif name == "RedisStorage":
        try:
            from arbiter_ai.storage.redis import RedisStorage

            return RedisStorage
        except ImportError as e:
            raise ImportError(
                "RedisStorage requires redis. Install with: pip install arbiter[redis]"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
