"""Base storage interface for evaluation results.

All storage backends must implement this async interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from arbiter_ai.core.models import BatchEvaluationResult, EvaluationResult


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    All storage operations are async to support high-performance I/O.
    Implementations must handle connection pooling and error recovery.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to storage backend.

        Should set up connection pools and verify connectivity.
        Called automatically before first use.

        Raises:
            StorageError: If connection fails
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close connections and cleanup resources.

        Should gracefully close connection pools.
        Called on shutdown or context manager exit.
        """
        pass

    @abstractmethod
    async def save_result(
        self,
        result: EvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save a single evaluation result.

        Args:
            result: Evaluation result to persist
            metadata: Optional additional metadata (e.g., user_id, session_id)

        Returns:
            Unique identifier for the saved result

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def save_batch_result(
        self,
        result: BatchEvaluationResult,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Save batch evaluation results.

        Args:
            result: Batch evaluation result to persist
            metadata: Optional additional metadata

        Returns:
            Unique identifier for the saved batch

        Raises:
            StorageError: If save operation fails
        """
        pass

    @abstractmethod
    async def get_result(self, result_id: str) -> Optional[EvaluationResult]:
        """Retrieve evaluation result by ID.

        Args:
            result_id: Unique identifier from save_result()

        Returns:
            EvaluationResult if found, None otherwise

        Raises:
            StorageError: If retrieval operation fails
        """
        pass

    @abstractmethod
    async def get_batch_result(self, batch_id: str) -> Optional[BatchEvaluationResult]:
        """Retrieve batch evaluation result by ID.

        Args:
            batch_id: Unique identifier from save_batch_result()

        Returns:
            BatchEvaluationResult if found, None otherwise

        Raises:
            StorageError: If retrieval operation fails
        """
        pass

    async def __aenter__(self) -> "StorageBackend":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


class StorageError(Exception):
    """Base exception for storage backend errors."""

    pass


class ConnectionError(StorageError):
    """Connection to storage backend failed."""

    pass


class SaveError(StorageError):
    """Failed to save result to storage."""

    pass


class RetrievalError(StorageError):
    """Failed to retrieve result from storage."""

    pass
