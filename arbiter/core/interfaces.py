"""Core interfaces and protocols for Arbiter components.

This module defines the abstract interfaces that all Arbiter components
must implement. Using protocols ensures type safety while allowing
flexibility in implementation.

## Key Interfaces:

- **BaseEvaluator**: Interface for all evaluators
- **StorageBackend**: Interface for storage implementations
- **MiddlewareProtocol**: Already defined in middleware.py

## Usage:

    >>> from arbiter.core.interfaces import BaseEvaluator
    >>>
    >>> class CustomEvaluator(BaseEvaluator):
    ...     @property
    ...     def name(self) -> str:
    ...         return "custom"
    ...
    ...     async def evaluate(self, output, reference):
    ...         # Implementation here
    ...         pass
"""

from abc import ABC, abstractmethod
from typing import Optional

from .models import EvaluationResult, Score

__all__ = ["BaseEvaluator", "StorageBackend"]


class BaseEvaluator(ABC):
    """Abstract base class for all evaluators.

    All evaluators must inherit from this class and implement the
    required methods. This ensures a consistent interface across
    different evaluation strategies.

    Example:
        >>> class SemanticEvaluator(BaseEvaluator):
        ...     @property
        ...     def name(self) -> str:
        ...         return "semantic"
        ...
        ...     async def evaluate(
        ...         self,
        ...         output: str,
        ...         reference: Optional[str],
        ...         criteria: Optional[str]
        ...     ) -> Score:
        ...         # Compute semantic similarity
        ...         score_value = await self._compute_similarity(output, reference)
        ...         return Score(
        ...             name="semantic_similarity",
        ...             value=score_value,
        ...             confidence=0.9
        ...         )
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the evaluator's unique identifier.

        This name is used for registration, configuration, and metrics.

        Returns:
            Unique string identifier for this evaluator

        Example:
            >>> evaluator = SemanticEvaluator()
            >>> print(evaluator.name)
            'semantic'
        """

    @abstractmethod
    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> Score:
        """Evaluate an output against reference or criteria.

        This is the main method that performs the actual evaluation.
        It must return a Score object with the computed value.

        Args:
            output: The LLM output to evaluate
            reference: Optional reference text for comparison
            criteria: Optional evaluation criteria for reference-free evaluation

        Returns:
            Score object with the evaluation result

        Raises:
            EvaluatorError: If evaluation fails

        Example:
            >>> score = await evaluator.evaluate(
            ...     output="Paris is the capital of France",
            ...     reference="The capital of France is Paris"
            ... )
            >>> print(f"Score: {score.value}")
            Score: 0.95
        """


class StorageBackend(ABC):
    """Abstract base class for storage backends.

    Defines the interface for persisting and retrieving evaluation
    results. Implementations can use various storage systems like
    memory, files, databases, etc.

    Example:
        >>> class RedisStorage(StorageBackend):
        ...     async def save(self, result: EvaluationResult) -> str:
        ...         result_id = generate_id()
        ...         await redis_client.set(result_id, result.model_dump_json())
        ...         return result_id
        ...
        ...     async def load(self, result_id: str) -> Optional[EvaluationResult]:
        ...         data = await redis_client.get(result_id)
        ...         if data:
        ...             return EvaluationResult.model_validate_json(data)
        ...         return None
    """

    @abstractmethod
    async def save(self, result: EvaluationResult) -> str:
        """Save an evaluation result.

        Persists the result and returns a unique identifier that can
        be used to retrieve it later.

        Args:
            result: The evaluation result to save

        Returns:
            Unique identifier for the saved result

        Raises:
            StorageError: If save operation fails

        Example:
            >>> result_id = await storage.save(eval_result)
            >>> print(f"Saved as: {result_id}")
            Saved as: eval_12345
        """

    @abstractmethod
    async def load(self, result_id: str) -> Optional[EvaluationResult]:
        """Load an evaluation result by ID.

        Retrieves a previously saved result using its unique identifier.

        Args:
            result_id: The unique identifier from save()

        Returns:
            The evaluation result if found, None otherwise

        Raises:
            StorageError: If load operation fails

        Example:
            >>> result = await storage.load("eval_12345")
            >>> if result:
            ...     print(f"Score: {result.overall_score}")
        """

    @abstractmethod
    async def delete(self, result_id: str) -> bool:
        """Delete an evaluation result by ID.

        Removes a previously saved result from storage.

        Args:
            result_id: The unique identifier of the result to delete

        Returns:
            True if deleted, False if not found

        Raises:
            StorageError: If delete operation fails

        Example:
            >>> deleted = await storage.delete("eval_12345")
            >>> print(f"Deleted: {deleted}")
            Deleted: True
        """

    @abstractmethod
    async def list_ids(self, limit: int = 100, offset: int = 0) -> list[str]:
        """List stored result IDs.

        Returns a paginated list of result identifiers.

        Args:
            limit: Maximum number of IDs to return
            offset: Number of IDs to skip

        Returns:
            List of result IDs

        Raises:
            StorageError: If list operation fails

        Example:
            >>> ids = await storage.list_ids(limit=10)
            >>> print(f"Found {len(ids)} results")
            Found 10 results
        """
