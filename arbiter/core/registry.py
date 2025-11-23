"""Evaluator registry for dynamic evaluator discovery and registration.

This module provides a registry system for evaluators, allowing:
- Automatic discovery of built-in evaluators
- Registration of custom evaluators
- Validation of evaluator names
- Type-safe evaluator access

Example:
    >>> from arbiter.core.registry import register_evaluator, get_evaluator_class
    >>> from arbiter.evaluators import BasePydanticEvaluator
    >>>
    >>> # Register a custom evaluator
    >>> class MyEvaluator(BasePydanticEvaluator):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_evaluator"
    ...     # ... implement required methods
    >>>
    >>> register_evaluator("my_evaluator", MyEvaluator)
    >>>
    >>> # Use in evaluate()
    >>> result = await evaluate(
    ...     output="test",
    ...     evaluators=["my_evaluator"]
    ... )
"""

from typing import Dict, List, Optional, Type

from .exceptions import ValidationError
from .interfaces import BaseEvaluator

__all__ = [
    "AVAILABLE_EVALUATORS",
    "register_evaluator",
    "get_evaluator_class",
    "get_available_evaluators",
    "validate_evaluator_name",
]


# Registry mapping evaluator names to their classes
AVAILABLE_EVALUATORS: Dict[str, Type[BaseEvaluator]] = {}


def _initialize_builtin_evaluators() -> None:
    """Initialize registry with built-in evaluators.

    This function is called automatically when the module is imported.
    It registers all built-in evaluators that are available by default.
    """
    # Import here to avoid circular dependencies
    from ..evaluators import (
        CustomCriteriaEvaluator,
        FactualityEvaluator,
        GroundednessEvaluator,
        RelevanceEvaluator,
        SemanticEvaluator,
    )

    AVAILABLE_EVALUATORS["semantic"] = SemanticEvaluator
    AVAILABLE_EVALUATORS["custom_criteria"] = CustomCriteriaEvaluator
    AVAILABLE_EVALUATORS["factuality"] = FactualityEvaluator
    AVAILABLE_EVALUATORS["groundedness"] = GroundednessEvaluator
    AVAILABLE_EVALUATORS["relevance"] = RelevanceEvaluator


def register_evaluator(name: str, evaluator_class: Type[BaseEvaluator]) -> None:
    """Register a custom evaluator.

    This allows users to add their own evaluators to the registry,
    making them available for use in the evaluate() function.

    Args:
        name: Unique name for the evaluator (e.g., "my_evaluator")
        evaluator_class: Class that implements BaseEvaluator interface

    Raises:
        ValueError: If name is already registered

    Example:
        >>> from arbiter.core.registry import register_evaluator
        >>> from arbiter.evaluators import BasePydanticEvaluator
        >>>
        >>> class MyEvaluator(BasePydanticEvaluator):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_evaluator"
        ...     # ... implement required methods
        >>>
        >>> register_evaluator("my_evaluator", MyEvaluator)
        >>>
        >>> # Now can use in evaluate()
        >>> result = await evaluate(
        ...     output="test",
        ...     evaluators=["my_evaluator"]
        ... )
    """
    if name in AVAILABLE_EVALUATORS:
        raise ValueError(
            f"Evaluator '{name}' is already registered. "
            f"Use a different name or unregister the existing evaluator first."
        )

    if not issubclass(evaluator_class, BaseEvaluator):
        raise ValueError(
            f"Evaluator class must inherit from BaseEvaluator. "
            f"Got: {evaluator_class.__name__}"
        )

    AVAILABLE_EVALUATORS[name] = evaluator_class


def get_evaluator_class(name: str) -> Optional[Type[BaseEvaluator]]:
    """Get evaluator class by name.

    Args:
        name: Evaluator name

    Returns:
        Evaluator class if found, None otherwise

    Example:
        >>> from arbiter.core.registry import get_evaluator_class
        >>> SemanticEvaluator = get_evaluator_class("semantic")
        >>> assert SemanticEvaluator is not None
    """
    return AVAILABLE_EVALUATORS.get(name)


def get_available_evaluators() -> List[str]:
    """Get list of all registered evaluator names.

    Returns:
        List of evaluator names, sorted alphabetically

    Example:
        >>> from arbiter.core.registry import get_available_evaluators
        >>> evaluators = get_available_evaluators()
        >>> print(evaluators)
        ['custom_criteria', 'semantic']
    """
    return sorted(AVAILABLE_EVALUATORS.keys())


def validate_evaluator_name(name: str) -> None:
    """Validate that an evaluator name is registered.

    Args:
        name: Evaluator name to validate

    Raises:
        ValidationError: If evaluator name is not registered

    Example:
        >>> from arbiter.core.registry import validate_evaluator_name
        >>> validate_evaluator_name("semantic")  # OK
        >>> validate_evaluator_name("unknown")  # Raises ValidationError
    """
    if name not in AVAILABLE_EVALUATORS:
        available = get_available_evaluators()
        available_str = ", ".join(f"'{e}'" for e in available)
        raise ValidationError(
            f"Unknown evaluator: '{name}'. "
            f"Available evaluators: [{available_str}]. "
            f"Use register_evaluator() to add custom evaluators."
        )


# Initialize built-in evaluators when module is imported
_initialize_builtin_evaluators()
