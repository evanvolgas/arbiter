"""Type definitions for Arbiter core components.

This module provides TypedDict definitions and type aliases used throughout
the Arbiter codebase for better type safety and IDE support.
"""

from typing import Any, Dict, List, Union

try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict

__all__ = ["MiddlewareContext"]


class MiddlewareContext(TypedDict, total=False):
    """Context dictionary passed between middleware components.

    This typed dictionary defines the standard keys that middleware can
    use to share data. All keys are optional to allow flexibility.

    Common keys:
        evaluators: List of evaluator names being used
        metrics: List of metric names being computed
        config: Configuration object for the evaluation
        llm_calls: Counter for total LLM API calls made
        start_time: Timestamp when evaluation started
        model: LLM model being used
        temperature: Temperature parameter for LLM
        is_pairwise_comparison: Whether this is a pairwise comparison
        pairwise_data: Data for pairwise comparisons

    Custom middleware can add their own keys as needed.
    """

    evaluators: List[Union[str, Any]]
    metrics: List[str]
    config: Any
    llm_calls: int
    start_time: float
    model: str
    temperature: float
    is_pairwise_comparison: bool
    pairwise_data: Dict[str, Any]
