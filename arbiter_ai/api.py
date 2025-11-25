"""Main API for Arbiter evaluation framework.

This module provides the primary entry points for evaluating LLM outputs.

## Quick Start:

    >>> from arbiter import evaluate
    >>> from arbiter_ai.core import LLMManager
    >>>
    >>> # Evaluate semantic similarity
    >>> client = await LLMManager.get_client(model="gpt-4o")
    >>> result = await evaluate(
    ...     output="Paris is the capital of France",
    ...     reference="The capital of France is Paris",
    ...     evaluators=["semantic"],
    ...     llm_client=client
    ... )
    >>>
    >>> print(f"Overall Score: {result.overall_score:.2f}")
    >>> print(f"LLM Calls: {len(result.interactions)}")
    >>> print(f"Total Cost: ${result.total_llm_cost():.4f}")

## With Multiple Evaluators:

    >>> result = await evaluate(
    ...     output="The quick brown fox jumps over the lazy dog",
    ...     reference="A fast brown fox leaps above a sleepy canine",
    ...     evaluators=["semantic"],  # More evaluators coming soon
    ...     llm_client=client
    ... )
"""

import asyncio
import logging
import time
from typing import Any, Callable, Dict, List, Optional

from .core import (
    LLMClient,
    LLMManager,
    Provider,
    get_evaluator_class,
    validate_evaluator_name,
)
from .core.exceptions import ArbiterError, EvaluatorError, ValidationError
from .core.middleware import MiddlewarePipeline
from .core.models import (
    BatchEvaluationResult,
    ComparisonResult,
    EvaluationResult,
    LLMInteraction,
    Metric,
    Score,
)
from .evaluators import PairwiseComparisonEvaluator

logger = logging.getLogger(__name__)

__all__ = ["evaluate", "compare", "batch_evaluate"]


async def evaluate(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
    middleware: Optional[MiddlewarePipeline] = None,
) -> EvaluationResult:
    """Evaluate an LLM output with automatic interaction tracking.

    This is the main entry point for Arbiter evaluations. It runs one or more
    evaluators on the output and returns comprehensive results including all
    LLM interactions for full transparency.

    Args:
        output: The LLM output to evaluate
        reference: Optional reference text for comparison-based evaluation
        criteria: Optional evaluation criteria for reference-free evaluation
        evaluators: List of evaluator names to run (default: ["semantic"])
        llm_client: Optional pre-configured LLM client (will create if not provided)
        model: Model to use if creating new client (default: "gpt-4o")
        provider: Provider to use if creating new client (default: OPENAI)
        threshold: Score threshold for pass/fail (default: 0.7)
        middleware: Optional middleware pipeline for cross-cutting concerns

    Returns:
        EvaluationResult with scores, metrics, and complete LLM interaction tracking.
        If some evaluators fail, result.partial will be True and result.errors will
        contain error messages keyed by evaluator name.

        Overall Score Behavior:
        - The overall_score is calculated as the average of all successful evaluator scores
        - Failed evaluators are excluded from the average calculation
        - If all evaluators fail, an EvaluatorError is raised instead
        - Example: If evaluators=["semantic", "custom_criteria"] and custom_criteria fails,
          overall_score = semantic_score (not average of semantic and 0)

    Raises:
        ValidationError: If input validation fails or evaluator name is not recognized.
            Error messages include available evaluator names for helpful debugging.
        EvaluatorError: If all evaluators fail (partial results return successfully)

    Example:
        >>> # Basic semantic evaluation
        >>> result = await evaluate(
        ...     output="Paris is the capital of France",
        ...     reference="The capital of France is Paris",
        ...     evaluators=["semantic"]
        ... )
        >>> print(f"Score: {result.overall_score:.2f}")
        >>> print(f"Passed: {result.passed}")
        >>>
        >>> # Check LLM usage
        >>> for interaction in result.interactions:
        ...     print(f"Purpose: {interaction.purpose}")
        ...     print(f"Latency: {interaction.latency:.2f}s")
        ...     print(f"Model: {interaction.model}")
        >>>
        >>> # Calculate cost
        >>> cost = result.total_llm_cost(cost_per_1k_tokens=0.03)
        >>> print(f"Total cost: ${cost:.4f}")
    """
    # Input validation
    if not output or not output.strip():
        raise ValidationError("output cannot be empty or whitespace")

    if reference is not None and not reference.strip():
        raise ValidationError("reference cannot be empty or whitespace if provided")

    if criteria is not None and not criteria.strip():
        raise ValidationError("criteria cannot be empty or whitespace if provided")

    # If middleware is provided, use it to wrap the evaluation
    if middleware:

        async def _evaluate_core(
            output: str, reference: Optional[str]
        ) -> EvaluationResult:
            return await _evaluate_impl(
                output=output,
                reference=reference,
                criteria=criteria,
                evaluators=evaluators,
                llm_client=llm_client,
                model=model,
                provider=provider,
                threshold=threshold,
            )

        return await middleware.execute(output, reference, _evaluate_core)

    # No middleware, run directly
    return await _evaluate_impl(
        output=output,
        reference=reference,
        criteria=criteria,
        evaluators=evaluators,
        llm_client=llm_client,
        model=model,
        provider=provider,
        threshold=threshold,
    )


async def _evaluate_impl(
    output: str,
    reference: Optional[str] = None,
    criteria: Optional[str] = None,
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
) -> EvaluationResult:
    """Internal implementation of evaluation (called directly or via middleware)."""
    start_time = time.time()

    # Default to semantic evaluator if none specified
    if evaluators is None:
        evaluators = ["semantic"]

    # Create LLM client if not provided
    if llm_client is None:
        llm_client = await LLMManager.get_client(
            provider=provider, model=model, temperature=0.0
        )

    # Initialize evaluator instances using registry
    from .core.interfaces import BaseEvaluator

    evaluator_instances: List[BaseEvaluator] = []
    for evaluator_name in evaluators:
        # Validate evaluator name
        validate_evaluator_name(evaluator_name)

        # Get evaluator class from registry
        evaluator_class = get_evaluator_class(evaluator_name)
        if evaluator_class is None:
            # This should not happen if validate_evaluator_name worked correctly
            from .core.registry import get_available_evaluators

            available = get_available_evaluators()
            available_str = ", ".join(f"'{e}'" for e in available)
            raise ValidationError(
                f"Evaluator '{evaluator_name}' not found in registry. "
                f"Available evaluators: [{available_str}]."
            )

        # Special validation for custom_criteria evaluator
        if evaluator_name == "custom_criteria" and not criteria:
            raise ValidationError(
                "custom_criteria evaluator requires criteria parameter. "
                "Provide criteria as a string or dict."
            )

        # Instantiate evaluator
        evaluator_instances.append(evaluator_class(llm_client=llm_client))

    # Run evaluations and collect scores
    scores: List[Score] = []
    metrics: List[Metric] = []
    all_interactions: List[LLMInteraction] = []
    evaluator_names: List[str] = []
    errors: Dict[str, str] = {}

    for evaluator in evaluator_instances:
        # Clear previous interactions
        evaluator.clear_interactions()

        try:
            # Run evaluation
            eval_start = time.time()
            score = await evaluator.evaluate(output, reference, criteria)
            eval_time = time.time() - eval_start

            # Collect results
            scores.append(score)
            evaluator_names.append(evaluator.name)

            # Collect interactions from this evaluator
            interactions = evaluator.get_interactions()
            all_interactions.extend(interactions)

            # Create metric metadata
            metric = Metric(
                name=evaluator.name,
                evaluator=evaluator.name,
                model=llm_client.model,
                processing_time=eval_time,
                tokens_used=sum(i.tokens_used for i in interactions),
                metadata={
                    "interaction_count": len(interactions),
                    "has_reference": reference is not None,
                    "has_criteria": criteria is not None,
                },
            )
            metrics.append(metric)

        except ArbiterError as e:
            # Track evaluator-specific errors
            error_msg = str(e)
            if hasattr(e, "details") and isinstance(e.details, dict):
                # Extract more detailed error message if available
                error_msg = e.details.get("error", error_msg)
            errors[evaluator.name] = error_msg
            logger.warning(
                f"Evaluator '{evaluator.name}' failed: {error_msg}",
                extra={"evaluator": evaluator.name, "error_type": type(e).__name__},
            )
            # Continue with other evaluators

        except Exception as e:
            # Catch any unexpected errors
            error_msg = f"Unexpected error: {str(e)}"
            errors[evaluator.name] = error_msg
            logger.error(
                f"Unexpected error in evaluator '{evaluator.name}': {type(e).__name__}: {str(e)}",
                exc_info=True,
                extra={"evaluator": evaluator.name, "error_type": type(e).__name__},
            )
            # Continue with other evaluators

    # Check if we have any successful evaluations
    if not scores and errors:
        # All evaluators failed - raise an error
        error_summary = "; ".join([f"{name}: {msg}" for name, msg in errors.items()])
        raise EvaluatorError(
            f"All evaluators failed: {error_summary}",
            details={"errors": errors, "evaluator_count": len(evaluator_instances)},
        )

    # Calculate overall score (average of successful evaluators only)
    # Note: This only includes scores from successful evaluators. If some evaluators
    # failed, their scores are not included in the average. This ensures the overall
    # score reflects only the evaluations that completed successfully.
    overall_score = sum(s.value for s in scores) / len(scores) if scores else 0.0

    # Determine if passed threshold
    passed = overall_score >= threshold

    # Calculate totals
    total_tokens = sum(i.tokens_used for i in all_interactions)
    processing_time = time.time() - start_time

    # Determine if this is a partial result
    partial = len(errors) > 0

    return EvaluationResult(
        output=output,
        reference=reference,
        criteria=criteria,
        scores=scores,
        overall_score=overall_score,
        passed=passed,
        errors=errors,
        partial=partial,
        metrics=metrics,
        evaluator_names=evaluator_names,
        total_tokens=total_tokens,
        processing_time=processing_time,
        interactions=all_interactions,
        metadata={
            "evaluator_count": len(evaluator_instances),
            "successful_evaluators": len(scores),
            "failed_evaluators": len(errors),
            "threshold": threshold,
        },
    )


async def compare(
    output_a: str,
    output_b: str,
    criteria: Optional[str] = None,
    reference: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    middleware: Optional[MiddlewarePipeline] = None,
) -> ComparisonResult:
    """Compare two LLM outputs to determine which is better.

    This function performs pairwise comparison between two outputs, determining
    which is superior or if they're equivalent. Useful for A/B testing, model
    comparison, and output selection.

    Args:
        output_a: First output to compare
        output_b: Second output to compare
        criteria: Optional criteria for comparison (e.g., "accuracy, clarity, completeness")
        reference: Optional reference context (e.g., user question or ground truth)
        llm_client: Optional pre-configured LLM client (will create if not provided)
        model: Model to use if creating new client (default: "gpt-4o")
        provider: Provider to use if creating new client (default: OPENAI)
        middleware: Optional middleware pipeline for cross-cutting concerns

    Returns:
        ComparisonResult with winner, confidence, reasoning, and aspect scores

    Raises:
        ValidationError: If input validation fails
        EvaluatorError: If comparison fails

    Example:
        >>> comparison = await compare(
        ...     output_a="GPT-4 response: Paris is the capital of France, founded in 3rd century BC.",
        ...     output_b="Claude response: The capital of France is Paris, established around 250 BC.",
        ...     criteria="accuracy, clarity, completeness",
        ...     reference="What is the capital of France?"
        ... )
        >>> print(f"Winner: {comparison.winner}")
        >>> print(f"Confidence: {comparison.confidence:.2f}")
        >>> print(f"Reasoning: {comparison.reasoning}")
        >>>
        >>> # Check aspect scores
        >>> for aspect, scores in comparison.aspect_scores.items():
        ...     print(f"{aspect}: A={scores['output_a']:.2f}, B={scores['output_b']:.2f}")
    """
    # Input validation
    if not output_a or not output_a.strip():
        raise ValidationError("output_a cannot be empty or whitespace")
    if not output_b or not output_b.strip():
        raise ValidationError("output_b cannot be empty or whitespace")

    if criteria is not None and not criteria.strip():
        raise ValidationError("criteria cannot be empty or whitespace if provided")

    if reference is not None and not reference.strip():
        raise ValidationError("reference cannot be empty or whitespace if provided")

    # If middleware is provided, use it to wrap the comparison
    if middleware:

        async def _compare_final_handler(
            out_a: str, out_b: str, crit: Optional[str], ref: Optional[str]
        ) -> ComparisonResult:
            """Final handler for middleware pipeline."""
            return await _compare_impl(
                output_a=out_a,
                output_b=out_b,
                criteria=crit,
                reference=ref,
                llm_client=llm_client,
                model=model,
                provider=provider,
            )

        # Use the middleware pipeline's execute_comparison adapter
        return await middleware.execute_comparison(
            output_a=output_a,
            output_b=output_b,
            criteria=criteria,
            reference=reference,
            final_handler=_compare_final_handler,
        )

    # No middleware, run directly
    return await _compare_impl(
        output_a=output_a,
        output_b=output_b,
        criteria=criteria,
        reference=reference,
        llm_client=llm_client,
        model=model,
        provider=provider,
    )


async def _compare_impl(
    output_a: str,
    output_b: str,
    criteria: Optional[str] = None,
    reference: Optional[str] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
) -> ComparisonResult:
    """Internal implementation of comparison."""
    # Create LLM client if not provided
    if llm_client is None:
        llm_client = await LLMManager.get_client(
            provider=provider, model=model, temperature=0.0
        )

    # Create evaluator
    evaluator = PairwiseComparisonEvaluator(llm_client)

    # Run comparison
    comparison = await evaluator.compare(
        output_a=output_a,
        output_b=output_b,
        criteria=criteria,
        reference=reference,
    )

    return comparison


async def batch_evaluate(
    items: List[Dict[str, Any]],
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
    max_concurrency: int = 10,
    progress_callback: Optional[
        Callable[[int, int, Optional["EvaluationResult"]], None]
    ] = None,
    middleware: Optional[MiddlewarePipeline] = None,
) -> "BatchEvaluationResult":
    """Evaluate multiple LLM outputs in parallel with progress tracking.

    Efficient batch processing while maintaining individual result fidelity.
    Failed items return partial results without failing the entire batch.

    Args:
        items: List of evaluation items, each dict with keys:
            - "output" (required): The LLM output to evaluate
            - "reference" (optional): Reference text for comparison
            - "criteria" (optional): Evaluation criteria
        evaluators: List of evaluator names to run (default: ["semantic"])
        llm_client: Optional pre-configured LLM client (shared across batch)
        model: Model to use if creating new client (default: "gpt-4o")
        provider: Provider to use if creating new client (default: OPENAI)
        threshold: Score threshold for pass/fail (default: 0.7)
        max_concurrency: Maximum parallel evaluations (default: 10)
        progress_callback: Optional callback(completed, total, latest_result)
        middleware: Optional middleware pipeline for cross-cutting concerns

    Returns:
        BatchEvaluationResult with results list, errors, and aggregate statistics.
        Results list preserves order - None for failed items.

    Raises:
        ValidationError: If input validation fails (empty items, invalid format)

    Example:
        >>> # Basic batch evaluation
        >>> results = await batch_evaluate(
        ...     items=[
        ...         {"output": "Paris is capital of France", "reference": "Paris is France's capital"},
        ...         {"output": "Tokyo is capital of Japan", "reference": "Tokyo is Japan's capital"},
        ...         {"output": "Berlin is capital of Germany", "reference": "Berlin is Germany's capital"},
        ...     ],
        ...     evaluators=["semantic"],
        ...     model="gpt-4o-mini",
        ...     max_concurrency=5
        ... )
        >>> print(f"Success: {results.successful_items}/{results.total_items}")
        >>> print(f"Total cost: ${await results.total_llm_cost():.4f}")
        >>>
        >>> # With progress tracking
        >>> def on_progress(completed, total, result):
        ...     print(f"Progress: {completed}/{total}")
        ...     if result:
        ...         print(f"Latest score: {result.overall_score:.2f}")
        >>>
        >>> results = await batch_evaluate(
        ...     items=[...],
        ...     progress_callback=on_progress
        ... )
        >>>
        >>> # Access individual results
        >>> for i, result in enumerate(results.results):
        ...     if result:
        ...         print(f"Item {i}: {result.overall_score:.2f}")
        ...     else:
        ...         error = results.get_error(i)
        ...         print(f"Item {i}: FAILED - {error['error']}")
    """
    # Input validation
    if not items:
        raise ValidationError("items list cannot be empty")

    # Validate each item structure
    for i, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValidationError(f"Item {i} must be a dictionary, got {type(item)}")
        if "output" not in item:
            raise ValidationError(f"Item {i} missing required 'output' key")
        if not item["output"] or not str(item["output"]).strip():
            raise ValidationError(f"Item {i} 'output' cannot be empty or whitespace")

    # Delegate to implementation
    return await _batch_evaluate_impl(
        items=items,
        evaluators=evaluators,
        llm_client=llm_client,
        model=model,
        provider=provider,
        threshold=threshold,
        max_concurrency=max_concurrency,
        progress_callback=progress_callback,
        middleware=middleware,
    )


async def _batch_evaluate_impl(
    items: List[Dict[str, Any]],
    evaluators: Optional[List[str]] = None,
    llm_client: Optional[LLMClient] = None,
    model: str = "gpt-4o",
    provider: Provider = Provider.OPENAI,
    threshold: float = 0.7,
    max_concurrency: int = 10,
    progress_callback: Optional[
        Callable[[int, int, Optional[EvaluationResult]], None]
    ] = None,
    middleware: Optional[MiddlewarePipeline] = None,
) -> "BatchEvaluationResult":
    """Internal implementation of batch evaluation."""

    start_time = time.time()
    total_items = len(items)

    # Create shared LLM client if not provided
    if llm_client is None:
        llm_client = await LLMManager.get_client(
            provider=provider, model=model, temperature=0.0
        )

    # Set up concurrency control
    semaphore = asyncio.Semaphore(max_concurrency)

    # Track completion for progress callback
    completed_count = [0]  # Use list for closure mutability

    def update_progress(result: Optional[EvaluationResult]) -> None:
        """Helper to update progress counter and invoke callback."""
        completed_count[0] += 1
        if progress_callback:
            progress_callback(completed_count[0], total_items, result)

    async def evaluate_item(
        index: int, item: Dict[str, Any]
    ) -> tuple[int, Optional[EvaluationResult], Optional[str]]:
        """Evaluate single item with concurrency control."""
        async with semaphore:
            try:
                # Extract item fields
                output = item["output"]
                reference = item.get("reference")
                criteria = item.get("criteria")

                # Run evaluation (reuses existing evaluate function)
                result = await evaluate(
                    output=output,
                    reference=reference,
                    criteria=criteria,
                    evaluators=evaluators,
                    llm_client=llm_client,
                    model=model,
                    provider=provider,
                    threshold=threshold,
                    middleware=middleware,
                )

                update_progress(result)
                return (index, result, None)

            except Exception as e:
                # Track error but don't fail entire batch
                error_msg = str(e)
                logger.warning(
                    f"Batch item {index} failed: {error_msg}",
                    extra={"batch_index": index, "error_type": type(e).__name__},
                )

                update_progress(None)
                return (index, None, error_msg)

    # Create tasks for all items
    tasks = [evaluate_item(i, item) for i, item in enumerate(items)]

    # Run all evaluations in parallel (respecting semaphore limit)
    results_with_indices = await asyncio.gather(*tasks)

    # Build results list (preserving order, None for failures)
    results: List[Optional[EvaluationResult]] = [None] * total_items
    errors: List[Dict[str, Any]] = []
    total_tokens = 0

    for index, result, error_msg in results_with_indices:
        if result:
            results[index] = result
            total_tokens += result.total_tokens
        else:
            errors.append(
                {
                    "index": index,
                    "item": items[index],
                    "error": error_msg or "Unknown error",
                }
            )

    # Calculate statistics
    successful_items = sum(1 for r in results if r is not None)
    failed_items = len(errors)
    processing_time = time.time() - start_time

    return BatchEvaluationResult(
        results=results,
        errors=errors,
        total_items=total_items,
        successful_items=successful_items,
        failed_items=failed_items,
        processing_time=processing_time,
        total_tokens=total_tokens,
        metadata={
            "evaluators": evaluators or ["semantic"],
            "model": model,
            "provider": provider.value if hasattr(provider, "value") else str(provider),
            "max_concurrency": max_concurrency,
            "threshold": threshold,
        },
    )
