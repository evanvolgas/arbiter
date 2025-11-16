"""Base evaluator class with PydanticAI integration and automatic tracking.

This module provides the foundation for all evaluators in Arbiter.
It handles LLM interaction tracking, structured outputs via PydanticAI,
and consistent error handling.

## Key Features:

- **Automatic Tracking**: All LLM calls are automatically recorded
- **Structured Outputs**: PydanticAI agents for type-safe responses
- **Error Handling**: Consistent error handling across evaluators
- **Observability**: Complete transparency in evaluation process

## Usage:

    >>> class MyEvaluator(BasePydanticEvaluator):
    ...     @property
    ...     def name(self) -> str:
    ...         return "my_evaluator"
    ...
    ...     def _get_system_prompt(self) -> str:
    ...         return "You are an expert evaluator..."
    ...
    ...     def _get_user_prompt(self, output: str, reference: Optional[str]) -> str:
    ...         return f"Evaluate this output: {output}"
    ...
    ...     async def _compute_score(self, response: BaseModel) -> Score:
    ...         return Score(name="my_metric", value=response.score)
"""

import logging
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional, Type

from pydantic import BaseModel, Field

from ..core.exceptions import EvaluatorError
from ..core.interfaces import BaseEvaluator
from ..core.llm_client import LLMClient
from ..core.models import LLMInteraction, Score

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.types import Provider

__all__ = ["BasePydanticEvaluator", "EvaluatorResponse"]


class EvaluatorResponse(BaseModel):
    """Standard response format for evaluators.

    This is the base response model that evaluators can extend.
    It ensures all evaluators return structured data with scores
    and explanations.
    """

    score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence in this score"
    )
    explanation: str = Field(..., description="Human-readable explanation")
    metadata: dict = Field(default_factory=dict, description="Additional data")


class BasePydanticEvaluator(BaseEvaluator):
    """Base class for evaluators using PydanticAI for structured outputs.

    This class provides the foundation for building evaluators that:
    - Use LLMs to compute evaluation scores
    - Return structured, type-safe responses
    - Automatically track all LLM interactions
    - Handle errors consistently

    Subclasses must implement:
    - name: Unique identifier
    - _get_system_prompt(): System prompt defining evaluator behavior
    - _get_user_prompt(): User prompt with output/reference
    - _get_response_type(): Pydantic model for structured response
    - _compute_score(): Extract Score from structured response

    Example:
        >>> class FactualityEvaluator(BasePydanticEvaluator):
        ...     @property
        ...     def name(self) -> str:
        ...         return "factuality"
        ...
        ...     def _get_system_prompt(self) -> str:
        ...         return "You evaluate factual accuracy of statements."
        ...
        ...     def _get_user_prompt(self, output, reference, criteria):
        ...         return f"Output: {output}\\nReference: {reference}"
        ...
        ...     def _get_response_type(self) -> Type[BaseModel]:
        ...         return EvaluatorResponse
        ...
        ...     async def _compute_score(self, response: EvaluatorResponse) -> Score:
        ...         return Score(
        ...             name="factuality",
        ...             value=response.score,
        ...             confidence=response.confidence,
        ...             explanation=response.explanation
        ...         )
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        model: Optional[str] = None,
        provider: Optional["Provider"] = None,
        temperature: float = 0.0,
    ):
        """Initialize evaluator with LLM client or model parameters.

        You can either provide a pre-configured LLMClient, or provide model
        parameters to create one automatically.

        Args:
            llm_client: Pre-configured LLM client (if provided, model/provider/temperature are ignored)
            model: Model name (e.g., "gpt-4o-mini") - creates client automatically if llm_client not provided
            provider: Provider enum (auto-detected from API keys if not provided)
            temperature: Temperature for generation (default: 0.0 for deterministic evaluation)

        Example:
            >>> # Option 1: Provide client directly
            >>> client = await LLMManager.get_client(model="gpt-4o")
            >>> evaluator = SemanticEvaluator(llm_client=client)
            >>>
            >>> # Option 2: Provide model, client created automatically
            >>> evaluator = SemanticEvaluator(model="gpt-4o-mini")
            >>> # This will work in sync context and create client on first use

        Raises:
            ValueError: If neither llm_client nor model is provided
        """
        if llm_client is None and model is None:
            raise ValueError(
                "Must provide either llm_client or model parameter. "
                "Example: SemanticEvaluator(model='gpt-4o-mini')"
            )

        self._llm_client = llm_client
        self._model = model
        self._provider = provider
        self._temperature = temperature
        self.interactions: list[LLMInteraction] = []

    @property
    def llm_client(self) -> LLMClient:
        """Get or create LLM client.

        Returns:
            LLMClient instance

        Raises:
            RuntimeError: If client needs to be created but we're not in async context
        """
        if self._llm_client is None:
            # Need to create client - this requires async context
            raise RuntimeError(
                "LLM client not initialized. Call evaluator.evaluate() which will "
                "create the client automatically, or provide llm_client in __init__."
            )
        return self._llm_client

    async def _ensure_client(self) -> None:
        """Ensure LLM client is initialized, creating if needed."""
        if self._llm_client is None:
            from ..core.llm_client import LLMManager

            self._llm_client = await LLMManager.get_client(
                model=self._model,
                provider=self._provider,
                temperature=self._temperature,
            )

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt that defines evaluator behavior.

        This prompt establishes the evaluator's role and approach.

        Returns:
            System prompt string

        Example:
            >>> return "You are an expert at evaluating text quality..."
        """

    @abstractmethod
    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get the user prompt for a specific evaluation.

        This prompt contains the actual output to evaluate and any
        reference text or criteria.

        Args:
            output: The text to evaluate
            reference: Optional reference text
            criteria: Optional evaluation criteria

        Returns:
            User prompt string

        Example:
            >>> return f"Evaluate this output: {output}\\nReference: {reference}"
        """

    def _get_response_type(self) -> Type[BaseModel]:
        """Get the Pydantic model for structured responses.

        Override this to use a custom response model. Defaults to
        EvaluatorResponse.

        Returns:
            Pydantic model class

        Example:
            >>> class CustomResponse(BaseModel):
            ...     score: float
            ...     reasoning: str
            >>> return CustomResponse
        """
        return EvaluatorResponse

    @abstractmethod
    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract a Score from the structured LLM response.

        This method transforms the PydanticAI response into a Score
        object that can be included in evaluation results.

        Args:
            response: Structured response from LLM

        Returns:
            Score object

        Example:
            >>> return Score(
            ...     name=self.name,
            ...     value=response.score,
            ...     confidence=response.confidence,
            ...     explanation=response.explanation
            ... )
        """

    async def evaluate(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
    ) -> Score:
        """Evaluate an output and return a score.

        This is the main entry point for evaluation. It:
        1. Ensures LLM client is initialized (creates if needed)
        2. Creates a PydanticAI agent with structured output
        3. Runs evaluation with automatic tracking
        4. Records the LLM interaction
        5. Computes and returns the score

        Args:
            output: The text to evaluate
            reference: Optional reference text for comparison
            criteria: Optional evaluation criteria

        Returns:
            Score object with evaluation result

        Raises:
            EvaluatorError: If evaluation fails

        Example:
            >>> # Option 1: With model parameter (client created automatically)
            >>> evaluator = SemanticEvaluator(model="gpt-4o-mini")
            >>> score = await evaluator.evaluate(
            ...     output="Paris is the capital of France",
            ...     reference="The capital of France is Paris"
            ... )
            >>> print(f"Score: {score.value}")
            >>>
            >>> # Option 2: With pre-configured client
            >>> client = await LLMManager.get_client(model="gpt-4o")
            >>> evaluator = SemanticEvaluator(llm_client=client)
            >>> score = await evaluator.evaluate(output, reference)
        """
        start_time = time.time()

        try:
            # Ensure client is initialized
            await self._ensure_client()
            # Get prompts
            system_prompt = self._get_system_prompt()
            user_prompt = self._get_user_prompt(output, reference, criteria)

            # Create PydanticAI agent with structured output
            response_type = self._get_response_type()
            agent = self.llm_client.create_agent(system_prompt, response_type)

            # Run evaluation
            result = await agent.run(user_prompt)

            # Extract detailed token usage from PydanticAI result
            input_tokens = 0
            output_tokens = 0
            cached_tokens = 0
            tokens_used = 0  # Backward compatibility

            try:
                if hasattr(result, "usage"):
                    usage = result.usage()  # Call as function
                    if usage:
                        # PydanticAI usage object structure
                        input_tokens = getattr(usage, "request_tokens", 0)
                        output_tokens = getattr(usage, "response_tokens", 0)
                        tokens_used = getattr(usage, "total_tokens", 0)

                        # Some providers support cached tokens
                        if hasattr(usage, "cached_tokens"):
                            cached_tokens = getattr(usage, "cached_tokens", 0)
            except Exception:
                # Fallback if usage() call fails or not available
                pass

            # Calculate cost using cost calculator
            cost = None
            try:
                from arbiter.core.cost_calculator import get_cost_calculator

                calc = get_cost_calculator()
                await calc.ensure_loaded()

                cost = calc.calculate_cost(
                    model=self.llm_client.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                )
            except Exception as e:
                # If cost calculation fails, continue without cost
                logger.warning(
                    f"Cost calculation failed for model {self.llm_client.model}: {e}"
                )
                pass

            # Record interaction for transparency
            latency = time.time() - start_time
            interaction = LLMInteraction(
                prompt=user_prompt,
                response=result.output.model_dump_json() if hasattr(result.output, 'model_dump_json') else str(result.output),
                model=self.llm_client.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
                tokens_used=tokens_used or (input_tokens + output_tokens),  # Backward compat
                cost=cost,
                latency=latency,
                purpose=f"{self.name}_evaluation",
                metadata={
                    "evaluator": self.name,
                    "system_prompt": system_prompt,
                    "has_reference": reference is not None,
                    "has_criteria": criteria is not None,
                },
            )
            self.interactions.append(interaction)

            # Compute score from structured response
            score = await self._compute_score(result.output)
            return score

        except Exception as e:
            raise EvaluatorError(
                f"Evaluation failed in {self.name}",
                details={"error": str(e), "evaluator": self.name},
            ) from e

    def get_interactions(self) -> list[LLMInteraction]:
        """Get all LLM interactions recorded by this evaluator.

        Returns:
            List of LLM interactions

        Example:
            >>> evaluator = SemanticEvaluator(llm_client)
            >>> await evaluator.evaluate(output, reference)
            >>> interactions = evaluator.get_interactions()
            >>> print(f"Made {len(interactions)} LLM calls")
        """
        return self.interactions.copy()

    def clear_interactions(self) -> None:
        """Clear recorded interactions.

        Useful when reusing an evaluator for multiple evaluations.
        """
        self.interactions.clear()
