"""Cost calculation using LiteLLM's bundled pricing database.

This module provides accurate LLM cost calculation using LiteLLM's bundled
model_cost database, which contains up-to-date pricing for all major providers.

## Benefits of using LiteLLM pricing:

- **No External API Calls**: Pricing bundled with package (no network requests)
- **Comprehensive Coverage**: All major providers (OpenAI, Anthropic, Google, etc.)
- **Exact Model Matching**: LiteLLM handles model ID normalization
- **Cache Pricing Support**: Includes cache read and creation costs
- **Simple Updates**: Run `uv update litellm` to get latest pricing

## Usage:

    >>> calc = get_cost_calculator()
    >>> cost = calc.calculate_cost(
    ...     model="gpt-4o-mini",
    ...     input_tokens=1000,
    ...     output_tokens=500
    ... )
    >>> print(f"${cost:.6f}")

## Consistency with Conduit:

This module uses the same pricing source as Conduit's routing system,
ensuring consistent cost calculations across both frameworks. Both use
LiteLLM's bundled model_cost database rather than external pricing APIs.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import litellm
from pydantic import BaseModel, Field

__all__ = ["ModelPricing", "CostCalculator", "get_cost_calculator"]

logger = logging.getLogger(__name__)


class ModelPricing(BaseModel):
    """Pricing information for a specific LLM model.

    Attributes:
        id: Model identifier (e.g., "gpt-4o-mini")
        vendor: Provider name (e.g., "openai", "anthropic")
        name: Human-readable model name
        input: Cost per 1 million input tokens (USD)
        output: Cost per 1 million output tokens (USD)
        input_cached: Cost per 1 million cached input tokens (USD), if applicable
        cache_creation: Cost per 1 million cache creation tokens (USD), if applicable
        last_updated: When this pricing data was loaded

    Example:
        >>> pricing = ModelPricing(
        ...     id="claude-sonnet-4-5-20250929",
        ...     vendor="anthropic",
        ...     name="Claude Sonnet 4.5",
        ...     input=3.0,
        ...     output=15.0,
        ...     input_cached=0.3,
        ...     last_updated=datetime.now()
        ... )
    """

    id: str = Field(..., description="Model identifier")
    vendor: str = Field(..., description="Provider name")
    name: str = Field(..., description="Human-readable model name")
    input: float = Field(..., ge=0, description="Cost per 1M input tokens (USD)")
    output: float = Field(..., ge=0, description="Cost per 1M output tokens (USD)")
    input_cached: Optional[float] = Field(
        None, ge=0, description="Cost per 1M cached input tokens (USD)"
    )
    cache_creation: Optional[float] = Field(
        None, ge=0, description="Cost per 1M cache creation tokens (USD)"
    )
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When pricing was loaded",
    )


class CostCalculator:
    """Calculate accurate LLM costs using LiteLLM's bundled pricing database.

    This calculator uses LiteLLM's bundled model_cost dictionary, which contains
    pricing data for all major LLM providers. No external API calls are needed.

    Example:
        >>> calc = CostCalculator()
        >>> calc.ensure_loaded()  # Optional, loads automatically on first use
        >>> cost = calc.calculate_cost(
        ...     model="gpt-4o-mini",
        ...     input_tokens=1000,
        ...     output_tokens=500
        ... )
        >>> print(f"Cost: ${cost:.6f}")
    """

    def __init__(self) -> None:
        """Initialize calculator with empty cache."""
        self._pricing_cache: Dict[str, ModelPricing] = {}
        self._loaded: bool = False

    async def ensure_loaded(self) -> None:
        """Ensure pricing data is loaded from LiteLLM.

        This method loads pricing data from LiteLLM's bundled database.
        Since the data is bundled with the package, this is synchronous
        and always succeeds.

        The async signature is maintained for backward compatibility with
        existing code that awaits this method.
        """
        if self._loaded:
            return

        self._load_pricing_data()

    def _load_pricing_data(self) -> None:
        """Load pricing data from LiteLLM's bundled database."""
        if self._loaded:
            return

        self._pricing_cache = {}
        loaded_count = 0

        for model_id, model_info in litellm.model_cost.items():
            # Skip non-chat models and sample spec
            if model_id == "sample_spec":
                continue

            mode = model_info.get("mode", "")
            if mode and mode != "chat":
                continue

            # Skip models without input pricing
            input_cost_per_token = model_info.get("input_cost_per_token")
            if input_cost_per_token is None:
                continue

            output_cost_per_token = model_info.get("output_cost_per_token", 0.0)
            cache_read_cost = model_info.get("cache_read_input_token_cost")
            cache_creation_cost = model_info.get("cache_creation_input_token_cost")

            # Extract vendor from litellm_provider or model_id
            vendor = model_info.get("litellm_provider", "")
            if not vendor and "/" in model_id:
                vendor = model_id.split("/")[0]

            pricing = ModelPricing(
                id=model_id,
                vendor=vendor,
                name=model_id,  # LiteLLM uses model_id as name
                input=input_cost_per_token * 1_000_000,
                output=output_cost_per_token * 1_000_000,
                input_cached=(cache_read_cost * 1_000_000 if cache_read_cost else None),
                cache_creation=(
                    cache_creation_cost * 1_000_000 if cache_creation_cost else None
                ),
                last_updated=datetime.now(timezone.utc),
            )
            self._pricing_cache[model_id] = pricing
            loaded_count += 1

        self._loaded = True
        logger.info(f"Loaded pricing data for {loaded_count} models from LiteLLM")

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a specific model.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "claude-sonnet-4-5-20250929")

        Returns:
            ModelPricing if found, None otherwise

        Example:
            >>> pricing = calc.get_pricing("gpt-4o-mini")
            >>> if pricing:
            ...     print(f"Input: ${pricing.input}/M, Output: ${pricing.output}/M")
        """
        # Ensure data is loaded
        if not self._loaded:
            self._load_pricing_data()

        # Try exact match first (most common case)
        if model in self._pricing_cache:
            return self._pricing_cache[model]

        # Try LiteLLM's model_cost directly (handles aliases)
        model_info = litellm.model_cost.get(model)
        if model_info and model_info.get("input_cost_per_token") is not None:
            input_cost = model_info.get("input_cost_per_token", 0.0)
            output_cost = model_info.get("output_cost_per_token", 0.0)
            cache_read = model_info.get("cache_read_input_token_cost")
            cache_creation = model_info.get("cache_creation_input_token_cost")
            vendor = model_info.get("litellm_provider", "")

            pricing = ModelPricing(
                id=model,
                vendor=vendor,
                name=model,
                input=input_cost * 1_000_000,
                output=output_cost * 1_000_000,
                input_cached=cache_read * 1_000_000 if cache_read else None,
                cache_creation=cache_creation * 1_000_000 if cache_creation else None,
                last_updated=datetime.now(timezone.utc),
            )
            # Cache for future lookups
            self._pricing_cache[model] = pricing
            return pricing

        logger.debug(f"No pricing found for model: {model}")
        return None

    def calculate_cost(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> float:
        """Calculate cost for an LLM call.

        Args:
            model: Model identifier (e.g., "gpt-4o-mini")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens (if applicable)

        Returns:
            Cost in USD

        Example:
            >>> cost = calc.calculate_cost(
            ...     model="claude-sonnet-4-5-20250929",
            ...     input_tokens=1000,
            ...     output_tokens=500
            ... )
            >>> print(f"${cost:.6f}")  # $0.010500
        """
        pricing = self.get_pricing(model)

        if pricing:
            # Use actual pricing data
            cost = 0.0

            # Input tokens (non-cached)
            non_cached_input = max(0, input_tokens - cached_tokens)
            cost += (non_cached_input / 1_000_000) * pricing.input

            # Cached input tokens (if pricing available)
            if cached_tokens > 0 and pricing.input_cached is not None:
                cost += (cached_tokens / 1_000_000) * pricing.input_cached
            elif cached_tokens > 0:
                # No cached pricing, treat as regular input
                cost += (cached_tokens / 1_000_000) * pricing.input

            # Output tokens
            cost += (output_tokens / 1_000_000) * pricing.output

            return cost
        else:
            # Fallback: Conservative estimate
            # Assume $10/M input, $30/M output (roughly GPT-4 tier pricing)
            logger.warning(
                f"No pricing for {model}, using conservative fallback estimate"
            )
            return self._fallback_estimate(input_tokens, output_tokens, cached_tokens)

    def _fallback_estimate(
        self, input_tokens: int, output_tokens: int, cached_tokens: int
    ) -> float:
        """Conservative cost estimate when pricing data unavailable.

        Uses GPT-4 tier pricing as a conservative upper bound:
        - Input: $10 per 1M tokens
        - Output: $30 per 1M tokens
        - Cached: $1 per 1M tokens

        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            cached_tokens: Number of cached input tokens

        Returns:
            Estimated cost in USD
        """
        INPUT_COST_PER_1M = 10.0
        OUTPUT_COST_PER_1M = 30.0
        CACHED_COST_PER_1M = 1.0

        cost = 0.0

        # Non-cached input
        non_cached = max(0, input_tokens - cached_tokens)
        cost += (non_cached / 1_000_000) * INPUT_COST_PER_1M

        # Cached input
        if cached_tokens > 0:
            cost += (cached_tokens / 1_000_000) * CACHED_COST_PER_1M

        # Output
        cost += (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M

        return cost

    @property
    def is_loaded(self) -> bool:
        """Whether pricing data was successfully loaded."""
        return self._loaded

    @property
    def model_count(self) -> int:
        """Number of models with pricing data."""
        if not self._loaded:
            self._load_pricing_data()
        return len(self._pricing_cache)


# Global singleton instance
_cost_calculator: Optional[CostCalculator] = None


def get_cost_calculator() -> CostCalculator:
    """Get the global cost calculator instance.

    Returns:
        Singleton CostCalculator instance

    Example:
        >>> calc = get_cost_calculator()
        >>> cost = calc.calculate_cost("gpt-4o-mini", 1000, 500)
    """
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator
