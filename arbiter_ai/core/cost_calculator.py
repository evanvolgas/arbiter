"""Cost calculation using llm-prices data.

This module provides accurate LLM cost calculation by fetching pricing data
from https://www.llm-prices.com and caching it for the session.

## Key Features:

- **Lazy Loading**: Fetches pricing data once on first use
- **Accurate Pricing**: Uses real-world pricing for 100+ models
- **Input/Output Differentiation**: Properly accounts for different token costs
- **Cache Support**: Handles cached token pricing (e.g., Anthropic prompt caching)
- **Fuzzy Matching**: Matches model variants (e.g., "gpt-4o-mini-2024-07-18" → "gpt-4o-mini")
- **Graceful Fallback**: Conservative estimates if pricing data unavailable

## Usage:

    >>> calc = get_cost_calculator()
    >>> await calc.ensure_loaded()
    >>> cost = calc.calculate_cost(
    ...     model="gpt-4o-mini",
    ...     input_tokens=1000,
    ...     output_tokens=500
    ... )
    >>> print(f"${cost:.6f}")

## Architecture:

The cost calculator fetches data once and caches it in memory. If the fetch
fails, it falls back to conservative estimates based on typical pricing tiers.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import httpx
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
        last_updated: When this pricing data was fetched

    Example:
        >>> pricing = ModelPricing(
        ...     id="claude-3-5-sonnet",
        ...     vendor="anthropic",
        ...     name="Claude 3.5 Sonnet",
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
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When pricing was fetched",
    )


class CostCalculator:
    """Calculate accurate LLM costs using llm-prices.com data.

    This calculator fetches pricing data once on first use and caches it
    for the session. If the fetch fails, it falls back to conservative
    estimates.

    Example:
        >>> calc = CostCalculator()
        >>> await calc.ensure_loaded()
        >>> cost = calc.calculate_cost(
        ...     model="gpt-4o-mini",
        ...     input_tokens=1000,
        ...     output_tokens=500
        ... )
        >>> print(f"Cost: ${cost:.6f}")
    """

    PRICING_URL = "https://www.llm-prices.com/current-v1.json"
    TIMEOUT = 10.0  # seconds

    def __init__(self) -> None:
        """Initialize calculator with empty cache."""
        self._pricing_cache: Dict[str, ModelPricing] = {}
        self._loaded: bool = False
        self._load_attempted: bool = False
        self._load_lock: asyncio.Lock = asyncio.Lock()

    async def ensure_loaded(self) -> None:
        """Ensure pricing data is loaded (lazy loading with thread safety).

        Fetches pricing data on first call, then uses cached data.
        Uses asyncio.Lock to prevent duplicate fetches in concurrent scenarios.
        If fetch fails, logs warning and uses fallback estimates.
        """
        # Fast path: Already loaded (no await needed)
        if self._load_attempted:
            return

        # Slow path: Need to load (use lock for thread safety)
        async with self._load_lock:
            # Double-check after acquiring lock (another coroutine may have loaded)
            if self._load_attempted:
                return  # type: ignore[unreachable]

            self._load_attempted = True

            try:
                await self._fetch_pricing_data()
                self._loaded = True
                logger.info(
                    f"Loaded pricing data for {len(self._pricing_cache)} models"
                )
            except Exception as e:
                logger.info(
                    f"Using fallback cost estimates (pricing API unavailable: {e})"
                )
                self._loaded = False

    async def _fetch_pricing_data(self) -> None:
        """Fetch latest pricing data from llm-prices.com with retry logic.

        Uses exponential backoff to handle transient network failures.

        Raises:
            httpx.HTTPError: If all retry attempts fail
            ValueError: If JSON is invalid
        """
        max_retries = 3
        base_delay = 1.0
        backoff = 2.0

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.TIMEOUT) as client:
                    response = await client.get(self.PRICING_URL)
                    response.raise_for_status()
                    data = response.json()

                    # Parse and cache pricing data
                    # Handle both old format (list) and new format (dict with "prices" key)
                    self._pricing_cache = {}
                    prices = (
                        data.get("prices", data) if isinstance(data, dict) else data
                    )
                    for model_data in prices:
                        pricing = ModelPricing(
                            id=model_data["id"],
                            vendor=model_data["vendor"],
                            name=model_data["name"],
                            input=model_data["input"],
                            output=model_data["output"],
                            input_cached=model_data.get("input_cached"),
                            last_updated=datetime.now(timezone.utc),
                        )
                        self._pricing_cache[pricing.id] = pricing

                    logger.debug(
                        f"Successfully fetched pricing data on attempt {attempt}"
                    )
                    return  # Success!

            except (httpx.HTTPError, httpx.ConnectError, asyncio.TimeoutError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = base_delay * (backoff ** (attempt - 1))
                    logger.warning(
                        f"Pricing data fetch failed (attempt {attempt}/{max_retries}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Failed to fetch pricing data after {max_retries} attempts: {e}"
                    )
                    raise

        # Should never reach here, but helps type checker
        if last_error:
            raise last_error

    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        """Get pricing for a specific model.

        Uses fuzzy matching to handle model variants (e.g., date suffixes).

        Args:
            model: Model identifier (e.g., "gpt-4o-mini", "gpt-4o-mini-2024-07-18")

        Returns:
            ModelPricing if found, None otherwise

        Example:
            >>> pricing = calc.get_pricing("gpt-4o-mini")
            >>> if pricing:
            ...     print(f"Input: ${pricing.input}/M, Output: ${pricing.output}/M")
        """
        if not self._loaded:
            return None

        # Try exact match first
        if model in self._pricing_cache:
            return self._pricing_cache[model]

        # Try fuzzy match (model name might have date/version suffix)
        # e.g., "gpt-4o-mini-2024-07-18" should match "gpt-4o-mini"
        # Require separator to avoid false positives (e.g., "gpt-4" shouldn't match "gpt-4o")
        for cached_id, pricing in self._pricing_cache.items():
            if model.startswith(cached_id + "-") or model.startswith(cached_id + "_"):
                logger.debug(f"Fuzzy matched '{model}' to '{cached_id}'")
                return pricing

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
            ...     model="claude-3-5-sonnet",
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
        return len(self._pricing_cache)


# Global singleton instance
_cost_calculator: Optional[CostCalculator] = None


def get_cost_calculator() -> CostCalculator:
    """Get the global cost calculator instance.

    Returns:
        Singleton CostCalculator instance

    Example:
        >>> calc = get_cost_calculator()
        >>> await calc.ensure_loaded()
        >>> cost = calc.calculate_cost("gpt-4o-mini", 1000, 500)
    """
    global _cost_calculator
    if _cost_calculator is None:
        _cost_calculator = CostCalculator()
    return _cost_calculator
