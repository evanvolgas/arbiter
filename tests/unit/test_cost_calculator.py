"""Unit tests for cost calculator functionality.

Tests the LiteLLM-based cost calculator which provides pricing data
from LiteLLM's bundled model_cost database.
"""

from unittest.mock import patch

import pytest

from arbiter_ai.core.cost_calculator import (
    CostCalculator,
    ModelPricing,
    get_cost_calculator,
)


class TestModelPricing:
    """Tests for ModelPricing model."""

    def test_model_pricing_creation(self):
        """Test creating a ModelPricing instance."""
        from datetime import datetime

        pricing = ModelPricing(
            id="gpt-4o-mini",
            vendor="openai",
            name="GPT-4o Mini",
            input=0.150,
            output=0.600,
        )

        assert pricing.id == "gpt-4o-mini"
        assert pricing.vendor == "openai"
        assert pricing.name == "GPT-4o Mini"
        assert pricing.input == 0.150
        assert pricing.output == 0.600
        assert pricing.input_cached is None
        assert isinstance(pricing.last_updated, datetime)

    def test_model_pricing_with_cached(self):
        """Test ModelPricing with cached token pricing."""
        pricing = ModelPricing(
            id="claude-sonnet-4-5-20250929",
            vendor="anthropic",
            name="Claude Sonnet 4.5",
            input=3.0,
            output=15.0,
            input_cached=0.3,
        )

        assert pricing.input_cached == 0.3

    def test_model_pricing_with_cache_creation(self):
        """Test ModelPricing with cache creation pricing."""
        pricing = ModelPricing(
            id="claude-sonnet-4-5-20250929",
            vendor="anthropic",
            name="Claude Sonnet 4.5",
            input=3.0,
            output=15.0,
            input_cached=0.3,
            cache_creation=3.75,
        )

        assert pricing.cache_creation == 3.75


class TestCostCalculator:
    """Tests for CostCalculator functionality."""

    @pytest.fixture
    def mock_litellm_model_cost(self):
        """Mock LiteLLM model_cost data."""
        return {
            "gpt-4o-mini": {
                "input_cost_per_token": 0.150 / 1_000_000,
                "output_cost_per_token": 0.600 / 1_000_000,
                "litellm_provider": "openai",
                "mode": "chat",
            },
            "claude-sonnet-4-5-20250929": {
                "input_cost_per_token": 3.0 / 1_000_000,
                "output_cost_per_token": 15.0 / 1_000_000,
                "cache_read_input_token_cost": 0.3 / 1_000_000,
                "litellm_provider": "anthropic",
                "mode": "chat",
            },
            "gpt-4o": {
                "input_cost_per_token": 2.50 / 1_000_000,
                "output_cost_per_token": 10.0 / 1_000_000,
                "litellm_provider": "openai",
                "mode": "chat",
            },
            "sample_spec": {
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            },
            "text-embedding-3-small": {
                "input_cost_per_token": 0.02 / 1_000_000,
                "output_cost_per_token": 0.0,
                "mode": "embedding",
            },
        }

    @pytest.mark.asyncio
    async def test_ensure_loaded_success(self, mock_litellm_model_cost):
        """Test successfully loading pricing data from LiteLLM."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost

            await calc.ensure_loaded()

            assert calc.is_loaded
            # Should load chat models only (excludes sample_spec and embedding)
            assert calc.model_count == 3
            assert calc.get_pricing("gpt-4o-mini") is not None
            assert calc.get_pricing("claude-sonnet-4-5-20250929") is not None

    @pytest.mark.asyncio
    async def test_ensure_loaded_only_once(self, mock_litellm_model_cost):
        """Test that pricing data is only loaded once."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost

            # Call ensure_loaded multiple times
            await calc.ensure_loaded()
            await calc.ensure_loaded()
            await calc.ensure_loaded()

            # Should still be loaded with same count
            assert calc.is_loaded
            assert calc.model_count == 3

    @pytest.mark.asyncio
    async def test_ensure_loaded_concurrent_calls(self, mock_litellm_model_cost):
        """Test that concurrent calls handle loading correctly."""
        import asyncio

        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost

            # Call ensure_loaded concurrently 5 times
            results = await asyncio.gather(
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
                calc.ensure_loaded(),
            )

            # All should complete without error
            assert len(results) == 5

            # Should be loaded correctly
            assert calc.is_loaded
            assert calc.model_count == 3

    def test_get_pricing_exact_match(self, mock_litellm_model_cost):
        """Test getting pricing with exact model match."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            pricing = calc.get_pricing("gpt-4o-mini")
            assert pricing is not None
            assert pricing.id == "gpt-4o-mini"
            assert abs(pricing.input - 0.150) < 0.001
            assert abs(pricing.output - 0.600) < 0.001

    def test_get_pricing_with_cached(self, mock_litellm_model_cost):
        """Test getting pricing with cached token pricing."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            pricing = calc.get_pricing("claude-sonnet-4-5-20250929")
            assert pricing is not None
            assert pricing.input_cached is not None
            assert abs(pricing.input_cached - 0.3) < 0.001

    def test_get_pricing_not_found(self, mock_litellm_model_cost):
        """Test getting pricing for unknown model."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            pricing = calc.get_pricing("unknown-model")
            assert pricing is None

    def test_get_pricing_lazy_load(self, mock_litellm_model_cost):
        """Test that get_pricing triggers lazy loading."""
        calc = CostCalculator()
        assert not calc.is_loaded

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost

            # Should trigger loading
            pricing = calc.get_pricing("gpt-4o-mini")
            assert calc.is_loaded
            assert pricing is not None

    def test_get_pricing_dynamic_lookup(self, mock_litellm_model_cost):
        """Test that get_pricing can look up models not in initial cache."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # Add a new model to litellm.model_cost after loading
            mock_litellm.model_cost["gpt-4-turbo"] = {
                "input_cost_per_token": 10.0 / 1_000_000,
                "output_cost_per_token": 30.0 / 1_000_000,
                "litellm_provider": "openai",
                "mode": "chat",
            }

            # Should find the new model via dynamic lookup
            pricing = calc.get_pricing("gpt-4-turbo")
            assert pricing is not None
            assert abs(pricing.input - 10.0) < 0.001

    def test_calculate_cost_with_pricing(self, mock_litellm_model_cost):
        """Test cost calculation with pricing data available."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # Calculate cost for 1000 input + 500 output tokens
            cost = calc.calculate_cost(
                model="gpt-4o-mini", input_tokens=1000, output_tokens=500
            )

            # Expected: (1000 / 1M * 0.15) + (500 / 1M * 0.60)
            # = 0.00015 + 0.0003 = 0.00045
            assert abs(cost - 0.00045) < 0.000001

    def test_calculate_cost_with_cached_tokens(self, mock_litellm_model_cost):
        """Test cost calculation with cached tokens."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # Calculate cost: 1000 input (500 cached) + 500 output
            cost = calc.calculate_cost(
                model="claude-sonnet-4-5-20250929",
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=500,
            )

            # Expected:
            # - Non-cached input: (500 / 1M * 3.0) = 0.0015
            # - Cached input: (500 / 1M * 0.3) = 0.00015
            # - Output: (500 / 1M * 15.0) = 0.0075
            # Total: 0.0015 + 0.00015 + 0.0075 = 0.00915
            assert abs(cost - 0.00915) < 0.000001

    def test_calculate_cost_cached_tokens_no_cached_pricing(
        self, mock_litellm_model_cost
    ):
        """Test cost calculation with cached tokens but no cached pricing."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # gpt-4o doesn't have cached pricing in mock data
            cost = calc.calculate_cost(
                model="gpt-4o",
                input_tokens=1000,
                output_tokens=500,
                cached_tokens=300,
            )

            # Expected (cached tokens treated as regular input):
            # - Non-cached input: ((1000-300) / 1M * 2.5) = 0.00175
            # - Cached input (no cached pricing, use regular): (300 / 1M * 2.5) = 0.00075
            # - Output: (500 / 1M * 10.0) = 0.005
            # Total: 0.00175 + 0.00075 + 0.005 = 0.0075
            assert abs(cost - 0.0075) < 0.000001

    def test_calculate_cost_fallback(self):
        """Test cost calculation fallback when pricing unavailable."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = {}  # Empty - no pricing data

            cost = calc.calculate_cost(
                model="unknown-model", input_tokens=1000, output_tokens=500
            )

            # Fallback uses conservative estimates:
            # Input: $10/M, Output: $30/M
            # Expected: (1000/1M * 10) + (500/1M * 30) = 0.01 + 0.015 = 0.025
            assert abs(cost - 0.025) < 0.000001

    def test_calculate_cost_zero_tokens(self, mock_litellm_model_cost):
        """Test cost calculation with zero tokens."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            cost = calc.calculate_cost(
                model="gpt-4o-mini", input_tokens=0, output_tokens=0
            )
            assert cost == 0.0

    def test_fallback_estimate(self):
        """Test fallback cost estimation."""
        calc = CostCalculator()

        # Test with various token counts
        cost = calc._fallback_estimate(
            input_tokens=1000, output_tokens=500, cached_tokens=0
        )

        # Input: $10/M * 1000 = $0.01
        # Output: $30/M * 500 = $0.015
        # Total: $0.025
        assert abs(cost - 0.025) < 0.000001

    def test_fallback_estimate_with_cached(self):
        """Test fallback estimation with cached tokens."""
        calc = CostCalculator()

        cost = calc._fallback_estimate(
            input_tokens=1000, output_tokens=500, cached_tokens=500
        )

        # Non-cached input: $10/M * 500 = $0.005
        # Cached input: $1/M * 500 = $0.0005
        # Output: $30/M * 500 = $0.015
        # Total: $0.0205
        assert abs(cost - 0.0205) < 0.000001

    def test_skips_sample_spec(self, mock_litellm_model_cost):
        """Test that sample_spec is skipped during loading."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # sample_spec should not be in cache
            assert "sample_spec" not in calc._pricing_cache

    def test_skips_non_chat_models(self, mock_litellm_model_cost):
        """Test that non-chat models (embeddings) are skipped during loading."""
        calc = CostCalculator()

        with patch("arbiter_ai.core.cost_calculator.litellm") as mock_litellm:
            mock_litellm.model_cost = mock_litellm_model_cost
            calc._load_pricing_data()

            # Embedding model should not be in cache
            assert "text-embedding-3-small" not in calc._pricing_cache


class TestGlobalCostCalculator:
    """Tests for global cost calculator singleton."""

    def test_get_cost_calculator_singleton(self):
        """Test that get_cost_calculator returns singleton."""
        # Reset global state for this test
        import arbiter_ai.core.cost_calculator as cc

        cc._cost_calculator = None

        calc1 = get_cost_calculator()
        calc2 = get_cost_calculator()

        assert calc1 is calc2

    def test_get_cost_calculator_returns_instance(self):
        """Test that get_cost_calculator returns CostCalculator."""
        calc = get_cost_calculator()

        assert isinstance(calc, CostCalculator)


class TestIntegrationWithEvaluationResult:
    """Integration tests with EvaluationResult."""

    @pytest.mark.asyncio
    async def test_evaluation_result_cost_calculation(self):
        """Test cost calculation in EvaluationResult."""
        from arbiter_ai.core.models import EvaluationResult, LLMInteraction, Score

        # Create evaluation result with interactions
        result = EvaluationResult(
            output="test output",
            scores=[Score(name="semantic", value=0.85)],
            overall_score=0.85,
            passed=True,
            processing_time=1.5,
            interactions=[
                LLMInteraction(
                    prompt="test prompt",
                    response="test response",
                    model="gpt-4o-mini",
                    input_tokens=1000,
                    output_tokens=500,
                    latency=1.2,
                    purpose="scoring",
                    cost=0.00045,  # Pre-calculated cost
                )
            ],
        )

        # Calculate total cost
        cost = await result.total_llm_cost()
        assert abs(cost - 0.00045) < 0.000001

    @pytest.mark.asyncio
    async def test_evaluation_result_cost_breakdown(self):
        """Test cost breakdown in EvaluationResult."""
        from arbiter_ai.core.models import EvaluationResult, LLMInteraction, Score

        result = EvaluationResult(
            output="test output",
            scores=[Score(name="semantic", value=0.85)],
            overall_score=0.85,
            passed=True,
            processing_time=1.5,
            interactions=[
                LLMInteraction(
                    prompt="prompt1",
                    response="response1",
                    model="gpt-4o-mini",
                    input_tokens=1000,
                    output_tokens=500,
                    latency=1.0,
                    purpose="semantic",
                    cost=0.00045,
                ),
                LLMInteraction(
                    prompt="prompt2",
                    response="response2",
                    model="claude-sonnet-4-5-20250929",
                    input_tokens=1500,
                    output_tokens=750,
                    latency=1.2,
                    purpose="factuality",
                    cost=0.015,
                ),
            ],
        )

        breakdown = await result.cost_breakdown()

        assert "total" in breakdown
        assert "by_evaluator" in breakdown
        assert "by_model" in breakdown
        assert "token_breakdown" in breakdown

        # Check total
        assert abs(breakdown["total"] - 0.01545) < 0.000001

        # Check by evaluator
        assert "semantic" in breakdown["by_evaluator"]
        assert "factuality" in breakdown["by_evaluator"]

        # Check by model
        assert "gpt-4o-mini" in breakdown["by_model"]
        assert "claude-sonnet-4-5-20250929" in breakdown["by_model"]

        # Check token breakdown
        assert breakdown["token_breakdown"]["input_tokens"] == 2500
        assert breakdown["token_breakdown"]["output_tokens"] == 1250
        assert breakdown["token_breakdown"]["total_tokens"] == 3750


class TestLiteLLMIntegration:
    """Tests for LiteLLM integration (uses real LiteLLM data)."""

    def test_real_litellm_has_common_models(self):
        """Test that real LiteLLM has pricing for common models."""
        import litellm

        # These models should exist in LiteLLM's bundled database
        common_models = ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"]

        for model in common_models:
            assert (
                model in litellm.model_cost
            ), f"Expected {model} in litellm.model_cost"
            assert litellm.model_cost[model].get("input_cost_per_token") is not None

    def test_real_calculator_loads_many_models(self):
        """Test that real calculator loads many models from LiteLLM."""
        calc = CostCalculator()
        calc._load_pricing_data()

        # Should load hundreds of models
        assert calc.model_count > 100, f"Expected >100 models, got {calc.model_count}"

    def test_real_calculator_cost_calculation(self):
        """Test cost calculation with real LiteLLM pricing."""
        calc = CostCalculator()
        calc._load_pricing_data()

        # Calculate cost for a known model
        cost = calc.calculate_cost(
            model="gpt-4o-mini",
            input_tokens=1000,
            output_tokens=500,
        )

        # Should return a reasonable cost (not zero, not huge)
        assert cost > 0, "Cost should be positive"
        assert cost < 1.0, "Cost should be less than $1 for this small usage"
