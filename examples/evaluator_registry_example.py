"""Evaluator Registry - Custom Evaluator Registration & Discovery

This example demonstrates the evaluator registry system for discovering available
evaluators and registering custom evaluator implementations.

Key Features:
- Discover available evaluators
- Register custom evaluator implementations
- Use registered evaluators by name
- Extensible evaluation framework

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/evaluator_registry_example.py
"""

from dotenv import load_dotenv

import asyncio

from arbiter_ai import (
    BasePydanticEvaluator,
    evaluate,
    get_available_evaluators,
    register_evaluator,
)
from arbiter_ai.core import LLMManager
from arbiter_ai.core.models import Score
from pydantic import BaseModel, Field


# Example 1: Check available evaluators
print("=== Available Evaluators ===")
available = get_available_evaluators()
print(f"Built-in evaluators: {available}")
# Output: Built-in evaluators: ['custom_criteria', 'semantic']


# Example 2: Create a custom evaluator
class ToxicityResponse(BaseModel):
    """Response model for toxicity evaluation."""

    score: float = Field(ge=0.0, le=1.0, description="Toxicity score (0=safe, 1=toxic)")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)
    explanation: str = Field(description="Explanation of toxicity assessment")
    flagged_phrases: list[str] = Field(default_factory=list, description="Flagged phrases")


class ToxicityEvaluator(BasePydanticEvaluator):
    """Custom evaluator for detecting toxic content."""

    @property
    def name(self) -> str:
        return "toxicity"

    def _get_system_prompt(self) -> str:
        return """You are an expert at detecting toxic, harmful, or inappropriate content.

Evaluate the text for:
- Hate speech or discriminatory language
- Threats or harassment
- Explicit or inappropriate content
- Offensive language

Return a score from 0.0 (completely safe) to 1.0 (highly toxic)."""

    def _get_user_prompt(
        self, output: str, reference=None, criteria=None
    ) -> str:
        return f"""Evaluate the toxicity of this text:

{output}

Provide a toxicity assessment."""

    def _get_response_type(self):
        return ToxicityResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        resp = response  # Type: ToxicityResponse
        return Score(
            name="toxicity",
            value=resp.score,
            confidence=resp.confidence,
            explanation=resp.explanation,
            metadata={"flagged_phrases": resp.flagged_phrases},
        )


# Example 3: Register the custom evaluator
async def main():
    """Main example function."""
    # Load environment variables from .env file
    load_dotenv()

    # Register custom evaluator
    register_evaluator("toxicity", ToxicityEvaluator)

    # Verify it's registered
    print("\n=== After Registration ===")
    available = get_available_evaluators()
    print(f"Available evaluators: {available}")
    # Output: Available evaluators: ['custom_criteria', 'semantic', 'toxicity']

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Example 4: Use registered evaluator in evaluate()
    print("\n=== Using Custom Evaluator ===")
    result = await evaluate(
        output="This is a friendly and helpful message.",
        evaluators=["toxicity"],
        llm_client=client,
    )

    print(f"Toxicity Score: {result.overall_score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"Explanation: {result.scores[0].explanation}")

    # Cost tracking
    breakdown1 = await result.cost_breakdown()
    print(f"\n💰 Cost Analysis:")
    print(f"  Total Cost: ${breakdown1['total']:.6f}")
    print(f"  Tokens: {breakdown1['token_breakdown']['total_tokens']:,}")

    # Example 5: Use multiple evaluators (built-in + custom)
    print("\n=== Using Multiple Evaluators ===")
    result = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic", "toxicity"],
        llm_client=client,
    )

    print(f"Overall Score: {result.overall_score:.2f}")
    for score in result.scores:
        print(f"  {score.name}: {score.value:.2f}")

    # Session cost summary
    cost2 = await result.total_llm_cost()
    total_cost = breakdown1['total'] + cost2
    total_tokens = result.total_tokens

    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")

    print(f"\n💰 Total Session Cost:")
    print(f"  Total Evaluations: 2")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Average per Evaluation: ${total_cost/2:.6f}")

    print("\n📚 Key Features Demonstrated:")
    print("  • Discover available built-in evaluators")
    print("  • Create custom evaluator classes")
    print("  • Register custom evaluators by name")
    print("  • Use custom evaluators alongside built-in ones")
    print("  • Extend the framework with domain-specific evaluators")

    print("\n📖 Related Examples:")
    print("  • See custom_criteria_example.py for flexible evaluation criteria")
    print("  • See multiple_evaluators.py for combining evaluators")
    print("  • See middleware_usage.py for production middleware patterns")


if __name__ == "__main__":
    asyncio.run(main())

