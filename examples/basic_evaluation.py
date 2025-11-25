"""Basic Evaluation - Getting Started with Arbiter

This example provides a comprehensive introduction to Arbiter's core evaluation
features, demonstrating semantic similarity assessment and interaction tracking.

Key Features:
- Semantic similarity evaluation
- Automatic LLM interaction tracking
- Cost calculation from token usage
- Score confidence levels and explanations
- Both high-level API and direct evaluator usage

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/basic_evaluation.py
"""

from dotenv import load_dotenv

import asyncio
import os

from arbiter_ai import evaluate
from arbiter_ai.core import LLMManager


async def main():
    """Run a basic evaluation with interaction tracking."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Basic Evaluation Example")
    print("=" * 50)

    # Example 1: High semantic similarity
    print("\n📝 Example 1: High Semantic Similarity")
    print("-" * 50)

    result1 = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini",  # Use mini for cost efficiency
    )

    print(f"Output: {result1.output}")
    print(f"Reference: {result1.reference}")
    print(f"\n📊 Results:")
    print(f"  Overall Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")
    print(f"  Processing Time: {result1.processing_time:.2f}s")

    # Show detailed score info
    for score in result1.scores:
        print(f"\n  {score.name}:")
        print(f"    Value: {score.value:.3f}")
        if score.confidence:
            print(f"    Confidence: {score.confidence:.3f}")
        if score.explanation:
            print(f"    Explanation: {score.explanation[:100]}...")

    # Show LLM interaction tracking
    print(f"\n🔬 LLM Interaction Tracking:")
    print(f"  Total LLM Calls: {len(result1.interactions)}")

    for i, interaction in enumerate(result1.interactions, 1):
        print(f"\n  Call {i}:")
        print(f"    Purpose: {interaction.purpose}")
        print(f"    Model: {interaction.model}")
        print(f"    Latency: {interaction.latency:.2f}s")
        print(f"    Tokens: {interaction.tokens_used:,}")
        print(f"      • Input: {interaction.input_tokens:,}")
        print(f"      • Output: {interaction.output_tokens:,}")
        print(f"    Timestamp: {interaction.timestamp.strftime('%H:%M:%S')}")

    # Cost breakdown
    print(f"\n💰 Cost Analysis:")
    breakdown1 = await result1.cost_breakdown()
    print(f"  Total Cost: ${breakdown1['total']:.6f}")
    print(f"  Token Breakdown:")
    print(f"    • Input tokens: {breakdown1['token_breakdown']['input_tokens']:,}")
    print(f"    • Output tokens: {breakdown1['token_breakdown']['output_tokens']:,}")
    print(f"    • Total tokens: {breakdown1['token_breakdown']['total_tokens']:,}")

    # Example 2: Lower semantic similarity
    print("\n\n📝 Example 2: Lower Semantic Similarity")
    print("-" * 50)

    result2 = await evaluate(
        output="The weather is nice today",
        reference="Python is a programming language",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result2.output}")
    print(f"Reference: {result2.reference}")
    print(f"\n📊 Results:")
    print(f"  Overall Score: {result2.overall_score:.3f}")
    print(f"  Passed: {'✅' if result2.passed else '❌'}")

    # Show cost for comparison
    cost2 = await result2.total_llm_cost()
    print(f"\n💰 Cost: ${cost2:.6f}")
    print(f"  Tokens: {result2.total_tokens:,}")

    # Example 3: Using the evaluator directly for more control
    print("\n\n📝 Example 3: Direct Evaluator Usage")
    print("-" * 50)

    from arbiter_ai.evaluators import SemanticEvaluator

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = SemanticEvaluator(client)

    # Run evaluation
    score = await evaluator.evaluate(
        output="The quick brown fox jumps over the lazy dog",
        reference="A fast brown fox leaps above a sleepy canine",
    )

    print(f"Semantic Score: {score.value:.3f}")
    print(f"Confidence: {score.confidence:.3f}")
    print(f"\nExplanation:")
    print(f"  {score.explanation}")

    # Access interactions directly from evaluator
    print(f"\n🔬 Evaluator Interactions:")
    interactions = evaluator.get_interactions()
    total_tokens = sum(i.tokens_used for i in interactions)
    print(f"  Total Calls: {len(interactions)}")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    # Summary with total costs
    print("\n\n" + "=" * 50)
    print("✅ Examples Complete!")

    # Calculate total session cost
    total_cost = breakdown1['total'] + cost2
    total_tokens_all = result1.total_tokens + result2.total_tokens
    print(f"\n💰 Total Session Cost:")
    print(f"  Total Evaluations: 3")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens_all:,}")
    print(f"  Average per Evaluation: ${total_cost / 3:.6f}")

    print("\n📚 Key Features Demonstrated:")
    print("  • Semantic similarity evaluation")
    print("  • Automatic LLM interaction tracking")
    print("  • Detailed cost tracking and token breakdown")
    print("  • Score confidence levels and explanations")
    print("  • Both high-level API and direct evaluator usage")

    print("\n📖 Next Steps:")
    print("  • See observability_example.py for comprehensive cost analysis")
    print("  • See batch_evaluation_example.py for parallel processing")
    print("  • See multiple_evaluators.py for multi-perspective evaluation")


if __name__ == "__main__":
    asyncio.run(main())
