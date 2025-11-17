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

from arbiter import evaluate
from arbiter.core import LLMManager


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
        print(f"    Tokens Used: {interaction.tokens_used}")
        print(f"    Timestamp: {interaction.timestamp.strftime('%H:%M:%S')}")

    # Calculate cost
    cost1 = await result1.total_llm_cost()
    print(f"\n💰 Estimated Cost: ${cost1:.6f}")

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

    # Example 3: Using the evaluator directly for more control
    print("\n\n📝 Example 3: Direct Evaluator Usage")
    print("-" * 50)

    from arbiter.evaluators import SemanticEvaluator

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
    print(f"  Total Calls: {len(interactions)}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    # Summary
    print("\n\n" + "=" * 50)
    print("✅ Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Semantic similarity evaluation")
    print("  • Automatic LLM interaction tracking")
    print("  • Cost calculation from token usage")
    print("  • Score confidence levels")
    print("  • Both high-level API and direct evaluator usage")


if __name__ == "__main__":
    asyncio.run(main())
