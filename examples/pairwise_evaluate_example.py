"""Pairwise Evaluator with evaluate() Method

This example demonstrates the new evaluate() functionality for PairwiseComparisonEvaluator.
When evaluate() is called with a single output, it compares that output against the
reference text to determine quality.

Key Features:
- Single output comparison against reference
- Automatic score conversion (winner → high score, loser → low score, tie → medium)
- Works seamlessly with the standard evaluate() API
- Compatible with other evaluators in multi-evaluator workflows

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/pairwise_evaluate_example.py
"""

import asyncio
import os

from dotenv import load_dotenv

from arbiter import PairwiseComparisonEvaluator
from arbiter.core import LLMManager


async def main():
    """Run pairwise evaluate() examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Pairwise evaluate() Method Example")
    print("=" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = PairwiseComparisonEvaluator(client)

    # Example 1: Output better than reference
    print("\n📝 Example 1: Output Better Than Reference")
    print("-" * 60)

    score1 = await evaluator.evaluate(
        output="Paris is the capital of France. It was founded around 250 BC by the Parisii tribe and became the capital in 508 CE.",
        reference="Paris is the capital of France.",
        criteria="completeness, accuracy, clarity",
    )

    print(f"Output: Paris is the capital of France. [detailed version]")
    print(f"Reference: Paris is the capital of France. [basic version]")
    print(f"\n📊 Evaluation Results:")
    print(f"  Score: {score1.value:.3f} (high = output better)")
    print(f"  Confidence: {score1.confidence:.3f}")
    print(f"  Winner: {score1.metadata.get('winner')}")
    print(f"\n  Explanation:")
    explanation_lines = score1.explanation.split("\n")
    for line in explanation_lines[:3]:  # Show first 3 lines
        print(f"    {line}")

    # Example 2: Reference better than output
    print("\n\n📝 Example 2: Reference Better Than Output")
    print("-" * 60)

    score2 = await evaluator.evaluate(
        output="France has a capital.",
        reference="Paris is the capital of France, a major European city with over 2 million residents.",
        criteria="completeness, specificity, informativeness",
    )

    print(f"Output: France has a capital. [vague]")
    print(f"Reference: Paris is the capital... [detailed]")
    print(f"\n📊 Evaluation Results:")
    print(f"  Score: {score2.value:.3f} (low = reference better)")
    print(f"  Confidence: {score2.confidence:.3f}")
    print(f"  Winner: {score2.metadata.get('winner')}")

    # Example 3: Tie (equivalent quality)
    print("\n\n📝 Example 3: Equivalent Quality (Tie)")
    print("-" * 60)

    score3 = await evaluator.evaluate(
        output="The capital of France is Paris.",
        reference="Paris is the capital of France.",
        criteria="accuracy, clarity",
    )

    print(f"Output: The capital of France is Paris.")
    print(f"Reference: Paris is the capital of France.")
    print(f"\n📊 Evaluation Results:")
    print(f"  Score: {score3.value:.3f} (medium = equivalent quality)")
    print(f"  Confidence: {score3.confidence:.3f}")
    print(f"  Winner: {score3.metadata.get('winner')}")

    # Example 4: Without criteria (overall quality comparison)
    print("\n\n📝 Example 4: Overall Quality Comparison (No Criteria)")
    print("-" * 60)

    score4 = await evaluator.evaluate(
        output="Machine learning is a subset of AI that uses algorithms to learn from data.",
        reference="ML is part of AI.",
    )

    print(f"Output: Machine learning is a subset... [detailed explanation]")
    print(f"Reference: ML is part of AI. [brief]")
    print(f"\n📊 Evaluation Results:")
    print(f"  Score: {score4.value:.3f}")
    print(f"  Confidence: {score4.confidence:.3f}")
    print(f"  Winner: {score4.metadata.get('winner')}")

    # Example 5: Integration with multi-evaluator workflow
    print("\n\n📝 Example 5: Multi-Evaluator Workflow")
    print("-" * 60)
    print("Demonstrating evaluate() compatibility with other evaluators:")

    from arbiter import SemanticEvaluator

    semantic = SemanticEvaluator(client)

    test_output = "Python is a high-level programming language."
    test_reference = "Python is a programming language used for software development."

    # Run both evaluators
    pairwise_score = await evaluator.evaluate(
        output=test_output, reference=test_reference, criteria="accuracy, completeness"
    )

    semantic_score = await semantic.evaluate(
        output=test_output, reference=test_reference
    )

    print(f"\nOutput: {test_output[:50]}...")
    print(f"Reference: {test_reference[:50]}...")
    print(f"\n  Pairwise Score: {pairwise_score.value:.3f} (winner: {pairwise_score.metadata.get('winner')})")
    print(f"  Semantic Score: {semantic_score.value:.3f}")

    # Calculate cost and interactions
    print("\n\n🔬 Evaluator Statistics:")
    print("-" * 60)
    interactions = evaluator.get_interactions()
    print(f"  Total LLM Calls: {len(interactions)}")
    print(f"  Total Tokens: {sum(i.tokens_used for i in interactions)}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    if interactions and interactions[0].cost:
        total_cost = sum(i.cost for i in interactions if i.cost)
        print(f"  Estimated Cost: ${total_cost:.6f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • evaluate() compares output against reference")
    print("  • Winner → Score mapping (high/medium/low)")
    print("  • Confidence-modulated scoring")
    print("  • Works with or without criteria")
    print("  • Compatible with multi-evaluator workflows")
    print("  • Automatic interaction tracking")

    print("\nWhen to Use evaluate() vs compare():")
    print("  • evaluate(output, reference): Compare one output against reference")
    print("  • compare(output_a, output_b): Explicit A/B comparison of two outputs")

    print("\n📖 Related Examples:")
    print("  • See pairwise_comparison_example.py for compare() usage")
    print("  • See multiple_evaluators.py for multi-evaluator workflows")
    print("  • See semantic_example.py for semantic similarity evaluation")


if __name__ == "__main__":
    asyncio.run(main())
