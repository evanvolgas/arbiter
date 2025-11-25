"""Pairwise Comparison - A/B Testing & Model Selection

This example demonstrates comparing two LLM outputs to determine which is better,
essential for A/B testing, model selection, and prompt optimization.

Key Features:
- Direct A/B comparison of outputs
- Preference scoring with confidence levels
- Detailed reasoning for preferences
- Production-ready model selection workflows

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/pairwise_comparison_example.py
"""

from dotenv import load_dotenv

import asyncio
import os

from arbiter_ai import compare, PairwiseComparisonEvaluator
from arbiter_ai.core import LLMManager


async def main():
    """Run pairwise comparison examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Pairwise Comparison Example")
    print("=" * 60)

    # Example 1: Basic comparison with criteria
    print("\n📝 Example 1: Basic Comparison with Criteria")
    print("-" * 60)

    comparison1 = await compare(
        output_a="GPT-4 response: Paris is the capital of France, founded in 3rd century BC.",
        output_b="Claude response: The capital of France is Paris, established around 250 BC.",
        criteria="accuracy, clarity, completeness",
        reference="What is the capital of France?",
        model="gpt-4o-mini",
    )

    print(f"Output A: {comparison1.output_a[:60]}...")
    print(f"Output B: {comparison1.output_b[:60]}...")
    print("\n📊 Comparison Results:")
    print(f"  Winner: {comparison1.winner.upper()}")
    print(f"  Confidence: {comparison1.confidence:.3f}")
    print(f"  Processing Time: {comparison1.processing_time:.2f}s")

    print("\n  Reasoning:")
    print(f"    {comparison1.reasoning[:200]}...")

    if comparison1.aspect_scores:
        print("\n  Aspect Scores:")
        for aspect, scores in comparison1.aspect_scores.items():
            print(f"    {aspect}:")
            print(f"      Output A: {scores['output_a']:.3f}")
            print(f"      Output B: {scores['output_b']:.3f}")
            diff = scores["output_a"] - scores["output_b"]
            if abs(diff) > 0.05:
                better = "A" if diff > 0 else "B"
                print(f"      → {better} is better by {abs(diff):.3f}")

    # Example 2: Model comparison (no criteria)
    print("\n\n📝 Example 2: Model Comparison (Overall Quality)")
    print("-" * 60)

    comparison2 = await compare(
        output_a="""Our product is amazing! It has lots of features and everyone loves it.
You should definitely buy it because it's the best thing ever!""",
        output_b="""Our product offers advanced features including real-time collaboration,
end-to-end encryption, and seamless integration with popular tools. It's designed
to improve productivity for teams of all sizes.""",
        model="gpt-4o-mini",
    )

    print(f"Output A: {comparison2.output_a[:80]}...")
    print(f"Output B: {comparison2.output_b[:80]}...")
    print("\n📊 Comparison Results:")
    print(f"  Winner: {comparison2.winner.upper()}")
    print(f"  Confidence: {comparison2.confidence:.3f}")

    print("\n  Reasoning:")
    print(f"    {comparison2.reasoning[:200]}...")

    # Example 3: Tie case
    print("\n\n📝 Example 3: Equivalent Outputs (Tie)")
    print("-" * 60)

    comparison3 = await compare(
        output_a="The weather today is sunny and warm, perfect for outdoor activities.",
        output_b="Today's weather is warm and sunny, ideal for activities outside.",
        criteria="clarity, completeness",
        model="gpt-4o-mini",
    )

    print(f"Output A: {comparison3.output_a}")
    print(f"Output B: {comparison3.output_b}")
    print("\n📊 Comparison Results:")
    print(f"  Winner: {comparison3.winner.upper()}")
    print(f"  Confidence: {comparison3.confidence:.3f}")

    if comparison3.winner == "tie":
        print("\n  ✅ Outputs are equivalent in quality")

    # Example 4: Using evaluator directly for more control
    print("\n\n📝 Example 4: Direct Evaluator Usage")
    print("-" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = PairwiseComparisonEvaluator(client)

    # Run comparison
    comparison4 = await evaluator.compare(
        output_a="""To install the package, run: pip install mypackage.
Then import it: import mypackage. Use it like: mypackage.doSomething()""",
        output_b="""Installation:
1. Run: pip install mypackage
2. Import: import mypackage
3. Usage: mypackage.doSomething()

For more details, see the documentation.""",
        criteria="clarity, completeness, helpfulness",
        reference="How do I install and use mypackage?",
    )

    print("Output A: Installation instructions (see code)")
    print("Output B: Installation instructions with steps (see code)")
    print("\n📊 Comparison Results:")
    print(f"  Winner: {comparison4.winner.upper()}")
    print(f"  Confidence: {comparison4.confidence:.3f}")

    if comparison4.aspect_scores:
        print("\n  Detailed Aspect Comparison:")
        for aspect, scores in comparison4.aspect_scores.items():
            a_score = scores["output_a"]
            b_score = scores["output_b"]
            winner_aspect = "A" if a_score > b_score else "B" if b_score > a_score else "Tie"
            print(f"    {aspect}:")
            print(f"      A: {a_score:.3f} | B: {b_score:.3f} → {winner_aspect}")

    # Access interactions directly
    print("\n🔬 Evaluator Interactions:")
    interactions = evaluator.get_interactions()
    print(f"  Total Calls: {len(interactions)}")
    print(f"  Total Tokens: {sum(i.tokens_used for i in interactions)}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    # Calculate cost
    cost = await comparison4.total_llm_cost()
    print(f"  Estimated Cost: ${cost:.6f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Pairwise comparison with criteria")
    print("  • Overall quality comparison (no criteria)")
    print("  • Tie detection for equivalent outputs")
    print("  • Aspect-level scoring and comparison")
    print("  • Both high-level API and direct evaluator usage")
    print("  • Automatic interaction tracking and cost calculation")
    print("\nUse Cases:")
    print("  • A/B testing different prompts")
    print("  • Comparing model outputs")
    print("  • Selecting best output from candidates")
    print("  • Evaluating relative quality differences")

    print("\n📖 Related Examples:")
    print("  • See batch_evaluation_example.py for comparing multiple outputs")
    print("  • See multiple_evaluators.py for multi-evaluator comparison")
    print("  • See semantic_example.py for similarity-based evaluation")


if __name__ == "__main__":
    asyncio.run(main())
