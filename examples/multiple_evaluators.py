"""Multiple Evaluators - Comprehensive Multi-Perspective Evaluation

This example demonstrates combining multiple evaluators to assess LLM outputs
from different perspectives, providing comprehensive quality analysis.

Key Features:
- Multi-evaluator composition (semantic + custom criteria)
- Aggregate scoring across different perspectives
- Individual score access and analysis
- Performance comparison between evaluators

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/multiple_evaluators.py
"""

from dotenv import load_dotenv

import asyncio
import os

from arbiter_ai import evaluate
from arbiter_ai.core import LLMManager


async def main():
    """Run multiple evaluators examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Multiple Evaluators Example")
    print("=" * 60)

    # Example 1: Combining semantic + custom criteria
    print("\n📝 Example 1: Semantic + Custom Criteria Evaluation")
    print("-" * 60)

    result1 = await evaluate(
        output="""Our new product offers revolutionary features including:
- Advanced AI-powered analytics
- Real-time collaboration tools
- Enterprise-grade security
- Seamless integration with popular platforms""",
        reference="Product description for enterprise software",
        criteria="Technical accuracy, clarity, completeness, professional tone",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result1.output[:80]}...")
    print(f"\n📊 Results:")
    print(f"  Overall Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")
    print(f"  Partial: {'⚠️ Yes' if result1.partial else '✅ No'}")
    print(f"  Evaluators Used: {', '.join(result1.evaluator_names)}")

    print("\n  Individual Scores:")
    for score in result1.scores:
        print(f"\n    {score.name}:")
        print(f"      Value: {score.value:.3f}")
        if score.confidence:
            print(f"      Confidence: {score.confidence:.3f}")
        if score.explanation:
            print(f"      Explanation: {score.explanation[:100]}...")

    if result1.errors:
        print("\n  ⚠️ Errors:")
        for evaluator, error_msg in result1.errors.items():
            print(f"    - {evaluator}: {error_msg}")

    # Cost tracking
    breakdown1 = await result1.cost_breakdown()
    print(f"\n💰 Cost Analysis:")
    print(f"  Total Cost: ${breakdown1['total']:.6f}")
    print(f"  Tokens: {breakdown1['token_breakdown']['total_tokens']:,}")

    # Example 2: Understanding combined scores
    print("\n\n📝 Example 2: Understanding Combined Scores")
    print("-" * 60)

    print("""
When using multiple evaluators:
- Each evaluator provides its own score
- Overall score = average of all successful evaluator scores
- Failed evaluators are excluded from the average
- You can inspect individual scores for detailed analysis

Example:
  evaluators = ["semantic", "custom_criteria"]

  If both succeed:
  - semantic_score = 0.85
  - custom_criteria_score = 0.75
  - overall_score = (0.85 + 0.75) / 2 = 0.80

  If custom_criteria fails:
  - semantic_score = 0.85
  - overall_score = 0.85 (not average with 0)
  - result.partial = True
  - result.errors = {"custom_criteria": "error message"}
    """)

    # Example 3: Using scores for decision making
    print("\n📝 Example 3: Using Multiple Scores for Decision Making")
    print("-" * 60)

    result2 = await evaluate(
        output="""The patient should take 500mg of ibuprofen every 6 hours.
This is a safe dosage for adults. Make sure to take with food to avoid
stomach upset.""",
        reference="Ibuprofen dosage guidelines",
        criteria="Medical accuracy, patient safety, clarity, appropriate tone",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result2.output[:80]}...")
    print(f"\n📊 Multi-Evaluator Analysis:")

    # Analyze each score
    semantic_score = next((s for s in result2.scores if s.name == "semantic_similarity"), None)
    custom_score = next((s for s in result2.scores if s.name == "custom_criteria"), None)

    if semantic_score:
        print(f"\n  Semantic Similarity: {semantic_score.value:.3f}")
        print(f"    → Measures how well output matches reference meaning")

    if custom_score:
        print(f"\n  Custom Criteria: {custom_score.value:.3f}")
        if custom_score.metadata.get("criteria_met"):
            print(f"    ✅ Met: {', '.join(custom_score.metadata['criteria_met'])}")
        if custom_score.metadata.get("criteria_not_met"):
            print(f"    ❌ Not Met: {', '.join(custom_score.metadata['criteria_not_met'])}")

    print(f"\n  Overall Score: {result2.overall_score:.3f}")
    print(f"    → Average of successful evaluators: {', '.join(result2.evaluator_names)}")

    # Decision logic
    if result2.overall_score >= 0.8:
        print("\n  ✅ Decision: HIGH QUALITY - Use this output")
    elif result2.overall_score >= 0.6:
        print("\n  ⚠️ Decision: MODERATE QUALITY - Review before use")
    else:
        print("\n  ❌ Decision: LOW QUALITY - Needs improvement")

    # Example 4: Handling partial results
    print("\n\n📝 Example 4: Handling Partial Results")
    print("-" * 60)

    print("""
When using multiple evaluators, some may fail. Here's how to handle it:

1. Check result.partial to see if any evaluators failed
2. Check result.errors for specific failure details
3. Use successful scores even if some failed
4. Log or alert on failures for monitoring

Best Practice:
  if result.partial:
      logger.warning(f"Partial result: {len(result.errors)} evaluators failed")
      for evaluator, error in result.errors.items():
          logger.warning(f"  {evaluator}: {error}")

  # Still use the result if we have scores
  if result.scores:
      use_score = result.overall_score
    """)

    # Example 5: Real-world use case - Content quality check
    print("\n📝 Example 5: Real-World Use Case - Content Quality Check")
    print("-" * 60)

    result3 = await evaluate(
        output="""Introducing our revolutionary new smartphone! With cutting-edge
technology and sleek design, this phone will transform how you communicate.
Features include 128GB storage, 48MP camera, and all-day battery life.
Order now and get 50% off!""",
        criteria="Accuracy, persuasiveness, brand voice, completeness",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result3.output[:80]}...")
    print(f"\n📊 Quality Assessment:")

    for score in result3.scores:
        print(f"\n  {score.name}: {score.value:.3f}")
        if score.metadata.get("criteria_met"):
            print(f"    ✅ Strengths: {', '.join(score.metadata['criteria_met'])}")
        if score.metadata.get("criteria_not_met"):
            print(f"    ⚠️ Areas for Improvement: {', '.join(score.metadata['criteria_not_met'])}")

    print(f"\n  Overall Quality: {result3.overall_score:.3f}")
    if result3.overall_score >= 0.8:
        print("    → Ready for publication")
    elif result3.overall_score >= 0.6:
        print("    → Needs revision")
    else:
        print("    → Requires significant improvement")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")

    # Session cost summary
    cost2 = await result2.total_llm_cost()
    cost3 = await result3.total_llm_cost()
    total_cost = breakdown1['total'] + cost2 + cost3
    total_tokens = result1.total_tokens + result2.total_tokens + result3.total_tokens

    print(f"\n💰 Total Session Cost:")
    print(f"  Total Evaluations: 5")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens:,}")
    print(f"  Average per Evaluation: ${total_cost / 5:.6f}")

    print("\n📚 Key Features Demonstrated:")
    print("  • Combining multiple evaluators for comprehensive assessment")
    print("  • Understanding how overall_score is calculated")
    print("  • Using individual scores for detailed analysis")
    print("  • Handling partial results when some evaluators fail")
    print("  • Real-world decision-making based on multiple scores")
    print("  • Automatic cost tracking across all evaluators")

    print("\n📖 Related Examples:")
    print("  • See custom_criteria_example.py for detailed criteria evaluation")
    print("  • See error_handling_example.py for robust error management")
    print("  • See rag_evaluation.py for multi-evaluator RAG assessment")


if __name__ == "__main__":
    asyncio.run(main())

