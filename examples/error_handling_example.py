"""Error Handling - Graceful Degradation & Partial Failures

This example demonstrates Arbiter's robust error handling with graceful
degradation when evaluators fail, ensuring reliable evaluation pipelines.

Key Features:
- Partial failure handling (continue with successful evaluators)
- Detailed error reporting and diagnostics
- Graceful degradation strategies
- Production-ready error resilience

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/error_handling_example.py
"""

from dotenv import load_dotenv

import asyncio
import os

from arbiter_ai import evaluate
from arbiter_ai.core import LLMManager
from arbiter_ai.core.exceptions import EvaluatorError


async def main():
    """Run error handling examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Error Handling Example")
    print("=" * 60)

    # Example 1: Successful evaluation (no errors)
    print("\n📝 Example 1: Successful Evaluation (No Errors)")
    print("-" * 60)

    result1 = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result1.output}")
    print(f"\n📊 Results:")
    print(f"  Overall Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")
    print(f"  Partial: {'⚠️ Yes' if result1.partial else '✅ No'}")
    print(f"  Errors: {len(result1.errors)}")
    print(f"  Successful Evaluators: {result1.metadata.get('successful_evaluators', 0)}")
    print(f"  Failed Evaluators: {result1.metadata.get('failed_evaluators', 0)}")

    if result1.errors:
        print("\n  ⚠️ Errors Encountered:")
        for evaluator, error_msg in result1.errors.items():
            print(f"    - {evaluator}: {error_msg}")

    # Cost tracking
    breakdown1 = await result1.cost_breakdown()
    print(f"\n💰 Cost Analysis:")
    print(f"  Total Cost: ${breakdown1['total']:.6f}")
    print(f"  Tokens: {breakdown1['token_breakdown']['total_tokens']:,}")

    # Example 2: Demonstrating partial results concept
    print("\n\n📝 Example 2: Understanding Partial Results")
    print("-" * 60)

    print("""
When using multiple evaluators, if one fails:
- ✅ Successful evaluators still return scores
- ⚠️ Failed evaluators are tracked in result.errors
- 📊 result.partial = True if any errors occurred
- 🎯 Overall score is calculated from successful evaluators only

Example scenario:
  evaluators = ["semantic", "factuality", "toxicity"]

  If factuality fails:
  - result.scores = [semantic_score, toxicity_score]
  - result.errors = {"factuality": "API timeout"}
  - result.partial = True
  - result.overall_score = average(semantic_score, toxicity_score)
    """)

    # Example 3: Checking for errors in results
    print("\n📝 Example 3: Checking for Errors in Results")
    print("-" * 60)

    result2 = await evaluate(
        output="Medical advice about diabetes management",
        criteria="Medical accuracy, HIPAA compliance, appropriate tone",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result2.output[:60]}...")
    print(f"\n📊 Results:")
    print(f"  Partial: {'⚠️ Yes - Some evaluators failed' if result2.partial else '✅ No - All succeeded'}")

    # Check for errors
    if result2.partial:
        print("\n  ⚠️ Partial Result Detected:")
        print(f"    Successful: {result2.metadata.get('successful_evaluators', 0)}")
        print(f"    Failed: {result2.metadata.get('failed_evaluators', 0)}")
        if result2.errors:
            print("\n    Errors:")
            for evaluator, error_msg in result2.errors.items():
                print(f"      - {evaluator}: {error_msg}")
    else:
        print("\n  ✅ All evaluators succeeded")
        print(f"    Scores: {len(result2.scores)}")
        for score in result2.scores:
            print(f"      - {score.name}: {score.value:.3f}")

    # Example 4: Error handling best practices
    print("\n\n📝 Example 4: Error Handling Best Practices")
    print("-" * 60)

    print("""
Best Practices:

1. Always check result.partial after evaluation:
   if result.partial:
       # Handle partial results
       logger.warning(f"Partial result: {len(result.errors)} evaluators failed")

2. Check result.errors for specific failures:
   if "factuality" in result.errors:
       # Factuality evaluator failed
       fallback_score = calculate_fallback_score()

3. Use successful scores even with partial results:
   if result.scores:
       # At least some evaluators succeeded
       overall_score = result.overall_score
       # Use this score, but note it's partial

4. Handle all-failure case (raises EvaluatorError):
   try:
       result = await evaluate(...)
   except EvaluatorError as e:
       if "All evaluators failed" in str(e):
           # Complete failure - handle accordingly
           logger.error("All evaluators failed")
    """)

    # Example 5: Practical error handling pattern
    print("\n📝 Example 5: Practical Error Handling Pattern")
    print("-" * 60)

    try:
        result3 = await evaluate(
            output="Test output for evaluation",
            reference="Test reference",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        # Check if we got a partial result
        if result3.partial:
            print("⚠️ Warning: Partial result received")
            print(f"   Successful evaluators: {result3.metadata.get('successful_evaluators', 0)}")
            print(f"   Failed evaluators: {result3.metadata.get('failed_evaluators', 0)}")

            # Log errors
            for evaluator, error_msg in result3.errors.items():
                print(f"   Error in {evaluator}: {error_msg}")

            # Decide if we can use the result
            if result3.scores:
                print(f"\n✅ Using partial result with score: {result3.overall_score:.3f}")
                print("   Note: Some evaluators failed, but we have usable scores")
            else:
                print("\n❌ No usable scores - all evaluators failed")
        else:
            print("✅ Complete result - all evaluators succeeded")
            print(f"   Score: {result3.overall_score:.3f}")

    except EvaluatorError as e:
        if "All evaluators failed" in str(e):
            print("❌ Complete failure - all evaluators failed")
            print(f"   Error: {e}")
        else:
            print(f"❌ Evaluation error: {e}")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")

    # Session cost summary
    cost2 = await result2.total_llm_cost()
    total_cost = breakdown1['total'] + cost2
    total_tokens = result1.total_tokens + result2.total_tokens

    print(f"\n💰 Total Session Cost:")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens:,}")

    print("\n📚 Key Features Demonstrated:")
    print("  • Partial result detection (result.partial)")
    print("  • Error tracking (result.errors)")
    print("  • Graceful degradation (use successful scores)")
    print("  • Error handling best practices")
    print("  • Automatic cost tracking even with errors")

    print("\n📖 Related Examples:")
    print("  • See multiple_evaluators.py for multi-evaluator error handling")
    print("  • See batch_evaluation_example.py for batch error scenarios")
    print("  • See middleware_usage.py for production error monitoring")


if __name__ == "__main__":
    asyncio.run(main())

