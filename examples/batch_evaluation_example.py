"""Batch Evaluation - Parallel Processing with Progress Tracking

This example demonstrates efficient batch evaluation with parallel processing,
progress tracking, and graceful handling of partial failures.

Key Features:
- Parallel batch processing with concurrency control
- Progress tracking with custom callbacks
- Graceful handling of partial failures
- Multiple evaluators per item
- Comprehensive cost tracking and breakdown
- Individual result and error access

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/batch_evaluation_example.py
"""

import asyncio
import os
from typing import Optional

from dotenv import load_dotenv

from arbiter_ai import EvaluationResult, batch_evaluate


async def main():
    """Run batch evaluation examples."""
    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Batch Evaluation Example")
    print("=" * 60)

    # Example 1: Basic batch evaluation
    print("\n📝 Example 1: Basic Batch Evaluation (5 items)")
    print("-" * 60)

    items = [
        {
            "output": "Paris is the capital of France",
            "reference": "The capital of France is Paris",
        },
        {
            "output": "Tokyo is the capital of Japan",
            "reference": "The capital of Japan is Tokyo",
        },
        {
            "output": "Berlin is the capital of Germany",
            "reference": "The capital of Germany is Berlin",
        },
        {
            "output": "London is the capital of England",
            "reference": "The capital of England is London",
        },
        {
            "output": "Rome is the capital of Italy",
            "reference": "The capital of Italy is Rome",
        },
    ]

    result = await batch_evaluate(
        items=items,
        evaluators=["semantic"],
        model="gpt-4o-mini",
        max_concurrency=3,  # Evaluate up to 3 items in parallel
    )

    print("\n📊 Results:")
    print(f"  Total Items: {result.total_items}")
    print(f"  Successful: {result.successful_items}")
    print(f"  Failed: {result.failed_items}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  Total Tokens: {result.total_tokens:,}")

    # Show individual scores
    print("\n📈 Individual Scores:")
    for i, eval_result in enumerate(result.results):
        if eval_result:
            print(f"  Item {i}: {eval_result.overall_score:.3f}")

    # Calculate cost
    cost = await result.total_llm_cost()
    print(f"\n💰 Total Cost: ${cost:.6f}")
    print(f"   Average per item: ${cost / result.total_items:.6f}")

    # Example 2: Batch evaluation with progress tracking
    print("\n\n📝 Example 2: With Progress Tracking")
    print("-" * 60)

    # Progress callback
    def on_progress(
        completed: int, total: int, latest_result: Optional[EvaluationResult]
    ) -> None:
        """Track progress of batch evaluation."""
        progress_pct = (completed / total) * 100
        status = "✅" if latest_result else "❌"
        score_str = f"{latest_result.overall_score:.2f}" if latest_result else "FAILED"
        print(
            f"  {status} Progress: {completed}/{total} ({progress_pct:.0f}%) | Latest: {score_str}"
        )

    items2 = [
        {
            "output": "The Eiffel Tower is in Paris",
            "reference": "The Eiffel Tower is located in Paris, France",
        },
        {
            "output": "The Great Wall is in China",
            "reference": "The Great Wall of China is in China",
        },
        {
            "output": "The Statue of Liberty is in New York",
            "reference": "The Statue of Liberty is in New York City",
        },
        {
            "output": "The Colosseum is in Rome",
            "reference": "The Colosseum is located in Rome, Italy",
        },
        {
            "output": "Big Ben is in London",
            "reference": "Big Ben is in London, England",
        },
    ]

    result2 = await batch_evaluate(
        items=items2,
        evaluators=["semantic"],
        model="gpt-4o-mini",
        max_concurrency=2,
        progress_callback=on_progress,
    )

    print("\n✅ Batch Complete!")
    print(
        f"   Success Rate: {result2.successful_items}/{result2.total_items} ({(result2.successful_items/result2.total_items)*100:.0f}%)"
    )

    # Example 3: Score variation across different quality
    print("\n\n📝 Example 3: Score Variation Analysis")
    print("-" * 60)

    items3 = [
        {
            "output": "Paris is the capital of France",
            "reference": "Paris is the capital of France",
        },
        {
            "output": "Paris might be in France somewhere",
            "reference": "Paris is the capital of France",
        },
        {
            "output": "Berlin is a city in Germany",
            "reference": "Paris is the capital of France",
        },
    ]

    result3 = await batch_evaluate(
        items=items3,
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print("\n📊 Results:")
    print(f"  Total Items: {result3.total_items}")
    print(f"  Successful: {result3.successful_items}")

    # Show score distribution
    print("\n📈 Score Distribution:")
    for i in range(result3.total_items):
        eval_result = result3.get_result(i)
        if eval_result:
            quality = "✅ High" if eval_result.overall_score > 0.8 else "⚠️  Medium" if eval_result.overall_score > 0.4 else "❌ Low"
            print(f"  Item {i}: {quality} - Score = {eval_result.overall_score:.2f}")

    # Example 4: Multiple evaluators in batch
    print("\n\n📝 Example 4: Multiple Evaluators per Item")
    print("-" * 60)

    items4 = [
        {
            "output": "Python is a programming language",
            "reference": "Python is a high-level programming language",
            "criteria": "Technical accuracy and clarity",
        },
        {
            "output": "JavaScript runs in browsers",
            "reference": "JavaScript is a programming language that runs in web browsers",
            "criteria": "Technical accuracy and clarity",
        },
    ]

    result4 = await batch_evaluate(
        items=items4,
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print("\n📊 Results with Multiple Evaluators:")
    for i, eval_result in enumerate(result4.results):
        if eval_result:
            print(f"\n  Item {i}:")
            for score in eval_result.scores:
                print(f"    {score.name}: {score.value:.2f}")
            print(f"    Overall: {eval_result.overall_score:.2f}")

    # Example 5: Cost breakdown
    print("\n\n📝 Example 5: Detailed Cost Breakdown")
    print("-" * 60)

    breakdown = await result4.cost_breakdown()
    print("\n💰 Cost Analysis:")
    print(f"  Total Cost: ${breakdown['total']:.6f}")
    print(f"  Per Item Average: ${breakdown['per_item_average']:.6f}")
    print(f"  Success Rate: {breakdown['success_rate']:.1%}")

    if breakdown["by_evaluator"]:
        print("\n  By Evaluator:")
        for evaluator, cost in breakdown["by_evaluator"].items():
            print(f"    {evaluator}: ${cost:.6f}")

    if breakdown["by_model"]:
        print("\n  By Model:")
        for model, cost in breakdown["by_model"].items():
            print(f"    {model}: ${cost:.6f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ All Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Parallel batch processing with concurrency control")
    print("  • Progress tracking with custom callbacks")
    print("  • Graceful handling of partial failures")
    print("  • Multiple evaluators per item")
    print("  • Comprehensive cost tracking and breakdown")
    print("  • Individual result and error access")

    print("\n📖 Related Examples:")
    print("  • See basic_evaluation.py for single evaluation getting started")
    print("  • See pairwise_comparison_example.py for comparing outputs")
    print("  • See error_handling_example.py for handling failures")


if __name__ == "__main__":
    asyncio.run(main())
