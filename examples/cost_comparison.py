"""Cost Comparison Example - See What Your Evaluations Actually Cost

This example demonstrates Arbiter's automatic cost tracking by comparing
evaluation costs across different models.

Key demonstrations:
1. Real-time cost calculation using live pricing data
2. Cost/quality trade-offs between models
3. Detailed cost breakdowns
4. Batch cost analysis

What makes this unique:
- No other evaluation framework shows costs this prominently
- Real pricing data from llm-prices.com (same source as Langfuse)
- Automatic tracking - no manual calculation needed
"""

import asyncio
from arbiter_ai import evaluate
from arbiter_ai.core import Provider


async def example1_basic_cost_tracking():
    """Example 1: Basic evaluation with automatic cost tracking."""
    print("=" * 80)
    print("Example 1: Basic Cost Tracking")
    print("=" * 80)

    result = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )

    # Cost tracking is automatic - just ask for it
    cost = await result.total_llm_cost()

    print(f"\n✓ Score: {result.overall_score:.2f}")
    print(f"💰 Cost: ${cost:.6f}")
    print(f"⏱️  Time: {result.processing_time:.2f}s")
    print(f"🔍 LLM Calls: {len(result.interactions)}")

    print("\n→ Arbiter tracks this automatically - no manual calculation!")


async def example2_model_comparison():
    """Example 2: Compare costs across different models."""
    print("\n" + "=" * 80)
    print("Example 2: Model Cost Comparison")
    print("=" * 80)

    output = "The Eiffel Tower stands 330 meters tall in Paris, France."
    reference = "The Eiffel Tower is a 330-meter tall iron lattice tower located in Paris."

    # Test with GPT-4o (expensive but high quality)
    print("\nTesting GPT-4o...")
    result_gpt4 = await evaluate(
        output=output,
        reference=reference,
        evaluators=["semantic"],
        model="gpt-4o"
    )

    # Test with GPT-4o-mini (cheaper, often similar quality)
    print("Testing GPT-4o-mini...")
    result_mini = await evaluate(
        output=output,
        reference=reference,
        evaluators=["semantic"],
        model="gpt-4o-mini"
    )

    # Compare costs and quality
    cost_gpt4 = await result_gpt4.total_llm_cost()
    cost_mini = await result_mini.total_llm_cost()

    print("\n" + "-" * 80)
    print("COMPARISON RESULTS")
    print("-" * 80)
    print(f"\nGPT-4o:")
    print(f"  Cost: ${cost_gpt4:.6f}")
    print(f"  Score: {result_gpt4.overall_score:.3f}")
    print(f"  Time: {result_gpt4.processing_time:.2f}s")

    print(f"\nGPT-4o-mini:")
    print(f"  Cost: ${cost_mini:.6f}")
    print(f"  Score: {result_mini.overall_score:.3f}")
    print(f"  Time: {result_mini.processing_time:.2f}s")

    savings_pct = ((cost_gpt4 - cost_mini) / cost_gpt4 * 100)
    score_diff = abs(result_gpt4.overall_score - result_mini.overall_score)

    print(f"\n💡 INSIGHTS:")
    print(f"   Cost savings: {savings_pct:.1f}%")
    print(f"   Score difference: {score_diff:.3f}")
    print(f"   {'→ Use GPT-4o-mini!' if score_diff < 0.05 else '→ Quality difference may justify GPT-4o cost'}")


async def example3_detailed_cost_breakdown():
    """Example 3: Detailed cost breakdown by evaluator and model."""
    print("\n" + "=" * 80)
    print("Example 3: Detailed Cost Breakdown")
    print("=" * 80)

    # Run multiple evaluators
    result = await evaluate(
        output="Machine learning enables computers to learn from data without explicit programming.",
        reference="Machine learning allows systems to automatically learn and improve from experience.",
        criteria="Technical accuracy, clarity for non-experts, conciseness",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini"
    )

    # Get detailed breakdown
    breakdown = await result.cost_breakdown()

    print(f"\n💰 COST BREAKDOWN")
    print("-" * 80)
    print(f"Total cost: ${breakdown['total']:.6f}")
    print(f"\nBy evaluator:")
    for evaluator, cost in breakdown['by_evaluator'].items():
        print(f"  {evaluator}: ${cost:.6f}")

    print(f"\nBy model:")
    for model, cost in breakdown['by_model'].items():
        print(f"  {model}: ${cost:.6f}")

    print(f"\nToken usage:")
    tokens = breakdown['token_breakdown']
    print(f"  Input: {tokens['input_tokens']:,}")
    print(f"  Output: {tokens['output_tokens']:,}")
    print(f"  Total: {tokens['total_tokens']:,}")

    print("\n→ This level of detail helps optimize evaluation costs!")


async def example4_provider_cost_comparison():
    """Example 4: Compare costs across different providers."""
    print("\n" + "=" * 80)
    print("Example 4: Provider Cost Comparison")
    print("=" * 80)

    output = "Python is a high-level programming language known for its simplicity and readability."
    reference = "Python is an interpreted high-level programming language emphasizing code readability."

    providers_to_test = [
        (Provider.OPENAI, "gpt-4o-mini"),
        (Provider.ANTHROPIC, "claude-3-5-sonnet-20241022"),
        (Provider.GOOGLE, "gemini-1.5-flash"),
    ]

    results = []

    for provider, model in providers_to_test:
        try:
            print(f"\nTesting {provider.value}/{model}...")
            result = await evaluate(
                output=output,
                reference=reference,
                evaluators=["semantic"],
                provider=provider,
                model=model
            )
            cost = await result.total_llm_cost()
            results.append({
                'provider': provider.value,
                'model': model,
                'cost': cost,
                'score': result.overall_score,
                'time': result.processing_time
            })
        except Exception as e:
            print(f"  Skipped (API key not configured or error): {e}")

    if results:
        print("\n" + "-" * 80)
        print("PROVIDER COMPARISON")
        print("-" * 80)
        for r in sorted(results, key=lambda x: x['cost']):
            print(f"\n{r['provider']}/{r['model']}:")
            print(f"  Cost: ${r['cost']:.6f}")
            print(f"  Score: {r['score']:.3f}")
            print(f"  Time: {r['time']:.2f}s")

        cheapest = min(results, key=lambda x: x['cost'])
        print(f"\n💡 Cheapest option: {cheapest['provider']}/{cheapest['model']} (${cheapest['cost']:.6f})")


async def example5_batch_cost_analysis():
    """Example 5: Batch evaluation cost analysis."""
    print("\n" + "=" * 80)
    print("Example 5: Batch Evaluation Cost Analysis")
    print("=" * 80)

    from arbiter_ai import batch_evaluate

    # Create test items
    items = [
        {
            "output": "Paris is the capital of France",
            "reference": "The capital of France is Paris"
        },
        {
            "output": "Tokyo is the capital of Japan",
            "reference": "Japan's capital city is Tokyo"
        },
        {
            "output": "Berlin is the capital of Germany",
            "reference": "The capital of Germany is Berlin"
        },
    ]

    print(f"\nEvaluating {len(items)} items in parallel...")

    result = await batch_evaluate(
        items=items,
        evaluators=["semantic"],
        model="gpt-4o-mini",
        max_concurrency=3
    )

    # Get batch cost breakdown
    breakdown = await result.cost_breakdown()
    total_cost = breakdown['total']
    per_item_cost = breakdown['per_item_average']

    print(f"\n💰 BATCH COST ANALYSIS")
    print("-" * 80)
    print(f"Total items: {result.total_items}")
    print(f"Successful: {result.successful_items}")
    print(f"Failed: {result.failed_items}")
    print(f"\nTotal cost: ${total_cost:.6f}")
    print(f"Per-item average: ${per_item_cost:.6f}")
    print(f"Success rate: {breakdown['success_rate']:.1%}")

    print(f"\nBy evaluator:")
    for evaluator, cost in breakdown['by_evaluator'].items():
        print(f"  {evaluator}: ${cost:.6f}")

    # Extrapolate costs
    print(f"\n📊 COST PROJECTIONS")
    print("-" * 80)
    scales = [100, 1000, 10000]
    for scale in scales:
        projected = per_item_cost * scale
        print(f"  {scale:,} evaluations: ${projected:.2f}")

    print("\n→ Arbiter makes it easy to estimate evaluation costs at scale!")


async def main():
    """Run all cost comparison examples."""
    print("\n" + "=" * 80)
    print("ARBITER COST COMPARISON EXAMPLES")
    print("=" * 80)
    print("\nDemonstrating automatic cost tracking - a unique Arbiter feature.")
    print("No other evaluation framework shows costs this prominently.")

    await example1_basic_cost_tracking()
    await example2_model_comparison()
    await example3_detailed_cost_breakdown()
    await example4_provider_cost_comparison()
    await example5_batch_cost_analysis()

    print("\n" + "=" * 80)
    print("💡 KEY TAKEAWAYS")
    print("=" * 80)
    print("""
1. Cost tracking is automatic - just call result.total_llm_cost()
2. Real pricing data from llm-prices.com (not estimates)
3. Detailed breakdowns help optimize evaluation strategies
4. GPT-4o-mini often gives 80%+ cost savings with similar quality
5. Batch operations make large-scale evaluation affordable

→ Use cost data to make informed decisions about model selection!
    """)


if __name__ == "__main__":
    asyncio.run(main())
