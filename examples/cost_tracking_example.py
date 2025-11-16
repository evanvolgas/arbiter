"""Example: Tracking LLM evaluation costs with accurate pricing data.

This example demonstrates Arbiter's cost tracking capabilities using
real-world pricing data from llm-prices.com.

Features demonstrated:
- Automatic cost calculation for each LLM interaction
- Detailed token tracking (input/output/cached)
- Cost breakdowns by evaluator and model
- Batch evaluation cost analysis
- Cost comparison across models

Requirements:
    - OpenAI API key: export OPENAI_API_KEY=your_key_here
    - Or Anthropic: export ANTHROPIC_API_KEY=your_key_here
"""

import asyncio
from typing import List

from arbiter import evaluate, get_cost_calculator


async def basic_cost_tracking():
    """Example 1: Basic cost tracking for a single evaluation."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Cost Tracking")
    print("=" * 70)

    # Perform evaluation
    result = await evaluate(
        output="The Eiffel Tower is 324 meters tall including antennas.",
        reference="How tall is the Eiffel Tower?",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    # Get total cost
    cost = await result.total_llm_cost()
    print(f"\nEvaluation Score: {result.overall_score:.2f}")
    print(f"Total Cost: ${cost:.6f}")

    # Show token breakdown
    print(f"\nToken Usage:")
    for interaction in result.interactions:
        print(f"  Model: {interaction.model}")
        print(f"  Input tokens: {interaction.input_tokens}")
        print(f"  Output tokens: {interaction.output_tokens}")
        print(f"  Cost: ${interaction.cost:.6f}" if interaction.cost else "  Cost: N/A")


async def detailed_cost_breakdown():
    """Example 2: Detailed cost breakdown by evaluator and model."""
    print("\n" + "=" * 70)
    print("Example 2: Detailed Cost Breakdown")
    print("=" * 70)

    # Evaluate with multiple evaluators
    result = await evaluate(
        output="Python is a high-level programming language known for simplicity.",
        reference="Python is a popular programming language.",
        evaluators=["semantic"],  # Add more when available
        model="gpt-4o-mini",
    )

    # Get detailed breakdown
    breakdown = await result.cost_breakdown()

    print(f"\n📊 Cost Analysis:")
    print(f"  Total Cost: ${breakdown['total']:.6f}")

    print(f"\n📈 By Evaluator:")
    for evaluator, cost in breakdown["by_evaluator"].items():
        print(f"  {evaluator}: ${cost:.6f}")

    print(f"\n🤖 By Model:")
    for model, cost in breakdown["by_model"].items():
        print(f"  {model}: ${cost:.6f}")

    print(f"\n🎯 Token Breakdown:")
    tokens = breakdown["token_breakdown"]
    print(f"  Input tokens: {tokens['input_tokens']:,}")
    print(f"  Output tokens: {tokens['output_tokens']:,}")
    print(f"  Total tokens: {tokens['total_tokens']:,}")
    if tokens.get("cached_tokens", 0) > 0:
        print(f"  Cached tokens: {tokens['cached_tokens']:,}")


async def batch_cost_analysis():
    """Example 3: Cost analysis for batch evaluations."""
    print("\n" + "=" * 70)
    print("Example 3: Batch Evaluation Cost Analysis")
    print("=" * 70)

    # Batch of outputs to evaluate
    test_cases = [
        {
            "output": "Water freezes at 0°C and boils at 100°C.",
            "reference": "Water freezes at 0 degrees Celsius.",
        },
        {
            "output": "The Earth orbits around the Sun once per year.",
            "reference": "Earth's orbital period is one year.",
        },
        {
            "output": "DNA contains genetic information for organisms.",
            "reference": "DNA stores genetic data.",
        },
    ]

    print(f"\nEvaluating {len(test_cases)} test cases...")

    results = []
    for i, case in enumerate(test_cases, 1):
        result = await evaluate(
            output=case["output"],
            reference=case["reference"],
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )
        results.append(result)
        print(f"  ✓ Case {i} evaluated (score: {result.overall_score:.2f})")

    # Calculate total costs
    total_cost = sum([await r.total_llm_cost() for r in results])
    total_tokens = sum([r.total_tokens for r in results])
    avg_cost = total_cost / len(results)

    print(f"\n💰 Batch Summary:")
    print(f"  Total evaluations: {len(results)}")
    print(f"  Total cost: ${total_cost:.6f}")
    print(f"  Average cost per evaluation: ${avg_cost:.6f}")
    print(f"  Total tokens used: {total_tokens:,}")
    print(f"  Average score: {sum([r.overall_score for r in results]) / len(results):.2f}")


async def model_cost_comparison():
    """Example 4: Compare costs across different models."""
    print("\n" + "=" * 70)
    print("Example 4: Model Cost Comparison")
    print("=" * 70)

    test_output = "Machine learning is a subset of artificial intelligence."
    test_reference = "ML is part of AI."

    models = ["gpt-4o-mini", "gpt-4o"]  # Add more models as needed

    print(f"\nComparing costs for {len(models)} models...\n")

    for model in models:
        try:
            result = await evaluate(
                output=test_output,
                reference=test_reference,
                evaluators=["semantic"],
                model=model,
            )

            cost = await result.total_llm_cost()
            print(f"📊 {model}:")
            print(f"  Score: {result.overall_score:.2f}")
            print(f"  Cost: ${cost:.6f}")
            print(f"  Tokens: {result.total_tokens:,}")
            print()

        except Exception as e:
            print(f"⚠️  {model}: Error - {e}\n")


async def cost_calculator_inspection():
    """Example 5: Inspect pricing data from llm-prices.com."""
    print("\n" + "=" * 70)
    print("Example 5: Pricing Data Inspection")
    print("=" * 70)

    # Get cost calculator
    calc = get_cost_calculator()
    await calc.ensure_loaded()

    if calc.is_loaded:
        print(f"\n✅ Pricing data loaded successfully")
        print(f"   Models available: {calc.model_count}")

        # Show pricing for specific models
        models_to_check = ["gpt-4o-mini", "claude-3-5-sonnet", "gpt-4o"]

        print(f"\n💵 Sample Pricing (per 1M tokens):")
        for model_id in models_to_check:
            pricing = calc.get_pricing(model_id)
            if pricing:
                print(f"\n  {pricing.name} ({pricing.vendor}):")
                print(f"    Input: ${pricing.input:.2f}")
                print(f"    Output: ${pricing.output:.2f}")
                if pricing.input_cached:
                    print(f"    Cached: ${pricing.input_cached:.2f}")
            else:
                print(f"\n  {model_id}: Pricing not available")

        # Calculate example costs
        print(f"\n🧮 Example Cost Calculation:")
        print(f"   For 10,000 input + 2,000 output tokens:")

        for model_id in models_to_check:
            cost = calc.calculate_cost(
                model=model_id, input_tokens=10000, output_tokens=2000
            )
            print(f"     {model_id}: ${cost:.4f}")

    else:
        print("\n⚠️  Pricing data not available")
        print("   Using fallback estimates")


async def cached_tokens_example():
    """Example 6: Cost savings with cached tokens (e.g., Anthropic)."""
    print("\n" + "=" * 70)
    print("Example 6: Cost Savings with Cached Tokens")
    print("=" * 70)

    calc = get_cost_calculator()
    await calc.ensure_loaded()

    # Simulate costs with and without caching
    model = "claude-3-5-sonnet"
    input_tokens = 10000
    output_tokens = 1000

    # Without caching
    cost_no_cache = calc.calculate_cost(
        model=model, input_tokens=input_tokens, output_tokens=output_tokens, cached_tokens=0
    )

    # With 50% cache hit rate
    cached_amount = input_tokens // 2
    cost_with_cache = calc.calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_amount,
    )

    savings = cost_no_cache - cost_with_cache
    savings_percent = (savings / cost_no_cache) * 100 if cost_no_cache > 0 else 0

    print(f"\n💾 Cache Impact Analysis:")
    print(f"   Model: {model}")
    print(f"   Input tokens: {input_tokens:,}")
    print(f"   Output tokens: {output_tokens:,}")
    print(f"   Cached tokens: {cached_amount:,} (50% hit rate)")
    print(f"\n   Cost without caching: ${cost_no_cache:.6f}")
    print(f"   Cost with caching: ${cost_with_cache:.6f}")
    print(f"   Savings: ${savings:.6f} ({savings_percent:.1f}%)")


async def main():
    """Run all cost tracking examples."""
    print("\n" + "=" * 70)
    print("Arbiter Cost Tracking Examples")
    print("Using real-world pricing data from llm-prices.com")
    print("=" * 70)

    try:
        # Run examples
        await basic_cost_tracking()
        await detailed_cost_breakdown()
        await batch_cost_analysis()
        await model_cost_comparison()
        await cost_calculator_inspection()
        await cached_tokens_example()

        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
