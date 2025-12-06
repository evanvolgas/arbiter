"""Complete Observability Example - Interaction Tracking, Cost Analysis, and Debugging

This example demonstrates Arbiter's comprehensive observability features that provide
complete transparency into LLM evaluation operations.

Key Features:
- Automatic interaction tracking (no instrumentation needed)
- Real-time cost calculation with LiteLLM's bundled pricing database
- Detailed token usage breakdown (input/output/cached)
- Performance monitoring and latency tracking
- Complete audit trail for compliance
- Debugging support (inspect prompts/responses)

Use Cases:
- Cost optimization: Track and optimize token usage
- Debugging: See exactly why scores are what they are
- Compliance: Complete audit trail for regulatory requirements
- Performance: Identify bottlenecks and slow operations
- Transparency: Show stakeholders how evaluations work

Requirements:
    export OPENAI_API_KEY=your_key_here
    # OR: export ANTHROPIC_API_KEY=your_key_here

Run with:
    python examples/observability_example.py
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv

from arbiter_ai import evaluate, get_cost_calculator
from arbiter_ai.core.models import LLMInteraction


def format_timestamp(dt: datetime) -> str:
    """Format timestamp for display."""
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def analyze_interactions(interactions: List[LLMInteraction]) -> Dict:
    """Analyze interactions and return summary statistics."""
    if not interactions:
        return {}

    total_tokens = sum(i.tokens_used for i in interactions)
    total_latency = sum(i.latency for i in interactions)
    avg_latency = total_latency / len(interactions) if interactions else 0

    # Group by purpose
    by_purpose: Dict[str, List[LLMInteraction]] = {}
    for interaction in interactions:
        purpose = interaction.purpose
        if purpose not in by_purpose:
            by_purpose[purpose] = []
        by_purpose[purpose].append(interaction)

    return {
        "total_interactions": len(interactions),
        "total_tokens": total_tokens,
        "total_latency": total_latency,
        "avg_latency": avg_latency,
        "by_purpose": by_purpose,
    }


async def example1_basic_tracking():
    """Example 1: Basic interaction tracking and cost calculation."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Interaction Tracking & Cost Calculation")
    print("=" * 70)

    result = await evaluate(
        output="Paris is the capital of France and a major European city.",
        reference="The capital of France is Paris, which is located in Europe.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")
    print(f"   Score: {result.overall_score:.3f}")
    print(f"   Processing Time: {result.processing_time:.3f}s")

    # Interaction tracking (automatic - no instrumentation)
    print(f"\n🔍 Interaction Tracking:")
    print(f"   Total LLM Calls: {len(result.interactions)}")

    for i, interaction in enumerate(result.interactions, 1):
        print(f"\n   Call {i}:")
        print(f"     Purpose: {interaction.purpose}")
        print(f"     Model: {interaction.model}")
        print(f"     Latency: {interaction.latency:.3f}s")
        print(f"     Tokens: {interaction.tokens_used:,}")
        print(f"     Timestamp: {format_timestamp(interaction.timestamp)}")

    # Cost analysis
    total_cost = await result.total_llm_cost()
    print(f"\n💰 Cost Analysis:")
    print(f"   Total Tokens: {result.total_tokens:,}")
    print(f"   Estimated Cost: ${total_cost:.6f}")


async def example2_multi_evaluator_transparency():
    """Example 2: Multi-evaluator transparency with cost breakdown."""
    print("\n" + "=" * 70)
    print("Example 2: Multi-Evaluator Transparency")
    print("=" * 70)

    result = await evaluate(
        output="Our product is revolutionary and will change everything!",
        criteria="Professional tone, factual accuracy, no hyperbole",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Evaluation Complete!")
    print(f"   Score: {result.overall_score:.3f}")
    print(f"   Evaluators: {', '.join(result.evaluator_names)}")

    # Analyze interactions
    analysis = analyze_interactions(result.interactions)

    print(f"\n📊 Interaction Analysis:")
    print(f"   Total LLM Calls: {analysis['total_interactions']}")
    print(f"   Total Tokens: {analysis['total_tokens']:,}")
    print(f"   Total Latency: {analysis['total_latency']:.3f}s")
    print(f"   Average Latency: {analysis['avg_latency']:.3f}s")

    # Show by-purpose breakdown
    print(f"\n📋 By Purpose:")
    for purpose, interactions in analysis['by_purpose'].items():
        tokens = sum(i.tokens_used for i in interactions)
        latency = sum(i.latency for i in interactions)
        print(f"   • {purpose}:")
        print(f"     Calls: {len(interactions)}, Tokens: {tokens:,}, Latency: {latency:.3f}s")

    # Cost breakdown
    breakdown = await result.cost_breakdown()
    print(f"\n💰 Cost Breakdown:")
    print(f"   Total: ${breakdown['total']:.6f}")

    if breakdown.get("by_evaluator"):
        print(f"   By Evaluator:")
        for evaluator, cost in breakdown["by_evaluator"].items():
            print(f"     • {evaluator}: ${cost:.6f}")


async def example3_debugging():
    """Example 3: Debugging with full prompt/response inspection."""
    print("\n" + "=" * 70)
    print("Example 3: Debugging - Inspect Prompts & Responses")
    print("=" * 70)

    result = await evaluate(
        output="The quick brown fox jumps over the lazy dog",
        reference="A fast brown fox leaps above a sleepy canine",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    print(f"\n✅ Score: {result.overall_score:.3f}")

    if result.interactions:
        interaction = result.interactions[0]

        print(f"\n🔍 Full Interaction Details:")
        print(f"\n   📝 Complete Prompt:")
        print(f"   {'-' * 66}")
        print(f"   {interaction.prompt}")
        print(f"   {'-' * 66}")

        print(f"\n   📤 Complete Response:")
        print(f"   {'-' * 66}")
        print(f"   {interaction.response}")
        print(f"   {'-' * 66}")

        print(f"\n   💡 Use Case:")
        print(f"      • See exactly what was sent to the LLM")
        print(f"      • Understand how the score was computed")
        print(f"      • Debug unexpected results")


async def example4_performance_monitoring():
    """Example 4: Performance monitoring and optimization insights."""
    print("\n" + "=" * 70)
    print("Example 4: Performance Monitoring")
    print("=" * 70)

    result = await evaluate(
        output="Machine learning is a subset of artificial intelligence that enables "
               "systems to learn and improve from experience without being explicitly programmed.",
        reference="ML is part of AI and allows systems to learn from data.",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )

    if result.interactions:
        interaction = result.interactions[0]
        tokens_per_second = interaction.tokens_used / interaction.latency if interaction.latency > 0 else 0

        print(f"\n⚡ Performance Metrics:")
        print(f"   Latency: {interaction.latency:.3f}s")
        print(f"   Tokens: {interaction.tokens_used:,}")
        print(f"   Throughput: {tokens_per_second:.1f} tokens/second")
        print(f"   Model: {interaction.model}")

        # Performance insights
        print(f"\n   💡 Performance Insights:")
        if interaction.latency > 2.0:
            print(f"      ⚠️  High latency ({interaction.latency:.3f}s)")
            print(f"         Consider: Faster model or prompt optimization")
        elif interaction.latency < 0.5:
            print(f"      ✅ Excellent latency!")

        if interaction.tokens_used > 1000:
            print(f"      ⚠️  High token usage ({interaction.tokens_used:,})")
            print(f"         Consider: Shorter prompts or response caching")


async def example5_audit_trail():
    """Example 5: Complete audit trail for compliance."""
    print("\n" + "=" * 70)
    print("Example 5: Audit Trail - Compliance & Transparency")
    print("=" * 70)

    result = await evaluate(
        output="Take 2 aspirin and call me in the morning.",
        criteria="Medical accuracy, appropriate disclaimer, professional tone",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"\n📋 Complete Audit Trail:")
    print(f"   Evaluation ID: {id(result)}")
    print(f"   Timestamp: {format_timestamp(result.timestamp)}")
    print(f"   Score: {result.overall_score:.3f}")
    print(f"   Evaluators: {', '.join(result.evaluator_names)}")
    print(f"   Processing Time: {result.processing_time:.3f}s")
    print(f"   Total Tokens: {result.total_tokens:,}")

    print(f"\n   🔍 All LLM Interactions:")
    for i, interaction in enumerate(result.interactions, 1):
        print(f"   {i}. {interaction.purpose}")
        print(f"      Model: {interaction.model}")
        print(f"      Tokens: {interaction.tokens_used:,}")
        print(f"      Latency: {interaction.latency:.3f}s")
        print(f"      Timestamp: {format_timestamp(interaction.timestamp)}")

    print(f"\n   💡 Use Case:")
    print(f"      • Complete audit trail for regulatory compliance")
    print(f"      • Timestamped records of all LLM usage")
    print(f"      • Full transparency for audits")


async def example6_cost_calculator():
    """Example 6: Cost calculator with real pricing data."""
    print("\n" + "=" * 70)
    print("Example 6: Cost Calculator - Real Pricing Data")
    print("=" * 70)

    # Get cost calculator
    calc = get_cost_calculator()
    await calc.ensure_loaded()

    if calc.is_loaded:
        print(f"\n✅ Pricing data loaded successfully")
        print(f"   Models available: {calc.model_count}")

        # Show pricing for common models
        models = ["gpt-4o-mini", "gpt-4o", "claude-3-5-sonnet"]

        print(f"\n💵 Sample Pricing (per 1M tokens):")
        for model_id in models:
            pricing = calc.get_pricing(model_id)
            if pricing:
                print(f"\n   {pricing.name}:")
                print(f"     Input: ${pricing.input:.2f}")
                print(f"     Output: ${pricing.output:.2f}")
                if pricing.input_cached:
                    print(f"     Cached: ${pricing.input_cached:.2f}")

        # Calculate example costs
        print(f"\n🧮 Example Cost (10K input + 2K output tokens):")
        for model_id in models:
            cost = calc.calculate_cost(model=model_id, input_tokens=10000, output_tokens=2000)
            print(f"   {model_id}: ${cost:.4f}")

    else:
        print(f"\n⚠️  Using fallback cost estimates")


async def example7_batch_cost_analysis():
    """Example 7: Cost analysis for batch evaluations."""
    print("\n" + "=" * 70)
    print("Example 7: Batch Cost Analysis")
    print("=" * 70)

    test_cases = [
        {"output": "Water freezes at 0°C", "reference": "Water freezes at 0 degrees Celsius"},
        {"output": "Earth orbits the Sun", "reference": "Earth's orbital period is one year"},
        {"output": "DNA stores genetic data", "reference": "DNA contains genetic information"},
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
        print(f"  ✓ Case {i}: {result.overall_score:.2f}")

    # Aggregate statistics
    total_cost = sum([await r.total_llm_cost() for r in results])
    total_tokens = sum([r.total_tokens for r in results])
    avg_score = sum([r.overall_score for r in results]) / len(results)

    print(f"\n💰 Batch Cost Summary:")
    print(f"   Total evaluations: {len(results)}")
    print(f"   Total cost: ${total_cost:.6f}")
    print(f"   Average per evaluation: ${total_cost / len(results):.6f}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Average score: {avg_score:.2f}")


async def main():
    """Run all observability examples."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("\n" + "=" * 70)
    print("Arbiter - Complete Observability Examples")
    print("Interaction Tracking • Cost Analysis • Debugging")
    print("=" * 70)

    try:
        await example1_basic_tracking()
        await example2_multi_evaluator_transparency()
        await example3_debugging()
        await example4_performance_monitoring()
        await example5_audit_trail()
        await example6_cost_calculator()
        await example7_batch_cost_analysis()

        print("\n\n" + "=" * 70)
        print("✅ All Examples Complete!")
        print("=" * 70)

        print("\n🎯 Key Takeaways:")
        print("   • Automatic tracking - no manual instrumentation")
        print("   • Real-time cost calculation with accurate pricing")
        print("   • Complete transparency into all LLM interactions")
        print("   • Debugging support - inspect prompts and responses")
        print("   • Audit trails for compliance and transparency")
        print("   • Performance monitoring and optimization insights")

        print("\n📚 Related Examples:")
        print("   • Basic usage: examples/basic_evaluation.py")
        print("   • Batch processing: examples/batch_evaluation_example.py")
        print("   • Error handling: examples/error_handling_example.py")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
