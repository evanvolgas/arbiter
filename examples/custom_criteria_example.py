"""Custom Criteria Evaluation - Domain-Specific Quality Assessment

This example demonstrates how to evaluate LLM outputs against custom criteria
without needing reference text, enabling flexible domain-specific evaluation.

Key Features:
- Custom criteria evaluation (no reference needed)
- Flexible quality assessment for any domain
- Confidence scoring and detailed explanations
- Direct evaluator usage for fine-grained control

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/custom_criteria_example.py
"""

from dotenv import load_dotenv

import asyncio
import os

from arbiter_ai import evaluate, CustomCriteriaEvaluator
from arbiter_ai.core import LLMManager


async def main():
    """Run custom criteria evaluation examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Custom Criteria Evaluation Example")
    print("=" * 60)

    # Example 1: Single criteria evaluation (medical domain)
    print("\n📝 Example 1: Medical Domain Evaluation")
    print("-" * 60)

    result1 = await evaluate(
        output="""Diabetes management requires careful attention to diet, exercise, and medication.
Patients should monitor their blood glucose levels regularly and work closely with
their healthcare providers to develop a personalized treatment plan.""",
        criteria="Medical accuracy, HIPAA compliance, appropriate tone for patients, clarity",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result1.output[:100]}...")
    print("\n📊 Results:")
    print(f"  Overall Score: {result1.overall_score:.3f}")
    print(f"  Passed: {'✅' if result1.passed else '❌'}")

    for score in result1.scores:
        print(f"\n  {score.name}:")
        print(f"    Value: {score.value:.3f}")
        if score.confidence:
            print(f"    Confidence: {score.confidence:.3f}")
        if score.metadata.get("criteria_met"):
            print(f"    ✅ Criteria Met: {', '.join(score.metadata['criteria_met'])}")
        if score.metadata.get("criteria_not_met"):
            print(f"    ❌ Criteria Not Met: {', '.join(score.metadata['criteria_not_met'])}")
        if score.explanation:
            print(f"    Explanation: {score.explanation[:150]}...")

    # Cost tracking
    breakdown1 = await result1.cost_breakdown()
    print(f"\n💰 Cost Analysis:")
    print(f"  Total Cost: ${breakdown1['total']:.6f}")
    print(f"  Tokens: {breakdown1['token_breakdown']['total_tokens']:,}")

    # Example 2: Technical accuracy evaluation
    print("\n\n📝 Example 2: Technical Accuracy Evaluation")
    print("-" * 60)

    result2 = await evaluate(
        output="""Our software solution provides enterprise-grade security with
end-to-end encryption, multi-factor authentication, and comprehensive audit logs.
The system supports SAML 2.0 and OAuth 2.0 for single sign-on capabilities.""",
        criteria="Technical accuracy, security focus, enterprise positioning, clear value proposition",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result2.output[:80]}...")
    print("\n📊 Results:")
    print(f"  Overall Score: {result2.overall_score:.3f}")
    print(f"  Passed: {'✅' if result2.passed else '❌'}")

    for score in result2.scores:
        print(f"\n  {score.name}:")
        print(f"    Value: {score.value:.3f}")
        if score.confidence:
            print(f"    Confidence: {score.confidence:.3f}")
        if score.metadata.get("criteria_met"):
            print(f"    ✅ Criteria Met: {', '.join(score.metadata['criteria_met'])}")
        if score.metadata.get("criteria_not_met"):
            print(f"    ❌ Criteria Not Met: {', '.join(score.metadata['criteria_not_met'])}")
        if score.explanation:
            print(f"    Explanation: {score.explanation[:150]}...")

    # Example 3: Brand voice evaluation
    print("\n\n📝 Example 3: Brand Voice Evaluation")
    print("-" * 60)

    result3 = await evaluate(
        output="""Hey there! Our product is super cool and you should totally buy it!
It's like, the best thing ever and everyone loves it. Get yours now!!!""",
        criteria="Professional tone, appropriate for B2B audience, clear value proposition, no excessive enthusiasm",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"Output: {result3.output}")
    print("\n📊 Results:")
    print(f"  Overall Score: {result3.overall_score:.3f}")
    print(f"  Passed: {'✅' if result3.passed else '❌'}")

    for score in result3.scores:
        if score.metadata.get("criteria_not_met"):
            print("\n  ⚠️  Issues Found:")
            for criterion in score.metadata["criteria_not_met"]:
                print(f"    - {criterion}")

    # Example 4: Using evaluator directly for more control with reference
    print("\n\n📝 Example 4: Direct Evaluator Usage with Reference")
    print("-" * 60)

    # Get LLM client
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Create evaluator
    evaluator = CustomCriteriaEvaluator(client)

    score = await evaluator.evaluate(
        output="The patient should take 500mg of ibuprofen every 6 hours for pain relief.",
        reference="Ibuprofen dosage for adults: 200-400mg every 4-6 hours as needed. Maximum daily dose: 1200mg for OTC use.",
        criteria="Medical accuracy, appropriate dosage recommendations, patient safety",
    )

    print("Output: The patient should take 500mg of ibuprofen...")
    print("Reference: Ibuprofen dosage for adults: 200-400mg...")
    print(f"\n📊 Score: {score.value:.3f}")
    print(f"Confidence: {score.confidence:.3f}")

    if score.metadata.get("criteria_not_met"):
        print("\n⚠️ Concerns:")
        for criterion in score.metadata["criteria_not_met"]:
            print(f"  - {criterion}")

    print("\nExplanation:")
    print(f"  {score.explanation[:200]}...")

    # Access interactions directly
    print("\n🔬 Evaluator Interactions:")
    interactions = evaluator.get_interactions()
    total_tokens_eval = sum(i.tokens_used for i in interactions)
    print(f"  Total Calls: {len(interactions)}")
    print(f"  Total Tokens: {total_tokens_eval:,}")
    print(f"  Total Latency: {sum(i.latency for i in interactions):.2f}s")

    # Session summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")

    # Calculate total session cost
    cost2 = await result2.total_llm_cost()
    cost3 = await result3.total_llm_cost()
    total_cost = breakdown1['total'] + cost2 + cost3
    total_tokens_all = result1.total_tokens + result2.total_tokens + result3.total_tokens

    print(f"\n💰 Total Session Cost:")
    print(f"  Total Evaluations: 4")
    print(f"  Total Cost: ${total_cost:.6f}")
    print(f"  Total Tokens: {total_tokens_all:,}")
    print(f"  Average per Evaluation: ${total_cost / 4:.6f}")

    print("\n📚 Key Features Demonstrated:")
    print("  • Domain-specific criteria evaluation (medical, technical, brand voice)")
    print("  • Reference-free evaluation (no ground truth needed)")
    print("  • Reference-based evaluation (comparing against expectations)")
    print("  • Detailed criteria breakdown (met/not met)")
    print("  • Both high-level API and direct evaluator usage")
    print("  • Automatic cost tracking and token analysis")

    print("\n📖 Related Examples:")
    print("  • See multiple_evaluators.py for combining criteria with other evaluators")
    print("  • See rag_evaluation.py for evaluating RAG systems with custom criteria")
    print("  • See factuality_example.py for fact-checking with criteria")


if __name__ == "__main__":
    asyncio.run(main())
