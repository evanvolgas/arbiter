"""Multi-call LLM system observability example.

Demonstrates automatic tracking across 15 LLM calls in a simulated
customer support agent with 5 stages.

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/debugging_multi_call.py
"""

import asyncio
import os
from typing import Dict, List

from dotenv import load_dotenv

from arbiter_ai import evaluate
from arbiter_ai.core.models import EvaluationResult


async def run_multi_stage_agent(user_query: str) -> Dict:
    """Run a 15-call agent simulation."""
    print(f"\nProcessing: '{user_query}'")
    print("Running 15 LLM calls across 5 stages...\n")

    results = []

    # Stage 1: Intent Classification (3 calls)
    print("Stage 1/5: Intent Classification")

    results.append(("routing", await evaluate(
        output="technical_support",
        reference="billing, technical_support, general_inquiry",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )))

    results.append(("sentiment", await evaluate(
        output="frustrated but professional",
        reference="positive, neutral, negative, frustrated",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )))

    results.append(("priority", await evaluate(
        output="high priority - blocking issue",
        reference="low, medium, high, critical",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )))

    # Stage 2: Knowledge Retrieval (4 calls)
    print("Stage 2/5: Knowledge Retrieval")

    results.append(("query_expansion", await evaluate(
        output="login error authentication failure password reset",
        reference=user_query,
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )))

    results.append(("semantic_search", await evaluate(
        output="Found 5 relevant knowledge base articles",
        criteria="relevance to authentication issues",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("reranking", await evaluate(
        output="Article 3 is most relevant (authentication troubleshooting)",
        criteria="ranking accuracy for user issue",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("context_assembly", await evaluate(
        output="Compiled context from top 3 articles about auth issues",
        criteria="completeness and coherence",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    # Stage 3: Response Generation (3 calls)
    print("Stage 3/5: Response Generation")

    results.append(("draft_response", await evaluate(
        output="Try clearing browser cache and resetting password via email link",
        reference="Professional technical support response with clear steps",
        evaluators=["semantic"],
        model="gpt-4o-mini",
    )))

    results.append(("fact_check", await evaluate(
        output="Instructions verified against knowledge base",
        criteria="factual accuracy of troubleshooting steps",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("tone_adjustment", await evaluate(
        output="Empathetic opening + clear steps + reassurance",
        criteria="professional, empathetic, helpful tone",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    # Stage 4: Quality Checks (2 calls)
    print("Stage 4/5: Quality Checks")

    results.append(("accuracy_check", await evaluate(
        output="All steps verified, no misleading information",
        criteria="technical accuracy, no false promises",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("policy_compliance", await evaluate(
        output="Response follows company guidelines and privacy policy",
        criteria="compliance with support policies",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    # Stage 5: Final Review (3 calls)
    print("Stage 5/5: Final Review")

    results.append(("grammar_check", await evaluate(
        output="No grammatical errors, clear and professional",
        criteria="grammar, spelling, clarity",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("tone_match", await evaluate(
        output="Tone matches user sentiment (empathetic to frustration)",
        criteria="appropriate emotional response to user state",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    results.append(("final_approval", await evaluate(
        output="Response approved for sending",
        criteria="ready to send, meets all quality standards",
        evaluators=["custom_criteria"],
        model="gpt-4o-mini",
    )))

    print("Done.\n")
    return {"results": results}


def print_interaction_breakdown(results: List[tuple[str, EvaluationResult]]):
    """Print call-by-call breakdown."""
    print("=" * 80)
    print("Call Breakdown")
    print("=" * 80)

    total_calls = sum(len(r.interactions) for _, r in results)
    print(f"\nTotal calls: {total_calls}\n")

    stage_markers = {
        "routing": "Stage 1: Intent Classification",
        "query_expansion": "Stage 2: Knowledge Retrieval",
        "draft_response": "Stage 3: Response Generation",
        "accuracy_check": "Stage 4: Quality Checks",
        "grammar_check": "Stage 5: Final Review",
    }

    current_stage = None
    for name, result in results:
        if name in stage_markers and stage_markers[name] != current_stage:
            current_stage = stage_markers[name]
            print(f"\n{current_stage}")

        if result.interactions:
            interaction = result.interactions[0]
            print(f"  {name:20s} {interaction.tokens_used:4d} tokens  {interaction.latency:.2f}s")


async def print_cost_analysis(results: List[tuple[str, EvaluationResult]]):
    """Print cost breakdown."""
    print("\n" + "=" * 80)
    print("Cost Analysis")
    print("=" * 80)

    total_cost = sum([await r.total_llm_cost() for _, r in results])
    total_tokens = sum([r.total_tokens for _, r in results])

    print(f"\nTotal cost: ${total_cost:.6f}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Average per call: ${total_cost / len(results):.6f}")

    print("\nBy stage:")
    stages = {
        "Intent Classification (1-3)": (0, 3),
        "Knowledge Retrieval (4-7)": (3, 7),
        "Response Generation (8-10)": (7, 10),
        "Quality Checks (11-12)": (10, 12),
        "Final Review (13-15)": (12, 15),
    }

    for stage_name, (start, end) in stages.items():
        cost = sum([await r.total_llm_cost() for _, r in results[start:end]])
        pct = (cost / total_cost * 100) if total_cost > 0 else 0
        print(f"  {stage_name:30s} ${cost:.6f} ({pct:.1f}%)")


def print_performance_analysis(results: List[tuple[str, EvaluationResult]]):
    """Print performance metrics."""
    print("\n" + "=" * 80)
    print("Performance Analysis")
    print("=" * 80)

    total_time = sum([r.processing_time for _, r in results])
    print(f"\nTotal time: {total_time:.2f}s")
    print(f"Average per call: {total_time / len(results):.2f}s")

    slowest = sorted(
        [(name, r.interactions[0].latency if r.interactions else 0)
         for name, r in results],
        key=lambda x: x[1],
        reverse=True
    )[:3]

    print("\nSlowest calls:")
    for name, latency in slowest:
        print(f"  {name}: {latency:.2f}s")


def print_debug_example(results: List[tuple[str, EvaluationResult]]):
    """Show how to inspect individual calls."""
    print("\n" + "=" * 80)
    print("Debug Example: Inspecting Call #9 (fact_check)")
    print("=" * 80)

    fact_check = results[8][1]  # 9th call

    if fact_check.interactions:
        interaction = fact_check.interactions[0]
        print(f"\nPurpose: {interaction.purpose}")
        print(f"Model: {interaction.model}")
        print(f"Tokens: {interaction.tokens_used:,}")
        print(f"Latency: {interaction.latency:.2f}s")

        print(f"\n--- Full Prompt ---")
        print(interaction.prompt)

        print(f"\n--- Full Response ---")
        print(interaction.response)

        print(f"\n--- Score Details ---")
        for score in fact_check.scores:
            print(f"Score: {score.value:.2f}")
            print(f"Confidence: {score.confidence:.2f}")
            print(f"Explanation: {score.explanation}")
            if score.metadata:
                print(f"Metadata: {score.metadata}")


async def main():
    """Run the multi-call example."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        return

    print("=" * 80)
    print("Multi-Call LLM System Observability")
    print("=" * 80)
    print("\nWithout Arbiter:")
    print("  - Manual tracking per call")
    print("  - Calculate costs yourself")
    print("  - Add logging everywhere")
    print("  - No automatic aggregation")
    print("\nWith Arbiter:")
    print("  - Just call evaluate()")
    print("  - Everything tracked automatically")
    print("  - Full interaction history in result.interactions")
    print("  - Cost calculation built-in")

    try:
        # Run the agent
        data = await run_multi_stage_agent(
            "I can't log in to my account, getting authentication error"
        )

        results = data["results"]

        # Show what Arbiter tracked
        print_interaction_breakdown(results)
        await print_cost_analysis(results)
        print_performance_analysis(results)
        print_debug_example(results)

        print("\n" + "=" * 80)
        print("How This Works")
        print("=" * 80)
        print("\nEach evaluate() call returns an EvaluationResult with:")
        print("  result.interactions       - List of all LLM calls")
        print("  result.total_tokens        - Sum of all tokens used")
        print("  result.processing_time     - Total latency")
        print("  result.total_llm_cost()    - Calculated cost (async)")
        print("\nEach interaction contains:")
        print("  interaction.purpose        - What this call was for")
        print("  interaction.model          - Which model was used")
        print("  interaction.prompt         - Full prompt sent")
        print("  interaction.response       - Full response received")
        print("  interaction.tokens_used    - Token count")
        print("  interaction.latency        - Call duration")
        print("  interaction.timestamp      - When it happened")
        print("\nAll tracked automatically in BasePydanticEvaluator.evaluate()")
        print("See: arbiter/evaluators/base.py lines 293-372")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
