"""Factuality Evaluation - Hallucination Detection & Fact-Checking

This example demonstrates using FactualityEvaluator for detecting hallucinations
and verifying factual claims in LLM outputs.

Key Features:
- Hallucination detection via reference comparison
- Standalone fact-checking using general knowledge
- RAG output validation
- Factual claim extraction and categorization (factual/non-factual/uncertain)
- Focus on specific fact types (dates, numbers, names, etc.)

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/factuality_example.py
"""

import asyncio
import os

from arbiter import evaluate


async def example_1_detect_hallucination():
    """Example 1: Detect hallucinations by comparing to reference text."""
    print("=" * 60)
    print("Example 1: Detect Hallucinations with Reference")
    print("=" * 60)

    # Output with a hallucinated date
    output = "The Eiffel Tower was built in 1889 and is 330 meters tall"

    # Reference with correct information
    reference = "The Eiffel Tower was constructed in 1889 and stands 300 meters tall (324 meters including antennas)"

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput: {output}")
    print(f"Reference: {reference}")
    print(f"\nFactuality Score: {result.overall_score:.2f}")
    print(f"Passed: {result.passed}")
    print(f"\nScore Details:")
    score = result.scores[0]
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Factual Claims: {score.metadata.get('factual_count', 0)}")
    print(f"  Non-Factual Claims: {score.metadata.get('non_factual_count', 0)}")

    if score.metadata.get("non_factual_claims"):
        print(f"\n  Hallucinated/Incorrect:")
        for claim in score.metadata["non_factual_claims"]:
            print(f"    - {claim}")


async def example_2_standalone_factchecking():
    """Example 2: Standalone fact-checking without reference text."""
    print("\n" + "=" * 60)
    print("Example 2: Standalone Fact-Checking (No Reference)")
    print("=" * 60)

    # Scientific claim to verify using general knowledge
    output = "Water boils at 100°C (212°F) at sea level under standard atmospheric pressure"

    result = await evaluate(
        output=output,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput: {output}")
    print(f"\nFactuality Score: {result.overall_score:.2f}")
    print(f"\nScore Details:")
    score = result.scores[0]
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Factual Claims: {score.metadata.get('factual_count', 0)}")

    if score.metadata.get("factual_claims"):
        print(f"\n  Verified Claims:")
        for claim in score.metadata["factual_claims"]:
            print(f"    - {claim}")


async def example_3_rag_validation():
    """Example 3: Validate RAG (Retrieval-Augmented Generation) output."""
    print("\n" + "=" * 60)
    print("Example 3: RAG Output Validation")
    print("=" * 60)

    # RAG output that should be grounded in source
    output = """Python was created by Guido van Rossum and first released in 1991.
It emphasizes code readability with significant whitespace.
Python 3.0 was released in 2008."""

    # Source document from retrieval
    reference = """Python is a high-level programming language created by
Guido van Rossum. The language was first released in 1991.
Python uses indentation to define code blocks.
Python 3.0, a major revision, was released on December 3, 2008."""

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nRAG Output:\n{output}")
    print(f"\nSource Document:\n{reference}")
    print(f"\nFactuality Score: {result.overall_score:.2f}")
    print(f"Grounded in Source: {result.passed}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Factual (Grounded): {score.metadata.get('factual_count', 0)}")
    print(f"  Non-Factual (Hallucinated): {score.metadata.get('non_factual_count', 0)}")
    print(f"  Uncertain: {score.metadata.get('uncertain_count', 0)}")


async def example_4_specific_criteria():
    """Example 4: Focus on specific types of facts (dates, numbers, etc.)."""
    print("\n" + "=" * 60)
    print("Example 4: Fact-Checking with Specific Criteria")
    print("=" * 60)

    # Medical claim with specific facts to verify
    output = """The human heart has four chambers and beats approximately
100,000 times per day. It pumps about 2,000 gallons of blood daily."""

    criteria = "Medical accuracy: Focus on verifying anatomical facts and quantitative measurements"

    result = await evaluate(
        output=output,
        criteria=criteria,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput: {output}")
    print(f"\nCriteria: {criteria}")
    print(f"\nFactuality Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Confidence: {score.confidence:.2f}")

    if score.metadata.get("factual_claims"):
        print(f"\n  Verified Medical Facts:")
        for claim in score.metadata["factual_claims"]:
            print(f"    ✓ {claim}")

    if score.metadata.get("non_factual_claims"):
        print(f"\n  Questionable/Incorrect:")
        for claim in score.metadata["non_factual_claims"]:
            print(f"    ✗ {claim}")


async def example_5_multiple_hallucinations():
    """Example 5: Detect multiple hallucinations in a single output."""
    print("\n" + "=" * 60)
    print("Example 5: Multiple Hallucinations Detection")
    print("=" * 60)

    # Output with multiple factual errors
    output = """The Great Wall of China was built in 1492 by Emperor Napoleon.
It is 50,000 kilometers long and can be seen from the Moon with the naked eye.
The wall was originally painted bright red."""

    # Correct reference
    reference = """The Great Wall of China was built over many centuries,
with major construction occurring during the Ming Dynasty (1368-1644).
It spans approximately 21,000 kilometers.
Contrary to popular belief, it is not visible from the Moon with the naked eye."""

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput (with hallucinations):\n{output}")
    print(f"\nFactuality Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nDetected Issues:")

    if score.metadata.get("non_factual_claims"):
        print(f"  Hallucinations Found: {score.metadata['non_factual_count']}")
        for claim in score.metadata["non_factual_claims"]:
            print(f"    ✗ {claim}")


async def example_6_cost_tracking():
    """Example 6: Track LLM costs and interactions."""
    print("\n" + "=" * 60)
    print("Example 6: LLM Interaction Tracking")
    print("=" * 60)

    output = "Claude AI was created by Anthropic and released in 2023"
    reference = "Claude is an AI assistant created by Anthropic, first released in March 2023"

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["factuality"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nFactuality Score: {result.overall_score:.2f}")
    print(f"\nLLM Interaction Details:")
    print(f"  Total Tokens Used: {result.total_tokens}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  LLM Calls Made: {len(result.interactions)}")

    if result.interactions:
        interaction = result.interactions[0]
        print(f"\nFirst Interaction:")
        print(f"  Model: {interaction.model}")
        print(f"  Tokens: {interaction.tokens_used}")
        print(f"  Latency: {interaction.latency:.2f}s")
        print(f"  Purpose: {interaction.purpose}")


async def main():
    """Run all factuality evaluation examples."""
    print("\n🔍 Arbiter - Factuality Evaluation Examples")
    print("=" * 60)
    print("\nThese examples demonstrate hallucination detection and fact-checking")
    print("using the FactualityEvaluator.\n")

    # Run all examples
    await example_1_detect_hallucination()
    await example_2_standalone_factchecking()
    await example_3_rag_validation()
    await example_4_specific_criteria()
    await example_5_multiple_hallucinations()
    await example_6_cost_tracking()

    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
