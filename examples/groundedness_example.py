"""Groundedness Evaluation - RAG System Validation & Citation Tracking

This example demonstrates using GroundednessEvaluator to validate that RAG
outputs are properly grounded in source documents with accurate citations.

Key Features:
- RAG output validation against source documents
- Hallucination and ungrounded claim detection
- Citation mapping (statements → source documents)
- Partial groundedness evaluation for mixed outputs
- Statement categorization (grounded/ungrounded with citations)

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/groundedness_example.py
"""

import asyncio
import os

from arbiter_ai import evaluate


async def example_1_basic_rag_validation():
    """Example 1: Basic RAG output validation against source documents."""
    print("=" * 60)
    print("Example 1: Basic RAG Output Validation")
    print("=" * 60)

    # RAG output that should be grounded in source
    output = "The Eiffel Tower was built in 1889 and is located in Paris, France"

    # Source document from retrieval
    reference = """The Eiffel Tower is a wrought-iron lattice tower located in Paris, France.
It was constructed from 1887 to 1889 as the entrance arch for the 1889 World's Fair."""

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nRAG Output: {output}")
    print(f"\nSource Document:\n{reference}")
    print(f"\nGroundedness Score: {result.overall_score:.2f}")
    print(f"Fully Grounded: {result.passed}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Grounded Statements: {score.metadata.get('grounded_count', 0)}")
    print(f"  Ungrounded Statements: {score.metadata.get('ungrounded_count', 0)}")

    if score.metadata.get("citations"):
        print(f"\n  Citation Mapping:")
        for stmt, source in score.metadata["citations"].items():
            print(f"    '{stmt}' → '{source[:50]}...'")


async def example_2_detect_hallucination():
    """Example 2: Detect hallucinated content not supported by sources."""
    print("\n" + "=" * 60)
    print("Example 2: Detect Hallucination in RAG Output")
    print("=" * 60)

    # Output with hallucinated height
    output = "The Eiffel Tower is 330 meters tall and was built in 1889"

    # Source with correct information
    reference = "The Eiffel Tower was constructed in 1889 and stands 300 meters tall"

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput: {output}")
    print(f"Source: {reference}")
    print(f"\nGroundedness Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Grounded: {score.metadata.get('grounded_count', 0)}")
    print(f"  Ungrounded: {score.metadata.get('ungrounded_count', 0)}")

    if score.metadata.get("grounded_statements"):
        print(f"\n  Grounded Statements:")
        for stmt in score.metadata["grounded_statements"]:
            print(f"    ✓ {stmt}")

    if score.metadata.get("ungrounded_statements"):
        print(f"\n  Ungrounded/Hallucinated:")
        for stmt in score.metadata["ungrounded_statements"]:
            print(f"    ✗ {stmt}")


async def example_3_multi_source_validation():
    """Example 3: Validate output against multiple source documents."""
    print("\n" + "=" * 60)
    print("Example 3: Multi-Source RAG Validation")
    print("=" * 60)

    # Complex RAG output synthesizing multiple sources
    output = """Python is a high-level programming language created by Guido van Rossum.
It was first released in 1991 and emphasizes code readability.
Python 3.0 was released in 2008, introducing significant changes."""

    # Multiple source documents (concatenated for this example)
    reference = """
Source 1: Python is a high-level programming language created by Guido van Rossum.
The language was first released in 1991.

Source 2: Python emphasizes code readability with its use of significant indentation.
The language design philosophy emphasizes code clarity.

Source 3: Python 3.0 was released on December 3, 2008, and was a major revision
that was not completely backward-compatible with earlier versions.
"""

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nRAG Output:\n{output}")
    print(f"\nSource Documents:\n{reference}")
    print(f"\nGroundedness Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Total Statements: {score.metadata.get('total_statements', 0)}")
    print(f"  Grounded: {score.metadata.get('grounded_count', 0)}")
    print(f"  Ungrounded: {score.metadata.get('ungrounded_count', 0)}")

    if score.metadata.get("citations"):
        print(f"\n  Statement → Source Attribution:")
        for stmt, source in score.metadata["citations"].items():
            print(f"    • {stmt}")
            print(f"      ↳ {source[:80]}...")


async def example_4_partial_groundedness():
    """Example 4: Handle partially grounded output with mixed content."""
    print("\n" + "=" * 60)
    print("Example 4: Partial Groundedness Detection")
    print("=" * 60)

    # Output with mix of grounded and ungrounded statements
    output = """The human heart has four chambers and beats approximately
60-100 times per minute at rest. It pumps about 5 liters of blood per minute.
The heart is located in the chest cavity and weighs around 500 grams."""

    # Source with partial information
    reference = """The human heart is a muscular organ with four chambers:
two atria and two ventricles. A normal resting heart rate for adults ranges
from 60 to 100 beats per minute. The heart pumps blood throughout the body
via the circulatory system."""

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput:\n{output}")
    print(f"\nSource:\n{reference}")
    print(f"\nGroundedness Score: {result.overall_score:.2f}")
    print(f"Fully Grounded: {result.passed}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Grounded: {score.metadata.get('grounded_count', 0)}")
    print(f"  Ungrounded: {score.metadata.get('ungrounded_count', 0)}")

    if score.metadata.get("grounded_statements"):
        print(f"\n  Grounded (Supported by Source):")
        for stmt in score.metadata["grounded_statements"]:
            print(f"    ✓ {stmt}")

    if score.metadata.get("ungrounded_statements"):
        print(f"\n  Ungrounded (Not in Source):")
        for stmt in score.metadata["ungrounded_statements"]:
            print(f"    ? {stmt}")


async def example_5_specific_criteria():
    """Example 5: Focus groundedness evaluation on specific criteria."""
    print("\n" + "=" * 60)
    print("Example 5: Groundedness with Specific Criteria")
    print("=" * 60)

    # Technical documentation output
    output = """The API supports GET and POST requests. It uses JSON for data exchange
and requires OAuth 2.0 authentication. Rate limits are set at 1000 requests per hour."""

    # Source documentation
    reference = """API Endpoints:
The API accepts GET and POST requests at https://api.example.com/v1/.
All request and response bodies must be in JSON format.
Authentication uses OAuth 2.0 bearer tokens."""

    criteria = "Focus on technical specifications: request methods, data formats, and authentication"

    result = await evaluate(
        output=output,
        reference=reference,
        criteria=criteria,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nOutput: {output}")
    print(f"\nSource: {reference}")
    print(f"\nCriteria: {criteria}")
    print(f"\nGroundedness Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Grounded: {score.metadata.get('grounded_count', 0)}")
    print(f"  Ungrounded: {score.metadata.get('ungrounded_count', 0)}")


async def example_6_observability():
    """Example 6: Inspect LLM interactions and token usage."""
    print("\n" + "=" * 60)
    print("Example 6: Observability - LLM Interaction Tracking")
    print("=" * 60)

    output = "Paris is the capital of France with a population of 2.2 million"
    reference = "Paris, the capital city of France, has a population of 2.16 million residents"

    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=["groundedness"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nGroundedness Score: {result.overall_score:.2f}")
    print(f"\nObservability Metrics:")
    print(f"  Total Tokens: {result.total_tokens:,}")
    print(f"  Processing Time: {result.processing_time:.2f}s")
    print(f"  LLM Interactions: {len(result.interactions)}")

    # Detailed interaction tracking
    for i, interaction in enumerate(result.interactions, 1):
        print(f"\n  Interaction {i}:")
        print(f"    Model: {interaction.model}")
        print(f"    Purpose: {interaction.purpose}")
        print(f"    Tokens: {interaction.tokens_used}")
        print(f"    Latency: {interaction.latency:.2f}s")
        print(f"    Timestamp: {interaction.timestamp}")


async def main():
    """Run all groundedness evaluation examples."""
    print("\n" + "=" * 60)
    print("GROUNDEDNESS EVALUATOR EXAMPLES")
    print("Validating RAG Outputs & Detecting Hallucinations")
    print("=" * 60)

    await example_1_basic_rag_validation()
    await example_2_detect_hallucination()
    await example_3_multi_source_validation()
    await example_4_partial_groundedness()
    await example_5_specific_criteria()
    await example_6_observability()

    print("\n" + "=" * 60)
    print("All groundedness examples completed!")
    print("=" * 60)

    print("\n📖 Related Examples:")
    print("  • See factuality_example.py for hallucination detection")
    print("  • See rag_evaluation.py for comprehensive RAG evaluation")
    print("  • See relevance_example.py for query-output alignment")


if __name__ == "__main__":
    asyncio.run(main())
