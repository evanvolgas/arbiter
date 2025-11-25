"""Relevance Evaluation - Query-Output Alignment Assessment

This example demonstrates using RelevanceEvaluator to assess whether LLM outputs
are relevant to queries, identifying addressed points and detecting off-topic content.

Key Features:
- Query-output relevance assessment
- Addressed vs missing query point identification
- Irrelevant/off-topic content detection
- Response completeness evaluation
- Detailed breakdown of coverage and gaps

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/relevance_example.py
"""

import asyncio
import os

from arbiter_ai import evaluate


async def example_1_basic_query_relevance():
    """Example 1: Basic query-output relevance assessment."""
    print("=" * 60)
    print("Example 1: Basic Query-Output Relevance")
    print("=" * 60)

    # Simple query-answer pair
    output = "Paris is the capital of France"
    query = "What is the capital of France?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery: {query}")
    print(f"Output: {output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")
    print(f"Fully Relevant: {result.passed}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Addressed Points: {score.metadata.get('addressed_count', 0)}")
    print(f"  Missing Points: {score.metadata.get('missing_count', 0)}")

    if score.metadata.get("addressed_points"):
        print(f"\n  What Was Addressed:")
        for point in score.metadata["addressed_points"]:
            print(f"    ✓ {point}")


async def example_2_incomplete_answer():
    """Example 2: Detect incomplete answers with missing points."""
    print("\n" + "=" * 60)
    print("Example 2: Incomplete Answer Detection")
    print("=" * 60)

    # Query asks for multiple pieces of information
    output = "The Eiffel Tower is 300 meters tall"
    query = "How tall is the Eiffel Tower and when was it built?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery: {query}")
    print(f"Output: {output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Addressed: {score.metadata.get('addressed_count', 0)}")
    print(f"  Missing: {score.metadata.get('missing_count', 0)}")

    if score.metadata.get("addressed_points"):
        print(f"\n  Addressed Points:")
        for point in score.metadata["addressed_points"]:
            print(f"    ✓ {point}")

    if score.metadata.get("missing_points"):
        print(f"\n  Missing Points:")
        for point in score.metadata["missing_points"]:
            print(f"    ✗ {point}")


async def example_3_irrelevant_content():
    """Example 3: Detect irrelevant or off-topic content."""
    print("\n" + "=" * 60)
    print("Example 3: Irrelevant Content Detection")
    print("=" * 60)

    # Output contains off-topic information
    output = """Paris is the capital of France. It's also known for its amazing restaurants,
the Eiffel Tower attracts millions of tourists annually, and the weather in spring is lovely."""
    query = "What is the capital of France?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery: {query}")
    print(f"Output:\n{output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Relevant Points: {score.metadata.get('addressed_count', 0)}")
    print(f"  Irrelevant Items: {score.metadata.get('irrelevant_count', 0)}")

    if score.metadata.get("addressed_points"):
        print(f"\n  Relevant Content:")
        for point in score.metadata["addressed_points"]:
            print(f"    ✓ {point}")

    if score.metadata.get("irrelevant_content"):
        print(f"\n  Irrelevant/Off-Topic:")
        for content in score.metadata["irrelevant_content"]:
            print(f"    ⚠ {content}")


async def example_4_multi_aspect_query():
    """Example 4: Evaluate complex multi-aspect queries."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Aspect Query Evaluation")
    print("=" * 60)

    # Complex query with multiple aspects
    output = """Python is a high-level programming language created by Guido van Rossum.
It emphasizes code readability and uses indentation for code blocks."""
    query = "Who created Python, when was it released, and what are its main features?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery:\n{query}")
    print(f"\nOutput:\n{output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Total Query Aspects: {score.metadata.get('total_points', 0)}")
    print(f"  Addressed: {score.metadata.get('addressed_count', 0)}")
    print(f"  Missing: {score.metadata.get('missing_count', 0)}")

    if score.metadata.get("addressed_points"):
        print(f"\n  Addressed Aspects:")
        for point in score.metadata["addressed_points"]:
            print(f"    ✓ {point}")

    if score.metadata.get("missing_points"):
        print(f"\n  Missing Aspects:")
        for point in score.metadata["missing_points"]:
            print(f"    ? {point}")


async def example_5_completely_off_topic():
    """Example 5: Detect completely off-topic responses."""
    print("\n" + "=" * 60)
    print("Example 5: Completely Off-Topic Response")
    print("=" * 60)

    # Completely irrelevant answer
    output = "The weather today is sunny with a high of 75°F"
    query = "What is the capital of France?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery: {query}")
    print(f"Output: {output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")
    print(f"On-Topic: {result.passed}")

    score = result.scores[0]
    if score.metadata.get("missing_points"):
        print(f"\n  What Should Have Been Addressed:")
        for point in score.metadata["missing_points"]:
            print(f"    ✗ {point}")


async def example_6_criteria_based_relevance():
    """Example 6: Evaluate relevance based on specific criteria."""
    print("\n" + "=" * 60)
    print("Example 6: Criteria-Based Relevance")
    print("=" * 60)

    # Evaluate against specific criteria
    output = "Python is good for beginners"
    query = "What programming language should I learn?"
    criteria = "Provide specific recommendations with reasoning and examples"

    result = await evaluate(
        output=output,
        reference=query,
        criteria=criteria,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery: {query}")
    print(f"Output: {output}")
    print(f"Criteria: {criteria}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Confidence: {score.confidence:.2f}")
    print(f"  Addressed: {score.metadata.get('addressed_count', 0)}")
    print(f"  Missing: {score.metadata.get('missing_count', 0)}")


async def example_7_question_answering_system():
    """Example 7: Quality control for question-answering systems."""
    print("\n" + "=" * 60)
    print("Example 7: QA System Quality Control")
    print("=" * 60)

    # Realistic QA scenario
    output = """The Battle of Hastings took place in 1066 in England.
It was fought between the Norman-French army of William, the Duke of Normandy,
and an English army under the Anglo-Saxon King Harold Godwinson."""
    query = "When did the Battle of Hastings occur and who were the main participants?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nQuery:\n{query}")
    print(f"\nOutput:\n{output}")
    print(f"\nRelevance Score: {result.overall_score:.2f}")
    print(f"Quality Check: {'PASS' if result.passed else 'FAIL'}")

    score = result.scores[0]
    print(f"\nScore Details:")
    print(f"  Addressed Points: {score.metadata.get('addressed_count', 0)}")
    print(f"  Missing Points: {score.metadata.get('missing_count', 0)}")


async def example_8_observability():
    """Example 8: Inspect LLM interactions and token usage."""
    print("\n" + "=" * 60)
    print("Example 8: Observability - LLM Interaction Tracking")
    print("=" * 60)

    output = "Paris is the capital of France"
    query = "What is the capital of France?"

    result = await evaluate(
        output=output,
        reference=query,
        evaluators=["relevance"],
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    )

    print(f"\nRelevance Score: {result.overall_score:.2f}")
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
    """Run all relevance evaluation examples."""
    print("\n" + "=" * 60)
    print("RELEVANCE EVALUATOR EXAMPLES")
    print("Assessing Query-Output Alignment & Completeness")
    print("=" * 60)

    await example_1_basic_query_relevance()
    await example_2_incomplete_answer()
    await example_3_irrelevant_content()
    await example_4_multi_aspect_query()
    await example_5_completely_off_topic()
    await example_6_criteria_based_relevance()
    await example_7_question_answering_system()
    await example_8_observability()

    print("\n" + "=" * 60)
    print("All relevance examples completed!")
    print("=" * 60)

    print("\n📖 Related Examples:")
    print("  • See factuality_example.py for fact-checking and hallucination detection")
    print("  • See groundedness_example.py for RAG output validation")
    print("  • See semantic_example.py for meaning-based similarity")


if __name__ == "__main__":
    asyncio.run(main())
