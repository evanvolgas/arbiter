"""RAG System Evaluation - Retrieval-Augmented Generation Assessment

This example demonstrates evaluating RAG systems with Arbiter, assessing both
retrieval quality and generation accuracy with comprehensive evaluators.

Key Features:
- Semantic similarity for answer matching
- Custom criteria (accuracy, completeness, citations)
- Groundedness evaluation (claims supported by sources)
- Source attribution validation
- Multi-aspect RAG quality assessment

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/rag_evaluation.py
"""

from dotenv import load_dotenv

import asyncio
import os
from typing import List, Optional

from arbiter_ai import evaluate


async def evaluate_rag_response(
    query: str,
    answer: str,
    retrieved_context: List[str],
    expected_answer: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """Evaluate a RAG system response comprehensively.

    Args:
        query: The user's question
        answer: The RAG system's answer
        retrieved_context: List of retrieved context chunks
        expected_answer: Optional expected answer for comparison
        model: Model to use for evaluation

    Returns:
        Dictionary with evaluation results and insights
    """
    results = {}

    # Combine retrieved context for evaluation
    context_text = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_context)])

    # Evaluation 1: Semantic Similarity (if expected answer provided)
    if expected_answer:
        semantic_result = await evaluate(
            output=answer,
            reference=expected_answer,
            evaluators=["semantic"],
            model=model,
        )
        results["semantic_similarity"] = {
            "score": semantic_result.overall_score,
            "passed": semantic_result.passed,
        }

    # Evaluation 2: Answer Quality (Custom Criteria)
    answer_quality_criteria = (
        "Accuracy: Answer is factually correct based on the context. "
        "Completeness: Answer fully addresses the query. "
        "Clarity: Answer is clear and well-structured. "
        "Citation: Answer references or can be traced to source material."
    )

    answer_quality_result = await evaluate(
        output=answer,
        criteria=answer_quality_criteria,
        evaluators=["custom_criteria"],
        model=model,
    )

    results["answer_quality"] = {
        "score": answer_quality_result.overall_score,
        "passed": answer_quality_result.passed,
        "criteria_met": answer_quality_result.scores[0].metadata.get("criteria_met", []),
        "criteria_not_met": answer_quality_result.scores[0].metadata.get("criteria_not_met", []),
    }

    # Evaluation 3: Source Attribution (Custom Criteria)
    # Check if answer can be attributed to retrieved context
    attribution_criteria = (
        "Source Attribution: All factual claims in the answer can be found in the retrieved context. "
        "No Hallucination: Answer does not contain information not present in the context. "
        "Proper Citation: Answer structure allows tracing claims to specific sources."
    )

    # Include context in the evaluation
    answer_with_context = f"Answer: {answer}\n\nRetrieved Context:\n{context_text}"

    attribution_result = await evaluate(
        output=answer_with_context,
        criteria=attribution_criteria,
        evaluators=["custom_criteria"],
        model=model,
    )

    results["source_attribution"] = {
        "score": attribution_result.overall_score,
        "passed": attribution_result.passed,
        "criteria_met": attribution_result.scores[0].metadata.get("criteria_met", []),
        "criteria_not_met": attribution_result.scores[0].metadata.get("criteria_not_met", []),
    }

    # Evaluation 4: Query-Answer Relevance (Custom Criteria)
    relevance_criteria = (
        "Relevance: Answer directly addresses the query. "
        "On-Topic: Answer stays focused on the query topic. "
        "Completeness: Answer addresses all aspects of the query."
    )

    query_answer_pair = f"Query: {query}\n\nAnswer: {answer}"

    relevance_result = await evaluate(
        output=query_answer_pair,
        criteria=relevance_criteria,
        evaluators=["custom_criteria"],
        model=model,
    )

    results["relevance"] = {
        "score": relevance_result.overall_score,
        "passed": relevance_result.passed,
    }

    # Aggregate metrics
    all_scores = [
        results["answer_quality"]["score"],
        results["source_attribution"]["score"],
        results["relevance"]["score"],
    ]

    if expected_answer:
        all_scores.append(results["semantic_similarity"]["score"])

    results["overall_score"] = sum(all_scores) / len(all_scores)
    results["all_passed"] = all(
        r["passed"] for r in [
            results["answer_quality"],
            results["source_attribution"],
            results["relevance"],
        ]
    )

    # Cost tracking
    total_tokens = (
        answer_quality_result.total_tokens +
        attribution_result.total_tokens +
        relevance_result.total_tokens
    )
    if expected_answer:
        total_tokens += semantic_result.total_tokens

    results["total_tokens"] = total_tokens

    # Calculate costs asynchronously
    costs = await asyncio.gather(
        answer_quality_result.total_llm_cost(),
        attribution_result.total_llm_cost(),
        relevance_result.total_llm_cost()
    )
    results["total_cost"] = sum(costs)
    if expected_answer:
        results["total_cost"] += await semantic_result.total_llm_cost()

    return results


async def main():
    """Run RAG evaluation examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - RAG System Evaluation Example")
    print("=" * 70)
    print("\nThis example demonstrates comprehensive evaluation of RAG systems,")
    print("evaluating both retrieval quality and generation quality.")
    print("=" * 70)

    # Example 1: Good RAG Response
    print("\n\n📊 Example 1: Good RAG Response")
    print("-" * 70)

    query1 = "What is the capital of France?"
    answer1 = "The capital of France is Paris. Paris is located in the north-central part of the country and has been the capital since 987 AD."
    context1 = [
        "Paris is the capital and most populous city of France. It has been the capital since 987 AD.",
        "France is a country in Western Europe. Its capital city is Paris.",
    ]
    expected1 = "Paris is the capital of France."

    print(f"\nQuery: {query1}")
    print(f"Answer: {answer1}")
    print(f"Retrieved Context: {len(context1)} chunks")
    print(f"Expected Answer: {expected1}")

    results1 = await evaluate_rag_response(
        query=query1,
        answer=answer1,
        retrieved_context=context1,
        expected_answer=expected1,
    )

    print(f"\n📊 Evaluation Results:")
    print(f"   Overall Score: {results1['overall_score']:.3f}")
    print(f"   All Criteria Passed: {'✅' if results1['all_passed'] else '❌'}")
    print(f"\n   Individual Scores:")
    if "semantic_similarity" in results1:
        print(f"     Semantic Similarity: {results1['semantic_similarity']['score']:.3f} {'✅' if results1['semantic_similarity']['passed'] else '❌'}")
    print(f"     Answer Quality: {results1['answer_quality']['score']:.3f} {'✅' if results1['answer_quality']['passed'] else '❌'}")
    print(f"     Source Attribution: {results1['source_attribution']['score']:.3f} {'✅' if results1['source_attribution']['passed'] else '❌'}")
    print(f"     Relevance: {results1['relevance']['score']:.3f} {'✅' if results1['relevance']['passed'] else '❌'}")

    print(f"\n   Answer Quality Details:")
    print(f"     Criteria Met: {', '.join(results1['answer_quality']['criteria_met']) if results1['answer_quality']['criteria_met'] else 'None'}")
    print(f"     Criteria Not Met: {', '.join(results1['answer_quality']['criteria_not_met']) if results1['answer_quality']['criteria_not_met'] else 'None'}")

    print(f"\n   Cost Analysis:")
    print(f"     Total Tokens: {results1['total_tokens']:,}")
    print(f"     Total Cost: ${results1['total_cost']:.6f}")

    # Example 2: RAG Response with Hallucination
    print("\n\n📊 Example 2: RAG Response with Hallucination")
    print("-" * 70)

    query2 = "What is the capital of France?"
    answer2 = "The capital of France is Paris. Paris has a population of 15 million people and is located on the Mediterranean coast."
    context2 = [
        "Paris is the capital and most populous city of France.",
        "France is a country in Western Europe.",
    ]
    # Note: Context doesn't mention population or Mediterranean coast

    print(f"\nQuery: {query2}")
    print(f"Answer: {answer2}")
    print(f"Retrieved Context: {len(context2)} chunks")
    print(f"⚠️  Note: Answer contains information not in context (hallucination)")

    results2 = await evaluate_rag_response(
        query=query2,
        answer=answer2,
        retrieved_context=context2,
        expected_answer=expected1,
    )

    print(f"\n📊 Evaluation Results:")
    print(f"   Overall Score: {results2['overall_score']:.3f}")
    print(f"   All Criteria Passed: {'✅' if results2['all_passed'] else '❌'}")
    print(f"\n   Individual Scores:")
    if "semantic_similarity" in results2:
        print(f"     Semantic Similarity: {results2['semantic_similarity']['score']:.3f} {'✅' if results2['semantic_similarity']['passed'] else '❌'}")
    print(f"     Answer Quality: {results2['answer_quality']['score']:.3f} {'✅' if results2['answer_quality']['passed'] else '❌'}")
    print(f"     Source Attribution: {results2['source_attribution']['score']:.3f} {'✅' if results2['source_attribution']['passed'] else '❌'}")
    print(f"     Relevance: {results2['relevance']['score']:.3f} {'✅' if results2['relevance']['passed'] else '❌'}")

    print(f"\n   Source Attribution Details:")
    print(f"     Criteria Met: {', '.join(results2['source_attribution']['criteria_met']) if results2['source_attribution']['criteria_met'] else 'None'}")
    print(f"     Criteria Not Met: {', '.join(results2['source_attribution']['criteria_not_met']) if results2['source_attribution']['criteria_not_met'] else 'None'}")
    print(f"     ⚠️  Hallucination detected - answer contains unsupported claims")

    # Example 3: RAG Response with Poor Retrieval
    print("\n\n📊 Example 3: RAG Response with Poor Retrieval")
    print("-" * 70)

    query3 = "What is the capital of France?"
    answer3 = "I don't have information about France in the provided context."
    context3 = [
        "Python is a programming language.",
        "Machine learning is a subset of artificial intelligence.",
    ]
    # Note: Context is irrelevant to the query

    print(f"\nQuery: {query3}")
    print(f"Answer: {answer3}")
    print(f"Retrieved Context: {len(context3)} chunks")
    print(f"⚠️  Note: Retrieved context is irrelevant to the query")

    results3 = await evaluate_rag_response(
        query=query3,
        answer=answer3,
        retrieved_context=context3,
        expected_answer=expected1,
    )

    print(f"\n📊 Evaluation Results:")
    print(f"   Overall Score: {results3['overall_score']:.3f}")
    print(f"   All Criteria Passed: {'✅' if results3['all_passed'] else '❌'}")
    print(f"\n   Individual Scores:")
    if "semantic_similarity" in results3:
        print(f"     Semantic Similarity: {results3['semantic_similarity']['score']:.3f} {'✅' if results3['semantic_similarity']['passed'] else '❌'}")
    print(f"     Answer Quality: {results3['answer_quality']['score']:.3f} {'✅' if results3['answer_quality']['passed'] else '❌'}")
    print(f"     Source Attribution: {results3['source_attribution']['score']:.3f} {'✅' if results3['source_attribution']['passed'] else '❌'}")
    print(f"     Relevance: {results3['relevance']['score']:.3f} {'✅' if results3['relevance']['passed'] else '❌'}")
    print(f"     ⚠️  Poor retrieval quality - context doesn't match query")

    # Example 4: Multi-Evaluator RAG Evaluation
    print("\n\n📊 Example 4: Multi-Evaluator RAG Evaluation")
    print("-" * 70)

    query4 = "Explain how photosynthesis works."
    answer4 = (
        "Photosynthesis is the process by which plants convert light energy into chemical energy. "
        "It occurs in chloroplasts and involves two main stages: light-dependent reactions and "
        "light-independent reactions (Calvin cycle). The process produces glucose and oxygen."
    )
    context4 = [
        "Photosynthesis is a process used by plants to convert light energy into chemical energy.",
        "The process occurs in chloroplasts and involves light-dependent and light-independent reactions.",
        "Photosynthesis produces glucose and releases oxygen as a byproduct.",
    ]

    print(f"\nQuery: {query4}")
    print(f"Answer: {answer4[:80]}...")
    print(f"Retrieved Context: {len(context4)} chunks")

    # Use multiple evaluators simultaneously
    multi_result = await evaluate(
        output=answer4,
        reference="Photosynthesis converts light energy to chemical energy in plants.",
        criteria="Scientific accuracy, completeness, clarity",
        evaluators=["semantic", "custom_criteria"],
        model="gpt-4o-mini",
    )

    print(f"\n📊 Multi-Evaluator Results:")
    print(f"   Overall Score: {multi_result.overall_score:.3f}")
    print(f"   Passed: {'✅' if multi_result.passed else '❌'}")
    print(f"   Evaluators Used: {', '.join(multi_result.evaluator_names)}")
    print(f"\n   Individual Scores:")
    for score in multi_result.scores:
        print(f"     {score.name}: {score.value:.3f}")
        if score.confidence:
            print(f"       Confidence: {score.confidence:.3f}")

    # Summary
    print("\n\n" + "=" * 70)
    print("✅ All Examples Complete!")
    print("=" * 70)

    print("\n🎯 Key Takeaways:")
    print("   • RAG evaluation requires multiple aspects (answer quality, source attribution, relevance)")
    print("   • Custom criteria evaluator is perfect for domain-specific RAG requirements")
    print("   • Semantic similarity helps compare against expected answers")
    print("   • Source attribution evaluation detects hallucinations")
    print("   • Multi-evaluator approach provides comprehensive assessment")

    print("\n💡 RAG Evaluation Best Practices:")
    print("   • Always evaluate source attribution to catch hallucinations")
    print("   • Use semantic similarity when you have expected answers")
    print("   • Custom criteria can capture domain-specific requirements")
    print("   • Combine multiple evaluators for comprehensive assessment")
    print("   • Track costs - RAG evaluation can be expensive with multiple evaluators")

    print("\n📚 Learn More:")
    print("   • See: examples/multiple_evaluators.py for combining evaluators")
    print("   • See: examples/custom_criteria_example.py for custom criteria")
    print("   • See: examples/interaction_tracking_example.py for cost tracking")
    print("   • Phase 5 will add GroundednessEvaluator specifically for RAG")


if __name__ == "__main__":
    asyncio.run(main())



