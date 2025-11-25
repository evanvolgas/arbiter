"""Example demonstrating factuality evaluation with external verification plugins.

This example shows how to enhance factuality evaluation by combining LLM judgment
with external verification from multiple sources:
- SearchVerifier: Web search validation via Tavily API
- CitationVerifier: Source attribution checking for RAG systems
- KnowledgeBaseVerifier: Wikipedia-based fact verification

The final score is a weighted average: LLM (50%) + External verification (50%)
"""

import asyncio
import os

from arbiter_ai import FactualityEvaluator, LLMManager
from arbiter_ai.verifiers import (
    CitationVerifier,
    KnowledgeBaseVerifier,
    SearchVerifier,
)


async def example_factuality_llm_only():
    """Example 1: Factuality evaluation with LLM only (no plugins)."""
    print("\n" + "=" * 60)
    print("Example 1: LLM-Only Factuality Evaluation")
    print("=" * 60)

    # Create evaluator without verifiers (traditional approach)
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = FactualityEvaluator(llm_client=client)

    # Test with a claim containing both factual and non-factual information
    output = """Paris is the capital of France and was founded in 1985.
    The city is known for the Eiffel Tower, which is 500 meters tall."""

    reference = """Paris is the capital of France, founded in ancient times.
    The Eiffel Tower stands approximately 300 meters tall."""

    score = await evaluator.evaluate(output=output, reference=reference)

    print(f"\nOutput:\n{output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"\nFactual Claims: {score.metadata.get('factual_claims', [])}")
    print(f"Non-Factual Claims: {score.metadata.get('non_factual_claims', [])}")


async def example_factuality_with_citation_verifier():
    """Example 2: Factuality evaluation with CitationVerifier."""
    print("\n" + "=" * 60)
    print("Example 2: Factuality with Citation Verification")
    print("=" * 60)

    # Create evaluator with CitationVerifier
    client = await LLMManager.get_client(model="gpt-4o-mini")
    citation_verifier = CitationVerifier()
    evaluator = FactualityEvaluator(
        llm_client=client, verifiers=[citation_verifier]
    )

    # RAG system output example
    output = "The Eiffel Tower is located in Paris and is approximately 300 meters tall."
    source_documents = """The Eiffel Tower is a wrought-iron lattice tower on the
    Champ de Mars in Paris, France. It was constructed from 1887 to 1889 and stands
    approximately 300 meters (984 feet) tall."""

    score = await evaluator.evaluate(output=output, reference=source_documents)

    print(f"\nOutput:\n{output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"Verification Used: {score.metadata.get('verification_used', False)}")

    if score.metadata.get("verification_used"):
        print(
            f"Verification Sources: {score.metadata.get('verification_sources', [])}"
        )


async def example_factuality_with_wikipedia():
    """Example 3: Factuality evaluation with KnowledgeBaseVerifier (Wikipedia)."""
    print("\n" + "=" * 60)
    print("Example 3: Factuality with Wikipedia Verification")
    print("=" * 60)

    # Create evaluator with WikipediaVerifier
    client = await LLMManager.get_client(model="gpt-4o-mini")
    wikipedia_verifier = KnowledgeBaseVerifier()
    evaluator = FactualityEvaluator(
        llm_client=client, verifiers=[wikipedia_verifier]
    )

    # Test with general knowledge claims
    output = "Paris is the capital of France."

    score = await evaluator.evaluate(output=output)

    print(f"\nOutput: {output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"Verification Used: {score.metadata.get('verification_used', False)}")


async def example_factuality_with_search(tavily_api_key: str):
    """Example 4: Factuality evaluation with SearchVerifier (Tavily).

    Args:
        tavily_api_key: Tavily API key for web search
    """
    print("\n" + "=" * 60)
    print("Example 4: Factuality with Web Search Verification")
    print("=" * 60)

    try:
        # Create evaluator with SearchVerifier
        client = await LLMManager.get_client(model="gpt-4o-mini")
        search_verifier = SearchVerifier(api_key=tavily_api_key)
        evaluator = FactualityEvaluator(
            llm_client=client, verifiers=[search_verifier]
        )

        # Test with verifiable claims
        output = "Water boils at 100 degrees Celsius at sea level under standard atmospheric pressure."

        score = await evaluator.evaluate(output=output)

        print(f"\nOutput: {output}")
        print(f"\nFactuality Score: {score.value:.2f}")
        print(f"Confidence: {score.confidence:.2f}")
        print(f"Verification Used: {score.metadata.get('verification_used', False)}")
        print(
            f"Verification Count: {score.metadata.get('verification_count', 0)} checks"
        )

    except ImportError as e:
        print(f"\nSkipping SearchVerifier example: {e}")
        print("Install with: pip install tavily-python")


async def example_factuality_with_multiple_verifiers(tavily_api_key: str):
    """Example 5: Factuality evaluation with multiple verification plugins.

    This demonstrates the recommended approach: combining multiple verification
    sources for maximum accuracy and confidence.

    Args:
        tavily_api_key: Tavily API key for web search
    """
    print("\n" + "=" * 60)
    print("Example 5: Factuality with Multiple Verifiers (Recommended)")
    print("=" * 60)

    # Create multiple verifiers
    verifiers = [
        CitationVerifier(),
        KnowledgeBaseVerifier(),
    ]

    # Optionally add SearchVerifier if Tavily API key is available
    try:
        verifiers.append(SearchVerifier(api_key=tavily_api_key))
        print("Using: CitationVerifier + KnowledgeBaseVerifier + SearchVerifier")
    except ImportError:
        print("Using: CitationVerifier + KnowledgeBaseVerifier")
        print("(SearchVerifier requires: pip install tavily-python)")

    # Create evaluator with all verifiers
    client = await LLMManager.get_client(model="gpt-4o-mini")
    evaluator = FactualityEvaluator(llm_client=client, verifiers=verifiers)

    # Test with mixed factual/non-factual content
    output = """The Eiffel Tower in Paris is approximately 300 meters tall
    and was completed in 1889 for the World's Fair."""

    source_docs = """The Eiffel Tower was constructed from 1887 to 1889 as the
    entrance to the 1889 World's Fair. It stands 300 meters (984 feet) tall."""

    score = await evaluator.evaluate(output=output, reference=source_docs)

    print(f"\nOutput:\n{output}")
    print(f"\nFactuality Score: {score.value:.2f}")
    print(f"LLM Score: {score.metadata.get('llm_score', 'N/A'):.2f}")
    print(f"Confidence: {score.confidence:.2f}")
    print(f"\nVerification Details:")
    print(f"  - Verification Used: {score.metadata.get('verification_used', False)}")
    print(
        f"  - Verification Count: {score.metadata.get('verification_count', 0)} checks"
    )
    print(
        f"  - Verification Sources: {score.metadata.get('verification_sources', [])}"
    )

    # Show LLM interaction tracking
    interactions = evaluator.get_interactions()
    print(f"\nLLM Interactions: {len(interactions)} calls")
    if interactions:
        print(f"  - Model: {interactions[0].model}")
        print(f"  - Total Tokens: {interactions[0].total_tokens}")


async def main():
    """Run all examples demonstrating factuality verification plugins."""
    print("\n" + "=" * 60)
    print("Factuality Evaluation with External Verification Plugins")
    print("=" * 60)

    # Get Tavily API key from environment (optional)
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Run examples
    await example_factuality_llm_only()
    await example_factuality_with_citation_verifier()
    await example_factuality_with_wikipedia()

    if tavily_api_key:
        await example_factuality_with_search(tavily_api_key)
        await example_factuality_with_multiple_verifiers(tavily_api_key)
    else:
        print("\n" + "=" * 60)
        print("Note: Set TAVILY_API_KEY environment variable to run")
        print("SearchVerifier examples. Get your key at https://tavily.com/")
        print("=" * 60)
        await example_factuality_with_multiple_verifiers("")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
