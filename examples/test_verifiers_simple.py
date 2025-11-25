"""Simple test of verification plugins without LLM dependency."""

import asyncio

from arbiter_ai.verifiers import CitationVerifier, KnowledgeBaseVerifier


async def test_citation_verifier():
    """Test CitationVerifier (no API key needed)."""
    print("\n" + "=" * 60)
    print("Testing CitationVerifier")
    print("=" * 60)

    verifier = CitationVerifier()

    # Test 1: Direct match
    result = await verifier.verify(
        claim="Paris is the capital of France",
        context="Paris is the capital of France and its largest city",
    )
    print(f"\nTest 1 - Direct Match:")
    print(f"  Verified: {result.is_verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation[:80]}...")

    # Test 2: Semantic match
    result = await verifier.verify(
        claim="The Eiffel Tower is in Paris",
        context="Paris has the famous Eiffel Tower landmark",
    )
    print(f"\nTest 2 - Semantic Match:")
    print(f"  Verified: {result.is_verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation[:80]}...")

    # Test 3: No match
    result = await verifier.verify(
        claim="The moon is made of cheese",
        context="The Earth has a rocky surface with water and continents",
    )
    print(f"\nTest 3 - No Match:")
    print(f"  Verified: {result.is_verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Explanation: {result.explanation[:80]}...")


async def test_knowledge_base_verifier():
    """Test KnowledgeBaseVerifier (Wikipedia API - free)."""
    print("\n" + "=" * 60)
    print("Testing KnowledgeBaseVerifier (Wikipedia)")
    print("=" * 60)

    verifier = KnowledgeBaseVerifier()

    # Test 1: Well-known fact
    result = await verifier.verify(claim="Paris is the capital of France")
    print(f"\nTest 1 - Well-known Fact:")
    print(f"  Verified: {result.is_verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Source: {result.source}")
    if result.evidence:
        print(f"  Evidence snippet: {result.evidence[0][:100]}...")

    # Test 2: Historical fact
    result = await verifier.verify(
        claim="World War II ended in 1945"
    )
    print(f"\nTest 2 - Historical Fact:")
    print(f"  Verified: {result.is_verified}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Source: {result.source}")


async def main():
    """Run all verifier tests."""
    print("\n" + "=" * 60)
    print("Verification Plugins Test Suite")
    print("Testing plugins directly without LLM dependency")
    print("=" * 60)

    await test_citation_verifier()
    await test_knowledge_base_verifier()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
