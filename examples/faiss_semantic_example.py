"""FAISS Semantic Similarity - Fast, Free, Deterministic Evaluation

This example demonstrates the FAISS backend for semantic evaluation: 20-60x faster,
$0 cost, deterministic results, and completely offline operation.

Key Features:
- 20-60x faster than LLM backend (50ms vs 2s)
- 100% cost reduction ($0 vs $0.001 per comparison)
- Deterministic results (reproducible scores)
- No API keys required (runs locally)
- Ideal for batch processing (1000s of comparisons)

Trade-offs:
- No explanations (scores only)
- Requires: pip install arbiter[scale]

Requirements:
    pip install arbiter[scale]  # Installs FAISS and sentence-transformers

Run with:
    # Install scale dependencies first
    pip install arbiter[scale]

    # Run example
    python examples/faiss_semantic_example.py
"""

import asyncio
import time
from dotenv import load_dotenv

from arbiter import LLMManager
from arbiter.evaluators import SemanticEvaluator


async def main():
    """Compare LLM vs FAISS backends for semantic similarity."""

    # Load environment (FAISS doesn't need it, but LLM does for comparison)
    load_dotenv()

    print("🚀 FAISS Semantic Similarity Example")
    print("=" * 60)
    print("\nThis example compares LLM vs FAISS backends:")
    print("- LLM: Rich explanations, slower (~2s), costs tokens (~$0.001)")
    print("- FAISS: Just scores, faster (~50ms), free ($0)")
    print()

    # Get LLM client (needed for initialization, but FAISS won't use it)
    client = await LLMManager.get_client(model="gpt-4o-mini")

    # Test cases
    test_cases = [
        {
            "output": "Paris is the capital of France",
            "reference": "The capital of France is Paris",
            "expected": "High similarity (same meaning, different wording)",
        },
        {
            "output": "The quick brown fox jumps over the lazy dog",
            "reference": "A fast brown fox leaps above a sleepy canine",
            "expected": "High similarity (paraphrase with synonyms)",
        },
        {
            "output": "Python is a programming language",
            "reference": "The weather is nice today",
            "expected": "Low similarity (completely different topics)",
        },
    ]

    # ========================================
    # Example 1: FAISS Backend (Fast & Free)
    # ========================================
    print("\n📦 Example 1: FAISS Backend (Fast & Free)")
    print("-" * 60)

    try:
        evaluator_faiss = SemanticEvaluator(client, backend="faiss")

        for i, case in enumerate(test_cases, 1):
            print(f"\n🔍 Test Case {i}: {case['expected']}")
            print(f"Output:    {case['output'][:50]}...")
            print(f"Reference: {case['reference'][:50]}...")

            # Time the evaluation
            start = time.time()
            score = await evaluator_faiss.evaluate(
                output=case["output"], reference=case["reference"]
            )
            elapsed = time.time() - start

            print(f"\n📊 FAISS Result:")
            print(f"  Score:      {score.value:.3f}")
            print(f"  Confidence: {score.confidence:.3f}")
            print(f"  Latency:    {elapsed*1000:.1f}ms")
            print(f"  Cost:       $0.00 (free!)")
            print(f"  Backend:    {score.metadata.get('backend', 'unknown')}")
            print(f"  Model:      {score.metadata.get('model', 'unknown')}")
            print(
                f"  Embed Dim:  {score.metadata.get('embedding_dim', 'unknown')}"
            )

    except ImportError as e:
        print("\n⚠️  FAISS backend not available:")
        print(f"   {e}")
        print("\n   Install with: pip install arbiter[scale]")
        return

    # ========================================
    # Example 2: LLM Backend (Rich Explanations)
    # ========================================
    print("\n\n💬 Example 2: LLM Backend (Rich Explanations)")
    print("-" * 60)
    print("(Running one comparison to show the difference)\n")

    evaluator_llm = SemanticEvaluator(client, backend="llm")

    case = test_cases[0]  # Use first test case
    print(f"Output:    {case['output']}")
    print(f"Reference: {case['reference']}")

    start = time.time()
    score_llm = await evaluator_llm.evaluate(
        output=case["output"], reference=case["reference"]
    )
    elapsed_llm = time.time() - start

    print(f"\n📊 LLM Result:")
    print(f"  Score:       {score_llm.value:.3f}")
    print(f"  Confidence:  {score_llm.confidence:.3f}")
    print(f"  Latency:     {elapsed_llm*1000:.1f}ms")
    print(f"  Cost:        ~$0.001")
    print(f"  Explanation: {score_llm.explanation[:200]}...")

    # ========================================
    # Example 3: Batch Processing (FAISS shines)
    # ========================================
    print("\n\n⚡ Example 3: Batch Processing Performance")
    print("-" * 60)

    batch_size = 100
    print(f"Evaluating {batch_size} comparisons...\n")

    # Generate test data
    outputs = [
        f"This is test sentence number {i}" for i in range(batch_size)
    ]
    references = [
        f"Test sentence {i} for comparison" for i in range(batch_size)
    ]

    # Batch evaluation with FAISS
    start = time.time()
    faiss_scores = []
    for output, reference in zip(outputs, references):
        score = await evaluator_faiss.evaluate(
            output=output, reference=reference
        )
        faiss_scores.append(score.value)
    faiss_elapsed = time.time() - start

    print(f"📦 FAISS Batch Results:")
    print(f"  Comparisons:  {batch_size}")
    print(f"  Total Time:   {faiss_elapsed:.2f}s")
    print(f"  Avg Latency:  {faiss_elapsed/batch_size*1000:.1f}ms per comparison")
    print(f"  Total Cost:   $0.00")
    print(f"  Avg Score:    {sum(faiss_scores)/len(faiss_scores):.3f}")

    # Estimate LLM costs
    llm_estimated_time = batch_size * 2.0  # 2s per comparison
    llm_estimated_cost = batch_size * 0.001  # $0.001 per comparison

    print(f"\n💬 LLM Estimated (for comparison):")
    print(f"  Comparisons:  {batch_size}")
    print(f"  Total Time:   ~{llm_estimated_time:.2f}s")
    print(
        f"  Avg Latency:  ~{llm_estimated_time/batch_size*1000:.1f}ms per comparison"
    )
    print(f"  Total Cost:   ~${llm_estimated_cost:.3f}")

    print(f"\n🚀 Performance Gains:")
    print(f"  Speed:  {llm_estimated_time/faiss_elapsed:.1f}x faster")
    print(f"  Cost:   100% reduction (${llm_estimated_cost:.3f} → $0.00)")

    # ========================================
    # Summary
    # ========================================
    print("\n\n" + "=" * 60)
    print("✅ FAISS Backend Summary")
    print("=" * 60)
    print("\n✨ Advantages:")
    print("  • 20-60x faster than LLM (50ms vs 2s)")
    print("  • 100% cost reduction ($0 vs $0.001)")
    print("  • Deterministic (same input → same score)")
    print("  • Offline (no API keys required)")
    print("\n⚠️  Limitations:")
    print("  • No explanations (just scores)")
    print("  • Requires reference text")
    print("  • Requires: pip install arbiter[scale]")
    print("\n🎯 Best Use Cases:")
    print("  • Batch processing (1000s of comparisons)")
    print("  • Development/testing (no API costs)")
    print("  • Simple similarity checks (don't need explanations)")
    print("\n💡 When to Use LLM Backend:")
    print("  • Need detailed explanations")
    print("  • Nuanced semantic analysis")
    print("  • Production where understanding matters")


if __name__ == "__main__":
    asyncio.run(main())
