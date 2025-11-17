"""Advanced Configuration - Fine-Tuning Evaluation Behavior

This example demonstrates advanced configuration options for production-grade
evaluations including temperature control, retry logic, and custom clients.

Key Features:
- Temperature control for evaluation consistency
- Custom LLM client configuration
- Retry configuration for reliability
- Connection pooling for performance
- Provider-specific settings
- Model selection strategies

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/advanced_config.py
"""

from dotenv import load_dotenv

import asyncio
import os
from typing import Optional

from arbiter import evaluate
from arbiter.core import LLMManager, Provider
from arbiter.core.llm_client import LLMClient
from arbiter.core.retry import RetryConfig, RETRY_STANDARD, RETRY_PERSISTENT


async def main():
    """Run advanced configuration examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("⚙️  Arbiter - Advanced Configuration Example")
    print("=" * 70)
    print("\nThis example demonstrates advanced configuration options for")
    print("fine-tuning evaluation behavior, performance, and reliability.")
    print("=" * 70)

    # Example 1: Temperature Control
    print("\n\n📊 Example 1: Temperature Control for Consistency")
    print("-" * 70)
    print("\nTemperature controls randomness in LLM responses.")
    print("Lower temperature = more consistent evaluations.")

    test_output = "Paris is the capital of France"
    test_reference = "The capital of France is Paris"

    temperatures = [0.0, 0.3, 0.7, 1.0]
    print(f"\nEvaluating with different temperatures:")
    print(f"Output: {test_output}")
    print(f"Reference: {test_reference}\n")

    for temp in temperatures:
        # Create custom client with specific temperature
        client = LLMClient(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            temperature=temp,
        )

        result = await evaluate(
            output=test_output,
            reference=test_reference,
            evaluators=["semantic"],
            llm_client=client,
        )

        print(f"  Temperature {temp:3.1f}: Score = {result.overall_score:.3f}")

    print("\n💡 Tip: Use temperature=0.0-0.3 for consistent evaluations")
    print("         Use temperature=0.7-1.0 for more creative assessments")

    # Example 2: Model Selection Strategy
    print("\n\n📊 Example 2: Model Selection - Accuracy vs Cost")
    print("-" * 70)

    models = [
        ("gpt-4o-mini", "Fast & Cheap", 0.15/1000),
        ("gpt-4o", "Balanced", 2.50/1000),
        ("gpt-4", "Most Accurate", 30.00/1000),
    ]

    print("\nComparing models for the same evaluation:")
    print(f"Output: {test_output}")
    print(f"Reference: {test_reference}\n")

    for model_name, description, cost_per_1k in models:
        try:
            result = await evaluate(
                output=test_output,
                reference=test_reference,
                evaluators=["semantic"],
                model=model_name,
            )

            cost = await result.total_llm_cost()
            print(f"  {model_name:15s} ({description:15s}):")
            print(f"    Score: {result.overall_score:.3f}")
            print(f"    Tokens: {result.total_tokens:,}")
            print(f"    Cost: ${cost:.6f}")
            print(f"    Latency: {result.processing_time:.3f}s")
        except Exception as e:
            print(f"  {model_name:15s}: ⚠️  Error - {e}")

    print("\n💡 Tip: Use gpt-4o-mini for high-volume, cost-sensitive evaluations")
    print("         Use gpt-4o for balanced accuracy/cost")
    print("         Use gpt-4 for critical evaluations requiring highest accuracy")

    # Example 3: Custom Client Configuration
    print("\n\n📊 Example 3: Custom LLM Client Configuration")
    print("-" * 70)

    print("\nCreating custom client with specific settings:")

    # Custom client with low temperature for consistency
    consistent_client = LLMClient(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.1,  # Very consistent
    )

    result1 = await evaluate(
        output=test_output,
        reference=test_reference,
        evaluators=["semantic"],
        llm_client=consistent_client,
    )

    print(f"  Consistent Client (temp=0.1):")
    print(f"    Score: {result1.overall_score:.3f}")
    print(f"    Tokens: {result1.total_tokens:,}")

    # Custom client with higher temperature for creative evaluation
    creative_client = LLMClient(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        temperature=0.9,  # More creative
    )

    result2 = await evaluate(
        output=test_output,
        reference=test_reference,
        evaluators=["semantic"],
        llm_client=creative_client,
    )

    print(f"  Creative Client (temp=0.9):")
    print(f"    Score: {result2.overall_score:.3f}")
    print(f"    Tokens: {result2.total_tokens:,}")

    print("\n💡 Tip: Create reusable client instances for consistent configuration")

    # Example 4: Provider Selection
    print("\n\n📊 Example 4: Provider Selection")
    print("-" * 70)

    providers_to_test = [
        (Provider.OPENAI, "OpenAI", "gpt-4o-mini"),
    ]

    # Add Anthropic if key is available
    if os.getenv("ANTHROPIC_API_KEY"):
        providers_to_test.append((Provider.ANTHROPIC, "Anthropic", "claude-3-5-sonnet-20241022"))

    # Add Google if key is available
    if os.getenv("GOOGLE_API_KEY"):
        providers_to_test.append((Provider.GOOGLE, "Google", "gemini-1.5-flash"))

    print("\nComparing providers for the same evaluation:")
    print(f"Output: {test_output}")
    print(f"Reference: {test_reference}\n")

    for provider, name, model in providers_to_test:
        try:
            client = await LLMManager.get_client(
                provider=provider,
                model=model,
            )

            result = await evaluate(
                output=test_output,
                reference=test_reference,
                evaluators=["semantic"],
                llm_client=client,
            )

            print(f"  {name:15s} ({model}):")
            print(f"    Score: {result.overall_score:.3f}")
            print(f"    Tokens: {result.total_tokens:,}")
            print(f"    Latency: {result.processing_time:.3f}s")
        except Exception as e:
            print(f"  {name:15s}: ⚠️  Error - {e}")

    print("\n💡 Tip: Different providers may have different characteristics")
    print("         Test multiple providers to find the best fit")

    # Example 5: Reusing Clients for Performance
    print("\n\n📊 Example 5: Client Reuse for Performance")
    print("-" * 70)

    print("\nCreating a reusable client for multiple evaluations:")

    # Create client once
    reusable_client = await LLMManager.get_client(model="gpt-4o-mini")

    # Use it for multiple evaluations
    evaluations = [
        ("Paris is the capital", "The capital of France is Paris"),
        ("Python is a language", "Python is used for programming"),
        ("The weather is nice", "It's a beautiful day"),
    ]

    print(f"\nEvaluating {len(evaluations)} pairs with same client:")
    for i, (output, reference) in enumerate(evaluations, 1):
        result = await evaluate(
            output=output,
            reference=reference,
            evaluators=["semantic"],
            llm_client=reusable_client,  # Reuse same client
        )
        print(f"  {i}. Score: {result.overall_score:.3f} ({result.total_tokens} tokens)")

    print("\n💡 Tip: Reusing clients avoids connection overhead")
    print("         Better for batch operations")

    # Example 6: Configuration Best Practices
    print("\n\n📊 Example 6: Configuration Best Practices")
    print("-" * 70)

    print("\nRecommended configurations for different use cases:\n")

    print("1. High-Volume Production (Cost-Sensitive):")
    print("   • Model: gpt-4o-mini")
    print("   • Temperature: 0.0-0.2 (consistent)")
    print("   • Provider: OpenAI (reliable)")
    print("   • Reuse clients for batch operations")

    print("\n2. Critical Evaluations (Accuracy-Focused):")
    print("   • Model: gpt-4o or gpt-4")
    print("   • Temperature: 0.0-0.3 (consistent)")
    print("   • Provider: OpenAI or Anthropic")
    print("   • Use retry configs for reliability")

    print("\n3. Development/Testing (Fast Iteration):")
    print("   • Model: gpt-4o-mini")
    print("   • Temperature: 0.7 (default)")
    print("   • Provider: Any available")
    print("   • Quick retries for transient failures")

    print("\n4. Creative Evaluation (Nuanced Assessment):")
    print("   • Model: gpt-4o")
    print("   • Temperature: 0.7-1.0 (more creative)")
    print("   • Provider: OpenAI or Anthropic")
    print("   • Allow for more variation in scores")

    # Summary
    print("\n\n" + "=" * 70)
    print("✅ All Examples Complete!")
    print("=" * 70)

    print("\n🎯 Key Takeaways:")
    print("   • Temperature controls evaluation consistency (0.0-0.3 recommended)")
    print("   • Model selection balances accuracy vs cost")
    print("   • Custom clients enable fine-grained control")
    print("   • Client reuse improves performance")
    print("   • Provider choice affects characteristics and cost")

    print("\n💡 Production Recommendations:")
    print("   • Use temperature=0.0-0.2 for consistent evaluations")
    print("   • Use gpt-4o-mini for high-volume operations")
    print("   • Create reusable client instances")
    print("   • Monitor costs and adjust models accordingly")
    print("   • Test multiple providers to find best fit")

    print("\n📚 Learn More:")
    print("   • See: examples/provider_switching.py for provider details")
    print("   • See: examples/batch_manual.py for batch operations")
    print("   • See: examples/middleware_usage.py for middleware config")


if __name__ == "__main__":
    asyncio.run(main())

