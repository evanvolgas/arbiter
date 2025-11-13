"""Provider switching example.

This example demonstrates Arbiter's provider-agnostic design, showing
how easy it is to switch between different LLM providers.

Run with:
    python examples/provider_switching.py
"""

import asyncio
import os

from arbiter import evaluate
from arbiter.core import LLMManager, Provider


async def main():
    """Run provider switching examples."""

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))

    if not any([has_openai, has_anthropic, has_google]):
        print("⚠️  Please set at least one API key:")
        print("   - OPENAI_API_KEY")
        print("   - ANTHROPIC_API_KEY")
        print("   - GOOGLE_API_KEY")
        return

    print("🔍 Arbiter - Provider Switching Example")
    print("=" * 60)

    output = "Paris is the capital of France, founded in the 3rd century BC."
    reference = "The capital of France is Paris, established around 250 BC."

    # Example 1: Using OpenAI (default)
    if has_openai:
        print("\n📝 Example 1: OpenAI Provider")
        print("-" * 60)

        result1 = await evaluate(
            output=output,
            reference=reference,
            evaluators=["semantic"],
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
        )

        print(f"Provider: OpenAI")
        print(f"Model: {result1.interactions[0].model if result1.interactions else 'N/A'}")
        print(f"Score: {result1.overall_score:.3f}")
        print(f"Tokens Used: {result1.total_tokens}")
        print(f"Processing Time: {result1.processing_time:.3f}s")

    # Example 2: Using Anthropic (if available)
    if has_anthropic:
        print("\n\n📝 Example 2: Anthropic Provider")
        print("-" * 60)

        try:
            result2 = await evaluate(
                output=output,
                reference=reference,
                evaluators=["semantic"],
                provider=Provider.ANTHROPIC,
                model="claude-3-5-sonnet-20241022",
            )

            print(f"Provider: Anthropic")
            print(f"Model: {result2.interactions[0].model if result2.interactions else 'N/A'}")
            print(f"Score: {result2.overall_score:.3f}")
            print(f"Tokens Used: {result2.total_tokens}")
            print(f"Processing Time: {result2.processing_time:.3f}s")
        except Exception as e:
            print(f"⚠️  Anthropic evaluation failed: {e}")
            print("   (This is expected if API key is not configured)")

    # Example 3: Using Google Gemini (if available)
    if has_google:
        print("\n\n📝 Example 3: Google Provider")
        print("-" * 60)

        try:
            result3 = await evaluate(
                output=output,
                reference=reference,
                evaluators=["semantic"],
                provider=Provider.GOOGLE,
                model="gemini-1.5-pro",
            )

            print(f"Provider: Google")
            print(f"Model: {result3.interactions[0].model if result3.interactions else 'N/A'}")
            print(f"Score: {result3.overall_score:.3f}")
            print(f"Tokens Used: {result3.total_tokens}")
            print(f"Processing Time: {result3.processing_time:.3f}s")
        except Exception as e:
            print(f"⚠️  Google evaluation failed: {e}")
            print("   (This is expected if API key is not configured)")

    # Example 4: Automatic provider detection
    print("\n\n📝 Example 4: Automatic Provider Detection")
    print("-" * 60)

    if has_openai:
        print("Using model name to auto-detect provider...")

        # Just specify model name, provider is auto-detected
        result4 = await evaluate(
            output=output,
            reference=reference,
            evaluators=["semantic"],
            model="gpt-4o-mini",  # Provider auto-detected as OPENAI
        )

        print(f"Model: gpt-4o-mini")
        print(f"Auto-detected Provider: OpenAI")
        print(f"Score: {result4.overall_score:.3f}")

    # Example 5: Using LLMManager directly for more control
    print("\n\n📝 Example 5: Direct LLM Client Management")
    print("-" * 60)

    if has_openai:
        # Get client for a specific provider
        client = await LLMManager.get_client(
            provider=Provider.OPENAI,
            model="gpt-4o-mini",
            temperature=0.0,
        )

        print(f"Created LLM client:")
        print(f"  Provider: {client.provider}")
        print(f"  Model: {client.model}")
        print(f"  Temperature: {client.temperature}")

        # Use the client with evaluators
        from arbiter.evaluators import SemanticEvaluator

        evaluator = SemanticEvaluator(client)
        score = await evaluator.evaluate(
            output=output,
            reference=reference,
        )

        print(f"\nEvaluation Result:")
        print(f"  Score: {score.value:.3f}")
        print(f"  Confidence: {score.confidence:.3f}")

    # Example 6: Provider comparison
    print("\n\n📝 Example 6: Comparing Providers")
    print("-" * 60)

    print("""
You can easily compare different providers on the same task:

results = {}
for provider in [Provider.OPENAI, Provider.ANTHROPIC]:
    results[provider] = await evaluate(
        output=output,
        reference=reference,
        provider=provider,
        evaluators=["semantic"]
    )

# Compare results
for provider, result in results.items():
    print(f"{provider}: {result.overall_score:.3f}")

This allows you to:
- Compare quality across providers
- Choose the best provider for your use case
- Switch providers without code changes
- Avoid vendor lock-in
    """)

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Provider-agnostic design - same code works with any provider")
    print("  • Easy provider switching - just change provider parameter")
    print("  • Automatic provider detection - from model name")
    print("  • Direct client management - for advanced use cases")
    print("  • No vendor lock-in - switch providers anytime")
    print("\nSupported Providers:")
    print("  • OpenAI (GPT-4o, GPT-4, GPT-3.5-turbo)")
    print("  • Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)")
    print("  • Google (Gemini 1.5 Pro/Flash)")
    print("  • Groq (Llama 3.1, Mixtral)")
    print("  • Mistral AI")
    print("  • Cohere")


if __name__ == "__main__":
    asyncio.run(main())

