"""Health Check Example - Verify Provider Connectivity

This example demonstrates how to verify that your LLM providers are properly
configured and accessible before running evaluations.

Key Features:
- Provider connectivity verification
- API key validation
- Quick latency check
- Multi-provider health status

Use this pattern before batch evaluations or in service startup routines.

Requirements:
    export OPENAI_API_KEY=your_key_here
    # Optional: ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.

Run with:
    python examples/health_check_example.py
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from arbiter_ai import evaluate
from arbiter_ai.core import LLMManager
from arbiter_ai.core.exceptions import ArbiterError, ModelProviderError


@dataclass
class HealthStatus:
    """Health check result for a provider."""

    provider: str
    model: str
    healthy: bool
    latency_ms: Optional[float] = None
    error: Optional[str] = None


async def check_provider_health(
    model: str, timeout_seconds: float = 10.0
) -> HealthStatus:
    """Check if a provider is healthy by running a minimal evaluation.

    Args:
        model: Model identifier (e.g., "gpt-4o-mini", "claude-3-haiku")
        timeout_seconds: Maximum time to wait for response

    Returns:
        HealthStatus with connectivity and latency information
    """
    provider = model.split("-")[0] if "-" in model else "openai"

    start = time.time()
    try:
        # Run minimal evaluation to verify connectivity
        result = await asyncio.wait_for(
            evaluate(
                output="test",
                reference="test",
                evaluators=["semantic"],
                model=model,
            ),
            timeout=timeout_seconds,
        )

        latency_ms = (time.time() - start) * 1000

        # Verify we got a valid response
        if result.overall_score >= 0:
            return HealthStatus(
                provider=provider,
                model=model,
                healthy=True,
                latency_ms=latency_ms,
            )
        else:
            return HealthStatus(
                provider=provider,
                model=model,
                healthy=False,
                error="Invalid response score",
            )

    except asyncio.TimeoutError:
        return HealthStatus(
            provider=provider,
            model=model,
            healthy=False,
            error=f"Timeout after {timeout_seconds}s",
        )
    except ModelProviderError as e:
        return HealthStatus(
            provider=provider,
            model=model,
            healthy=False,
            error=f"Provider error: {e}",
        )
    except ArbiterError as e:
        return HealthStatus(
            provider=provider,
            model=model,
            healthy=False,
            error=f"Arbiter error: {e}",
        )
    except Exception as e:
        return HealthStatus(
            provider=provider,
            model=model,
            healthy=False,
            error=f"Unexpected error: {type(e).__name__}: {e}",
        )


async def check_all_providers() -> dict[str, HealthStatus]:
    """Check health of all configured providers.

    Returns:
        Dictionary mapping provider names to their health status
    """
    # Define models to check based on available API keys
    models_to_check = []

    if os.getenv("OPENAI_API_KEY"):
        models_to_check.append("gpt-4o-mini")

    if os.getenv("ANTHROPIC_API_KEY"):
        models_to_check.append("claude-3-haiku-20240307")

    if os.getenv("GOOGLE_API_KEY"):
        models_to_check.append("gemini-1.5-flash")

    if os.getenv("GROQ_API_KEY"):
        models_to_check.append("llama-3.1-8b-instant")

    if not models_to_check:
        print("No API keys configured. Set at least OPENAI_API_KEY.")
        return {}

    # Run health checks in parallel
    tasks = [check_provider_health(model) for model in models_to_check]
    results = await asyncio.gather(*tasks)

    return {status.model: status for status in results}


def print_health_report(statuses: dict[str, HealthStatus]) -> bool:
    """Print a formatted health report.

    Args:
        statuses: Dictionary of health statuses

    Returns:
        True if all providers are healthy, False otherwise
    """
    print("\n" + "=" * 60)
    print("ARBITER HEALTH CHECK REPORT")
    print("=" * 60)

    all_healthy = True

    for model, status in statuses.items():
        icon = "✅" if status.healthy else "❌"
        print(f"\n{icon} {status.provider.upper()} ({model})")

        if status.healthy:
            print(f"   Latency: {status.latency_ms:.0f}ms")
        else:
            print(f"   Error: {status.error}")
            all_healthy = False

    print("\n" + "-" * 60)

    if all_healthy:
        print("✅ All providers healthy - ready for evaluations")
    else:
        print("⚠️  Some providers unhealthy - check configuration")

    print("=" * 60 + "\n")

    return all_healthy


async def main() -> None:
    """Run health checks and display report."""
    load_dotenv()

    print("🏥 Arbiter Health Check")
    print("Verifying provider connectivity...")

    statuses = await check_all_providers()

    if not statuses:
        print("\n❌ No providers configured")
        print("Set API keys in your .env file:")
        print("  OPENAI_API_KEY=sk-...")
        print("  ANTHROPIC_API_KEY=sk-ant-...")
        return

    all_healthy = print_health_report(statuses)

    # Example: Use health check before batch operations
    if all_healthy:
        print("Example: Running evaluation with verified provider...")

        # Pick the fastest healthy provider
        fastest = min(
            (s for s in statuses.values() if s.healthy),
            key=lambda s: s.latency_ms or float("inf"),
        )

        result = await evaluate(
            output="The capital of France is Paris.",
            reference="Paris is the capital of France.",
            evaluators=["semantic"],
            model=fastest.model,
        )

        cost = await result.total_llm_cost()
        print(f"\n✅ Evaluation complete:")
        print(f"   Model: {fastest.model}")
        print(f"   Score: {result.overall_score:.2f}")
        print(f"   Cost: ${cost:.6f}")


if __name__ == "__main__":
    asyncio.run(main())
