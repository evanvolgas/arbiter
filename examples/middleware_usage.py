"""Middleware - Production Cross-Cutting Concerns

This example demonstrates using middleware for logging, metrics, caching,
and other production-grade cross-cutting concerns.

Key Features:
- Custom middleware for logging and metrics
- Pre/post-processing hooks
- Request/response transformation
- Production monitoring integration
- Composable middleware chains

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/middleware_usage.py
"""

from dotenv import load_dotenv

import asyncio
import logging
import os

from arbiter import evaluate
from arbiter.core import LLMManager, MiddlewarePipeline
from arbiter.core.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
)

# Configure logging to see middleware output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main():
    """Run middleware usage examples."""

    # Load environment variables from .env file
    load_dotenv()

    # Ensure API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔍 Arbiter - Middleware Usage Example")
    print("=" * 60)

    # Example 1: Basic logging middleware
    print("\n📝 Example 1: Logging Middleware")
    print("-" * 60)

    logging_pipeline = MiddlewarePipeline([LoggingMiddleware(log_level="INFO")])

    result1 = await evaluate(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        evaluators=["semantic"],
        middleware=logging_pipeline,
        model="gpt-4o-mini",
    )

    print(f"Output: {result1.output}")
    print(f"Score: {result1.overall_score:.3f}")
    print("\n💡 Check the logs above for middleware output!")

    # Example 2: Metrics middleware
    print("\n\n📝 Example 2: Metrics Middleware")
    print("-" * 60)

    metrics_pipeline = MiddlewarePipeline([MetricsMiddleware()])

    # Run multiple evaluations to accumulate metrics
    for i in range(3):
        await evaluate(
            output=f"Test output {i+1}",
            reference=f"Test reference {i+1}",
            evaluators=["semantic"],
            middleware=metrics_pipeline,
            model="gpt-4o-mini",
        )

    # Get metrics
    metrics_mw = metrics_pipeline.get_middleware(MetricsMiddleware)
    if metrics_mw:
        metrics = metrics_mw.get_metrics()
        print("\n📊 Accumulated Metrics:")
        print(f"  Total Requests: {metrics.get('total_requests', 0)}")
        print(f"  Successful Requests: {metrics.get('successful_requests', 0)}")
        print(f"  Failed Requests: {metrics.get('failed_requests', 0)}")
        if metrics.get("avg_processing_time"):
            print(f"  Avg Processing Time: {metrics['avg_processing_time']:.3f}s")
        if metrics.get("total_tokens"):
            print(f"  Total Tokens: {metrics['total_tokens']}")

    # Example 3: Caching middleware
    print("\n\n📝 Example 3: Caching Middleware")
    print("-" * 60)

    caching_pipeline = MiddlewarePipeline([CachingMiddleware(max_size=10)])

    output = "The quick brown fox jumps over the lazy dog"
    reference = "A fast brown fox leaps above a sleepy canine"

    print("First evaluation (will call LLM):")
    import time

    start1 = time.time()
    result2a = await evaluate(
        output=output,
        reference=reference,
        evaluators=["semantic"],
        middleware=caching_pipeline,
        model="gpt-4o-mini",
    )
    time1 = time.time() - start1
    print(f"  Score: {result2a.overall_score:.3f}")
    print(f"  Time: {time1:.3f}s")

    print("\nSecond evaluation (same inputs, should use cache):")
    start2 = time.time()
    result2b = await evaluate(
        output=output,
        reference=reference,
        evaluators=["semantic"],
        middleware=caching_pipeline,
        model="gpt-4o-mini",
    )
    time2 = time.time() - start2
    print(f"  Score: {result2b.overall_score:.3f}")
    print(f"  Time: {time2:.3f}s")

    if time2 < time1 * 0.5:  # Cache should be much faster
        print(f"\n  ✅ Cache hit! {time1/time2:.1f}x faster")
    else:
        print(f"\n  ⚠️ Cache may not have worked (times: {time1:.3f}s vs {time2:.3f}s)")

    # Example 4: Combined middleware pipeline
    print("\n\n📝 Example 4: Combined Middleware Pipeline")
    print("-" * 60)

    # Create a production-ready pipeline
    production_pipeline = MiddlewarePipeline(
        [
            LoggingMiddleware(log_level="INFO"),
            MetricsMiddleware(),
            CachingMiddleware(max_size=100),
        ]
    )

    print("Running evaluation with full middleware pipeline...")
    result3 = await evaluate(
        output="Our product offers advanced features and excellent performance.",
        criteria="Clarity, completeness, professionalism",
        evaluators=["custom_criteria"],
        middleware=production_pipeline,
        model="gpt-4o-mini",
    )

    print(f"\n📊 Results:")
    print(f"  Score: {result3.overall_score:.3f}")
    print(f"  Passed: {'✅' if result3.passed else '❌'}")

    # Get metrics from pipeline
    metrics_mw = production_pipeline.get_middleware(MetricsMiddleware)
    if metrics_mw:
        metrics = metrics_mw.get_metrics()
        print(f"\n  Metrics:")
        print(f"    Total Requests: {metrics.get('total_requests', 0)}")
        print(f"    Cache Hits: {metrics.get('cache_hits', 0)}")
        print(f"    Cache Misses: {metrics.get('cache_misses', 0)}")

    # Example 5: Custom middleware
    print("\n\n📝 Example 5: Custom Middleware")
    print("-" * 60)

    from typing import Callable, Dict, Optional

    from arbiter.core.middleware import Middleware
    from arbiter.core.models import EvaluationResult

    class TimingMiddleware(Middleware):
        """Custom middleware to track evaluation timing."""

        async def process(
            self,
            output: str,
            reference: Optional[str],
            next_handler: Callable,
            context: Dict,
        ) -> EvaluationResult:
            import time

            start = time.time()
            result = await next_handler(output, reference)
            elapsed = time.time() - start

            # Store timing in context
            context["evaluation_time"] = elapsed
            print(f"  ⏱️  Evaluation took {elapsed:.3f}s")

            return result

    custom_pipeline = MiddlewarePipeline([TimingMiddleware(), LoggingMiddleware()])

    result4 = await evaluate(
        output="Test output for custom middleware",
        reference="Test reference",
        evaluators=["semantic"],
        middleware=custom_pipeline,
        model="gpt-4o-mini",
    )

    print(f"Score: {result4.overall_score:.3f}")

    # Summary
    print("\n\n" + "=" * 60)
    print("✅ Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  • LoggingMiddleware - Log all evaluation operations")
    print("  • MetricsMiddleware - Collect performance metrics")
    print("  • CachingMiddleware - Cache results for faster repeated evaluations")
    print("  • Combined pipelines - Use multiple middleware together")
    print("  • Custom middleware - Build your own middleware components")
    print("\nProduction Benefits:")
    print("  • Complete observability with logging")
    print("  • Performance monitoring with metrics")
    print("  • Cost reduction with caching")
    print("  • Flexible extension with custom middleware")


if __name__ == "__main__":
    asyncio.run(main())

