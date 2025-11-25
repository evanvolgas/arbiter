"""Production Deployment Example - Enterprise-Ready Evaluation Setup

This example demonstrates how to configure Arbiter for production deployments
with PostgreSQL persistence, Redis caching, circuit breakers, and proper
error handling.

Key Features:
- PostgreSQL for persistent result storage
- Redis for fast caching layer
- Circuit breaker for fault tolerance
- Retry logic with exponential backoff
- Comprehensive error handling
- Health checks before processing
- Cost budgeting and monitoring

Requirements:
    pip install arbiter[postgres,redis]

    # Environment variables:
    export OPENAI_API_KEY=your_key_here
    export DATABASE_URL=postgresql://user:pass@localhost:5432/arbiter
    export REDIS_URL=redis://localhost:6379/0

Run with:
    python examples/production_deployment_example.py
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from dotenv import load_dotenv

from arbiter_ai import batch_evaluate, evaluate
from arbiter_ai.core import LLMManager
from arbiter_ai.core.circuit_breaker import CircuitBreaker
from arbiter_ai.core.exceptions import (
    ArbiterError,
    CircuitBreakerOpenError,
    ModelProviderError,
)
from arbiter_ai.core.middleware import (
    CachingMiddleware,
    LoggingMiddleware,
    MetricsMiddleware,
    MiddlewarePipeline,
    RateLimitingMiddleware,
)
from arbiter_ai.core.retry import RetryConfig, with_retry

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production configuration settings."""

    # Model settings
    model: str = "gpt-4o-mini"
    fallback_model: str = "gpt-4o-mini"

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 30.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    # Rate limiting
    max_requests_per_minute: int = 60

    # Caching
    cache_size: int = 1000

    # Cost budget (optional)
    max_cost_per_request: Optional[float] = 0.10
    max_daily_cost: Optional[float] = 100.0


class ProductionEvaluator:
    """Production-ready evaluation wrapper with resilience patterns."""

    def __init__(self, config: Optional[ProductionConfig] = None):
        self.config = config or ProductionConfig()
        self._setup_components()
        self._daily_cost = 0.0

    def _setup_components(self) -> None:
        """Initialize production components."""
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.failure_threshold,
            timeout=self.config.recovery_timeout,
        )

        # Retry configuration
        self.retry_config = RetryConfig(
            max_attempts=self.config.max_retries,
            delay=self.config.retry_delay,
            backoff=2.0,
        )

        # Middleware pipeline
        self.metrics = MetricsMiddleware()
        self.cache = CachingMiddleware(max_size=self.config.cache_size)

        self.pipeline = MiddlewarePipeline(
            [
                LoggingMiddleware(log_level="INFO"),
                self.metrics,
                self.cache,
                RateLimitingMiddleware(
                    max_requests_per_minute=self.config.max_requests_per_minute
                ),
            ]
        )

        # Storage backends (initialized lazily)
        self._postgres = None
        self._redis = None

    @asynccontextmanager
    async def storage_context(self) -> AsyncIterator["ProductionEvaluator"]:
        """Context manager for storage backend lifecycle."""
        try:
            # Initialize storage if configured
            if os.getenv("DATABASE_URL"):
                try:
                    from arbiter_ai.storage import PostgresStorage

                    self._postgres = PostgresStorage()
                    await self._postgres.connect()
                    logger.info("PostgreSQL storage connected")
                except ImportError:
                    logger.warning(
                        "PostgreSQL storage not available. "
                        "Install with: pip install arbiter[postgres]"
                    )

            if os.getenv("REDIS_URL"):
                try:
                    from arbiter_ai.storage import RedisStorage

                    self._redis = RedisStorage(ttl=3600)  # 1 hour TTL
                    await self._redis.connect()
                    logger.info("Redis storage connected")
                except ImportError:
                    logger.warning(
                        "Redis storage not available. "
                        "Install with: pip install arbiter[redis]"
                    )

            yield self

        finally:
            # Cleanup storage connections
            if self._postgres:
                await self._postgres.disconnect()
                logger.info("PostgreSQL storage disconnected")

            if self._redis:
                await self._redis.disconnect()
                logger.info("Redis storage disconnected")

    async def health_check(self) -> bool:
        """Verify system health before processing."""
        try:
            # Check circuit breaker state
            if self.circuit_breaker.is_open:
                logger.warning("Circuit breaker is OPEN")
                return False

            # Quick provider check
            result = await evaluate(
                output="health",
                reference="health",
                evaluators=["semantic"],
                model=self.config.model,
            )

            return result.overall_score >= 0

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def evaluate_with_resilience(
        self,
        output: str,
        reference: Optional[str] = None,
        criteria: Optional[str] = None,
        evaluators: Optional[list[str]] = None,
    ) -> dict:
        """Run evaluation with production resilience patterns.

        Args:
            output: The LLM output to evaluate
            reference: Reference text for comparison
            criteria: Custom evaluation criteria
            evaluators: List of evaluators to use

        Returns:
            Dictionary with evaluation results and metadata
        """
        evaluators = evaluators or ["semantic"]

        # Check circuit breaker
        if self.circuit_breaker.is_open:
            raise CircuitBreakerOpenError(
                "Circuit breaker is open. System is protecting against failures."
            )

        try:
            # Run evaluation with retry
            @with_retry(self.retry_config)
            async def run_eval() -> dict:
                result = await evaluate(
                    output=output,
                    reference=reference,
                    criteria=criteria,
                    evaluators=evaluators,
                    model=self.config.model,
                    middleware=self.pipeline,
                )

                cost = await result.total_llm_cost()

                # Check cost budget
                if (
                    self.config.max_cost_per_request
                    and cost > self.config.max_cost_per_request
                ):
                    logger.warning(
                        f"Request cost ${cost:.4f} exceeds budget "
                        f"${self.config.max_cost_per_request:.4f}"
                    )

                # Track daily cost
                self._daily_cost += cost
                if (
                    self.config.max_daily_cost
                    and self._daily_cost > self.config.max_daily_cost
                ):
                    logger.error(
                        f"Daily cost budget exceeded: ${self._daily_cost:.2f}"
                    )

                # Persist result if storage available
                result_id = None
                if self._postgres:
                    result_id = await self._postgres.save_result(result)

                # Cache in Redis
                if self._redis:
                    await self._redis.save_result(result)

                return {
                    "score": result.overall_score,
                    "passed": result.passed,
                    "cost": cost,
                    "tokens": result.total_tokens,
                    "latency": result.processing_time,
                    "result_id": result_id,
                    "interactions": len(result.interactions),
                    "scores": [
                        {"name": s.name, "value": s.value} for s in result.scores
                    ],
                }

            result = await run_eval()

            # Record success (internal method)
            self.circuit_breaker._on_success()

            return result

        except ModelProviderError as e:
            # Record failure for circuit breaker (internal method)
            self.circuit_breaker._on_failure()
            logger.error(f"Provider error: {e}")
            raise

        except ArbiterError as e:
            logger.error(f"Evaluation error: {e}")
            raise

    def get_metrics(self) -> dict:
        """Get current metrics."""
        metrics = self.metrics.get_metrics()
        cache_stats = self.cache.get_stats()

        return {
            "evaluations": metrics,
            "cache": cache_stats,
            "circuit_breaker": {
                "state": (
                    "OPEN"
                    if self.circuit_breaker.is_open
                    else "HALF_OPEN"
                    if self.circuit_breaker.is_half_open
                    else "CLOSED"
                ),
                "failure_count": self.circuit_breaker.failure_count,
            },
            "daily_cost": self._daily_cost,
        }


async def main() -> None:
    """Demonstrate production deployment patterns."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY environment variable")
        return

    print("🏭 Arbiter Production Deployment Example")
    print("=" * 60)

    # Create production evaluator
    config = ProductionConfig(
        model="gpt-4o-mini",
        max_requests_per_minute=30,
        max_cost_per_request=0.01,
        max_daily_cost=10.0,
    )

    evaluator = ProductionEvaluator(config)

    # Use storage context manager for proper lifecycle
    async with evaluator.storage_context():
        # Health check
        print("\n1. Running health check...")
        healthy = await evaluator.health_check()
        print(f"   System healthy: {'✅' if healthy else '❌'}")

        if not healthy:
            print("   System not healthy, exiting")
            return

        # Single evaluation with resilience
        print("\n2. Running single evaluation...")
        try:
            result = await evaluator.evaluate_with_resilience(
                output="Paris is the capital of France",
                reference="The capital of France is Paris",
                evaluators=["semantic"],
            )
            print(f"   Score: {result['score']:.2f}")
            print(f"   Cost: ${result['cost']:.6f}")
            print(f"   Latency: {result['latency']:.2f}s")
            if result["result_id"]:
                print(f"   Persisted ID: {result['result_id']}")

        except CircuitBreakerOpenError:
            print("   Circuit breaker open - system protecting against failures")

        # Batch evaluation
        print("\n3. Running batch evaluation...")
        items = [
            {
                "output": "Python is a programming language",
                "reference": "Python is a high-level programming language",
            },
            {
                "output": "Machine learning uses algorithms",
                "reference": "ML algorithms learn from data",
            },
            {
                "output": "The sky is blue",
                "reference": "The atmosphere scatters blue light",
            },
        ]

        batch_result = await batch_evaluate(
            items=items,
            evaluators=["semantic"],
            model="gpt-4o-mini",
            max_concurrency=3,
        )

        print(f"   Success: {batch_result.successful_items}/{batch_result.total_items}")
        total_cost = await batch_result.total_llm_cost()
        print(f"   Total cost: ${total_cost:.6f}")

        # Show metrics
        print("\n4. Production metrics:")
        metrics = evaluator.get_metrics()
        print(f"   Total requests: {metrics['evaluations']['total_requests']}")
        print(f"   Average score: {metrics['evaluations']['average_score']:.2f}")
        print(f"   Cache hit rate: {metrics['cache']['hit_rate']:.1%}")
        print(f"   Circuit breaker: {metrics['circuit_breaker']['state']}")
        print(f"   Daily cost: ${metrics['daily_cost']:.4f}")

    print("\n" + "=" * 60)
    print("✅ Production deployment example complete")
    print("\nKey patterns demonstrated:")
    print("  - Circuit breaker for fault tolerance")
    print("  - Retry with exponential backoff")
    print("  - Rate limiting protection")
    print("  - Result caching")
    print("  - Cost budget tracking")
    print("  - Storage backend integration")
    print("  - Comprehensive metrics collection")


if __name__ == "__main__":
    asyncio.run(main())
