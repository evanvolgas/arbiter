"""Circuit Breaker - Resilient Production Evaluation

This example demonstrates the circuit breaker pattern for preventing cascading
failures during LLM provider outages or degraded performance.

Key Features:
- Automatic failure detection and circuit opening
- Configurable failure thresholds and timeouts
- Graceful degradation during outages
- Provider recovery monitoring
- Production-grade resilience patterns

Requirements:
    export OPENAI_API_KEY=your_key_here

Run with:
    python examples/circuit_breaker_example.py
"""

import asyncio
import os
from dotenv import load_dotenv

from arbiter import evaluate
from arbiter.core.circuit_breaker import CircuitBreaker
from arbiter.core.exceptions import CircuitBreakerOpenError, ModelProviderError
from arbiter.core.llm_client import LLMClient, LLMManager
from arbiter.core.types import Provider


async def main():
    """Demonstrate circuit breaker functionality."""

    # Load environment variables
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Please set OPENAI_API_KEY environment variable")
        return

    print("🔌 Circuit Breaker Example - Resilient LLM Evaluation")
    print("=" * 70)

    # Example 1: Default circuit breaker (automatic)
    print("\n📝 Example 1: Default Circuit Breaker")
    print("-" * 70)
    print("By default, all LLMClients have circuit breaker protection.")
    print("Threshold: 5 failures, Timeout: 60 seconds\n")

    try:
        result = await evaluate(
            output="Paris is the capital of France.",
            reference="Paris is the capital of France.",
            model="gpt-4o-mini",
        )
        print(f"✅ Evaluation completed: {result.overall_score:.3f}")
        print("   Circuit breaker is active and monitoring failures.")
    except Exception as e:
        print(f"❌ Error: {e}")

    # Example 2: Custom circuit breaker configuration
    print("\n\n📝 Example 2: Custom Circuit Breaker Configuration")
    print("-" * 70)
    print("Configure circuit breaker with custom thresholds.")

    # Create custom circuit breaker
    custom_breaker = CircuitBreaker(
        failure_threshold=3,  # Open after 3 failures
        timeout=30.0,  # Wait 30 seconds before testing recovery
        half_open_max_calls=2,  # Allow 2 test calls in half-open state
    )

    # Create client with custom circuit breaker
    client = LLMClient(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        circuit_breaker=custom_breaker,
    )

    print(f"Circuit breaker configured:")
    print(f"  - Failure threshold: 3")
    print(f"  - Timeout: 30 seconds")
    print(f"  - Half-open test calls: 2")

    # Get initial stats
    stats = custom_breaker.get_stats()
    print(f"\nInitial state: {stats['state']}")
    print(f"Failures: {stats['failure_count']}")

    # Example 3: Simulating failures and recovery
    print("\n\n📝 Example 3: Failure Detection and Recovery")
    print("-" * 70)
    print("Demonstrating how circuit breaker handles failures.\n")

    # Create a test circuit breaker with low threshold
    test_breaker = CircuitBreaker(
        failure_threshold=2,
        timeout=2.0,  # Short timeout for demo
    )

    async def simulated_llm_call(should_fail: bool = False):
        """Simulate an LLM API call that might fail."""
        if should_fail:
            raise ModelProviderError("Simulated API failure")
        return "Success"

    # Simulate failures
    print("Simulating 2 failures to open the circuit...")
    for i in range(2):
        try:
            await test_breaker.call(simulated_llm_call, should_fail=True)
        except ModelProviderError:
            print(f"  Failure {i+1}: API call failed")

    stats = test_breaker.get_stats()
    print(f"\n✋ Circuit opened after {stats['failure_count']} failures")
    print(f"   State: {stats['state']}")

    # Try to make a call while circuit is open
    print("\nAttempting call while circuit is open...")
    try:
        await test_breaker.call(simulated_llm_call)
    except CircuitBreakerOpenError as e:
        print(f"  ⛔ Blocked: {e}")

    # Wait for timeout
    print(f"\n⏳ Waiting {test_breaker.timeout} seconds for circuit to enter half-open state...")
    await asyncio.sleep(test_breaker.timeout + 0.1)

    # Test recovery
    print("Testing recovery in half-open state...")
    try:
        result = await test_breaker.call(simulated_llm_call, should_fail=False)
        print(f"  ✅ Success! Circuit closed. Result: {result}")
    except Exception as e:
        print(f"  ❌ Recovery failed: {e}")

    stats = test_breaker.get_stats()
    print(f"\n🔄 Circuit recovered")
    print(f"   State: {stats['state']}")
    print(f"   Success count: {stats['success_count']}")

    # Example 4: Circuit breaker statistics
    print("\n\n📝 Example 4: Monitoring Circuit Breaker Statistics")
    print("-" * 70)

    monitoring_breaker = CircuitBreaker()

    # Make some successful calls
    for i in range(5):
        await monitoring_breaker.call(simulated_llm_call, should_fail=False)

    # Check stats
    stats = monitoring_breaker.get_stats()
    print("\nCircuit Breaker Statistics:")
    print(f"  State: {stats['state']}")
    print(f"  Total successes: {stats['success_count']}")
    print(f"  Total failures: {stats['failure_count']}")
    print(f"  Last failure: {stats['last_failure_time'] or 'None'}")

    # Example 5: Manual reset
    print("\n\n📝 Example 5: Manual Circuit Reset")
    print("-" * 70)

    reset_breaker = CircuitBreaker(failure_threshold=1)

    # Open the circuit
    try:
        await reset_breaker.call(simulated_llm_call, should_fail=True)
    except ModelProviderError:
        pass

    print(f"Circuit state before reset: {reset_breaker.state}")

    # Manual reset
    reset_breaker.reset()
    print(f"Circuit state after reset: {reset_breaker.state}")
    print("All counters have been reset to initial state.")

    # Example 6: Production usage pattern
    print("\n\n📝 Example 6: Production Usage Pattern")
    print("-" * 70)
    print("Recommended pattern for production deployments.\n")

    # Create circuit breaker with production settings
    production_breaker = CircuitBreaker(
        failure_threshold=5,  # Tolerate 5 failures
        timeout=60.0,  # Wait 60 seconds before retry
        half_open_max_calls=1,  # Test with 1 call
    )

    # Create client pool with shared circuit breaker
    production_client = LLMClient(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        circuit_breaker=production_breaker,
    )

    print("Production configuration:")
    print("  ✅ Shared circuit breaker across all clients")
    print("  ✅ Conservative failure threshold (5 failures)")
    print("  ✅ Reasonable timeout (60 seconds)")
    print("  ✅ Single test call in half-open state")
    print("\nThis configuration provides:")
    print("  - Protection against cascading failures")
    print("  - Automatic recovery testing")
    print("  - Minimal impact on normal operations")

    # Example 7: Disabling circuit breaker
    print("\n\n📝 Example 7: Disabling Circuit Breaker (Not Recommended)")
    print("-" * 70)

    # To disable circuit breaker entirely (not recommended for production)
    # Pass an empty CircuitBreaker that never opens:
    always_open_breaker = CircuitBreaker(
        failure_threshold=999999,  # Effectively never opens
    )

    disabled_client = LLMClient(
        provider=Provider.OPENAI,
        model="gpt-4o-mini",
        circuit_breaker=always_open_breaker,
    )

    print("⚠️  Circuit breaker effectively disabled (very high threshold)")
    print("   This removes protection against cascading failures.")
    print("   Only use for testing or special circumstances.")

    # Summary
    print("\n\n" + "=" * 70)
    print("✨ Circuit Breaker Summary")
    print("=" * 70)
    print("\n🎯 Key Benefits:")
    print("  1. Prevents cascading failures during provider outages")
    print("  2. Automatic recovery testing after timeout")
    print("  3. Protects your application from degraded performance")
    print("  4. Provides visibility into failure patterns")
    print("\n⚙️  Recommended Settings:")
    print("  - Failure threshold: 5 (balance between sensitivity and stability)")
    print("  - Timeout: 60 seconds (reasonable recovery time)")
    print("  - Half-open calls: 1 (minimal testing)")
    print("\n📊 Monitoring:")
    print("  - Use get_stats() to monitor circuit state")
    print("  - Log CircuitBreakerOpenError exceptions")
    print("  - Alert on frequent circuit openings")
    print("\n🔧 Integration:")
    print("  - Circuit breaker is enabled by default")
    print("  - Customize via LLMClient constructor")
    print("  - Share circuit breaker across client pool")
    print("\n✅ Production Ready!")


if __name__ == "__main__":
    asyncio.run(main())
