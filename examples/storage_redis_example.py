"""Example: Redis storage for fast evaluation result caching.

Demonstrates:
- Caching evaluation results in Redis
- TTL (Time To Live) for automatic expiration
- Fast retrieval from cache
- Key prefixing (arbiter:) for isolation

Prerequisites:
    1. pip install arbiter[redis]
    2. Set REDIS_URL in .env (can share with conduit)
"""

import asyncio

from arbiter import evaluate
from arbiter.storage import RedisStorage


async def main() -> None:
    """Demonstrate Redis caching for evaluation results."""
    print("=" * 60)
    print("Redis Storage Example")
    print("=" * 60)

    # Initialize Redis storage with 1-hour TTL
    # Uses REDIS_URL from environment (shared with conduit)
    # Keys prefixed with 'arbiter:' (isolated from conduit keys)
    storage = RedisStorage(ttl=3600)  # 1 hour cache

    async with storage:
        # 1. First evaluation (not cached)
        print("\n1. First evaluation (cache miss)...")
        result = await evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        print(f"   Score: {result.overall_score:.2f}")
        print(f"   Cost: ${await result.total_llm_cost():.6f}")

        # 2. Save to Redis cache
        print("\n2. Caching result in Redis...")
        result_id = await storage.save_result(result)
        print(f"   Cached with ID: {result_id}")
        print(f"   TTL: 1 hour")
        print(f"   Redis key: arbiter:result:{result_id}")

        # 3. Retrieve from cache (fast!)
        print("\n3. Retrieving from cache...")
        cached = await storage.get_result(result_id)

        if cached:
            print(f"   ✓ Cache hit!")
            print(f"   Score: {cached.overall_score:.2f}")
            print(f"   Match: {cached.overall_score == result.overall_score}")
        else:
            print("   ✗ Cache miss (expired or not found)")

        # 4. Demonstrate cache expiration
        print("\n4. Cache with short TTL...")
        short_ttl_storage = RedisStorage(ttl=2)  # 2 seconds

        async with short_ttl_storage:
            temp_id = await short_ttl_storage.save_result(result)
            print(f"   Cached with 2-second TTL")

            # Immediately retrieve (should work)
            immediate = await short_ttl_storage.get_result(temp_id)
            print(f"   Immediate retrieval: {'✓ Hit' if immediate else '✗ Miss'}")

            # Wait 3 seconds
            print(f"   Waiting 3 seconds...")
            await asyncio.sleep(3)

            # Try to retrieve (should be expired)
            expired = await short_ttl_storage.get_result(temp_id)
            print(
                f"   After expiration: {'✓ Hit (unexpected)' if expired else '✗ Miss (expected)'}"
            )

        # 5. Batch result caching
        print("\n5. Batch result caching...")
        from arbiter import batch_evaluate

        items = [
            {"output": "Tokyo is Japan's capital", "reference": "Capital of Japan is Tokyo"},
            {"output": "Berlin is Germany's capital", "reference": "Capital of Germany is Berlin"},
        ]

        batch_result = await batch_evaluate(
            items=items, evaluators=["semantic"], model="gpt-4o-mini"
        )

        batch_id = await storage.save_batch_result(batch_result)
        print(f"   Batch cached with ID: {batch_id}")

        # Retrieve batch
        cached_batch = await storage.get_batch_result(batch_id)
        if cached_batch:
            print(f"   ✓ Batch cache hit")
            print(f"   Items: {cached_batch.total_items}")
            print(f"   Success rate: {cached_batch.successful_items}/{cached_batch.total_items}")

    print("\n" + "=" * 60)
    print("Redis Caching Example Complete!")
    print("=" * 60)
    print("\nNotes:")
    print("- Results cached with TTL (auto-expiration)")
    print("- Keys prefixed with 'arbiter:' for isolation")
    print("- Safe to share Redis with conduit")
    print("- Ideal for frequently accessed results")


if __name__ == "__main__":
    asyncio.run(main())
