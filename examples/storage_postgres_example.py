"""Example: PostgreSQL storage for persistent evaluation results.

Demonstrates:
- Storing evaluation results in PostgreSQL
- Retrieving results by ID
- Schema isolation (arbiter schema, separate from conduit)
- Connection pool management

Prerequisites:
    1. pip install arbiter[postgres]
    2. Set DATABASE_URL in .env (can share with conduit)
    3. Run migrations: alembic upgrade head
"""

import asyncio

from arbiter import evaluate
from arbiter.storage import PostgresStorage


async def main() -> None:
    """Demonstrate PostgreSQL storage for evaluation results."""
    print("=" * 60)
    print("PostgreSQL Storage Example")
    print("=" * 60)

    # Initialize storage backend
    # Uses DATABASE_URL from environment (shared with conduit)
    # Results stored in 'arbiter' schema (isolated from conduit's 'public' schema)
    storage = PostgresStorage()

    async with storage:
        # 1. Evaluate with storage
        print("\n1. Running evaluation...")
        result = await evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
            evaluators=["semantic"],
            model="gpt-4o-mini",
        )

        print(f"   Score: {result.overall_score:.2f}")
        print(f"   Cost: ${await result.total_llm_cost():.6f}")

        # 2. Save to PostgreSQL
        print("\n2. Saving to PostgreSQL...")
        result_id = await storage.save_result(
            result,
            metadata={
                "environment": "development",
                "user_id": "demo_user",
                "application": "arbiter_examples",
            },
        )
        print(f"   Saved with ID: {result_id}")

        # 3. Retrieve from PostgreSQL
        print("\n3. Retrieving from PostgreSQL...")
        retrieved = await storage.get_result(result_id)

        if retrieved:
            print(f"   Retrieved score: {retrieved.overall_score:.2f}")
            print(f"   Retrieved cost: ${await retrieved.total_llm_cost():.6f}")
            print(f"   Match: {retrieved.overall_score == result.overall_score}")
        else:
            print("   ERROR: Could not retrieve result")

        # 4. Batch evaluation with storage
        print("\n4. Batch evaluation...")
        from arbiter import batch_evaluate

        items = [
            {
                "output": "Tokyo is Japan's capital",
                "reference": "The capital of Japan is Tokyo",
            },
            {
                "output": "Berlin is Germany's capital",
                "reference": "The capital of Germany is Berlin",
            },
        ]

        batch_result = await batch_evaluate(
            items=items, evaluators=["semantic"], model="gpt-4o-mini"
        )

        print(f"   Successful: {batch_result.successful_items}/{batch_result.total_items}")
        print(f"   Total cost: ${await batch_result.total_llm_cost():.6f}")

        # Save batch result
        batch_id = await storage.save_batch_result(
            batch_result, metadata={"batch_type": "capitals"}
        )
        print(f"   Batch saved with ID: {batch_id}")

    print("\n" + "=" * 60)
    print("Storage Example Complete!")
    print("=" * 60)
    print("\nNotes:")
    print("- Results stored in 'arbiter' schema in PostgreSQL")
    print("- Safe to share database with conduit (separate schemas)")
    print("- Query results: SELECT * FROM arbiter.evaluation_results;")


if __name__ == "__main__":
    asyncio.run(main())
