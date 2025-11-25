"""Comprehensive tests for PostgreSQL storage backend."""

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter_ai.core.models import (
    BatchEvaluationResult,
    EvaluationResult,
    LLMInteraction,
    Score,
)
from arbiter_ai.storage.base import ConnectionError, RetrievalError, SaveError

# Skip all tests if asyncpg is not installed
pytest.importorskip("asyncpg", reason="PostgreSQL storage requires asyncpg")

from arbiter_ai.storage.postgres import PostgresStorage


@pytest.fixture
def mock_eval_result():
    """Create a mock EvaluationResult for testing."""
    return EvaluationResult(
        output="Paris is the capital of France",
        reference="The capital of France is Paris",
        overall_score=0.92,
        passed=True,
        scores=[
            Score(
                name="semantic",
                value=0.92,
                confidence=0.95,
                explanation="High similarity",
            )
        ],
        interactions=[
            LLMInteraction(
                model="gpt-4o-mini",
                prompt="Test prompt",
                response="Test response",
                input_tokens=10,
                output_tokens=20,
                cached_tokens=0,
                tokens_used=30,
                cost=0.0015,
                latency=1.5,
                purpose="scoring",
            )
        ],
        total_input_tokens=10,
        total_output_tokens=20,
        total_cached_tokens=0,
        total_tokens_used=30,
        total_cost=0.0015,
        processing_time=1.5,
    )


@pytest.fixture
def mock_batch_result(mock_eval_result):
    """Create a mock BatchEvaluationResult for testing."""
    return BatchEvaluationResult(
        results=[mock_eval_result, mock_eval_result],
        total_items=2,
        successful_items=2,
        failed_items=0,
        errors=[],
        processing_time=3.0,
    )


def create_mock_pool(schema_exists=True):
    """Helper to create a properly mocked asyncpg pool."""
    pool = MagicMock()
    conn = MagicMock()

    # Set default return values
    conn.fetchval = AsyncMock(return_value=schema_exists)
    conn.fetchrow = AsyncMock(return_value=None)

    # Create async context manager for pool.acquire()
    class MockAcquire:
        def __init__(self, connection):
            self.connection = connection

        async def __aenter__(self):
            return self.connection

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    pool.acquire = MagicMock(return_value=MockAcquire(conn))
    pool.close = AsyncMock()
    pool._conn = conn  # Store for easy access in tests

    return pool


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg connection pool."""
    return create_mock_pool(schema_exists=True)


class TestPostgresStorageInit:
    """Test PostgresStorage initialization."""

    def test_init_with_url(self):
        """Test initialization with explicit database URL."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        assert storage.database_url == "postgresql://localhost/test"
        assert storage.schema == "arbiter"
        assert storage.min_pool_size == 2
        assert storage.max_pool_size == 10
        assert storage.pool is None

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with DATABASE_URL environment variable."""
        monkeypatch.setenv("DATABASE_URL", "postgresql://env/test")
        storage = PostgresStorage()
        assert storage.database_url == "postgresql://env/test"

    def test_init_without_url_raises_error(self, monkeypatch):
        """Test initialization without database URL raises ValueError."""
        monkeypatch.delenv("DATABASE_URL", raising=False)
        with pytest.raises(ValueError, match="DATABASE_URL must be provided"):
            PostgresStorage()

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        storage = PostgresStorage(
            database_url="postgresql://localhost/test",
            schema="custom_schema",
            min_pool_size=5,
            max_pool_size=20,
        )
        assert storage.schema == "custom_schema"
        assert storage.min_pool_size == 5
        assert storage.max_pool_size == 20


class TestPostgresStorageConnect:
    """Test PostgresStorage connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_pool):
        """Test successful connection establishes pool and verifies schema."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        async def mock_create_pool(*args, **kwargs):
            return mock_pool

        with patch(
            "arbiter.storage.postgres.asyncpg.create_pool", side_effect=mock_create_pool
        ):
            await storage.connect()

            assert storage.pool == mock_pool

    @pytest.mark.asyncio
    async def test_connect_schema_missing_raises_error(self):
        """Test connection fails if schema doesn't exist."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        # Create pool with schema not existing
        pool = create_mock_pool(schema_exists=False)

        async def mock_create_pool(*args, **kwargs):
            return pool

        with patch(
            "arbiter.storage.postgres.asyncpg.create_pool", side_effect=mock_create_pool
        ):
            with pytest.raises(
                ConnectionError, match="Schema 'arbiter' does not exist"
            ):
                await storage.connect()

    @pytest.mark.asyncio
    async def test_connect_database_error_raises_connection_error(self):
        """Test connection failure raises ConnectionError."""
        storage = PostgresStorage(database_url="postgresql://invalid/test")

        with patch(
            "arbiter.storage.postgres.asyncpg.create_pool",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(ConnectionError, match="PostgreSQL connection failed"):
                await storage.connect()

    @pytest.mark.asyncio
    async def test_close_closes_pool(self, mock_pool):
        """Test close() closes the connection pool."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        await storage.close()

        mock_pool.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_pool_does_nothing(self):
        """Test close() is safe when pool doesn't exist."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        await storage.close()  # Should not raise


class TestPostgresStorageSaveResult:
    """Test PostgresStorage save_result() method."""

    @pytest.mark.asyncio
    async def test_save_result_success(self, mock_pool, mock_eval_result):
        """Test successful result save returns UUID."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        # Mock database insert
        test_uuid = uuid.uuid4()
        mock_pool._conn.fetchrow = AsyncMock(return_value={"id": test_uuid})

        result_id = await storage.save_result(mock_eval_result)

        assert result_id == str(test_uuid)
        mock_pool._conn.fetchrow.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_result_with_metadata(self, mock_pool, mock_eval_result):
        """Test saving result with metadata."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        test_uuid = uuid.uuid4()
        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(return_value={"id": test_uuid})

        metadata = {"user_id": "user_123", "session_id": "session_456"}
        result_id = await storage.save_result(mock_eval_result, metadata=metadata)

        assert result_id == str(test_uuid)

    @pytest.mark.asyncio
    async def test_save_result_without_pool_raises_error(self, mock_eval_result):
        """Test save without connection raises ConnectionError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        with pytest.raises(ConnectionError, match="Not connected to database"):
            await storage.save_result(mock_eval_result)

    @pytest.mark.asyncio
    async def test_save_result_database_error_raises_save_error(
        self, mock_pool, mock_eval_result
    ):
        """Test database error during save raises SaveError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(SaveError, match="Failed to save result"):
            await storage.save_result(mock_eval_result)


class TestPostgresStorageSaveBatchResult:
    """Test PostgresStorage save_batch_result() method."""

    @pytest.mark.asyncio
    async def test_save_batch_result_success(self, mock_pool, mock_batch_result):
        """Test successful batch save returns UUID."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        test_uuid = uuid.uuid4()
        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(return_value={"id": test_uuid})

        batch_id = await storage.save_batch_result(mock_batch_result)

        assert batch_id == str(test_uuid)

    @pytest.mark.asyncio
    async def test_save_batch_result_with_metadata(self, mock_pool, mock_batch_result):
        """Test saving batch with metadata."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        test_uuid = uuid.uuid4()
        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(return_value={"id": test_uuid})

        metadata = {"experiment_id": "exp_789"}
        batch_id = await storage.save_batch_result(mock_batch_result, metadata=metadata)

        assert batch_id == str(test_uuid)

    @pytest.mark.asyncio
    async def test_save_batch_result_without_pool_raises_error(self, mock_batch_result):
        """Test save without connection raises ConnectionError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        with pytest.raises(ConnectionError, match="Not connected to database"):
            await storage.save_batch_result(mock_batch_result)

    @pytest.mark.asyncio
    async def test_save_batch_result_database_error_raises_save_error(
        self, mock_pool, mock_batch_result
    ):
        """Test database error during batch save raises SaveError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(SaveError, match="Failed to save batch result"):
            await storage.save_batch_result(mock_batch_result)


class TestPostgresStorageGetResult:
    """Test PostgresStorage get_result() method."""

    @pytest.mark.asyncio
    async def test_get_result_success(self, mock_pool, mock_eval_result):
        """Test successful result retrieval."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        result_data = mock_eval_result.model_dump(mode="json")
        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(
            return_value={"result_data": json.dumps(result_data)}
        )

        test_uuid = str(uuid.uuid4())
        retrieved = await storage.get_result(test_uuid)

        assert retrieved is not None
        assert retrieved.overall_score == mock_eval_result.overall_score

    @pytest.mark.asyncio
    async def test_get_result_not_found_returns_none(self, mock_pool):
        """Test retrieval of non-existent result returns None."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(return_value=None)

        test_uuid = str(uuid.uuid4())
        result = await storage.get_result(test_uuid)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_result_without_pool_raises_error(self):
        """Test get without connection raises ConnectionError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        with pytest.raises(ConnectionError, match="Not connected to database"):
            await storage.get_result(str(uuid.uuid4()))

    @pytest.mark.asyncio
    async def test_get_result_database_error_raises_retrieval_error(self, mock_pool):
        """Test database error during retrieval raises RetrievalError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        mock_pool._conn.fetchrow = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(RetrievalError, match="Failed to retrieve result"):
            await storage.get_result(str(uuid.uuid4()))


class TestPostgresStorageGetBatchResult:
    """Test PostgresStorage get_batch_result() method."""

    @pytest.mark.asyncio
    async def test_get_batch_result_success(self, mock_pool, mock_batch_result):
        """Test successful batch retrieval."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        result_data = mock_batch_result.model_dump(mode="json")
        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(
            return_value={"result_data": json.dumps(result_data)}
        )

        test_uuid = str(uuid.uuid4())
        retrieved = await storage.get_batch_result(test_uuid)

        assert retrieved is not None
        assert retrieved.total_items == mock_batch_result.total_items

    @pytest.mark.asyncio
    async def test_get_batch_result_not_found_returns_none(self, mock_pool):
        """Test retrieval of non-existent batch returns None."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        # Use mock_pool._conn instead
        mock_pool._conn.fetchrow = AsyncMock(return_value=None)

        test_uuid = str(uuid.uuid4())
        result = await storage.get_batch_result(test_uuid)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_batch_result_without_pool_raises_error(self):
        """Test get without connection raises ConnectionError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        with pytest.raises(ConnectionError, match="Not connected to database"):
            await storage.get_batch_result(str(uuid.uuid4()))

    @pytest.mark.asyncio
    async def test_get_batch_result_database_error_raises_retrieval_error(
        self, mock_pool
    ):
        """Test database error during batch retrieval raises RetrievalError."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")
        storage.pool = mock_pool

        mock_pool._conn.fetchrow = AsyncMock(side_effect=Exception("Database error"))

        with pytest.raises(RetrievalError, match="Failed to retrieve batch"):
            await storage.get_batch_result(str(uuid.uuid4()))


class TestPostgresStorageContextManager:
    """Test PostgresStorage async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self, mock_pool):
        """Test context manager connects on enter and closes on exit."""
        storage = PostgresStorage(database_url="postgresql://localhost/test")

        async def mock_create_pool(*args, **kwargs):
            return mock_pool

        with patch(
            "arbiter.storage.postgres.asyncpg.create_pool", side_effect=mock_create_pool
        ):
            async with storage:
                assert storage.pool == mock_pool

            mock_pool.close.assert_called_once()
