"""Comprehensive tests for Redis storage backend."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from arbiter_ai.core.models import (
    BatchEvaluationResult,
    EvaluationResult,
    LLMInteraction,
    Score,
)
from arbiter_ai.storage.base import ConnectionError, RetrievalError, SaveError

# Skip all tests if redis is not installed
pytest.importorskip("redis", reason="Redis storage requires redis")

from arbiter_ai.storage.redis import RedisStorage


@pytest.fixture
def mock_eval_result():
    """Create a mock EvaluationResult for testing."""
    return EvaluationResult(
        output="London is the capital of England",
        reference="The capital of England is London",
        overall_score=0.85,
        passed=True,
        scores=[
            Score(
                name="semantic",
                value=0.85,
                confidence=0.90,
                explanation="Good similarity",
            )
        ],
        interactions=[
            LLMInteraction(
                model="gpt-4o-mini",
                prompt="Test prompt",
                response="Test response",
                input_tokens=15,
                output_tokens=25,
                cached_tokens=0,
                tokens_used=40,
                cost=0.002,
                latency=2.0,
                purpose="scoring",
            )
        ],
        total_input_tokens=15,
        total_output_tokens=25,
        total_cached_tokens=0,
        total_tokens_used=40,
        total_cost=0.002,
        processing_time=2.0,
    )


@pytest.fixture
def mock_batch_result(mock_eval_result):
    """Create a mock BatchEvaluationResult for testing."""
    return BatchEvaluationResult(
        results=[mock_eval_result, mock_eval_result, mock_eval_result],
        total_items=3,
        successful_items=3,
        failed_items=0,
        errors=[],
        processing_time=6.0,
    )


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    client = AsyncMock()
    client.ping = AsyncMock(return_value=True)
    client.setex = AsyncMock()
    client.get = AsyncMock()
    client.close = AsyncMock()
    return client


class TestRedisStorageInit:
    """Test RedisStorage initialization."""

    def test_init_with_url(self):
        """Test initialization with explicit Redis URL."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        assert storage.redis_url == "redis://localhost:6379"
        assert storage.ttl == 86400  # 24 hours default
        assert storage.key_prefix == "arbiter:"
        assert storage.client is None

    def test_init_with_env_var(self, monkeypatch):
        """Test initialization with REDIS_URL environment variable."""
        monkeypatch.setenv("REDIS_URL", "redis://env:6379")
        storage = RedisStorage()
        assert storage.redis_url == "redis://env:6379"

    def test_init_without_url_raises_error(self, monkeypatch):
        """Test initialization without Redis URL raises ValueError."""
        monkeypatch.delenv("REDIS_URL", raising=False)
        with pytest.raises(ValueError, match="REDIS_URL must be provided"):
            RedisStorage()

    def test_init_with_custom_params(self):
        """Test initialization with custom parameters."""
        storage = RedisStorage(
            redis_url="redis://localhost:6379",
            ttl=3600,  # 1 hour
            key_prefix="test:",
        )
        assert storage.ttl == 3600
        assert storage.key_prefix == "test:"


class TestRedisStorageConnect:
    """Test RedisStorage connection handling."""

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis_client):
        """Test successful Redis connection."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        async def mock_from_url(*args, **kwargs):
            return mock_redis_client

        with patch("arbiter.storage.redis.redis.from_url", side_effect=mock_from_url):
            await storage.connect()

            assert storage.client == mock_redis_client
            mock_redis_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_raises_connection_error(self):
        """Test connection failure raises ConnectionError."""
        storage = RedisStorage(redis_url="redis://invalid:6379")

        with patch(
            "arbiter.storage.redis.redis.from_url",
            side_effect=Exception("Connection failed"),
        ):
            with pytest.raises(ConnectionError, match="Redis connection failed"):
                await storage.connect()

    @pytest.mark.asyncio
    async def test_close_closes_client(self, mock_redis_client):
        """Test close() closes the Redis client."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        await storage.close()

        mock_redis_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_without_client_does_nothing(self):
        """Test close() is safe when client doesn't exist."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        await storage.close()  # Should not raise


class TestRedisStorageMakeKey:
    """Test RedisStorage key generation."""

    def test_make_key_for_result(self):
        """Test key generation for regular results."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        key = storage._make_key("test_id", "result")
        assert key == "arbiter:result:test_id"

    def test_make_key_for_batch(self):
        """Test key generation for batch results."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        key = storage._make_key("batch_id", "batch")
        assert key == "arbiter:batch:batch_id"

    def test_make_key_with_custom_prefix(self):
        """Test key generation with custom prefix."""
        storage = RedisStorage(redis_url="redis://localhost:6379", key_prefix="custom:")
        key = storage._make_key("test_id", "result")
        assert key == "custom:result:test_id"


class TestRedisStorageSaveResult:
    """Test RedisStorage save_result() method."""

    @pytest.mark.asyncio
    async def test_save_result_success(self, mock_redis_client, mock_eval_result):
        """Test successful result save to Redis."""
        storage = RedisStorage(redis_url="redis://localhost:6379", ttl=3600)
        storage.client = mock_redis_client

        result_id = await storage.save_result(mock_eval_result)

        # Verify result_id is generated
        assert result_id is not None
        assert isinstance(result_id, str)

        # Verify setex was called with correct TTL
        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 3600  # TTL
        assert "arbiter:result:" in call_args[0][0]  # Key prefix

    @pytest.mark.asyncio
    async def test_save_result_with_metadata(self, mock_redis_client, mock_eval_result):
        """Test saving result with metadata."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        metadata = {"user_id": "user_789", "experiment": "test_exp"}
        result_id = await storage.save_result(mock_eval_result, metadata=metadata)

        assert result_id is not None
        mock_redis_client.setex.assert_called_once()

        # Verify metadata is included in stored data
        stored_data = json.loads(mock_redis_client.setex.call_args[0][2])
        assert stored_data["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_save_result_deterministic_id(
        self, mock_redis_client, mock_eval_result
    ):
        """Test that same result generates same ID (deterministic hashing)."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        result_id_1 = await storage.save_result(mock_eval_result)
        result_id_2 = await storage.save_result(mock_eval_result)

        assert result_id_1 == result_id_2

    @pytest.mark.asyncio
    async def test_save_result_without_client_raises_error(self, mock_eval_result):
        """Test save without connection raises ConnectionError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        with pytest.raises(ConnectionError, match="Not connected to Redis"):
            await storage.save_result(mock_eval_result)

    @pytest.mark.asyncio
    async def test_save_result_redis_error_raises_save_error(
        self, mock_redis_client, mock_eval_result
    ):
        """Test Redis error during save raises SaveError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.setex = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(SaveError, match="Failed to cache result"):
            await storage.save_result(mock_eval_result)


class TestRedisStorageSaveBatchResult:
    """Test RedisStorage save_batch_result() method."""

    @pytest.mark.asyncio
    async def test_save_batch_result_success(
        self, mock_redis_client, mock_batch_result
    ):
        """Test successful batch save to Redis."""
        storage = RedisStorage(redis_url="redis://localhost:6379", ttl=7200)
        storage.client = mock_redis_client

        batch_id = await storage.save_batch_result(mock_batch_result)

        assert batch_id is not None
        assert isinstance(batch_id, str)

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][1] == 7200  # TTL
        assert "arbiter:batch:" in call_args[0][0]  # Key prefix

    @pytest.mark.asyncio
    async def test_save_batch_result_with_metadata(
        self, mock_redis_client, mock_batch_result
    ):
        """Test saving batch with metadata."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        metadata = {"batch_type": "daily_eval"}
        batch_id = await storage.save_batch_result(mock_batch_result, metadata=metadata)

        assert batch_id is not None

        stored_data = json.loads(mock_redis_client.setex.call_args[0][2])
        assert stored_data["metadata"] == metadata

    @pytest.mark.asyncio
    async def test_save_batch_result_deterministic_id(
        self, mock_redis_client, mock_batch_result
    ):
        """Test that same batch generates same ID."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        batch_id_1 = await storage.save_batch_result(mock_batch_result)
        batch_id_2 = await storage.save_batch_result(mock_batch_result)

        assert batch_id_1 == batch_id_2

    @pytest.mark.asyncio
    async def test_save_batch_result_without_client_raises_error(
        self, mock_batch_result
    ):
        """Test save without connection raises ConnectionError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        with pytest.raises(ConnectionError, match="Not connected to Redis"):
            await storage.save_batch_result(mock_batch_result)

    @pytest.mark.asyncio
    async def test_save_batch_result_redis_error_raises_save_error(
        self, mock_redis_client, mock_batch_result
    ):
        """Test Redis error during batch save raises SaveError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.setex = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(SaveError, match="Failed to cache batch"):
            await storage.save_batch_result(mock_batch_result)


class TestRedisStorageGetResult:
    """Test RedisStorage get_result() method."""

    @pytest.mark.asyncio
    async def test_get_result_success(self, mock_redis_client, mock_eval_result):
        """Test successful result retrieval from Redis."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        # Mock stored data
        result_data = mock_eval_result.model_dump(mode="json")
        stored_data = json.dumps({"result": result_data, "metadata": None})
        mock_redis_client.get = AsyncMock(return_value=stored_data)

        # Generate ID to retrieve
        test_id = "test_result_id"
        retrieved = await storage.get_result(test_id)

        assert retrieved is not None
        assert retrieved.overall_score == mock_eval_result.overall_score

    @pytest.mark.asyncio
    async def test_get_result_not_found_returns_none(self, mock_redis_client):
        """Test retrieval of expired/missing result returns None."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.get = AsyncMock(return_value=None)

        result = await storage.get_result("nonexistent_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_result_without_client_raises_error(self):
        """Test get without connection raises ConnectionError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        with pytest.raises(ConnectionError, match="Not connected to Redis"):
            await storage.get_result("test_id")

    @pytest.mark.asyncio
    async def test_get_result_redis_error_raises_retrieval_error(
        self, mock_redis_client
    ):
        """Test Redis error during retrieval raises RetrievalError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.get = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(RetrievalError, match="Failed to retrieve cached result"):
            await storage.get_result("test_id")


class TestRedisStorageGetBatchResult:
    """Test RedisStorage get_batch_result() method."""

    @pytest.mark.asyncio
    async def test_get_batch_result_success(self, mock_redis_client, mock_batch_result):
        """Test successful batch retrieval from Redis."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        # Mock stored data
        result_data = mock_batch_result.model_dump(mode="json")
        stored_data = json.dumps({"result": result_data, "metadata": None})
        mock_redis_client.get = AsyncMock(return_value=stored_data)

        test_id = "test_batch_id"
        retrieved = await storage.get_batch_result(test_id)

        assert retrieved is not None
        assert retrieved.total_items == mock_batch_result.total_items

    @pytest.mark.asyncio
    async def test_get_batch_result_not_found_returns_none(self, mock_redis_client):
        """Test retrieval of expired/missing batch returns None."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.get = AsyncMock(return_value=None)

        result = await storage.get_batch_result("nonexistent_id")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_batch_result_without_client_raises_error(self):
        """Test get without connection raises ConnectionError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        with pytest.raises(ConnectionError, match="Not connected to Redis"):
            await storage.get_batch_result("test_id")

    @pytest.mark.asyncio
    async def test_get_batch_result_redis_error_raises_retrieval_error(
        self, mock_redis_client
    ):
        """Test Redis error during batch retrieval raises RetrievalError."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        mock_redis_client.get = AsyncMock(side_effect=Exception("Redis error"))

        with pytest.raises(RetrievalError, match="Failed to retrieve cached batch"):
            await storage.get_batch_result("test_id")


class TestRedisStorageContextManager:
    """Test RedisStorage async context manager."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_closes(self, mock_redis_client):
        """Test context manager connects on enter and closes on exit."""
        storage = RedisStorage(redis_url="redis://localhost:6379")

        async def mock_from_url(*args, **kwargs):
            return mock_redis_client

        with patch("arbiter.storage.redis.redis.from_url", side_effect=mock_from_url):
            async with storage:
                assert storage.client == mock_redis_client
                mock_redis_client.ping.assert_called_once()

            mock_redis_client.close.assert_called_once()


class TestRedisStorageIntegration:
    """Integration tests for full save/retrieve cycle."""

    @pytest.mark.asyncio
    async def test_save_and_retrieve_result_roundtrip(
        self, mock_redis_client, mock_eval_result
    ):
        """Test full cycle: save result and retrieve it back."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        # Mock save
        result_id = await storage.save_result(mock_eval_result)

        # Mock retrieve with saved data
        saved_call = mock_redis_client.setex.call_args[0]
        saved_data = saved_call[2]
        mock_redis_client.get = AsyncMock(return_value=saved_data)

        retrieved = await storage.get_result(result_id)

        assert retrieved is not None
        assert retrieved.overall_score == mock_eval_result.overall_score
        assert len(retrieved.scores) == len(mock_eval_result.scores)

    @pytest.mark.asyncio
    async def test_save_and_retrieve_batch_roundtrip(
        self, mock_redis_client, mock_batch_result
    ):
        """Test full cycle: save batch and retrieve it back."""
        storage = RedisStorage(redis_url="redis://localhost:6379")
        storage.client = mock_redis_client

        # Mock save
        batch_id = await storage.save_batch_result(mock_batch_result)

        # Mock retrieve with saved data
        saved_call = mock_redis_client.setex.call_args[0]
        saved_data = saved_call[2]
        mock_redis_client.get = AsyncMock(return_value=saved_data)

        retrieved = await storage.get_batch_result(batch_id)

        assert retrieved is not None
        assert retrieved.total_items == mock_batch_result.total_items
        assert retrieved.successful_items == mock_batch_result.successful_items
