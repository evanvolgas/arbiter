"""Tests for LLM client connection pooling.

Tests cover:
- Pool initialization and configuration
- Getting and returning clients
- Connection metrics tracking
- Pool size limits (min/max)
- Connection expiration and idle cleanup
- Warm-up functionality
- Background task management
- Global pool access
"""

import time
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio

from arbiter_ai.core.exceptions import ModelProviderError
from arbiter_ai.core.llm_client_pool import (
    ConnectionMetrics,
    LLMClientPool,
    PoolConfig,
    PooledConnection,
    get_global_pool,
)
from arbiter_ai.core.types import Provider


@pytest.fixture
def pool_config():
    """Create pool config with minimum valid timeouts for testing."""
    return PoolConfig(
        max_pool_size=5,
        min_pool_size=1,
        max_idle_time=10.0,  # Minimum allowed value
        max_connection_age=60.0,  # Minimum allowed value
        health_check_interval=10.0,  # Minimum allowed value
        connection_timeout=2.0,
        enable_health_checks=True,
        enable_metrics=True,
    )


@pytest.fixture
def pool(pool_config):
    """Create pool instance with test config."""
    return LLMClientPool(config=pool_config)


@pytest_asyncio.fixture
async def initialized_pool(pool):
    """Create and initialize pool."""
    await pool._initialize()
    yield pool
    await pool.close()


class TestPoolConfig:
    """Test suite for PoolConfig validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PoolConfig()
        assert config.max_pool_size == 10
        assert config.min_pool_size == 1
        assert config.max_idle_time == 300.0
        assert config.max_connection_age == 1800.0
        assert config.health_check_interval == 60.0
        assert config.enable_health_checks is True
        assert config.enable_metrics is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PoolConfig(
            max_pool_size=20,
            min_pool_size=5,
            max_idle_time=600.0,
            enable_metrics=False,
        )
        assert config.max_pool_size == 20
        assert config.min_pool_size == 5
        assert config.max_idle_time == 600.0
        assert config.enable_metrics is False

    def test_config_validation(self):
        """Test that invalid config values raise errors."""
        with pytest.raises(Exception):  # Pydantic validation error
            PoolConfig(max_pool_size=0)  # Below minimum

        with pytest.raises(Exception):
            PoolConfig(max_pool_size=101)  # Above maximum


class TestConnectionMetrics:
    """Test suite for ConnectionMetrics."""

    def test_metrics_initialization(self):
        """Test metrics start at zero."""
        metrics = ConnectionMetrics()
        assert metrics.total_connections == 0
        assert metrics.active_connections == 0
        assert metrics.idle_connections == 0
        assert metrics.pool_hits == 0
        assert metrics.pool_misses == 0

    def test_metrics_reset(self):
        """Test metrics reset functionality."""
        metrics = ConnectionMetrics()
        metrics.total_connections = 10
        metrics.pool_hits = 5
        metrics.pool_misses = 3

        metrics.reset()

        assert metrics.total_connections == 0
        assert metrics.pool_hits == 0
        assert metrics.pool_misses == 0
        assert metrics.last_reset > 0


class TestPooledConnection:
    """Test suite for PooledConnection."""

    @patch("arbiter.core.llm_client_pool.LLMClient")
    def test_connection_initialization(self, mock_client_class):
        """Test connection wrapper initialization."""
        mock_client = MagicMock()
        conn = PooledConnection(
            client=mock_client,
            created_at=time.time(),
            last_used=time.time(),
        )

        assert conn.client is mock_client
        assert conn.use_count == 0
        assert conn.is_healthy is True

    @patch("arbiter.core.llm_client_pool.LLMClient")
    def test_mark_used(self, mock_client_class):
        """Test marking connection as used."""
        mock_client = MagicMock()
        conn = PooledConnection(
            client=mock_client,
            created_at=time.time(),
            last_used=time.time() - 10,
        )

        initial_use_count = conn.use_count
        initial_last_used = conn.last_used

        conn.mark_used()

        assert conn.use_count == initial_use_count + 1
        assert conn.last_used > initial_last_used

    @patch("arbiter.core.llm_client_pool.LLMClient")
    def test_is_expired(self, mock_client_class):
        """Test connection expiration check."""
        mock_client = MagicMock()

        # Old connection
        old_conn = PooledConnection(
            client=mock_client,
            created_at=time.time() - 100,
            last_used=time.time(),
        )
        assert old_conn.is_expired(max_age=10.0) is True

        # Fresh connection
        new_conn = PooledConnection(
            client=mock_client,
            created_at=time.time(),
            last_used=time.time(),
        )
        assert new_conn.is_expired(max_age=10.0) is False

    @patch("arbiter.core.llm_client_pool.LLMClient")
    def test_is_idle(self, mock_client_class):
        """Test connection idle check."""
        mock_client = MagicMock()

        # Idle connection
        idle_conn = PooledConnection(
            client=mock_client,
            created_at=time.time(),
            last_used=time.time() - 100,
        )
        assert idle_conn.is_idle(max_idle=10.0) is True

        # Recently used connection
        active_conn = PooledConnection(
            client=mock_client,
            created_at=time.time(),
            last_used=time.time(),
        )
        assert active_conn.is_idle(max_idle=10.0) is False


class TestLLMClientPool:
    """Test suite for LLMClientPool basic operations."""

    def test_pool_initialization(self, pool_config):
        """Test pool initialization."""
        pool = LLMClientPool(config=pool_config)
        assert pool.config == pool_config
        assert pool._initialized is False
        assert pool._lock is None

    def test_pool_initialization_default_config(self):
        """Test pool with default config."""
        pool = LLMClientPool()
        assert pool.config is not None
        assert pool.config.max_pool_size == 10

    @pytest.mark.asyncio
    async def test_pool_async_initialization(self, pool):
        """Test async initialization."""
        assert pool._initialized is False
        await pool._initialize()
        assert pool._initialized is True
        assert pool._lock is not None
        await pool.close()

    @pytest.mark.asyncio
    async def test_get_pool_key(self, initialized_pool):
        """Test pool key generation."""
        key = initialized_pool._get_pool_key(Provider.OPENAI, "gpt-4o-mini", 0.7)
        assert key == "openai:gpt-4o-mini:0.7"

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_get_client_creates_new(self, mock_client_class, initialized_pool):
        """Test getting client when pool is empty creates new one."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        assert client is mock_client
        mock_client_class.assert_called_once_with(
            Provider.OPENAI, "gpt-4o-mini", 0.7, None
        )

        # Check metrics
        metrics = initialized_pool.get_metrics()
        assert metrics.pool_misses == 1
        assert metrics.created_connections == 1
        assert metrics.active_connections == 1

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_get_client_from_pool(self, mock_client_class, initialized_pool):
        """Test getting client from pool (cache hit)."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # First get creates client
        client1 = await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        # Return to pool
        await initialized_pool.return_client(client1)

        # Second get should reuse from pool
        client2 = await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        assert client2 is mock_client

        # Check metrics
        metrics = initialized_pool.get_metrics()
        assert metrics.pool_hits == 1
        assert metrics.pool_misses == 1

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_return_client_to_pool(self, mock_client_class, initialized_pool):
        """Test returning client to pool."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        client = await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        await initialized_pool.return_client(client)

        # Check metrics
        metrics = initialized_pool.get_metrics()
        assert metrics.returned_connections == 1
        assert metrics.idle_connections == 1
        assert metrics.active_connections == 0

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_return_unknown_client(self, mock_client_class, initialized_pool):
        """Test returning client not from pool (should be ignored)."""
        unknown_client = MagicMock()

        # Should not raise error, just ignore
        await initialized_pool.return_client(unknown_client)

        # Metrics should be unchanged
        metrics = initialized_pool.get_metrics()
        assert metrics.returned_connections == 0

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_pool_size_limit(self, mock_client_class, initialized_pool):
        """Test that pool respects max_pool_size."""
        initialized_pool.config.max_pool_size = 2

        # Create and return 3 clients
        clients = []
        for i in range(3):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            client = await initialized_pool.get_client(
                Provider.OPENAI, f"model-{i}", 0.7
            )
            clients.append(client)

        # Return all clients
        for client in clients:
            await initialized_pool.return_client(client)

        # Pool should only have max_pool_size connections per key
        # Since we used different models, each gets its own pool
        # But let's test with same model
        initialized_pool._pools.clear()
        initialized_pool._active_connections.clear()

        clients = []
        for _ in range(3):
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            client = await initialized_pool.get_client(
                Provider.OPENAI, "gpt-4o-mini", 0.7
            )
            clients.append(client)

        for client in clients:
            await initialized_pool.return_client(client)

        # Only 2 should be in pool, 1 destroyed
        pool_key = "openai:gpt-4o-mini:0.7"
        assert len(initialized_pool._pools.get(pool_key, [])) <= 2

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_warm_up(self, mock_client_class, initialized_pool):
        """Test pool warm-up functionality."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        await initialized_pool.warm_up(
            Provider.OPENAI, "gpt-4o-mini", 0.7, connections=3
        )

        # Pool reuses connections, so only 1 connection created
        # but it gets borrowed and returned 3 times
        pool_key = "openai:gpt-4o-mini:0.7"
        assert len(initialized_pool._pools.get(pool_key, [])) == 1

        # Check metrics - only 1 connection created, but multiple borrows/returns
        metrics = initialized_pool.get_metrics()
        assert metrics.created_connections == 1  # Only creates once
        assert metrics.pool_hits == 2  # Second and third calls hit the pool
        assert metrics.pool_misses == 1  # First call misses
        assert metrics.borrowed_connections == 3  # Borrowed 3 times total
        assert metrics.returned_connections == 3  # Returned 3 times
        assert metrics.idle_connections == 1  # 1 connection idle at end

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient", side_effect=Exception("API Error"))
    async def test_get_client_error_handling(self, mock_client_class, initialized_pool):
        """Test error handling when creating client fails."""
        with pytest.raises(ModelProviderError, match="Failed to create LLM client"):
            await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        # Check metrics
        metrics = initialized_pool.get_metrics()
        assert metrics.failed_connections == 1

    @pytest.mark.asyncio
    async def test_get_metrics(self, initialized_pool):
        """Test getting pool metrics."""
        metrics = initialized_pool.get_metrics()
        assert isinstance(metrics, ConnectionMetrics)
        assert metrics.total_connections >= 0


class TestPoolCleanup:
    """Test suite for connection cleanup and lifecycle."""

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_expired_connection_cleanup(self, mock_client_class):
        """Test that expired connections are cleaned up."""
        # Use minimum valid value for faster test
        config = PoolConfig(max_connection_age=60.0)
        pool = LLMClientPool(config=config)
        await pool._initialize()

        try:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Get and return client
            client = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)
            await pool.return_client(client)

            # Manually mark connection as expired by manipulating created_at
            pool_key = "openai:gpt-4o-mini:0.7"
            for conn in pool._pools.get(pool_key, []):
                conn.created_at = (
                    time.time() - 65
                )  # 65 seconds ago (> 60 second max age)

            # Trigger cleanup manually
            async with pool._lock:
                for pool_key, connections in list(pool._pools.items()):
                    valid_connections = []
                    for conn in connections:
                        if not conn.is_expired(pool.config.max_connection_age):
                            valid_connections.append(conn)
                    pool._pools[pool_key] = valid_connections

            # Pool should be empty
            pool_key = "openai:gpt-4o-mini:0.7"
            assert len(pool._pools.get(pool_key, [])) == 0
        finally:
            await pool.close()

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_idle_connection_cleanup(self, mock_client_class):
        """Test that idle connections are cleaned up."""
        # Use minimum valid value for faster test
        config = PoolConfig(max_idle_time=10.0)
        pool = LLMClientPool(config=config)
        await pool._initialize()

        try:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            # Get and return client
            client = await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)
            await pool.return_client(client)

            # Manually mark connection as idle by manipulating last_used
            pool_key = "openai:gpt-4o-mini:0.7"
            for conn in pool._pools.get(pool_key, []):
                conn.last_used = (
                    time.time() - 15
                )  # 15 seconds ago (> 10 second max idle)

            # Trigger cleanup manually
            async with pool._lock:
                for pool_key, connections in list(pool._pools.items()):
                    valid_connections = []
                    for conn in connections:
                        if not conn.is_idle(pool.config.max_idle_time):
                            valid_connections.append(conn)
                    pool._pools[pool_key] = valid_connections

            # Pool should be empty
            pool_key = "openai:gpt-4o-mini:0.7"
            assert len(pool._pools.get(pool_key, [])) == 0
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_pool_close(self, initialized_pool):
        """Test pool cleanup on close."""
        # Add some connections to pool
        with patch("arbiter.core.llm_client_pool.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            client = await initialized_pool.get_client(
                Provider.OPENAI, "gpt-4o-mini", 0.7
            )
            await initialized_pool.return_client(client)

        # Close pool
        await initialized_pool.close()

        # Pools should be cleared
        assert len(initialized_pool._pools) == 0
        assert len(initialized_pool._active_connections) == 0


class TestGlobalPool:
    """Test suite for global pool access."""

    def test_get_global_pool_creates_singleton(self):
        """Test that get_global_pool returns singleton instance."""
        pool1 = get_global_pool()
        pool2 = get_global_pool()

        assert pool1 is pool2


class TestPoolBackgroundTasks:
    """Test suite for background task management."""

    @pytest.mark.asyncio
    async def test_background_tasks_start(self, pool):
        """Test that background tasks start on initialization."""
        await pool._initialize()

        try:
            assert pool._cleanup_task is not None
            assert not pool._cleanup_task.done()

            if pool.config.enable_health_checks:
                assert pool._health_check_task is not None
                assert not pool._health_check_task.done()
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_background_tasks_cancel_on_close(self, initialized_pool):
        """Test that background tasks are cancelled on close."""
        cleanup_task = initialized_pool._cleanup_task
        health_task = initialized_pool._health_check_task

        await initialized_pool.close()

        # Tasks should be cancelled
        assert cleanup_task.cancelled() or cleanup_task.done()
        if health_task:
            assert health_task.cancelled() or health_task.done()

    @pytest.mark.asyncio
    async def test_pool_without_health_checks(self):
        """Test pool with health checks disabled."""
        config = PoolConfig(enable_health_checks=False)
        pool = LLMClientPool(config=config)
        await pool._initialize()

        try:
            assert pool._health_check_task is None
            assert pool._cleanup_task is not None
        finally:
            await pool.close()


class TestPoolMetrics:
    """Test suite for pool metrics tracking."""

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_metrics_tracking_disabled(self, mock_client_class):
        """Test pool with metrics disabled."""
        config = PoolConfig(enable_metrics=False)
        pool = LLMClientPool(config=config)
        await pool._initialize()

        try:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            await pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

            # Metrics should not increment
            metrics = pool.get_metrics()
            assert metrics.pool_misses == 0
            assert metrics.created_connections == 0
        finally:
            await pool.close()

    @pytest.mark.asyncio
    @patch("arbiter.core.llm_client_pool.LLMClient")
    async def test_comprehensive_metrics_tracking(
        self, mock_client_class, initialized_pool
    ):
        """Test comprehensive metrics tracking."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        # Get client (miss)
        client1 = await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        metrics = initialized_pool.get_metrics()
        assert metrics.pool_misses == 1
        assert metrics.created_connections == 1
        assert metrics.borrowed_connections == 1
        assert metrics.active_connections == 1

        # Return client
        await initialized_pool.return_client(client1)

        metrics = initialized_pool.get_metrics()
        assert metrics.returned_connections == 1
        assert metrics.idle_connections == 1
        assert metrics.active_connections == 0

        # Get client again (hit)
        await initialized_pool.get_client(Provider.OPENAI, "gpt-4o-mini", 0.7)

        metrics = initialized_pool.get_metrics()
        assert metrics.pool_hits == 1
        assert metrics.borrowed_connections == 2
        assert metrics.active_connections == 1
        assert metrics.idle_connections == 0
