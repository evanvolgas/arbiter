"""LLM client connection pooling for improved performance and resource management.

This module provides connection pooling for LLM clients to:
- Reduce connection setup overhead
- Manage concurrent connections efficiently
- Implement connection health checks
- Handle connection lifecycle management
- Provide connection statistics and monitoring

The pool maintains a set of ready-to-use connections for each provider/model
combination, automatically creating new connections when needed and cleaning
up idle connections based on configurable policies.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from .exceptions import ModelProviderError
from .llm_client import LLMClient
from .types import Provider

__all__ = ["LLMClientPool", "PoolConfig", "ConnectionMetrics", "get_global_pool"]


@dataclass
class ConnectionMetrics:
    """Metrics for connection pool monitoring."""

    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    created_connections: int = 0
    destroyed_connections: int = 0
    borrowed_connections: int = 0
    returned_connections: int = 0
    failed_connections: int = 0
    pool_hits: int = 0
    pool_misses: int = 0
    last_reset: float = field(default_factory=time.time)

    def reset(self) -> None:
        """Reset all metrics."""
        self.total_connections = 0
        self.active_connections = 0
        self.idle_connections = 0
        self.created_connections = 0
        self.destroyed_connections = 0
        self.borrowed_connections = 0
        self.returned_connections = 0
        self.failed_connections = 0
        self.pool_hits = 0
        self.pool_misses = 0
        self.last_reset = time.time()


@dataclass
class PooledConnection:
    """Wrapper for pooled LLM client connections."""

    client: LLMClient
    created_at: float
    last_used: float
    use_count: int = 0
    is_healthy: bool = True
    pool_key: str = ""

    def __post_init__(self) -> None:
        """Set creation timestamp."""
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_used == 0:
            self.last_used = time.time()

    def mark_used(self) -> None:
        """Mark connection as used."""
        self.use_count += 1
        self.last_used = time.time()

    def is_expired(self, max_age: float) -> bool:
        """Check if connection has expired."""
        return time.time() - self.created_at > max_age

    def is_idle(self, max_idle: float) -> bool:
        """Check if connection has been idle too long."""
        return time.time() - self.last_used > max_idle


class PoolConfig(BaseModel):
    """Configuration for LLM client connection pool."""

    max_pool_size: int = Field(
        default=10, ge=1, le=100, description="Maximum number of connections per pool"
    )

    min_pool_size: int = Field(
        default=1, ge=0, le=10, description="Minimum number of connections to maintain"
    )

    max_idle_time: float = Field(
        default=300.0,  # 5 minutes
        ge=10.0,
        le=3600.0,
        description="Maximum time a connection can be idle before cleanup",
    )

    max_connection_age: float = Field(
        default=1800.0,  # 30 minutes
        ge=60.0,
        le=7200.0,
        description="Maximum age of a connection before renewal",
    )

    health_check_interval: float = Field(
        default=60.0,  # 1 minute
        ge=10.0,
        le=600.0,
        description="Interval between health checks",
    )

    connection_timeout: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for creating new connections",
    )

    enable_health_checks: bool = Field(
        default=True, description="Whether to perform connection health checks"
    )

    enable_metrics: bool = Field(
        default=True, description="Whether to collect pool metrics"
    )


class LLMClientPool:
    """Connection pool for LLM clients with lifecycle management."""

    def __init__(self, config: Optional[PoolConfig] = None):
        """Initialize the connection pool.

        Args:
            config: Pool configuration. Uses defaults if not provided.
        """
        self.config = config or PoolConfig()
        self._pools: Dict[str, List[PooledConnection]] = {}
        self._active_connections: Dict[str, List[PooledConnection]] = {}
        self._metrics = ConnectionMetrics()
        self._lock: Optional[asyncio.Lock] = None
        self._health_check_task: Optional[asyncio.Task[None]] = None
        self._cleanup_task: Optional[asyncio.Task[None]] = None
        self._initialized = False

    async def _initialize(self) -> None:
        """Initialize async components if not already initialized."""
        if self._initialized:
            return

        self._lock = asyncio.Lock()

        # Start background tasks
        if self.config.enable_health_checks:
            self._health_check_task = asyncio.create_task(self._health_check_loop())

        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._initialized = True

    def _get_pool_key(self, provider: Provider, model: str, temperature: float) -> str:
        """Generate pool key for provider/model/temperature combination."""
        return f"{provider.value}:{model}:{temperature:.1f}"

    async def get_client(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> LLMClient:
        """Get a client from the pool or create a new one.

        Args:
            provider: LLM provider
            model: Model name
            temperature: Generation temperature
            api_key: API key override

        Returns:
            LLM client ready for use

        Raises:
            ModelProviderError: If unable to create connection
        """
        await self._initialize()
        pool_key = self._get_pool_key(provider, model, temperature)

        assert self._lock is not None
        async with self._lock:
            # Try to get from pool
            if pool_key in self._pools and self._pools[pool_key]:
                connection = self._pools[pool_key].pop()
                connection.mark_used()

                # Move to active connections
                if pool_key not in self._active_connections:
                    self._active_connections[pool_key] = []
                self._active_connections[pool_key].append(connection)

                if self.config.enable_metrics:
                    self._metrics.pool_hits += 1
                    self._metrics.borrowed_connections += 1
                    self._metrics.active_connections += 1
                    self._metrics.idle_connections -= 1

                return connection.client

            # Create new connection
            try:
                client = LLMClient(provider, model, temperature, api_key)
                connection = PooledConnection(
                    client=client,
                    created_at=time.time(),
                    last_used=time.time(),
                    pool_key=pool_key,
                )

                # Move to active connections
                if pool_key not in self._active_connections:
                    self._active_connections[pool_key] = []
                self._active_connections[pool_key].append(connection)

                if self.config.enable_metrics:
                    self._metrics.pool_misses += 1
                    self._metrics.created_connections += 1
                    self._metrics.borrowed_connections += 1
                    self._metrics.active_connections += 1
                    self._metrics.total_connections += 1

                return client

            except Exception as e:
                if self.config.enable_metrics:
                    self._metrics.failed_connections += 1
                raise ModelProviderError(
                    f"Failed to create LLM client: {e}",
                    details={"provider": provider.value},
                ) from e

    async def return_client(self, client: LLMClient) -> None:
        """Return a client to the pool.

        Args:
            client: The client to return to the pool
        """
        await self._initialize()
        assert self._lock is not None
        async with self._lock:
            # Find the connection in active connections
            connection = None
            pool_key = None

            for key, active_list in self._active_connections.items():
                for conn in active_list:
                    if conn.client is client:
                        connection = conn
                        pool_key = key
                        break
                if connection:
                    break

            if not connection or not pool_key:
                # Client not found in active connections, ignore
                return

            # Remove from active connections
            if connection in self._active_connections[pool_key]:
                self._active_connections[pool_key].remove(connection)

            # Check if connection is still healthy and not expired
            if connection.is_healthy and not connection.is_expired(
                self.config.max_connection_age
            ):
                # Return to pool if there's space
                if pool_key not in self._pools:
                    self._pools[pool_key] = []

                if len(self._pools[pool_key]) < self.config.max_pool_size:
                    self._pools[pool_key].append(connection)

                    if self.config.enable_metrics:
                        self._metrics.returned_connections += 1
                        self._metrics.active_connections -= 1
                        self._metrics.idle_connections += 1
                else:
                    # Pool is full, destroy connection
                    if self.config.enable_metrics:
                        self._metrics.destroyed_connections += 1
                        self._metrics.active_connections -= 1
                        self._metrics.total_connections -= 1
            else:
                # Connection unhealthy or expired, destroy it
                if self.config.enable_metrics:
                    self._metrics.destroyed_connections += 1
                    self._metrics.active_connections -= 1
                    self._metrics.total_connections -= 1

    async def warm_up(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        connections: int = 1,
    ) -> None:
        """Pre-create connections for a provider/model combination.

        Args:
            provider: LLM provider
            model: Model name
            temperature: Generation temperature
            connections: Number of connections to create
        """
        for _ in range(connections):
            client = await self.get_client(provider, model, temperature)
            await self.return_client(client)

    def get_metrics(self) -> ConnectionMetrics:
        """Get current pool metrics."""
        return self._metrics

    async def _health_check_loop(self) -> None:
        """Background task for connection health checks."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                # Health check logic could be added here
                # For now, we rely on connection age/idle checks
            except asyncio.CancelledError:
                break
            except Exception:
                # Silently continue on errors
                pass

    async def _cleanup_loop(self) -> None:
        """Background task for cleaning up idle and expired connections."""
        while True:
            try:
                await asyncio.sleep(60)  # Cleanup every minute

                if self._lock:
                    async with self._lock:
                        for pool_key, connections in list(self._pools.items()):
                            # Remove expired or idle connections
                            valid_connections = []
                            for conn in connections:
                                if conn.is_expired(
                                    self.config.max_connection_age
                                ) or conn.is_idle(self.config.max_idle_time):
                                    if self.config.enable_metrics:
                                        self._metrics.destroyed_connections += 1
                                        self._metrics.idle_connections -= 1
                                        self._metrics.total_connections -= 1
                                else:
                                    valid_connections.append(conn)

                            self._pools[pool_key] = valid_connections

            except asyncio.CancelledError:
                break
            except Exception:
                # Silently continue on errors
                pass

    async def close(self) -> None:
        """Close the pool and cleanup resources."""
        # Cancel background tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Clear all connections
        if self._lock:
            async with self._lock:
                self._pools.clear()
                self._active_connections.clear()
                if self.config.enable_metrics:
                    self._metrics.reset()


# Global pool instance
_global_pool: Optional[LLMClientPool] = None


def get_global_pool() -> LLMClientPool:
    """Get the global connection pool instance."""
    global _global_pool
    if _global_pool is None:
        _global_pool = LLMClientPool()
    return _global_pool
