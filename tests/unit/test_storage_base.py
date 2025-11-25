"""Tests for storage backend base classes."""

import pytest

from arbiter_ai.storage.base import (
    ConnectionError,
    RetrievalError,
    SaveError,
    StorageBackend,
    StorageError,
)


def test_storage_exceptions_inheritance():
    """Test that storage exceptions inherit correctly."""
    assert issubclass(ConnectionError, StorageError)
    assert issubclass(SaveError, StorageError)
    assert issubclass(RetrievalError, StorageError)
    assert issubclass(StorageError, Exception)


def test_storage_backend_is_abstract():
    """Test that StorageBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        StorageBackend()  # type: ignore


@pytest.mark.asyncio
async def test_storage_backend_context_manager():
    """Test that concrete implementations support async context manager."""

    class TestStorage(StorageBackend):
        def __init__(self):
            self.connected = False
            self.closed = False

        async def connect(self):
            self.connected = True

        async def close(self):
            self.closed = True

        async def save_result(self, result, metadata=None):
            return "test_id"

        async def save_batch_result(self, result, metadata=None):
            return "test_batch_id"

        async def get_result(self, result_id):
            return None

        async def get_batch_result(self, batch_id):
            return None

    storage = TestStorage()
    assert not storage.connected
    assert not storage.closed

    async with storage:
        assert storage.connected
        assert not storage.closed

    assert storage.connected
    assert storage.closed
