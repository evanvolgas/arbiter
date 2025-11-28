"""Tests for LLM client and manager.

Tests cover:
- LLMClient initialization with different providers
- API key retrieval and environment variable handling
- Provider model mapping
- Circuit breaker integration
- Agent creation for PydanticAI
- Completion requests with retry logic
- Error handling for different failure modes
- LLMManager pool integration
- Provider auto-detection
"""

import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel

from arbiter_ai.core.circuit_breaker import CircuitBreaker
from arbiter_ai.core.exceptions import ModelProviderError
from arbiter_ai.core.llm_client import LLMClient, LLMManager, LLMResponse
from arbiter_ai.core.types import Provider


class MockEvaluationResponse(BaseModel):
    """Mock response model for agent creation tests."""

    score: float
    explanation: str


class TestLLMClient:
    """Test suite for LLMClient."""

    def test_client_initialization_openai(self):
        """Test client initialization for OpenAI."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                client = LLMClient(
                    provider=Provider.OPENAI, model="gpt-4o-mini", temperature=0.7
                )

                assert client.provider == Provider.OPENAI
                assert client.model == "gpt-4o-mini"
                assert client.temperature == 0.7
                assert client.circuit_breaker is not None
                mock_openai.assert_called_once()

    def test_client_initialization_groq(self):
        """Test client initialization for Groq."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
                client = LLMClient(
                    provider=Provider.GROQ, model="gpt-4", temperature=0.5
                )

                assert client.provider == Provider.GROQ
                assert client.model == "gpt-4"
                assert client.temperature == 0.5
                # Groq should use Groq base URL
                call_kwargs = mock_openai.call_args[1]
                assert "groq" in call_kwargs["base_url"]

    def test_client_initialization_with_custom_circuit_breaker(self):
        """Test client initialization with custom circuit breaker."""
        custom_breaker = CircuitBreaker(failure_threshold=10)

        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                client = LLMClient(
                    provider=Provider.OPENAI,
                    model="gpt-4",
                    circuit_breaker=custom_breaker,
                )

                assert client.circuit_breaker is custom_breaker

    def test_client_initialization_with_api_key_override(self):
        """Test client initialization with API key override."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI") as mock_openai:
            _client = LLMClient(
                provider=Provider.OPENAI, model="gpt-4", api_key="custom-api-key"
            )

            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "custom-api-key"

    def test_get_api_key_openai(self):
        """Test API key retrieval for OpenAI."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test123"}):
            key = LLMClient._get_api_key(Provider.OPENAI)
            assert key == "sk-test123"

    def test_get_api_key_anthropic(self):
        """Test API key retrieval for Anthropic."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}):
            key = LLMClient._get_api_key(Provider.ANTHROPIC)
            assert key == "sk-ant-test"

    def test_get_api_key_groq(self):
        """Test API key retrieval for Groq."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk-test"}):
            key = LLMClient._get_api_key(Provider.GROQ)
            assert key == "gsk-test"

    def test_get_api_key_google(self):
        """Test API key retrieval for Google."""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "AIza-test"}):
            key = LLMClient._get_api_key(Provider.GOOGLE)
            assert key == "AIza-test"

    def test_get_api_key_missing(self):
        """Test API key retrieval when not in environment."""
        with patch.dict(os.environ, {}, clear=True):
            key = LLMClient._get_api_key(Provider.OPENAI)
            assert key is None

    def test_get_provider_model_openai(self):
        """Test provider model mapping for OpenAI."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")
                model = client._get_provider_model()
                assert model == "gpt-4"

    def test_get_provider_model_groq(self):
        """Test provider model mapping for Groq."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict(os.environ, {"GROQ_API_KEY": "test"}):
                client = LLMClient(Provider.GROQ, "gpt-4o-mini")
                model = client._get_provider_model()
                # Groq maps gpt-4o-mini to llama-3.1-8b-instant
                assert model == "llama-3.1-8b-instant"

    def test_get_provider_model_unmapped(self):
        """Test provider model for unmapped model name."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "custom-model-123")
                model = client._get_provider_model()
                # Should return original if not in mapping
                assert model == "custom-model-123"

    def test_create_agent(self):
        """Test agent creation for PydanticAI."""
        with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                with patch("arbiter_ai.core.llm_client.Agent") as mock_agent_class:
                    client = LLMClient(Provider.OPENAI, "gpt-4o-mini")

                    _agent = client.create_agent(
                        system_prompt="You are a test assistant",
                        result_type=MockEvaluationResponse,
                    )

                    # Verify Agent was created with correct parameters
                    mock_agent_class.assert_called_once()
                    call_kwargs = mock_agent_class.call_args[1]
                    assert call_kwargs["model"] == "openai:gpt-4o-mini"
                    assert call_kwargs["output_type"] == MockEvaluationResponse
                    assert call_kwargs["system_prompt"] == "You are a test assistant"

    @pytest.mark.asyncio
    async def test_execute_completion_success(self):
        """Test successful completion execution."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = Mock(
            spec=["prompt_tokens", "completion_tokens", "total_tokens"]
        )
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 100

        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")

                messages = [{"role": "user", "content": "test"}]
                result = await client._execute_completion("gpt-4", messages)

                assert isinstance(result, LLMResponse)
                assert result.content == "Test response"
                assert result.usage["total_tokens"] == 100
                assert result.usage["prompt_tokens"] == 50
                assert result.usage["completion_tokens"] == 50
                assert result.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_execute_completion_rate_limit_error(self):
        """Test completion with rate limit error."""
        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Rate limit exceeded")
            )
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")

                messages = [{"role": "user", "content": "test"}]

                with pytest.raises(ModelProviderError, match="Rate limit exceeded"):
                    await client._execute_completion("gpt-4", messages)

    @pytest.mark.asyncio
    async def test_execute_completion_auth_error(self):
        """Test completion with authentication error."""
        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Invalid API key")
            )
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")

                messages = [{"role": "user", "content": "test"}]

                with pytest.raises(ModelProviderError, match="Authentication failed"):
                    await client._execute_completion("gpt-4", messages)

    @pytest.mark.asyncio
    async def test_execute_completion_generic_error(self):
        """Test completion with generic error."""
        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Network error")
            )
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")

                messages = [{"role": "user", "content": "test"}]

                with pytest.raises(ModelProviderError, match="LLM API error"):
                    await client._execute_completion("gpt-4", messages)

    @pytest.mark.asyncio
    async def test_complete_with_circuit_breaker(self):
        """Test completion with circuit breaker."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.usage = Mock(
            spec=["prompt_tokens", "completion_tokens", "total_tokens"]
        )
        mock_response.usage.prompt_tokens = 25
        mock_response.usage.completion_tokens = 25
        mock_response.usage.total_tokens = 50

        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                client = LLMClient(Provider.OPENAI, "gpt-4")

                messages = [{"role": "user", "content": "test"}]
                result = await client.complete(messages)

                assert result.content == "Success"
                assert result.usage["total_tokens"] == 50

    @pytest.mark.asyncio
    async def test_complete_without_circuit_breaker(self):
        """Test completion without circuit breaker."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "No breaker"
        mock_response.usage = Mock(
            spec=["prompt_tokens", "completion_tokens", "total_tokens"]
        )
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 35
        mock_response.usage.total_tokens = 75

        with patch(
            "arbiter_ai.core.llm_client.openai.AsyncOpenAI"
        ) as mock_openai_class:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            mock_openai_class.return_value = mock_client

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
                # Create client without circuit breaker
                client = LLMClient(Provider.OPENAI, "gpt-4", circuit_breaker=None)
                client.circuit_breaker = None  # Explicitly set to None

                messages = [{"role": "user", "content": "test"}]
                result = await client.complete(messages)

                assert result.content == "No breaker"


class TestLLMManager:
    """Test suite for LLMManager."""

    @pytest.mark.asyncio
    async def test_set_pool(self):
        """Test setting custom pool."""
        from arbiter_ai.core.llm_client_pool import LLMClientPool

        custom_pool = LLMClientPool()
        LLMManager.set_pool(custom_pool)

        assert LLMManager._pool is custom_pool

        # Cleanup
        LLMManager._pool = None

    @pytest.mark.asyncio
    async def test_get_pool_creates_default(self):
        """Test that get_pool creates default pool if not set."""
        # Ensure no pool set
        LLMManager._pool = None

        pool = LLMManager.get_pool()
        assert pool is not None

        # Should return same pool on subsequent calls
        pool2 = LLMManager.get_pool()
        assert pool is pool2

        # Cleanup
        LLMManager._pool = None

    @pytest.mark.asyncio
    async def test_get_client_with_openai_key(self):
        """Test get_client with OpenAI API key."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
                with patch(
                    "arbiter_ai.core.llm_client_pool.LLMClient"
                ) as mock_client_class:
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    client = await LLMManager.get_client(model="gpt-4")
                    assert client is not None

    @pytest.mark.asyncio
    async def test_get_client_with_provider_string(self):
        """Test get_client with provider as string."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "test-key"}):
            with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
                with patch(
                    "arbiter_ai.core.llm_client_pool.LLMClient"
                ) as mock_client_class:
                    mock_client = MagicMock()
                    mock_client_class.return_value = mock_client

                    client = await LLMManager.get_client(
                        provider="groq", model="llama-3.1-8b"
                    )
                    assert client is not None

    @pytest.mark.asyncio
    async def test_get_client_no_api_key_raises_error(self):
        """Test get_client raises error when no API key found."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="No API key found"):
                await LLMManager.get_client()

    @pytest.mark.asyncio
    async def test_return_client(self):
        """Test returning client to pool."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test"}):
            with patch("arbiter_ai.core.llm_client.openai.AsyncOpenAI"):
                mock_client = MagicMock()

                # Mock the pool's return_client method
                with patch.object(LLMManager, "get_pool") as mock_get_pool:
                    mock_pool = AsyncMock()
                    mock_get_pool.return_value = mock_pool

                    await LLMManager.return_client(mock_client)
                    mock_pool.return_client.assert_called_once_with(mock_client)

    @pytest.mark.asyncio
    async def test_warm_up(self):
        """Test pool warm-up."""
        with patch.object(LLMManager, "get_pool") as mock_get_pool:
            mock_pool = AsyncMock()
            mock_get_pool.return_value = mock_pool

            await LLMManager.warm_up(
                Provider.OPENAI, "gpt-4", temperature=0.5, connections=3
            )

            mock_pool.warm_up.assert_called_once_with(Provider.OPENAI, "gpt-4", 0.5, 3)

    @pytest.mark.asyncio
    async def test_get_metrics(self):
        """Test getting pool metrics."""
        from arbiter_ai.core.llm_client_pool import ConnectionMetrics

        with patch.object(LLMManager, "get_pool") as mock_get_pool:
            mock_pool = MagicMock()
            mock_metrics = ConnectionMetrics()
            mock_pool.get_metrics.return_value = mock_metrics
            mock_get_pool.return_value = mock_pool

            metrics = LLMManager.get_metrics()
            assert metrics is mock_metrics

    @pytest.mark.asyncio
    async def test_close(self):
        """Test closing manager pool."""
        mock_pool = AsyncMock()
        LLMManager._pool = mock_pool

        await LLMManager.close()

        mock_pool.close.assert_called_once()
        assert LLMManager._pool is None

    @pytest.mark.asyncio
    async def test_close_when_no_pool(self):
        """Test closing when no pool exists."""
        LLMManager._pool = None

        # Should not raise error
        await LLMManager.close()

        assert LLMManager._pool is None


class TestLLMResponse:
    """Test suite for LLMResponse model."""

    def test_response_creation(self):
        """Test creating LLMResponse."""
        response = LLMResponse(
            content="Test content", usage={"total_tokens": 100}, model="gpt-4"
        )

        assert response.content == "Test content"
        assert response.usage["total_tokens"] == 100
        assert response.model == "gpt-4"

    def test_response_default_usage(self):
        """Test LLMResponse with default usage."""
        response = LLMResponse(content="Test", model="gpt-4")

        assert response.usage == {}

    @pytest.mark.asyncio
    async def test_cached_tokens_handling(self):
        """Test handling of cached_tokens in usage response."""
        client = LLMClient(provider=Provider.OPENAI, model="gpt-4o", api_key="test-key")

        # Mock response with cached tokens (Anthropic-style)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test response"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        # Add prompt_tokens_details with cached_tokens
        mock_response.usage.prompt_tokens_details = MagicMock()
        mock_response.usage.prompt_tokens_details.cached_tokens = 80

        with patch.object(
            client.client.chat.completions,
            "create",
            new=AsyncMock(return_value=mock_response),
        ):
            response = await client.complete(
                [{"role": "user", "content": "Test prompt"}]
            )

            assert response.usage["cached_tokens"] == 80
            assert response.usage["prompt_tokens"] == 100

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"OPENAI_API_KEY": "fallback-key"}, clear=True)
    async def test_fallback_client_creation_for_unsupported_provider(self):
        """Test fallback to OpenAI client for providers not in PROVIDER_URLS."""
        # ANTHROPIC is not in PROVIDER_URLS, so it should fallback
        client = LLMClient(
            provider=Provider.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

        # The client should be created (fallback path line 196)
        assert client.client is not None
