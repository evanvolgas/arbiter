"""Provider-agnostic LLM client with automatic provider detection and PydanticAI integration.

This module provides a unified interface for interacting with various LLM providers
(OpenAI, Anthropic, Google Gemini, Groq) while handling provider-specific details
internally. It integrates with PydanticAI for structured outputs.

## Key Features:

- **Automatic Provider Detection**: Infers provider from model name
- **Unified Interface**: Same API regardless of provider
- **Structured Outputs**: PydanticAI integration for type-safe responses
- **Retry Logic**: Built-in retry for transient failures
- **Unified Provider Support**: All providers route through PydanticAI

## Supported Providers:

All providers and models supported by PydanticAI work with Arbiter:

- **OpenAI**: Any GPT model (GPT-4o, GPT-4o-mini, o1, o3-mini, etc.)
- **Anthropic**: Any Claude model (Claude Sonnet 4.5, Claude 3.5/3 Opus/Sonnet/Haiku)
- **Google**: Any Gemini model (Gemini 2.0 Flash, Gemini 1.5 Pro/Flash)
- **Groq**: Fast inference (Llama, Mixtral, Gemma models)
- **Mistral**: Mistral Large/Medium/Small, Mixtral models
- **Cohere**: Command R/R+, Embed models

Model names are passed directly to the provider - use any model the provider offers.

## Usage:

    >>> from arbiter_ai.core.llm_client import LLMManager
    >>>
    >>> # Automatic provider detection
    >>> client = await LLMManager.get_client(model="gpt-4")
    >>> response = await client.complete(messages)
    >>>
    >>> # Structured output with PydanticAI
    >>> agent = client.create_agent(
    ...     system_prompt="You are an evaluation assistant",
    ...     result_type=EvaluationResponse
    ... )
    >>> result = await agent.run("Evaluate this output")

## Environment Variables:

Set API keys for each provider:
- OPENAI_API_KEY
- ANTHROPIC_API_KEY
- GOOGLE_API_KEY (for Gemini models)
- GROQ_API_KEY
- MISTRAL_API_KEY
- COHERE_API_KEY

The client will automatically use the appropriate key based on the model.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

import logfire
from dotenv import load_dotenv
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings

from .circuit_breaker import CircuitBreaker
from .exceptions import ModelProviderError
from .retry import RETRY_STANDARD, with_retry
from .types import Provider

if TYPE_CHECKING:
    from .llm_client_pool import ConnectionMetrics, LLMClientPool

# Load environment variables
load_dotenv()

# Configure logfire if token is available
if os.getenv("LOGFIRE_TOKEN"):
    logfire.configure(token=os.getenv("LOGFIRE_TOKEN"))

__all__ = ["LLMClient", "LLMManager", "LLMResponse", "Provider"]


class LLMResponse(BaseModel):
    """Standardized response from any LLM provider.

    Provides a consistent response format regardless of which provider
    is used, making it easy to switch between providers.

    Attributes:
        content: The generated text from the LLM
        usage: Token usage statistics with keys like 'prompt_tokens',
            'completion_tokens', and 'total_tokens'
        model: The actual model name used for generation

    Example:
        >>> response = LLMResponse(
        ...     content="Generated evaluation here",
        ...     usage={"prompt_tokens": 10, "completion_tokens": 20},
        ...     model="gpt-4"
        ... )
    """

    content: str
    usage: Dict[str, int] = Field(default_factory=dict)
    model: str


class LLMClient:
    """Unified client for interacting with multiple LLM providers.

    This client abstracts away provider-specific details and provides a
    consistent interface for all supported LLMs. It handles:
    - Provider-specific API endpoints and authentication
    - Model name mapping between providers
    - Retry logic for transient failures
    - Integration with PydanticAI for structured outputs

    All providers route through PydanticAI for consistency and unified
    message handling across all LLM providers.

    Example:
        >>> # Create client for OpenAI
        >>> client = LLMClient(
        ...     provider=Provider.OPENAI,
        ...     model="gpt-4",
        ...     temperature=0.7
        ... )
        >>>
        >>> # Use with messages
        >>> messages = [{"role": "user", "content": "Evaluate this"}]
        >>> response = await client.complete(messages)
        >>> print(response.content)

    Attributes:
        provider: The LLM provider being used
        model: The model name requested by the user
        temperature: Generation temperature (0.0-2.0)
    """

    # Model mappings
    MODEL_MAPPINGS = {
        Provider.OPENAI: {
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4o": "gpt-4o",
            "gpt-4": "gpt-4",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        },
        Provider.GROQ: {
            "gpt-4o-mini": "llama-3.1-8b-instant",
            "gpt-4": "llama-3.1-70b-versatile",
            "mixtral": "mixtral-8x7b-32768",
        },
    }

    def __init__(
        self,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        api_key: Optional[str] = None,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self._api_key = api_key or self._get_api_key(provider)

        # Initialize circuit breaker with default settings if not provided
        # Can be disabled by passing circuit_breaker=None explicitly
        self.circuit_breaker = (
            circuit_breaker
            if circuit_breaker is not None
            else CircuitBreaker(
                failure_threshold=5,
                timeout=60.0,
                half_open_max_calls=1,
            )
        )

        # All providers route through PydanticAI for consistency
        # PydanticAI handles provider-specific details internally

    @staticmethod
    def _get_api_key(provider: Provider) -> Optional[str]:
        """Retrieve API key for the specified provider from environment.

        Looks for provider-specific environment variables containing
        API keys. This allows users to configure multiple providers
        without code changes.

        Args:
            provider: The provider to get API key for

        Returns:
            API key string if found in environment, None otherwise

        Environment Variables:
            - OPENAI_API_KEY for OpenAI
            - ANTHROPIC_API_KEY for Anthropic
            - GOOGLE_API_KEY for Google Gemini
            - GROQ_API_KEY for Groq
            - MISTRAL_API_KEY for Mistral
            - COHERE_API_KEY for Cohere
        """
        env_keys = {
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.GOOGLE: "GOOGLE_API_KEY",
            Provider.GROQ: "GROQ_API_KEY",
            Provider.MISTRAL: "MISTRAL_API_KEY",
            Provider.COHERE: "COHERE_API_KEY",
        }
        env_var = env_keys.get(provider)
        return os.getenv(env_var) if env_var else None

    def _get_provider_model(self) -> str:
        """Map generic model names to provider-specific identifiers.

        Different providers use different names for similar models.
        This method translates common model names to provider-specific
        ones, allowing users to use familiar names across providers.

        Returns:
            Provider-specific model identifier

        Example:
            - "gpt-4" on Groq maps to "llama-3.1-70b-versatile"
            - "gpt-4o-mini" on Groq maps to "llama-3.1-8b-instant"
        """
        mappings = self.MODEL_MAPPINGS.get(self.provider, {})
        return mappings.get(self.model, self.model)

    def create_agent(self, system_prompt: str, result_type: type[BaseModel]) -> Agent:
        """Create a PydanticAI agent for structured outputs.

        Args:
            system_prompt: System prompt defining the agent's behavior
            result_type: Pydantic model defining the expected response structure

        Returns:
            Configured PydanticAI agent

        Example:
            >>> class ScoreResponse(BaseModel):
            ...     score: float
            ...     explanation: str
            >>>
            >>> agent = client.create_agent(
            ...     system_prompt="You are an evaluator",
            ...     result_type=ScoreResponse
            ... )
            >>> result = await agent.run("Evaluate this output")
        """
        # Map provider and model to PydanticAI format
        provider_model = self._get_provider_model()

        # Create model string for PydanticAI using provider enum value
        # This creates strings like "openai:gpt-4o", "google:gemini-pro", etc.
        model_str = f"{self.provider.value}:{provider_model}"

        # Create agent with structured output
        return Agent(
            model=model_str,
            output_type=result_type,  # type: ignore[arg-type]
            system_prompt=system_prompt,
            model_settings=ModelSettings(
                temperature=self.temperature,
            ),
        )

    async def _execute_completion(
        self,
        provider_model: str,
        typed_messages: List[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> LLMResponse:
        """Internal method to execute the actual API call.

        This method is separated to allow circuit breaker wrapping.

        Args:
            provider_model: Provider-specific model name
            typed_messages: Typed list of messages
            **kwargs: Additional API parameters

        Returns:
            Standardized LLM response

        Raises:
            ModelProviderError: If the API call fails
        """
        # All providers route through PydanticAI for consistency
        return await self._execute_pydanticai_completion(
            provider_model, typed_messages, **kwargs
        )

    async def _execute_pydanticai_completion(
        self,
        provider_model: str,
        typed_messages: List[ChatCompletionMessageParam],
        **kwargs: Any,
    ) -> LLMResponse:
        """Execute completion via PydanticAI for all providers."""
        try:
            # Build simple system prompt from first system message if present
            system_prompt = ""
            user_messages: List[str] = []
            for msg in typed_messages:
                role = msg.get("role")
                content = msg.get("content") or ""
                if role == "system" and not system_prompt:
                    system_prompt = str(content)
                elif role == "user":
                    user_messages.append(str(content))
                elif role == "assistant":
                    # Preserve prior assistant messages inline for context
                    user_messages.append(f"[assistant]\n{content}")
                else:
                    user_messages.append(str(content))

            # Fallback system prompt for providers that require it
            if not system_prompt:
                system_prompt = "You are a helpful assistant."

            # Flatten user messages into a single prompt; PydanticAI Agent.run
            # expects a single input payload.
            user_prompt = "\n\n".join(user_messages) if user_messages else ""

            # Create a lightweight agent for this call
            agent = Agent(
                model=f"{self.provider.value}:{provider_model}",
                output_type=str,
                system_prompt=system_prompt,
                model_settings=ModelSettings(
                    temperature=kwargs.get("temperature", self.temperature),
                ),
            )

            result = await agent.run(user_prompt)

            usage_dict: Dict[str, int] = {}
            raw = getattr(result, "raw_response", None)
            if raw is not None:
                usage = getattr(raw, "usage", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    total_tokens = getattr(usage, "total_tokens", None)
                    if prompt_tokens is not None:
                        usage_dict["prompt_tokens"] = prompt_tokens
                    if completion_tokens is not None:
                        usage_dict["completion_tokens"] = completion_tokens
                    if total_tokens is not None:
                        usage_dict["total_tokens"] = total_tokens

            content = result.output if hasattr(result, "output") else ""
            return LLMResponse(
                content=str(content),
                usage=usage_dict,
                model=f"{self.provider.value}:{provider_model}",
            )
        except Exception as e:
            error_msg = str(e).lower()
            details = {"provider": self.provider.value}

            if "rate limit" in error_msg:
                details["error_code"] = "rate_limit"
                raise ModelProviderError("Rate limit exceeded", details=details) from e
            elif (
                "api key" in error_msg
                or "unauthorized" in error_msg
                or "authentication" in error_msg
            ):
                details["error_code"] = "authentication"
                raise ModelProviderError(
                    "Authentication failed", details=details
                ) from e
            else:
                raise ModelProviderError(
                    f"LLM API error via PydanticAI: {e!s}", details=details
                ) from e

    @with_retry(RETRY_STANDARD)
    async def complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> LLMResponse:
        """Complete a conversation with the LLM with circuit breaker protection.

        The circuit breaker prevents cascading failures by temporarily blocking
        requests when too many failures occur. This protects against LLM provider
        outages and degraded performance.

        Args:
            messages: List of conversation messages
            **kwargs: Additional parameters to pass to the API

        Returns:
            Standardized LLM response

        Raises:
            ModelProviderError: If the API call fails
            CircuitBreakerOpenError: If circuit breaker is open
        """
        provider_model = self._get_provider_model()

        # Cast messages to the expected type
        typed_messages = cast(List[ChatCompletionMessageParam], messages)

        # Wrap API call with circuit breaker if available
        if self.circuit_breaker:
            return await self.circuit_breaker.call(
                self._execute_completion,
                provider_model,
                typed_messages,
                **kwargs,
            )
        else:
            # No circuit breaker, call directly
            return await self._execute_completion(
                provider_model, typed_messages, **kwargs
            )


class LLMManager:
    """Manager for LLM clients with connection pooling.

    Provides a centralized interface for obtaining and managing LLM
    clients with automatic connection pooling for improved performance.

    Example:
        >>> # Get a client
        >>> client = await LLMManager.get_client(model="gpt-4")
        >>>
        >>> # Use the client
        >>> response = await client.complete(messages)
        >>>
        >>> # Return when done (optional - pool handles this)
        >>> await LLMManager.return_client(client)
        >>>
        >>> # Get pool metrics
        >>> metrics = LLMManager.get_metrics()
        >>> print(f"Active connections: {metrics.active_connections}")
    """

    _pool: Optional["LLMClientPool"] = None

    @classmethod
    def set_pool(cls, pool: "LLMClientPool") -> None:
        """Set the connection pool for the manager."""
        cls._pool = pool

    @classmethod
    def get_pool(cls) -> "LLMClientPool":
        """Get the connection pool, creating one if needed."""
        if cls._pool is None:
            from .llm_client_pool import get_global_pool

            cls._pool = get_global_pool()
        return cls._pool

    @classmethod
    async def get_client(
        cls,
        provider: Optional[Union[str, Provider]] = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key: Optional[str] = None,
    ) -> LLMClient:
        """Get a client from the connection pool.

        Args:
            provider: LLM provider (auto-detected if None)
            model: Model name
            temperature: Generation temperature
            api_key: API key override

        Returns:
            LLM client ready for use

        Raises:
            ValueError: If no API key found for any provider
        """
        # Auto-detect provider if not specified
        if provider is None:
            if os.getenv("OPENAI_API_KEY"):
                provider = Provider.OPENAI
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = Provider.ANTHROPIC
            elif os.getenv("GROQ_API_KEY"):
                provider = Provider.GROQ
            elif os.getenv("GOOGLE_API_KEY"):
                provider = Provider.GOOGLE
            elif os.getenv("MISTRAL_API_KEY"):
                provider = Provider.MISTRAL
            elif os.getenv("COHERE_API_KEY"):
                provider = Provider.COHERE
            else:
                raise ValueError(
                    "No API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                    "GROQ_API_KEY, GOOGLE_API_KEY, MISTRAL_API_KEY, COHERE_API_KEY"
                )

        # Convert string to Provider enum
        if isinstance(provider, str):
            provider = Provider(provider.lower())

        # Get client from pool
        pool = cls.get_pool()
        return await pool.get_client(provider, model, temperature, api_key)

    @classmethod
    async def return_client(cls, client: LLMClient) -> None:
        """Return a client to the connection pool."""
        pool = cls.get_pool()
        await pool.return_client(client)

    @classmethod
    async def warm_up(
        cls,
        provider: Provider,
        model: str,
        temperature: float = 0.7,
        connections: int = 1,
    ) -> None:
        """Pre-create connections for a provider/model combination."""
        pool = cls.get_pool()
        await pool.warm_up(provider, model, temperature, connections)

    @classmethod
    def get_metrics(cls) -> "ConnectionMetrics":
        """Get connection pool metrics."""
        pool = cls.get_pool()
        return pool.get_metrics()

    @classmethod
    async def close(cls) -> None:
        """Close the connection pool."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
