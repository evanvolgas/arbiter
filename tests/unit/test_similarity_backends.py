"""Unit tests for similarity backends (LLM and FAISS)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter_ai.evaluators.semantic import SemanticEvaluator
from arbiter_ai.evaluators.similarity_backends import (
    FAISSSimilarityBackend,
    LLMSimilarityBackend,
    SimilarityResult,
)
from tests.conftest import MockAgentResult

# Check if sentence-transformers is available
try:
    import sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

# Skip marker for FAISS tests
requires_faiss = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="Requires sentence-transformers (pip install arbiter[scale])",
)


class TestLLMSimilarityBackend:
    """Test suite for LLMSimilarityBackend."""

    def test_backend_initialization(self, mock_llm_client):
        """Test LLM backend initialization."""
        backend = LLMSimilarityBackend(mock_llm_client)
        assert backend.llm_client == mock_llm_client

    @pytest.mark.asyncio
    async def test_compute_similarity_empty_text(self, mock_llm_client):
        """Test LLM backend with empty text."""
        backend = LLMSimilarityBackend(mock_llm_client)

        with pytest.raises(ValueError) as exc_info:
            await backend.compute_similarity("", "test")
        assert "non-empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            await backend.compute_similarity("test", "")
        assert "non-empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_compute_similarity_success(self, mock_llm_client):
        """Test LLM backend compute_similarity with mocked Agent."""
        from pydantic import BaseModel, Field

        # Create mock response matching SemanticResponse structure
        class MockSemanticResponse(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            confidence: float = Field(default=0.85, ge=0.0, le=1.0)
            explanation: str
            key_differences: list[str] = Field(default_factory=list)
            key_similarities: list[str] = Field(default_factory=list)

        mock_response = MockSemanticResponse(
            score=0.92,
            confidence=0.88,
            explanation="The texts are very similar",
            key_similarities=["Both discuss testing", "Same topic"],
            key_differences=["Different wording"],
        )

        # Mock the Agent and its result
        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        # Mock llm_client.create_agent to return our mock agent
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        backend = LLMSimilarityBackend(mock_llm_client)
        result = await backend.compute_similarity(
            "This is a test", "This is also a test"
        )

        # Verify result
        assert isinstance(result, SimilarityResult)
        assert result.score == 0.92
        assert result.confidence == 0.88
        assert "The texts are very similar" in result.explanation
        assert "Both discuss testing" in result.explanation
        assert "Different wording" in result.explanation
        assert result.metadata["backend"] == "llm"

        # Verify create_agent was called correctly
        mock_llm_client.create_agent.assert_called_once()
        call_args = mock_llm_client.create_agent.call_args
        assert "semantic similarity" in call_args[0][0]  # system_prompt

    @pytest.mark.asyncio
    async def test_compute_similarity_no_key_differences(self, mock_llm_client):
        """Test LLM backend with response containing no key differences or similarities."""
        from pydantic import BaseModel, Field

        class MockSemanticResponse(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            confidence: float = Field(default=0.85, ge=0.0, le=1.0)
            explanation: str
            key_differences: list[str] = Field(default_factory=list)
            key_similarities: list[str] = Field(default_factory=list)

        mock_response = MockSemanticResponse(
            score=0.75,
            confidence=0.80,
            explanation="Basic explanation",
            key_similarities=[],
            key_differences=[],
        )

        mock_agent_result = MockAgentResult(mock_response)
        mock_agent = AsyncMock()
        mock_agent.run = AsyncMock(return_value=mock_agent_result)

        # Mock llm_client.create_agent to return our mock agent
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        backend = LLMSimilarityBackend(mock_llm_client)
        result = await backend.compute_similarity("Text A", "Text B")

        # Should only contain base explanation, no similarity/difference sections
        assert result.explanation == "Basic explanation"
        assert "Key Similarities" not in result.explanation
        assert "Key Differences" not in result.explanation


class TestFAISSSimilarityBackend:
    """Test suite for FAISSSimilarityBackend."""

    def test_backend_initialization_missing_dependency(self):
        """Test FAISS backend initialization without sentence-transformers."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError) as exc_info:
                FAISSSimilarityBackend()
            assert "pip install arbiter[scale]" in str(exc_info.value)

    @requires_faiss
    def test_backend_initialization_success(self):
        """Test FAISS backend initialization with sentence-transformers."""
        backend = FAISSSimilarityBackend()
        assert backend.model is not None
        assert backend.model_name == "all-MiniLM-L6-v2"

    @requires_faiss
    def test_backend_custom_model(self):
        """Test FAISS backend with custom model."""
        backend = FAISSSimilarityBackend(model_name="all-MiniLM-L6-v2")
        assert backend.model_name == "all-MiniLM-L6-v2"

    @requires_faiss
    @pytest.mark.asyncio
    async def test_compute_similarity_identical_texts(self):
        """Test FAISS backend with identical texts (should be ~1.0)."""
        backend = FAISSSimilarityBackend()

        result = await backend.compute_similarity(
            "This is a test sentence", "This is a test sentence"
        )

        assert isinstance(result, SimilarityResult)
        assert result.score > 0.99  # Should be very close to 1.0
        assert result.confidence == 0.95  # FAISS is deterministic
        assert result.metadata["backend"] == "faiss"
        assert "embedding_dim" in result.metadata

    @requires_faiss
    @pytest.mark.asyncio
    async def test_compute_similarity_similar_texts(self):
        """Test FAISS backend with similar texts."""
        backend = FAISSSimilarityBackend()

        result = await backend.compute_similarity(
            "Paris is the capital of France", "The capital of France is Paris"
        )

        assert isinstance(result, SimilarityResult)
        assert 0.7 < result.score < 1.0  # Should be high similarity
        assert result.metadata["backend"] == "faiss"
        assert result.metadata["model"] == "all-MiniLM-L6-v2"

    @requires_faiss
    @pytest.mark.asyncio
    async def test_compute_similarity_different_texts(self):
        """Test FAISS backend with completely different texts."""
        backend = FAISSSimilarityBackend()

        result = await backend.compute_similarity(
            "The weather is nice today", "Python is a programming language"
        )

        assert isinstance(result, SimilarityResult)
        assert result.score < 0.5  # Should be low similarity
        assert result.confidence == 0.95

    @requires_faiss
    @pytest.mark.asyncio
    async def test_compute_similarity_empty_text(self):
        """Test FAISS backend with empty text."""
        backend = FAISSSimilarityBackend()

        with pytest.raises(ValueError) as exc_info:
            await backend.compute_similarity("", "test")
        assert "non-empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            await backend.compute_similarity("test", "")
        assert "non-empty" in str(exc_info.value)

    @requires_faiss
    @pytest.mark.asyncio
    async def test_compute_similarity_metadata(self):
        """Test FAISS backend metadata."""
        backend = FAISSSimilarityBackend()

        result = await backend.compute_similarity("test1", "test2")

        assert "backend" in result.metadata
        assert "model" in result.metadata
        assert "embedding_dim" in result.metadata
        assert "raw_cosine" in result.metadata
        assert result.metadata["backend"] == "faiss"


class TestSemanticEvaluatorWithBackends:
    """Test SemanticEvaluator with different backends."""

    def test_default_backend_is_llm(self, mock_llm_client):
        """Test that default backend is LLM."""
        evaluator = SemanticEvaluator(mock_llm_client)
        assert evaluator.backend_type == "llm"
        assert isinstance(evaluator._similarity_backend, LLMSimilarityBackend)

    def test_llm_backend_explicit(self, mock_llm_client):
        """Test explicit LLM backend selection."""
        evaluator = SemanticEvaluator(mock_llm_client, backend="llm")
        assert evaluator.backend_type == "llm"
        assert isinstance(evaluator._similarity_backend, LLMSimilarityBackend)

    @requires_faiss
    def test_faiss_backend_explicit(self, mock_llm_client):
        """Test explicit FAISS backend selection."""
        evaluator = SemanticEvaluator(mock_llm_client, backend="faiss")
        assert evaluator.backend_type == "faiss"
        assert isinstance(evaluator._similarity_backend, FAISSSimilarityBackend)

    def test_invalid_backend_raises_error(self, mock_llm_client):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SemanticEvaluator(mock_llm_client, backend="invalid")
        assert "Invalid backend" in str(exc_info.value)
        assert "llm" in str(exc_info.value)
        assert "faiss" in str(exc_info.value)

    @requires_faiss
    @pytest.mark.asyncio
    async def test_faiss_backend_evaluation(self, mock_llm_client):
        """Test evaluation with FAISS backend."""
        evaluator = SemanticEvaluator(mock_llm_client, backend="faiss")

        score = await evaluator.evaluate(
            output="Paris is the capital of France",
            reference="The capital of France is Paris",
        )

        assert 0.0 <= score.value <= 1.0
        assert score.confidence == 0.95  # FAISS is deterministic
        assert score.metadata["backend"] == "faiss"
        assert "embedding_dim" in score.metadata

    @requires_faiss
    @pytest.mark.asyncio
    async def test_faiss_backend_without_reference_raises_error(self, mock_llm_client):
        """Test that FAISS backend requires reference text."""
        evaluator = SemanticEvaluator(mock_llm_client, backend="faiss")

        with pytest.raises(ValueError) as exc_info:
            await evaluator.evaluate(output="Test output without reference")

        assert "requires a reference" in str(exc_info.value)
        assert "LLM backend" in str(exc_info.value)

    @requires_faiss
    @pytest.mark.asyncio
    async def test_faiss_backend_deterministic(self, mock_llm_client):
        """Test that FAISS backend is deterministic."""
        evaluator = SemanticEvaluator(mock_llm_client, backend="faiss")

        text1 = "This is a test sentence for determinism"
        text2 = "This is a test sentence for verification"

        # Run same comparison twice
        score1 = await evaluator.evaluate(output=text1, reference=text2)
        score2 = await evaluator.evaluate(output=text1, reference=text2)

        # Should get exact same score (deterministic)
        assert score1.value == score2.value
        assert score1.confidence == score2.confidence

    @pytest.mark.asyncio
    async def test_llm_backend_evaluation_with_reference(
        self, mock_llm_client, mock_agent
    ):
        """Test that LLM backend still works after adding backend support."""
        from pydantic import BaseModel, Field

        class MockSemanticResponse(BaseModel):
            score: float = Field(ge=0.0, le=1.0)
            confidence: float = 0.85
            explanation: str
            key_differences: list[str] = []
            key_similarities: list[str] = []

        mock_response = MockSemanticResponse(
            score=0.92,
            confidence=0.88,
            explanation="Very similar",
            key_similarities=["test"],
            key_differences=[],
        )

        mock_result = MockAgentResult(mock_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        # Default LLM backend
        evaluator = SemanticEvaluator(mock_llm_client)

        score = await evaluator.evaluate(
            output="test output", reference="test reference"
        )

        assert score.value == 0.92
        assert score.confidence == 0.88
        assert len(evaluator.interactions) == 1


class TestSimilarityResult:
    """Test suite for SimilarityResult model."""

    def test_result_creation(self):
        """Test creating a SimilarityResult."""
        result = SimilarityResult(
            score=0.85,
            confidence=0.9,
            explanation="Test explanation",
            metadata={"backend": "llm"},
        )

        assert result.score == 0.85
        assert result.confidence == 0.9
        assert result.explanation == "Test explanation"
        assert result.metadata["backend"] == "llm"

    def test_result_defaults(self):
        """Test SimilarityResult default values."""
        result = SimilarityResult(score=0.8)

        assert result.score == 0.8
        assert result.confidence == 0.85  # Default
        assert result.explanation == ""  # Default
        assert result.metadata == {}  # Default

    def test_result_score_validation(self):
        """Test that score must be between 0 and 1."""
        # Valid scores
        SimilarityResult(score=0.0)
        SimilarityResult(score=1.0)
        SimilarityResult(score=0.5)

        # Invalid scores
        with pytest.raises(Exception):  # Pydantic validation error
            SimilarityResult(score=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            SimilarityResult(score=1.1)
