"""Unit tests for factuality verification plugins."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from arbiter_ai.evaluators import FactualityEvaluator
from arbiter_ai.verifiers import (
    CitationVerifier,
    FactualityVerifier,
    KnowledgeBaseVerifier,
    SearchVerifier,
    VerificationResult,
)


class TestVerificationResult:
    """Test VerificationResult model."""

    def test_verification_result_creation(self):
        """Test creating a verification result."""
        result = VerificationResult(
            is_verified=True,
            confidence=0.9,
            evidence=["Supporting evidence"],
            explanation="Claim verified",
            source="test_verifier",
        )

        assert result.is_verified is True
        assert result.confidence == 0.9
        assert len(result.evidence) == 1
        assert result.explanation == "Claim verified"
        assert result.source == "test_verifier"

    def test_verification_result_validation(self):
        """Test validation of confidence scores."""
        # Valid confidence
        result = VerificationResult(
            is_verified=True,
            confidence=0.5,
            evidence=[],
            explanation="Test",
            source="test",
        )
        assert result.confidence == 0.5

        # Invalid confidence (below 0)
        with pytest.raises(Exception):
            VerificationResult(
                is_verified=True,
                confidence=-0.1,
                evidence=[],
                explanation="Test",
                source="test",
            )

        # Invalid confidence (above 1)
        with pytest.raises(Exception):
            VerificationResult(
                is_verified=True,
                confidence=1.5,
                evidence=[],
                explanation="Test",
                source="test",
            )


class TestSearchVerifier:
    """Test SearchVerifier with Tavily API.

    Note: These tests mock the Tavily client to avoid requiring the package.
    """

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", False)
    def test_search_verifier_requires_tavily_package(self):
        """Test that SearchVerifier raises ImportError without tavily package."""
        # SearchVerifier will raise ImportError if tavily is not installed
        with pytest.raises(ImportError, match="Tavily package not installed"):
            SearchVerifier(api_key="test_key")

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    @pytest.mark.asyncio
    async def test_search_verifier_with_many_results(self, mock_tavily_client):
        """Test search verification with many supporting results."""
        # Mock search results with multiple results
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {
            "results": [
                {"content": "Paris is indeed the capital of France"},
                {"content": "The capital city of France is Paris"},
                {"content": "Paris, France's capital, is known for..."},
            ]
        }
        mock_tavily_client.return_value = mock_client_instance

        verifier = SearchVerifier(api_key="test_key")
        result = await verifier.verify("Paris is the capital of France")

        assert result.is_verified is True
        assert result.confidence == 0.85
        assert len(result.evidence) == 3
        assert result.source == "tavily_search"
        assert "3 search results" in result.explanation

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    @pytest.mark.asyncio
    async def test_search_verifier_with_few_results(self, mock_tavily_client):
        """Test search verification with few supporting results."""
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {
            "results": [
                {"content": "Paris is the capital of France"},
            ]
        }
        mock_tavily_client.return_value = mock_client_instance

        verifier = SearchVerifier(api_key="test_key", max_results=5)
        result = await verifier.verify("Paris is the capital")

        assert result.is_verified is True
        assert result.confidence == 0.6
        assert len(result.evidence) == 1
        assert result.source == "tavily_search"

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    @pytest.mark.asyncio
    async def test_search_verifier_with_no_results(self, mock_tavily_client):
        """Test search verification with no results found."""
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {"results": []}
        mock_tavily_client.return_value = mock_client_instance

        verifier = SearchVerifier(api_key="test_key")
        result = await verifier.verify("Fake claim that doesn't exist")

        assert result.is_verified is False
        assert result.confidence == 0.3
        assert len(result.evidence) == 0
        assert "No supporting evidence" in result.explanation

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    @pytest.mark.asyncio
    async def test_search_verifier_with_context(self, mock_tavily_client):
        """Test search verification with additional context."""
        mock_client_instance = MagicMock()
        mock_client_instance.search.return_value = {
            "results": [{"content": "Context helps refine search"}]
        }
        mock_tavily_client.return_value = mock_client_instance

        verifier = SearchVerifier(api_key="test_key")
        await verifier.verify("Capital", context="France geography")

        # Verify query includes context
        mock_client_instance.search.assert_called_once()
        call_args = mock_client_instance.search.call_args
        assert "Capital France geography" in call_args[1]["query"]

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    @pytest.mark.asyncio
    async def test_search_verifier_search_failure(self, mock_tavily_client):
        """Test search verification handles search API failures."""
        mock_client_instance = MagicMock()
        mock_client_instance.search.side_effect = Exception("API rate limit exceeded")
        mock_tavily_client.return_value = mock_client_instance

        verifier = SearchVerifier(api_key="test_key")
        result = await verifier.verify("Some claim")

        assert result.is_verified is False
        assert result.confidence == 0.0
        assert len(result.evidence) == 0
        assert "Search verification failed" in result.explanation
        assert "API rate limit" in result.explanation

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    def test_search_verifier_requires_api_key(self, mock_tavily_client):
        """Test that SearchVerifier requires API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Tavily API key required"):
                SearchVerifier()

    @patch("arbiter_ai.verifiers.search_verifier.TAVILY_AVAILABLE", True)
    @patch("arbiter_ai.verifiers.search_verifier.TavilyClient")
    def test_search_verifier_uses_env_var(self, mock_tavily_client):
        """Test that SearchVerifier can use API key from environment."""
        with patch.dict(os.environ, {"TAVILY_API_KEY": "env_test_key"}):
            verifier = SearchVerifier()
            assert verifier.api_key == "env_test_key"


class TestCitationVerifier:
    """Test CitationVerifier."""

    @pytest.mark.asyncio
    async def test_citation_verifier_direct_match(self):
        """Test citation verification with direct substring match."""
        verifier = CitationVerifier()

        result = await verifier.verify(
            claim="Paris is the capital of France",
            context="Paris is the capital of France and its largest city",
        )

        assert result.is_verified is True
        assert result.confidence == 0.95  # High confidence for direct match
        assert len(result.evidence) > 0
        assert result.source == "citation_check"
        assert "directly found" in result.explanation

    @pytest.mark.asyncio
    async def test_citation_verifier_semantic_match(self):
        """Test citation verification with semantic similarity."""
        verifier = CitationVerifier(min_similarity=0.6)

        result = await verifier.verify(
            claim="The Eiffel Tower is in Paris",
            context="Paris has the famous Eiffel Tower landmark",
        )

        assert result.is_verified is True
        assert result.confidence > 0.6  # Above threshold
        assert result.source == "citation_check"
        assert "semantically similar" in result.explanation

    @pytest.mark.asyncio
    async def test_citation_verifier_no_match(self):
        """Test citation verification with contradicting information."""
        verifier = CitationVerifier(min_similarity=0.7)

        # Use completely different content to ensure no match
        result = await verifier.verify(
            claim="The moon is made of cheese",
            context="The Earth has a rocky surface with water and continents",
        )

        assert result.is_verified is False
        assert result.confidence < 0.7
        assert "not sufficiently supported" in result.explanation

    @pytest.mark.asyncio
    async def test_citation_verifier_no_context(self):
        """Test citation verification without context."""
        verifier = CitationVerifier()

        result = await verifier.verify(claim="Test claim", context=None)

        assert result.is_verified is False
        assert result.confidence == 0.0
        assert "No source context provided" in result.explanation


class TestKnowledgeBaseVerifier:
    """Test KnowledgeBaseVerifier with Wikipedia."""

    @pytest.mark.asyncio
    async def test_knowledge_base_verifier_successful(self):
        """Test successful Wikipedia verification."""
        verifier = KnowledgeBaseVerifier()

        # Mock Wikipedia search
        with patch.object(
            verifier, "_search_wikipedia", return_value=["Paris", "France"]
        ):
            # Mock Wikipedia content
            with patch.object(
                verifier,
                "_get_wikipedia_content",
                return_value="Paris is the capital and most populous city of France",
            ):
                result = await verifier.verify(
                    claim="Paris is the capital of France", context=None
                )

                assert result.is_verified is True
                assert result.confidence >= 0.7
                assert result.source == "wikipedia"
                assert len(result.evidence) > 0

    @pytest.mark.asyncio
    async def test_knowledge_base_verifier_no_articles(self):
        """Test Wikipedia verification with no articles found."""
        verifier = KnowledgeBaseVerifier()

        # Mock empty search results
        with patch.object(verifier, "_search_wikipedia", return_value=[]):
            result = await verifier.verify(claim="Fake claim", context=None)

            assert result.is_verified is False
            assert result.confidence == 0.3
            assert "No relevant Wikipedia articles" in result.explanation

    @pytest.mark.asyncio
    async def test_knowledge_base_verifier_api_failure(self):
        """Test Wikipedia verification when API fails."""
        verifier = KnowledgeBaseVerifier()

        # Mock API failure
        with patch.object(
            verifier, "_search_wikipedia", side_effect=Exception("API error")
        ):
            result = await verifier.verify(claim="Test claim", context=None)

            assert result.is_verified is False
            assert result.confidence == 0.0
            assert "Wikipedia verification failed" in result.explanation

    def test_knowledge_base_compute_similarity(self):
        """Test similarity computation."""
        verifier = KnowledgeBaseVerifier()

        # Direct match
        similarity = verifier._compute_similarity(
            claim="Paris is the capital of France",
            content="Paris is the capital of France and its largest city",
        )
        assert similarity == 0.95

        # Partial match
        similarity = verifier._compute_similarity(
            claim="Paris capital France", content="Paris is in France"
        )
        assert 0.0 < similarity < 1.0

        # No match
        similarity = verifier._compute_similarity(
            claim="Berlin Germany", content="Paris France"
        )
        assert similarity < 0.5

        # Empty claim words (all words too short)
        similarity = verifier._compute_similarity(claim="a is the", content="Paris")
        assert similarity == 0.0

    @pytest.mark.asyncio
    async def test_knowledge_base_high_similarity(self):
        """Test verification with high similarity (>= 0.7)."""
        verifier = KnowledgeBaseVerifier(max_results=2)

        with patch.object(verifier, "_search_wikipedia", return_value=["Paris"]):
            with patch.object(
                verifier,
                "_get_wikipedia_content",
                return_value="Paris is the capital of France and the largest city",
            ):
                result = await verifier.verify("Paris is the capital of France")

                assert result.is_verified is True
                assert result.confidence == 0.9
                assert "strongly supported" in result.explanation
                assert result.source == "wikipedia"

    @pytest.mark.asyncio
    async def test_knowledge_base_medium_similarity(self):
        """Test verification with medium similarity (>= 0.5, < 0.7)."""
        verifier = KnowledgeBaseVerifier(max_results=1)

        with patch.object(verifier, "_search_wikipedia", return_value=["Paris"]):
            with patch.object(
                verifier,
                "_get_wikipedia_content",
                return_value="Paris located capital France government tourism",
            ):
                result = await verifier.verify("Paris capital France major city")

                assert result.is_verified is True
                assert result.confidence == 0.7
                assert "partially supported" in result.explanation

    @pytest.mark.asyncio
    async def test_knowledge_base_low_similarity(self):
        """Test verification with low similarity (< 0.5)."""
        verifier = KnowledgeBaseVerifier(max_results=1)

        with patch.object(verifier, "_search_wikipedia", return_value=["Berlin"]):
            with patch.object(
                verifier,
                "_get_wikipedia_content",
                return_value="Berlin is the capital of Germany",
            ):
                result = await verifier.verify("Paris is the capital of France")

                assert result.is_verified is False
                assert result.confidence == 0.4
                assert "not well-supported" in result.explanation

    @patch("urllib.request.urlopen")
    def test_knowledge_base_search_wikipedia(self, mock_urlopen):
        """Test _search_wikipedia method with API mocking."""
        verifier = KnowledgeBaseVerifier()

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.read.return_value = (
            b'["paris", ["Paris", "Paris, France"], [], []]'
        )
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = verifier._search_wikipedia("Paris")
        assert results == ["Paris", "Paris, France"]

    @patch("urllib.request.urlopen")
    def test_knowledge_base_search_wikipedia_failure(self, mock_urlopen):
        """Test _search_wikipedia handles API failures."""
        verifier = KnowledgeBaseVerifier()

        # Mock API failure
        mock_urlopen.side_effect = Exception("Network error")

        results = verifier._search_wikipedia("Paris")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_knowledge_base_search_wikipedia_invalid_response(self, mock_urlopen):
        """Test _search_wikipedia handles invalid API responses."""
        verifier = KnowledgeBaseVerifier()

        # Mock invalid response format
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"invalid": "format"}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        results = verifier._search_wikipedia("Paris")
        assert results == []

    @patch("urllib.request.urlopen")
    def test_knowledge_base_get_content(self, mock_urlopen):
        """Test _get_wikipedia_content method with API mocking."""
        verifier = KnowledgeBaseVerifier()

        # Mock successful API response
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"query": {"pages": {"123": {"extract": "Paris is the capital of France"}}}}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        content = verifier._get_wikipedia_content("Paris")
        assert content == "Paris is the capital of France"

    @patch("urllib.request.urlopen")
    def test_knowledge_base_get_content_failure(self, mock_urlopen):
        """Test _get_wikipedia_content handles API failures."""
        verifier = KnowledgeBaseVerifier()

        # Mock API failure
        mock_urlopen.side_effect = Exception("Network error")

        content = verifier._get_wikipedia_content("Paris")
        assert content == ""

    @patch("urllib.request.urlopen")
    def test_knowledge_base_get_content_no_extract(self, mock_urlopen):
        """Test _get_wikipedia_content when article has no extract."""
        verifier = KnowledgeBaseVerifier()

        # Mock response without extract
        mock_response = MagicMock()
        mock_response.read.return_value = b'{"query": {"pages": {"123": {}}}}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        content = verifier._get_wikipedia_content("Paris")
        assert content == ""

    def test_knowledge_base_extract_snippet(self):
        """Test _extract_snippet method."""
        verifier = KnowledgeBaseVerifier()

        # Test snippet extraction with matching sentence
        content = (
            "Paris is a city. It is the capital of France. The Eiffel Tower is there."
        )
        snippet = verifier._extract_snippet("capital of France", content, length=100)
        assert "capital of France" in snippet or "capital" in snippet

        # Test fallback to content truncation when no good match
        content = "Some unrelated text about Berlin and Germany."
        snippet = verifier._extract_snippet("Paris capital", content, length=20)
        assert len(snippet) <= 20
        assert snippet == content[:20]


class TestFactualityEvaluatorWithVerifiers:
    """Integration tests for FactualityEvaluator with verification plugins."""

    @pytest.mark.asyncio
    async def test_factuality_evaluator_without_verifiers(self):
        """Test FactualityEvaluator without verifiers (LLM-only)."""
        mock_client = MagicMock()
        evaluator = FactualityEvaluator(llm_client=mock_client)

        assert evaluator.verifiers == []
        assert hasattr(evaluator, "_current_output")
        assert hasattr(evaluator, "_current_reference")

    @pytest.mark.asyncio
    async def test_factuality_evaluator_with_verifiers(self):
        """Test FactualityEvaluator with verification plugins."""
        mock_client = MagicMock()

        # Create mock verifiers
        mock_verifier1 = AsyncMock(spec=FactualityVerifier)
        mock_verifier1.verify = AsyncMock(
            return_value=VerificationResult(
                is_verified=True,
                confidence=0.9,
                evidence=["Evidence 1"],
                explanation="Verified",
                source="verifier1",
            )
        )

        mock_verifier2 = AsyncMock(spec=FactualityVerifier)
        mock_verifier2.verify = AsyncMock(
            return_value=VerificationResult(
                is_verified=True,
                confidence=0.8,
                evidence=["Evidence 2"],
                explanation="Verified",
                source="verifier2",
            )
        )

        evaluator = FactualityEvaluator(
            llm_client=mock_client, verifiers=[mock_verifier1, mock_verifier2]
        )

        assert len(evaluator.verifiers) == 2

    @pytest.mark.asyncio
    async def test_factuality_evaluator_stores_context(self):
        """Test that FactualityEvaluator stores output and reference."""
        from arbiter_ai.core.models import Score

        mock_client = MagicMock()
        evaluator = FactualityEvaluator(llm_client=mock_client)

        # Mock the parent evaluate to return a Score directly
        mock_score = Score(
            name="factuality",
            value=0.8,
            confidence=0.9,
            explanation="Test",
            metadata={},
        )

        with patch.object(
            evaluator.__class__.__bases__[0], "evaluate", return_value=mock_score
        ):
            await evaluator.evaluate(
                output="Test output", reference="Test reference", criteria=None
            )

            # Verify context was stored
            assert evaluator._current_output == "Test output"
            assert evaluator._current_reference == "Test reference"

    def test_factuality_evaluator_init_signature(self):
        """Test FactualityEvaluator accepts verifiers parameter."""
        mock_client = MagicMock()
        mock_verifier = MagicMock(spec=FactualityVerifier)

        # Should accept verifiers parameter
        evaluator = FactualityEvaluator(
            llm_client=mock_client, verifiers=[mock_verifier]
        )

        assert evaluator.verifiers == [mock_verifier]

        # Should default to empty list
        evaluator2 = FactualityEvaluator(llm_client=mock_client)
        assert evaluator2.verifiers == []

    @pytest.mark.asyncio
    async def test_factuality_evaluator_with_verification_integration(
        self, mock_llm_client, mock_agent
    ):
        """Test full integration of FactualityEvaluator with external verifiers."""
        from arbiter_ai.evaluators.factuality import FactualityResponse
        from tests.conftest import MockAgentResult

        # Create mock verifiers that will be called
        mock_verifier = AsyncMock(spec=FactualityVerifier)
        mock_verifier.verify = AsyncMock(
            return_value=VerificationResult(
                is_verified=True,
                confidence=0.9,
                evidence=["Web search confirms this"],
                explanation="Claim verified by search",
                source="web_search",
            )
        )

        # Mock the LLM response with claims
        mock_llm_response = FactualityResponse(
            score=0.7,
            confidence=0.85,
            explanation="LLM evaluation complete",
            factual_claims=["Paris is the capital of France"],
            non_factual_claims=["Paris was founded in 1985"],
            uncertain_claims=["Paris has 10 million residents"],
        )

        # Use MockAgentResult to wrap response
        mock_result = MockAgentResult(mock_llm_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        evaluator = FactualityEvaluator(
            llm_client=mock_llm_client, verifiers=[mock_verifier]
        )

        # Run evaluation - this should trigger verifier calls
        score = await evaluator.evaluate(
            output="Paris is the capital of France, founded in 1985",
            reference="Paris is the capital of France",
        )

        # Verify verifier was called for each claim
        assert mock_verifier.verify.called
        # Should be called 3 times (one for each claim)
        assert mock_verifier.verify.call_count == 3

        # Verify score combines LLM + verification
        # Score should be weighted: (0.7 * 0.5) + (0.9 * 0.5) = 0.8
        assert 0.75 <= score.value <= 0.85

        # Verify metadata includes verification info
        assert score.metadata["verification_used"] is True
        assert score.metadata["verification_count"] > 0
        assert "web_search" in score.metadata["verification_sources"]

    @pytest.mark.asyncio
    async def test_factuality_evaluator_verification_failures(
        self, mock_llm_client, mock_agent
    ):
        """Test that FactualityEvaluator handles verifier failures gracefully."""
        from arbiter_ai.evaluators.factuality import FactualityResponse
        from tests.conftest import MockAgentResult

        # Create mock verifier that fails
        mock_verifier = AsyncMock(spec=FactualityVerifier)
        mock_verifier.verify = AsyncMock(side_effect=Exception("Verification failed"))

        # Mock LLM response
        mock_llm_response = FactualityResponse(
            score=0.7,
            confidence=0.85,
            explanation="LLM evaluation",
            factual_claims=["Some claim"],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_llm_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        evaluator = FactualityEvaluator(
            llm_client=mock_llm_client, verifiers=[mock_verifier]
        )

        # Should not crash despite verifier failure
        score = await evaluator.evaluate(output="Some claim")

        # Should fall back to LLM score only
        assert score.value == 0.7
        assert score.metadata["verification_used"] is False

    @pytest.mark.asyncio
    async def test_factuality_evaluator_no_claims(self, mock_llm_client, mock_agent):
        """Test FactualityEvaluator when there are no claims to verify."""
        from arbiter_ai.evaluators.factuality import FactualityResponse
        from tests.conftest import MockAgentResult

        mock_verifier = AsyncMock(spec=FactualityVerifier)

        # Mock LLM response with no claims
        mock_llm_response = FactualityResponse(
            score=0.5,
            confidence=0.8,
            explanation="No verifiable claims",
            factual_claims=[],
            non_factual_claims=[],
            uncertain_claims=[],
        )

        mock_result = MockAgentResult(mock_llm_response)
        mock_agent.run = AsyncMock(return_value=mock_result)
        mock_llm_client.create_agent = MagicMock(return_value=mock_agent)

        evaluator = FactualityEvaluator(
            llm_client=mock_llm_client, verifiers=[mock_verifier]
        )

        score = await evaluator.evaluate(output="This is opinion, not fact")

        # Verifier should not be called (no claims)
        assert not mock_verifier.verify.called
        # Score should be LLM score only
        assert score.value == 0.5
        assert score.metadata["verification_used"] is False


class TestVerifierIntegration:
    """Integration tests combining multiple verifiers."""

    @pytest.mark.asyncio
    async def test_multiple_verifiers_combined(self):
        """Test using multiple verifiers together."""
        # Create verifiers
        citation_verifier = CitationVerifier()

        # Test with citation verifier
        result1 = await citation_verifier.verify(
            claim="Paris is the capital of France",
            context="Paris is the capital of France",
        )

        assert result1.is_verified is True
        assert result1.source == "citation_check"

    @pytest.mark.asyncio
    async def test_verifier_error_handling(self):
        """Test that verifiers handle errors gracefully using KnowledgeBaseVerifier."""
        verifier = KnowledgeBaseVerifier()

        # Mock API to raise exception
        with patch.object(
            verifier, "_search_wikipedia", side_effect=Exception("Network error")
        ):
            result = await verifier.verify(claim="Test", context=None)

            # Should return unverified result, not crash
            assert result.is_verified is False
            assert result.confidence == 0.0
            assert "failed" in result.explanation.lower()
