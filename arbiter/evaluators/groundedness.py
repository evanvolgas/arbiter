"""Groundedness evaluator for validating RAG system outputs.

This evaluator assesses whether LLM outputs are properly grounded in source
documents by identifying statements and verifying they can be attributed to
provided sources. It's essential for RAG systems to prevent hallucinations
and ensure outputs are based on retrieved context.

## When to Use:

- Validating RAG (Retrieval-Augmented Generation) outputs
- Ensuring AI responses are grounded in provided sources
- Detecting hallucinations in context-based generation
- Verifying citation accuracy and source attribution
- Quality assurance for document-based question answering

## Example:

    >>> evaluator = GroundednessEvaluator(llm_client)
    >>> score = await evaluator.evaluate(
    ...     output="Paris is the capital of France with population of 2.2M",
    ...     reference="Paris, the capital city of France, has a population of 2.16 million"
    ... )
    >>> print(f"Groundedness score: {score.value:.2f}")
    >>> print(f"Ungrounded: {score.metadata.get('ungrounded_statements', [])}")
    Groundedness score: 0.5
    Ungrounded statements: ['Paris has population of 2.2M']
"""

from typing import Dict, List, Optional, Type, cast

from pydantic import BaseModel, Field

from ..core.models import Score
from .base import BasePydanticEvaluator

__all__ = ["GroundednessEvaluator", "GroundednessResponse"]


class GroundednessResponse(BaseModel):
    """Structured response for groundedness evaluation."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Groundedness score (0=all statements ungrounded, 1=all statements grounded in sources)",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence in this groundedness assessment",
    )
    explanation: str = Field(
        ..., description="Explanation of the groundedness assessment and reasoning"
    )
    grounded_statements: List[str] = Field(
        default_factory=list,
        description="Statements that are properly supported by source documents",
    )
    ungrounded_statements: List[str] = Field(
        default_factory=list,
        description="Statements that lack source support or contradict sources",
    )
    citations: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from statements to their supporting source text",
    )


class GroundednessEvaluator(BasePydanticEvaluator):
    """Evaluates whether outputs are grounded in source documents.

    This evaluator validates RAG system outputs by checking if statements
    can be attributed to provided source documents. It identifies grounded
    vs ungrounded statements and tracks source citations.

    The evaluator:
    - Extracts statements from the output
    - Maps each statement to supporting source text (if available)
    - Categorizes statements as grounded or ungrounded
    - Provides a score based on the ratio of grounded to total statements
    - Returns detailed citation mappings for transparency

    Modes:
    - With reference: Verify all statements are grounded in reference sources
    - With criteria: Focus on specific types of statements (facts, claims, etc.)

    Example:
        >>> # Create evaluator
        >>> from arbiter import LLMManager
        >>> client = await LLMManager.get_client(model="gpt-4o")
        >>> evaluator = GroundednessEvaluator(client)
        >>>
        >>> # Validate RAG output against source documents
        >>> score = await evaluator.evaluate(
        ...     output="The Eiffel Tower was built in 1889 and is 324 meters tall",
        ...     reference="The Eiffel Tower, built in 1889, stands 300 meters tall"
        ... )
        >>>
        >>> print(f"Groundedness: {score.value:.2f}")
        >>> print(f"Ungrounded: {score.metadata['ungrounded_statements']}")
        >>> print(f"Citations: {score.metadata['citations']}")
        >>>
        >>> # Check LLM interactions
        >>> interactions = evaluator.get_interactions()
        >>> print(f"Made {len(interactions)} LLM calls")
    """

    @property
    def name(self) -> str:
        """Return evaluator identifier."""
        return "groundedness"

    def _get_system_prompt(self) -> str:
        """Get system prompt defining groundedness evaluation approach."""
        return """You are an expert in source attribution and groundedness evaluation for RAG systems.

Your task is to evaluate whether LLM outputs are properly grounded in source documents by:

1. EXTRACTING STATEMENTS: Identify all factual statements and claims in the output
2. VERIFYING SOURCES: Check if each statement can be attributed to the provided sources
3. CATEGORIZING STATEMENTS:
   - Grounded: Statements directly supported by source text (can cite specific source)
   - Ungrounded: Statements not supported by sources, contradicting sources, or hallucinated

4. CITATION MAPPING: For grounded statements, identify the specific source text that supports them

5. SCORING: Calculate groundedness score based on the ratio:
   score = grounded_statements / total_statements
   - All statements grounded = score 1.0
   - No statements grounded = score 0.0
   - Partial groundedness = proportional score

Important Guidelines:
- Be strict: A statement is only "grounded" if clearly supported by source text
- Exact match not required: Paraphrasing is acceptable if meaning is preserved
- Detect hallucinations: Flag any statements not supported by sources
- No inference: Don't mark statements as grounded if they require inferential leaps
- Track citations: Map each grounded statement to its supporting source passage
- Handle contradictions: Statements contradicting sources are ungrounded
- Partial support: If only part of a statement is grounded, mark as ungrounded

Provide:
- A groundedness score from 0.0 (completely ungrounded) to 1.0 (fully grounded)
- Your confidence in this assessment
- Clear categorization of each statement
- Citation mapping for grounded statements
- Detailed explanation of your reasoning"""

    def _get_user_prompt(
        self, output: str, reference: Optional[str], criteria: Optional[str]
    ) -> str:
        """Get user prompt for specific evaluation."""
        if not reference:
            return """Error: Groundedness evaluation requires reference source documents.

Please provide source documents in the 'reference' parameter to evaluate groundedness.

The reference should contain the source text(s) that the output should be grounded in."""

        if criteria:
            # Focus on specific types of statements
            return f"""Evaluate the groundedness of this OUTPUT against SOURCE DOCUMENTS, focusing on: {criteria}

OUTPUT (to evaluate):
{output}

SOURCE DOCUMENTS (ground truth):
{reference}

Your task:
1. Extract statements from OUTPUT related to the specified criteria
2. For each statement, check if it's supported by the SOURCE DOCUMENTS
3. Categorize statements as:
   - Grounded: Directly supported by sources (identify supporting text)
   - Ungrounded: Not supported by sources or contradicting sources

4. Create citation mapping: statement -> supporting source text
5. Calculate groundedness score = grounded / total

Pay special attention to the criteria areas when evaluating groundedness."""

        else:
            # Standard groundedness evaluation
            return f"""Evaluate the groundedness of this OUTPUT against SOURCE DOCUMENTS.

OUTPUT (to evaluate):
{output}

SOURCE DOCUMENTS (ground truth):
{reference}

Your task:
1. Extract all factual statements and claims from the OUTPUT
2. For each statement, verify if it's supported by the SOURCE DOCUMENTS
3. Categorize each statement as:
   - Grounded: Directly supported by source text (can cite specific passage)
   - Ungrounded: Not found in sources, contradicts sources, or is hallucinated

4. For grounded statements, identify the specific source text that supports them
5. Calculate groundedness score = grounded_statements / total_statements

Provide your assessment with clear categorization and citation mappings."""

    def _get_response_type(self) -> Type[BaseModel]:
        """Use groundedness response model."""
        return GroundednessResponse

    async def _compute_score(self, response: BaseModel) -> Score:
        """Extract Score from groundedness response."""
        groundedness_response = cast(GroundednessResponse, response)

        # Build detailed explanation
        explanation_parts = [groundedness_response.explanation]

        if (
            hasattr(groundedness_response, "grounded_statements")
            and groundedness_response.grounded_statements
        ):
            explanation_parts.append(
                "\n\nGrounded Statements (Supported by Sources):\n- "
                + "\n- ".join(groundedness_response.grounded_statements)
            )

        if (
            hasattr(groundedness_response, "ungrounded_statements")
            and groundedness_response.ungrounded_statements
        ):
            explanation_parts.append(
                "\n\nUngrounded Statements (Not Supported/Hallucinated):\n- "
                + "\n- ".join(groundedness_response.ungrounded_statements)
            )

        if (
            hasattr(groundedness_response, "citations")
            and groundedness_response.citations
        ):
            citations_text = "\n".join(
                f"- '{stmt}' → '{source}'"
                for stmt, source in groundedness_response.citations.items()
            )
            explanation_parts.append(f"\n\nCitation Mapping:\n{citations_text}")

        full_explanation = "".join(explanation_parts)

        return Score(
            name=self.name,
            value=groundedness_response.score,
            confidence=groundedness_response.confidence,
            explanation=full_explanation,
            metadata={
                "grounded_statements": groundedness_response.grounded_statements,
                "ungrounded_statements": groundedness_response.ungrounded_statements,
                "citations": groundedness_response.citations,
                "grounded_count": len(
                    getattr(groundedness_response, "grounded_statements", [])
                ),
                "ungrounded_count": len(
                    getattr(groundedness_response, "ungrounded_statements", [])
                ),
                "total_statements": len(
                    getattr(groundedness_response, "grounded_statements", [])
                )
                + len(getattr(groundedness_response, "ungrounded_statements", [])),
            },
        )
