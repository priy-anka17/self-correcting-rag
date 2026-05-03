"""Self-Correcting RAG Pipeline — orchestrates retrieval, generation, and critique."""

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from langchain_community.vectorstores import Chroma

from core.retriever import retrieve, retrieve_with_requery, RetrievalResult
from core.generator import generate_answer, GenerationResult
from core.critic import critique_answer, CriticResult, Verdict
from config import MAX_RETRIES


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


@dataclass
class IterationLog:
    """Log entry for a single iteration of the self-correction loop."""

    iteration: int
    query_used: str
    retrieval: RetrievalResult
    generation: GenerationResult
    critique: CriticResult

    @property
    def summary(self) -> str:
        return (
            f"Iteration {self.iteration}: "
            f"Query='{self.query_used}' | "
            f"Avg Relevance={self.retrieval.avg_score:.2f} | "
            f"Verdict={self.critique.verdict.value} "
            f"(confidence={self.critique.confidence:.0%})"
        )


@dataclass
class PipelineResult:
    """Final result of the self-correcting RAG pipeline."""

    status: PipelineStatus
    final_answer: str
    question: str
    iterations: List[IterationLog] = field(default_factory=list)
    total_iterations: int = 0

    @property
    def was_corrected(self) -> bool:
        return self.total_iterations > 1

    @property
    def correction_summary(self) -> str:
        if not self.was_corrected:
            return "Answer passed on first attempt — no corrections needed."
        lines = ["Self-correction trace:"]
        for log in self.iterations:
            lines.append(f"  {log.summary}")
        return "\n".join(lines)

    @property
    def sources(self) -> List[str]:
        if self.iterations:
            return self.iterations[-1].retrieval.source_list
        return []

    @property
    def final_critique(self) -> Optional[CriticResult]:
        if self.iterations:
            return self.iterations[-1].critique
        return None


def run_pipeline(
    vector_store: Chroma,
    question: str,
    max_retries: int = MAX_RETRIES,
    on_iteration: callable = None,
) -> PipelineResult:
    """
    Execute the self-correcting RAG pipeline.

    Flow:
    1. Retrieve relevant documents
    2. Generate an answer
    3. Critic evaluates the answer
    4. If FAIL → refine query and retry (up to max_retries)
    5. If PASS or PARTIAL → return the answer
    """
    iterations = []
    current_query = question

    for i in range(1, max_retries + 1):
        # Step 1: Retrieve
        if i == 1:
            retrieval = retrieve(vector_store, current_query)
        else:
            retrieval = retrieve_with_requery(
                vector_store, question, current_query
            )

        # Step 2: Generate
        if not retrieval.is_relevant:
            generation_context = retrieval.context_text
            if not generation_context.strip():
                generation_context = (
                    "No relevant documents found for this query."
                )
        else:
            generation_context = retrieval.context_text

        generation = generate_answer(question, generation_context)

        # Step 3: Critique
        critique = critique_answer(
            question=question,
            context=generation_context,
            answer=generation.answer,
        )

        # Log this iteration
        log = IterationLog(
            iteration=i,
            query_used=current_query,
            retrieval=retrieval,
            generation=generation,
            critique=critique,
        )
        iterations.append(log)

        # Notify callback if provided
        if on_iteration:
            on_iteration(log)

        # Step 4: Decide — pass, partial accept, or retry
        if critique.verdict == Verdict.PASS:
            return PipelineResult(
                status=PipelineStatus.SUCCESS,
                final_answer=generation.answer,
                question=question,
                iterations=iterations,
                total_iterations=i,
            )

        if critique.verdict == Verdict.PARTIAL and critique.confidence >= 0.7:
            return PipelineResult(
                status=PipelineStatus.PARTIAL,
                final_answer=generation.answer,
                question=question,
                iterations=iterations,
                total_iterations=i,
            )

        # Step 5: Refine query for next iteration
        if critique.suggested_requery:
            current_query = critique.suggested_requery
        else:
            # Append hallucination context to narrow the search
            issues = critique.hallucinations + critique.contradictions
            if issues:
                current_query = f"{question} (excluding: {', '.join(issues[:2])})"

    # Exhausted all retries — return best effort
    best = max(iterations, key=lambda x: x.critique.confidence)
    return PipelineResult(
        status=PipelineStatus.FAILED,
        final_answer=best.generation.answer,
        question=question,
        iterations=iterations,
        total_iterations=len(iterations),
    )
