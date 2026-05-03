"""Retriever module — fetches relevant chunks and filters by relevance."""

from typing import List, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from config import TOP_K, RELEVANCE_THRESHOLD


@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata."""

    documents: List[Document]
    scores: List[float]
    query: str
    avg_score: float
    is_relevant: bool

    @property
    def context_text(self) -> str:
        """Combine all retrieved documents into a single context string."""
        parts = []
        for i, doc in enumerate(self.documents):
            source = doc.metadata.get("source", "unknown")
            parts.append(f"[Source {i+1}: {source}]\n{doc.page_content}")
        return "\n\n---\n\n".join(parts)

    @property
    def source_list(self) -> List[str]:
        """Get list of unique sources."""
        sources = set()
        for doc in self.documents:
            sources.add(doc.metadata.get("source", "unknown"))
        return sorted(sources)


def retrieve(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> RetrievalResult:
    """Retrieve relevant documents for a query with relevance filtering."""
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)

    documents = []
    scores = []
    for doc, score in results:
        documents.append(doc)
        scores.append(score)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    is_relevant = avg_score >= threshold and len(documents) > 0

    return RetrievalResult(
        documents=documents,
        scores=scores,
        query=query,
        avg_score=avg_score,
        is_relevant=is_relevant,
    )


def retrieve_with_requery(
    vector_store: Chroma,
    original_query: str,
    refined_query: str,
    k: int = TOP_K,
    threshold: float = RELEVANCE_THRESHOLD,
) -> RetrievalResult:
    """Retrieve with a refined query (used during self-correction)."""
    return retrieve(vector_store, refined_query, k=k, threshold=threshold)
