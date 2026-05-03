"""ChromaDB vector store for document storage and retrieval."""

import os
from typing import List, Optional

import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, TOP_K


def get_embeddings() -> HuggingFaceEmbeddings:
    """Create the embedding model instance (local, no API key needed)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(
    documents: List[Document],
    collection_name: str = "default",
) -> Chroma:
    """Create a new vector store from documents."""
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"Created vector store with {len(documents)} documents")
    return vector_store


def load_vector_store(collection_name: str = "default") -> Optional[Chroma]:
    """Load an existing vector store."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    return vector_store


def similarity_search_with_scores(
    vector_store: Chroma,
    query: str,
    k: int = TOP_K,
) -> List[tuple]:
    """Search for similar documents with relevance scores."""
    results = vector_store.similarity_search_with_relevance_scores(query, k=k)
    return results


def delete_collection(collection_name: str = "default"):
    """Delete a vector store collection."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
    except Exception:
        print(f"Collection '{collection_name}' not found")
