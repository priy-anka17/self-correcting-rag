"""Document loader and chunker for various file formats."""

import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import CHUNK_SIZE, CHUNK_OVERLAP


LOADER_MAP = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".md": TextLoader,
}


def load_single_file(file_path: str) -> List[Document]:
    """Load a single file based on its extension."""
    ext = os.path.splitext(file_path)[1].lower()
    loader_cls = LOADER_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    loader = loader_cls(file_path)
    return loader.load()


def load_directory(directory_path: str) -> List[Document]:
    """Load all supported files from a directory."""
    documents = []
    for root, _, files in os.walk(directory_path):
        for file_name in files:
            ext = os.path.splitext(file_name)[1].lower()
            if ext in LOADER_MAP:
                file_path = os.path.join(root, file_name)
                try:
                    documents.extend(load_single_file(file_path))
                except Exception as e:
                    print(f"Warning: Failed to load {file_path}: {e}")
    return documents


def chunk_documents(
    documents: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def load_and_chunk(source: str) -> List[Document]:
    """Load documents from a file or directory and chunk them."""
    if os.path.isdir(source):
        docs = load_directory(source)
    elif os.path.isfile(source):
        docs = load_single_file(source)
    else:
        raise FileNotFoundError(f"Source not found: {source}")

    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} document(s) → {len(chunks)} chunks")
    return chunks
