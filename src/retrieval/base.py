# src/retrieval/base.py
"""Core data models and RetrievalService interface for the RAG layer."""
from __future__ import annotations
import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    """One page-level chunk of text from any document.

    Fields are kept general so any PDF — academic, textbook, manual, novel,
    report — can be indexed without forcing a class/subject hierarchy.
    """
    id: str           # unique identifier, e.g. "my_doc_p3_0"
    text: str
    source: str       # human-readable document name (filename stem)
    page: int         # 1-indexed page number

    # Legacy NCERT aliases kept for backward-compatibility with existing
    # chunks.jsonl files that carry klass/subject/chapter fields.
    # For new generic documents these default to empty / 0.
    klass: int = 0
    subject: str = ""
    chapter: str = ""


@dataclass(frozen=True)
class RetrievedChunk:
    """A Chunk retrieved from the FAISS index with a similarity score."""
    chunk: Chunk
    score: float      # cosine similarity in [0, 1] after L2 normalization

    def citation(self) -> str:
        """Return a human-readable citation string."""
        if self.chunk.subject:
            # Legacy NCERT format
            return f"{self.chunk.source}, page {self.chunk.page}"
        return f"{self.chunk.source}, page {self.chunk.page}"


@dataclass(frozen=True)
class CachedAnswer:
    """A cached query → answer entry."""
    query_hash: str
    answer_text: str
    audio_path: Optional[str]
    citations: list[str]
    created_at: float


def normalize_query(query: str) -> str:
    """Normalize a query for cache lookup."""
    q = query.lower()
    q = re.sub(r'[^\w\s]', '', q)
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def hash_query(query: str) -> str:
    """Return a sha256 hex digest of the normalized query."""
    normalized = normalize_query(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class RetrievalService(ABC):
    """Abstract interface for the RAG retrieval layer."""

    @abstractmethod
    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    def cache_get(self, query: str) -> Optional[CachedAnswer]:
        ...

    @abstractmethod
    def cache_put(self, query: str, answer: str, audio_pcm: Optional[bytes] = None) -> None:
        ...
