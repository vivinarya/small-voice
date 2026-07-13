# src/retrieval/service.py
"""Concrete RetrievalService implementations.

FAISSRetrievalService: full RAG — embed query → FAISS search → filter → cache
NullRetrievalService:  no-op for when index hasn't been built yet (graceful degradation)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from .base import CachedAnswer, RetrievalService, RetrievedChunk
from .cache import AnswerCache
from .embedder import Embedder
from .faiss_index import FAISSIndex

logger = logging.getLogger(__name__)


class FAISSRetrievalService(RetrievalService):
    """Full RAG retrieval using FAISS + sentence-transformer embeddings.
    
    Pipeline per query:
      1. Normalize query
      2. Embed query with Embedder (L2-normalized, shape (1, dim))
      3. Search FAISSIndex for top-k results (filtered by min_score)
      4. Return RetrievedChunk list with citations
    """

    def __init__(
        self,
        index_dir: str,
        embedder: Embedder,
        cache_max_entries: int = 256,
        min_score: Optional[float] = None,
    ) -> None:
        from .embedder import MIN_SCORE as _DEFAULT_MIN_SCORE  # noqa: PLC0415
        self._embedder = embedder
        self._min_score: float = _DEFAULT_MIN_SCORE if min_score is None else min_score
        embed_backend_name = "minilm" if "MiniLM" in type(embedder).__name__ else "bge_small"
        self._index = FAISSIndex.load(index_dir, embed_backend=embed_backend_name)
        cache_dir = os.path.join(index_dir, "cache")
        self._cache = AnswerCache(cache_dir=cache_dir, max_entries=cache_max_entries)
        logger.info(
            "FAISSRetrievalService ready: %d chunks, %d cache entries, min_score=%.2f",
            self._index.ntotal,
            len(self._cache),
            self._min_score,
        )

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        """Embed query and return top-k relevant chunks.
        
        Preconditions: 1 <= k <= 5
        Postconditions: <= k results, sorted desc by score, score in [0,1]
        """
        k = max(1, min(5, k))
        
        if self._index.ntotal == 0:
            return []
        
        # Embed query (L2-normalized, shape (1, dim))
        qv = self._embedder.embed([query])   # shape (1, dim)
        results = self._index.search(qv, k, min_score=self._min_score)
        
        logger.debug("Retrieved %d chunks for query '%s...'", len(results), query[:50])
        return results

    def cache_get(self, query: str) -> Optional[CachedAnswer]:
        """Return cached answer for query, or None."""
        return self._cache.get(query)

    def cache_put(self, query: str, answer: str, audio_pcm: Optional[bytes] = None) -> None:
        """Cache an answer for future instant replay."""
        self._cache.put(query, answer, audio_pcm=audio_pcm)

    def list_sources(self) -> list[dict]:
        """Return indexed documents as {"source", "page_count", "chunk_count"}."""
        return self._index.list_sources()

    def get_page(self, page: int, source: Optional[str] = None):
        """Return all chunks on the given page (optionally filtered by source)."""
        return self._index.get_page(page, source)


class NullRetrievalService(RetrievalService):
    """No-op RetrievalService used when the FAISS index hasn't been built yet.
    
    Allows the assistant to run without RAG — the model answers from its own
    parametric knowledge. No cache, no retrieval.
    """

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        return []

    def cache_get(self, query: str) -> Optional[CachedAnswer]:
        return None

    def cache_put(self, query: str, answer: str, audio_pcm: Optional[bytes] = None) -> None:
        pass  # No-op: nowhere to cache without an index directory

    def list_sources(self) -> list[dict]:
        return []  # No index built yet

    def get_page(self, page: int, source: Optional[str] = None):
        return []  # No index built yet
