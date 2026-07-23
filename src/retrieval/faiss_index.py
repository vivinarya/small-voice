# src/retrieval/faiss_index.py
"""FAISS-based on-disk index for the RAG layer.

Index artifacts (in index_dir/):
  faiss.index   — binary FAISS IndexFlatIP (inner product on L2-normalized vecs == cosine)
  chunks.jsonl  — one Chunk JSON per line, row-aligned to FAISS vector ids
  meta.json     — {embed_backend, dim, count, built_at, model_name}

Validation rules:
  - meta.embed_backend must match runtime embed_backend (refuse to load if mismatch)
  - FAISS vector count must equal chunks.jsonl line count
  
Requires: pip install faiss-cpu
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Chunk, RetrievedChunk
from .embedder import Embedder, MIN_SCORE

logger = logging.getLogger(__name__)

_FAISS_FILE = "faiss.index"
_CHUNKS_FILE = "chunks.jsonl"
_META_FILE = "meta.json"

BATCH_SIZE = 64   # batch size for embedding during build


class FAISSIndex:
    """Manages a FAISS IndexFlatIP over NCERT text chunks.
    
    Can be used in two modes:
      - Build mode: call build(chunks, embedder, index_dir) class method
      - Query mode: instantiate with load(index_dir, embed_backend) class method
    """
    
    def __init__(self, index, chunks: list[Chunk], meta: dict) -> None:
        self._index = index      # faiss.IndexFlatIP
        self._chunks = chunks    # aligned with FAISS vector ids
        self._meta = meta

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, index_dir: str) -> "FAISSIndex":
        """Build and persist a FAISS index from a list of chunks.
        
        Postconditions:
          - index_dir contains faiss.index, chunks.jsonl, meta.json
          - faiss.ntotal == len(chunks)
          - meta.embed_backend recorded for validation at load time
        """
        try:
            import faiss  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "faiss-cpu is required to build the index.\n"
                "Install it with:  pip install faiss-cpu"
            )
        
        os.makedirs(index_dir, exist_ok=True)
        
        if not chunks:
            logger.warning("No chunks provided to build_index — writing empty index.")
        
        dim = embedder.dim
        index = faiss.IndexFlatIP(dim)
        
        # Embed in batches to bound peak RAM
        all_vectors = np.empty((len(chunks), dim), dtype=np.float32)
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start:batch_start + BATCH_SIZE]
            vecs = embedder.embed([c.text for c in batch])
            all_vectors[batch_start:batch_start + len(batch)] = vecs
            logger.debug("Embedded batch %d-%d of %d", batch_start, batch_start + len(batch), len(chunks))
        
        if len(chunks) > 0:
            index.add(all_vectors)
        
        assert index.ntotal == len(chunks), f"FAISS ntotal={index.ntotal} != chunks={len(chunks)}"
        
        # Persist
        idx_path = os.path.join(index_dir, _FAISS_FILE)
        faiss.write_index(index, idx_path)
        
        chunks_path = os.path.join(index_dir, _CHUNKS_FILE)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps({
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "page": chunk.page,
                    # Legacy NCERT fields preserved for backward-compat
                    "klass": chunk.klass,
                    "subject": chunk.subject,
                    "chapter": chunk.chapter,
                }) + "\n")
        
        meta = {
            "embed_backend": type(embedder).__name__.replace("Embedder", "").lower(),
            "model_name": getattr(embedder, "_MODEL_NAME", "unknown"),
            "dim": dim,
            "count": len(chunks),
            "built_at": time.time(),
        }
        # Normalize embed_backend to match config keys (MiniLM → minilm, BGESmall → bge_small)
        cls_name = type(embedder).__name__
        if "MiniLM" in cls_name:
            meta["embed_backend"] = "minilm"
        elif "BGE" in cls_name or "Bge" in cls_name:
            meta["embed_backend"] = "bge_small"
        
        meta_path = os.path.join(index_dir, _META_FILE)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        
        logger.info(
            "Built FAISS index: %d chunks, dim=%d, backend=%s -> '%s'",
            len(chunks), dim, meta["embed_backend"], index_dir,
        )
        return cls(index, chunks, meta)

    @classmethod
    def load(cls, index_dir: str, embed_backend: str) -> "FAISSIndex":
        """Load an existing on-disk index, validating embed_backend consistency.
        
        Raises:
          FileNotFoundError: if index artifacts are missing
          ValueError: if meta.embed_backend != embed_backend (dimension/space mismatch)
        """
        try:
            import faiss  # noqa: PLC0415
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")
        
        idx_path = os.path.join(index_dir, _FAISS_FILE)
        chunks_path = os.path.join(index_dir, _CHUNKS_FILE)
        meta_path = os.path.join(index_dir, _META_FILE)
        
        for p in [idx_path, chunks_path, meta_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Index artifact missing: '{p}'. "
                    f"Rebuild with: python scripts/build_index.py --src data/ncert --out {index_dir}"
                )
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        stored_backend = meta.get("embed_backend", "")
        if stored_backend != embed_backend:
            raise ValueError(
                f"Index embed_backend '{stored_backend}' does not match runtime "
                f"embed_backend '{embed_backend}'. "
                f"This would cause a dimension/space mismatch. "
                f"Rebuild the index with: python scripts/build_index.py --embed {embed_backend}"
            )
        
        index = faiss.read_index(idx_path)
        
        chunks: list[Chunk] = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                # Support both new generic schema (source) and legacy NCERT schema
                source = d.get("source") or d.get("chapter") or d.get("subject") or d.get("id", "")
                chunks.append(Chunk(
                    id=d["id"],
                    text=d["text"],
                    source=source,
                    page=d.get("page", 1),
                    klass=d.get("klass", 0),
                    subject=d.get("subject", ""),
                    chapter=d.get("chapter", ""),
                ))
        
        if index.ntotal != len(chunks):
            raise ValueError(
                f"FAISS index has {index.ntotal} vectors but chunks.jsonl has {len(chunks)} lines. "
                "The index artifacts are inconsistent. Rebuild the index."
            )
        
        logger.info(
            "Loaded FAISS index: %d chunks, dim=%d, backend=%s from '%s'",
            len(chunks), index.d, embed_backend, index_dir,
        )
        return cls(index, chunks, meta)

    def search(self, query_vector: np.ndarray, k: int, min_score: Optional[float] = None) -> list[RetrievedChunk]:
        """Search for the top-k nearest chunks.
        
        Preconditions: query_vector is L2-normalized float32 of shape (1, dim); 1 <= k <= 5
        Postconditions: result len <= k, sorted desc by score, each score in [0, 1]
        
        Args:
            query_vector: L2-normalized float32 query embedding of shape (1, dim).
            k: Maximum number of results to return (clamped to [1, 5]).
            min_score: Minimum cosine-similarity score for a chunk to be included.
                       Defaults to the module-level MIN_SCORE constant (0.25) when None.
        """
        if self._index.ntotal == 0:
            return []
        
        threshold = MIN_SCORE if min_score is None else min_score
        k = max(1, min(k, 5))
        k_actual = min(k, self._index.ntotal)
        
        scores, ids = self._index.search(query_vector, k_actual)
        
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:           # FAISS pads with -1 when fewer than k hits
                continue
            score_f = float(score)
            if score_f < threshold:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx], score=score_f))
        
        # Already sorted desc by FAISS, but re-sort to be explicit
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def list_sources(self) -> list[dict]:
        """Aggregate indexed chunks by source document.

        Returns a list of {"source", "page_count", "chunk_count"} dicts where
        page_count is the highest page number seen for that source.
        """
        stats: dict[str, dict] = {}
        for c in self._chunks:
            s = stats.get(c.source)
            if s is None:
                s = {"source": c.source, "page_count": 0, "chunk_count": 0}
                stats[c.source] = s
            s["chunk_count"] += 1
            if c.page > s["page_count"]:
                s["page_count"] = c.page
        return list(stats.values())

    def get_page(self, page: int, source: Optional[str] = None) -> list[Chunk]:
        """Return all chunks on a given page, ordered by their position.

        If `source` is provided, only chunks whose source name contains it
        (case-insensitive) are returned. Chunk ids look like
        "<doc>_p<page>_<localidx>", so sorting by id preserves page order.
        """
        src_lc = source.lower() if source else None
        hits = [
            c for c in self._chunks
            if c.page == page and (src_lc is None or src_lc in c.source.lower())
        ]
        hits.sort(key=lambda c: c.id)
        return hits
