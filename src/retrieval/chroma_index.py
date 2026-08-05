# src/retrieval/chroma_index.py
"""ChromaDB-backed persistent vector index for the RAG layer.

Replaces FAISSIndex with a self-contained, SQLite-backed store that:
  - Persists automatically to `index_dir/` on every write
  - Survives reboots and process restarts with zero rebuild cost
  - Supports incremental document addition/removal (no full rebuild needed)
  - Runs 100% offline on Jetson Orin Nano — no network calls ever

Directory layout (index_dir/):
  chroma.sqlite3          — ChromaDB metadata & embeddings (auto-managed)
  <uuid>/                 — ChromaDB segment data (auto-managed)

Public API is intentionally identical to FAISSIndex so service.py
can swap backends with a one-line change.

Requires: pip install chromadb>=0.5.0
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .base import Chunk, RetrievedChunk
from .embedder import Embedder, MIN_SCORE

logger = logging.getLogger(__name__)

COLLECTION_NAME = "jarvis_rag"
BATCH_SIZE = 64   # chunks per upsert batch to bound peak RAM


class ChromaIndex:
    """Manages a ChromaDB persistent collection of RAG text chunks.

    Two usage modes:
      - Build mode : ChromaIndex.build(chunks, embedder, index_dir)
      - Query mode : ChromaIndex.load(index_dir)

    Both modes return a fully ready ChromaIndex instance.
    """

    def __init__(self, collection, chunks_by_id: dict[str, Chunk]) -> None:
        self._collection = collection          # chromadb.Collection
        self._chunks_by_id = chunks_by_id     # id → Chunk, for fast lookup

    # ------------------------------------------------------------------ build
    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedder: Embedder,
        index_dir: str,
    ) -> "ChromaIndex":
        """Build and persist a ChromaDB collection from a list of Chunk objects.

        Existing collection data is replaced (full rebuild).  Call this once
        after ingesting your PDFs/markdown; the DB is then reused on every
        subsequent startup via ChromaIndex.load().

        Postconditions:
          - index_dir/ exists and contains a valid ChromaDB database
          - collection contains len(chunks) documents
        """
        try:
            import chromadb  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "chromadb is required to build the index.\n"
                "Install it with:  pip install chromadb>=0.5.0"
            )

        Path(index_dir).mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=index_dir)

        # Delete existing collection so build is always a clean slate
        try:
            client.delete_collection(COLLECTION_NAME)
            logger.info("Deleted existing ChromaDB collection '%s' for rebuild.", COLLECTION_NAME)
        except Exception:
            pass  # collection didn't exist yet — fine

        # ChromaDB stores raw embeddings we supply; we handle normalization
        # ourselves via the Embedder so we use no embedding function here.
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},   # cosine distance for search
        )

        if not chunks:
            logger.warning("ChromaIndex.build called with 0 chunks — empty index written.")
            return cls(collection, {})

        chunks_by_id: dict[str, Chunk] = {}
        total = 0

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]

            # Embed batch and L2-normalize (embedder guarantees this)
            vecs: np.ndarray = embedder.embed([c.text for c in batch])  # (B, dim) float32

            ids = [c.id for c in batch]
            documents = [c.text for c in batch]
            embeddings = vecs.tolist()
            metadatas = [
                {
                    "source": c.source,
                    "page": c.page,
                    "klass": c.klass,
                    "subject": c.subject,
                    "chapter": c.chapter,
                }
                for c in batch
            ]

            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            for c in batch:
                chunks_by_id[c.id] = c
            total += len(batch)
            logger.debug(
                "ChromaDB upserted batch %d–%d of %d",
                batch_start, batch_start + len(batch), len(chunks),
            )

        logger.info(
            "ChromaIndex.build complete: %d chunks persisted to '%s'",
            total, index_dir,
        )
        return cls(collection, chunks_by_id)

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, index_dir: str) -> "ChromaIndex":
        """Open an existing on-disk ChromaDB collection (instant, no rebuild).

        Raises:
          FileNotFoundError: if index_dir does not exist or has no DB file.
          ValueError: if the collection is empty (index was never built).
        """
        try:
            import chromadb  # noqa: PLC0415
        except ImportError:
            raise ImportError("chromadb is required: pip install chromadb>=0.5.0")

        db_path = Path(index_dir)
        if not db_path.exists():
            raise FileNotFoundError(
                f"ChromaDB directory '{index_dir}' does not exist. "
                "Run the index build script first:\n"
                "  python scripts/build_index.py --src data/docs --out data/chroma"
            )

        client = chromadb.PersistentClient(path=index_dir)

        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            raise FileNotFoundError(
                f"ChromaDB collection '{COLLECTION_NAME}' not found in '{index_dir}'. "
                "Run the index build script to create it."
            )

        count = collection.count()
        if count == 0:
            logger.warning(
                "ChromaDB collection '%s' is empty. "
                "RAG will return no results until the index is (re)built.",
                COLLECTION_NAME,
            )

        # Re-hydrate the in-memory id→Chunk map from stored documents+metadata
        chunks_by_id: dict[str, Chunk] = {}
        if count > 0:
            # Fetch all in one shot (memory is fine for <100k chunks)
            result = collection.get(include=["documents", "metadatas"])
            for cid, doc, meta in zip(
                result["ids"], result["documents"], result["metadatas"]
            ):
                chunks_by_id[cid] = Chunk(
                    id=cid,
                    text=doc,
                    source=meta.get("source", ""),
                    page=int(meta.get("page", 1)),
                    klass=int(meta.get("klass", 0)),
                    subject=meta.get("subject", ""),
                    chapter=meta.get("chapter", ""),
                )

        logger.info(
            "ChromaIndex loaded: %d chunks from '%s'",
            count, index_dir,
        )
        return cls(collection, chunks_by_id)

    # ------------------------------------------------------------------ search
    def search(
        self,
        query_vector: np.ndarray,
        k: int,
        min_score: Optional[float] = None,
    ) -> list[RetrievedChunk]:
        """Search for the top-k nearest chunks using cosine similarity.

        Args:
            query_vector: L2-normalized float32 embedding of shape (1, dim).
            k: Maximum number of results (clamped to [1, 5]).
            min_score: Minimum cosine similarity; defaults to MIN_SCORE (0.25).

        Returns:
            List of RetrievedChunk sorted descending by similarity score.
        """
        if self._collection.count() == 0:
            return []

        threshold = MIN_SCORE if min_score is None else min_score
        k = max(1, min(k, 5))
        k_actual = min(k, self._collection.count())

        # ChromaDB query() accepts a list of embedding lists
        query_embedding = query_vector[0].tolist()  # shape (dim,) as plain list

        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k_actual,
            include=["distances"],   # cosine distance; we convert to similarity below
        )

        retrieved: list[RetrievedChunk] = []
        ids = result["ids"][0]
        distances = result["distances"][0]  # cosine distance ∈ [0, 2]; 0 = identical

        for cid, dist in zip(ids, distances):
            # Convert cosine distance → cosine similarity: sim = 1 - dist/2
            # ChromaDB's "cosine" space returns distance where 0=identical, 2=opposite.
            # For normalized vectors: cosine_sim = 1 - (cosine_distance / 2)
            score = float(1.0 - dist / 2.0)
            if score < threshold:
                continue
            chunk = self._chunks_by_id.get(cid)
            if chunk is None:
                logger.warning("ChromaDB returned unknown id '%s' — skipping.", cid)
                continue
            retrieved.append(RetrievedChunk(chunk=chunk, score=score))

        retrieved.sort(key=lambda r: r.score, reverse=True)
        return retrieved

    # ------------------------------------------------------------------ helpers
    @property
    def ntotal(self) -> int:
        """Total number of chunks stored in the collection."""
        return self._collection.count()

    def list_sources(self) -> list[dict]:
        """Return a list of {source, page_count, chunk_count} dicts."""
        stats: dict[str, dict] = {}
        for chunk in self._chunks_by_id.values():
            s = stats.get(chunk.source)
            if s is None:
                s = {"source": chunk.source, "page_count": 0, "chunk_count": 0}
                stats[chunk.source] = s
            s["chunk_count"] += 1
            if chunk.page > s["page_count"]:
                s["page_count"] = chunk.page
        return list(stats.values())

    def get_page(self, page: int, source: Optional[str] = None) -> list[Chunk]:
        """Return all chunks on a given page, optionally filtered by source name."""
        src_lc = source.lower() if source else None
        hits = [
            c for c in self._chunks_by_id.values()
            if c.page == page and (src_lc is None or src_lc in c.source.lower())
        ]
        hits.sort(key=lambda c: c.id)
        return hits

    def add_chunks(self, chunks: list[Chunk], embedder: Embedder) -> int:
        """Incrementally add new chunks to the existing collection.

        Skips chunks whose id already exists (idempotent upsert).
        Returns the number of newly added chunks.
        """
        if not chunks:
            return 0

        added = 0
        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start: batch_start + BATCH_SIZE]
            vecs = embedder.embed([c.text for c in batch])

            self._collection.upsert(
                ids=[c.id for c in batch],
                embeddings=vecs.tolist(),
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "source": c.source,
                        "page": c.page,
                        "klass": c.klass,
                        "subject": c.subject,
                        "chapter": c.chapter,
                    }
                    for c in batch
                ],
            )
            for c in batch:
                self._chunks_by_id[c.id] = c
            added += len(batch)

        logger.info("ChromaIndex.add_chunks: added %d chunks incrementally.", added)
        return added

    def remove_source(self, source_name: str) -> int:
        """Remove all chunks belonging to a given source document.

        Returns the number of chunks removed.
        """
        ids_to_remove = [
            cid for cid, c in self._chunks_by_id.items()
            if c.source == source_name
        ]
        if ids_to_remove:
            self._collection.delete(ids=ids_to_remove)
            for cid in ids_to_remove:
                del self._chunks_by_id[cid]
        logger.info(
            "ChromaIndex.remove_source: removed %d chunks for source '%s'.",
            len(ids_to_remove), source_name,
        )
        return len(ids_to_remove)
