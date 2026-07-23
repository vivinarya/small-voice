# src/retrieval/embedder.py
"""Embedding models for the RAG layer.

Default: all-MiniLM-L6-v2 (384-dim)
Alternative: BAAI/bge-small-en-v1.5 (384-dim)

Both produce L2-normalized float32 vectors suitable for inner-product
similarity search in FAISS IndexFlatIP.

Requires: pip install sentence-transformers

Offline mode:
  TRANSFORMERS_OFFLINE=1 and HF_DATASETS_OFFLINE=1 are set at module
  load time so the embedder NEVER makes network calls at runtime.
  Models must be pre-downloaded to the HuggingFace cache before first use:

      python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

  After that one-time download, the Jetson Orin Nano can operate with
  no internet connection and the model loads instantly from local cache.
"""
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

# ── Enforce fully offline HuggingFace model loading ──────────────────────────
# This prevents sentence-transformers / transformers from attempting any
# network request at runtime. If the model is not already cached locally,
# an OSError is raised immediately with a clear message rather than
# silently hanging on a connection attempt.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

logger = logging.getLogger(__name__)

# Minimum relevance threshold for retrieved chunks (cosine similarity)
MIN_SCORE: float = 0.25


class Embedder(ABC):
    """Pluggable embedding model interface."""

    dim: int  # embedding dimension, set by subclass

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized float32 array of shape (len(texts), dim).

        Postconditions: result.dtype == float32, result.shape == (len(texts), dim),
                        each row has L2 norm == 1.0
        """
        ...


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize a batch of vectors in-place, returning the result."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return (vectors / norms).astype(np.float32)


class MiniLMEmbedder(Embedder):
    """all-MiniLM-L6-v2 embedder (384-dim, fast, CPU-friendly).

    Defers sentence-transformers import to __init__ so the module loads cleanly
    without requiring the library unless this class is instantiated.
    """

    dim: int = 384
    _MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for MiniLMEmbedder but is not installed.\n"
                "Install it with:\n\n"
                "    pip install sentence-transformers\n"
            ) from exc
        self._model = SentenceTransformer(self._MODEL_NAME)
        logger.info("MiniLMEmbedder loaded model '%s'", self._MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts and return L2-normalized float32 vectors."""
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vectors)


class BGESmallEmbedder(Embedder):
    """BAAI/bge-small-en-v1.5 embedder (384-dim).

    Slightly better quality than MiniLM at the same dimension.
    """

    dim: int = 384
    _MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for BGESmallEmbedder.\n"
                "Install it with:  pip install sentence-transformers"
            ) from exc
        self._model = SentenceTransformer(self._MODEL_NAME)
        logger.info("BGESmallEmbedder loaded model '%s'", self._MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _l2_normalize(vectors)


def build_embedder(backend: str) -> Embedder:
    """Factory: return an Embedder for the given backend name."""
    if backend == "minilm":
        return MiniLMEmbedder()
    elif backend == "bge_small":
        return BGESmallEmbedder()
    else:
        raise ValueError(f"Unknown embed_backend '{backend}'. Choose 'minilm' or 'bge_small'.")
