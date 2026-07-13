# src/retrieval/cache.py
"""LRU disk cache for query → answer pairs.

Cache entries are persisted as JSON files under cache_dir/ (default: data/index/cache/).
An in-memory LRU index tracks recency; oldest entries are evicted when capacity is exceeded.

Design rules:
  - Bounded: max_entries entries (default 256)
  - Persisted: survives process restart
  - Query normalization: lowercase, strip punctuation, collapse whitespace
  - Idempotent: cache_get after cache_put returns the same answer_text
"""
from __future__ import annotations

import json
import logging
import os
import time
from collections import OrderedDict
from typing import Optional

from .base import CachedAnswer, hash_query, normalize_query

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 256
_CACHE_SUBDIR = "cache"
_INDEX_FILE = "cache_index.json"   # maps query_hash → {created_at, answer_text[:50], ...}


class AnswerCache:
    """LRU disk cache for voice assistant answers.
    
    Properties satisfied:
      - cache_put(q, a, _) then cache_get(q) returns CachedAnswer with answer_text == a
      - repeated cache_get(q) is side-effect-free (except LRU recency update)
    """

    def __init__(self, cache_dir: str = "data/index/cache", max_entries: int = _DEFAULT_MAX_ENTRIES) -> None:
        self._cache_dir = cache_dir
        self._max_entries = max_entries
        os.makedirs(cache_dir, exist_ok=True)
        # In-memory LRU: OrderedDict maps query_hash → created_at (for eviction)
        self._lru: OrderedDict[str, float] = OrderedDict()
        self._load_index()
    
    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------
    
    def _entry_path(self, query_hash: str) -> str:
        return os.path.join(self._cache_dir, f"{query_hash}.json")
    
    def _index_path(self) -> str:
        return os.path.join(self._cache_dir, _INDEX_FILE)
    
    def _load_index(self) -> None:
        """Load the LRU order from disk at startup."""
        idx_path = self._index_path()
        if not os.path.exists(idx_path):
            return
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # Restore LRU order sorted by created_at ascending (oldest first)
            entries = sorted(raw.items(), key=lambda kv: kv[1].get("created_at", 0))
            for qhash, meta in entries:
                self._lru[qhash] = meta.get("created_at", 0.0)
        except Exception as exc:
            logger.warning("Failed to load cache index: %s", exc)
    
    def _save_index(self) -> None:
        """Persist the LRU order to disk."""
        try:
            raw = {qhash: {"created_at": ts} for qhash, ts in self._lru.items()}
            with open(self._index_path(), "w", encoding="utf-8") as f:
                json.dump(raw, f)
        except Exception as exc:
            logger.warning("Failed to save cache index: %s", exc)
    
    def _evict_oldest(self) -> None:
        """Remove the least-recently-used entry (both from memory and disk)."""
        if not self._lru:
            return
        oldest_hash, _ = next(iter(self._lru.items()))
        self._lru.pop(oldest_hash)
        entry_path = self._entry_path(oldest_hash)
        try:
            if os.path.exists(entry_path):
                os.remove(entry_path)
            logger.debug("Evicted cache entry %s", oldest_hash[:16])
        except Exception as exc:
            logger.warning("Failed to evict cache entry %s: %s", oldest_hash[:16], exc)
    
    # ---------------------------------------------------------------
    # Public interface
    # ---------------------------------------------------------------
    
    def get(self, query: str) -> Optional[CachedAnswer]:
        """Return a cached answer for the query, or None if not found."""
        qhash = hash_query(query)
        if qhash not in self._lru:
            return None
        
        entry_path = self._entry_path(qhash)
        if not os.path.exists(entry_path):
            # Stale index entry — clean up
            self._lru.pop(qhash, None)
            return None
        
        try:
            with open(entry_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Update LRU recency
            self._lru.move_to_end(qhash)
            return CachedAnswer(
                query_hash=data["query_hash"],
                answer_text=data["answer_text"],
                audio_path=data.get("audio_path"),
                citations=data.get("citations", []),
                created_at=data["created_at"],
            )
        except Exception as exc:
            logger.warning("Failed to read cache entry %s: %s", qhash[:16], exc)
            return None
    
    def put(self, query: str, answer: str, audio_pcm: Optional[bytes] = None, citations: Optional[list[str]] = None) -> None:
        """Store an answer in the cache, evicting oldest entry if at capacity."""
        qhash = hash_query(query)
        
        # Evict if at capacity and this is a new entry
        if qhash not in self._lru and len(self._lru) >= self._max_entries:
            self._evict_oldest()
        
        audio_path: Optional[str] = None
        if audio_pcm is not None:
            audio_path = self._entry_path(qhash).replace(".json", ".pcm")
            try:
                with open(audio_path, "wb") as f:
                    f.write(audio_pcm)
            except Exception as exc:
                logger.warning("Failed to write audio cache: %s", exc)
                audio_path = None
        
        entry = {
            "query_hash": qhash,
            "answer_text": answer,
            "audio_path": audio_path,
            "citations": citations or [],
            "created_at": time.time(),
        }
        try:
            with open(self._entry_path(qhash), "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to write cache entry: %s", exc)
            return
        
        # Update LRU (move to most-recent end)
        self._lru[qhash] = entry["created_at"]
        self._lru.move_to_end(qhash)
        self._save_index()
        logger.debug("Cached answer for query hash %s", qhash[:16])
    
    def __len__(self) -> int:
        return len(self._lru)
