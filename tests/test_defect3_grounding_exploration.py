"""Bug-condition exploration test for Defect 3 — Grounding / Hallucination (Property 3).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 3 (design.md): Bug Condition — Silent RAG Disable, Uncited Answers, No Hot-Reload.

**Validates: Requirements 1.7, 1.8, 1.9**

CRITICAL — this is a BUGFIX *exploration* test run on the UNFIXED code. Per
bugfix exploration-test semantics, demonstrating the defect IS the success
case. DO NOT "fix" the code or weaken the assertions to change these outcomes.

Bug condition (bugfix.md `isGroundingBug`):
    (a) a textbook-answerable query is answered with ZERO retrieved chunks
        (RAG silently disabled because the FAISS index is missing), OR
    (b) grounded content exists but the answer/prompt carries no citation, AND
    (c) the build-once-at-startup gap: after a successful rebuild the running
        retrieval reference is never replaced, so the new on-disk index is
        ignored until the process restarts, AND
    (d) the only failure signal is an `app.log` WARNING the user never sees.

This test encodes three concrete states, fully OFFLINE, with no faiss /
sentence-transformers / litert / whisper installed:

  1. `data/index/` absent  -> build_retrieval returns NullRetrievalService,
     retrieve() == [], and build_prompt(query, []) returns the BARE question
     (no "Context:" block, no citation instruction).
  2. LiteRTEngine system prompt contains NO anti-hallucination /
     answer-only-from-context guard.
  3. Build-once gap: a NullRetrievalService created at "startup" keeps
     returning [] even after a faiss.index appears on disk, while a freshly
     built retrieval service WOULD see chunks — the running reference is stale.
"""
from __future__ import annotations

import os
import sys
import types

import pytest

# --------------------------------------------------------------------------
# Make `src/` importable (matches the other exploration tests).
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --------------------------------------------------------------------------
# Dependency-light stand-ins, installed BEFORE importing code under test.
#
#  * `numpy` is needed only as a module object to import retrieval.* (no array
#    math runs at import time); install a stub only if a real one is absent.
#  * `faiss` / `sentence_transformers` are imported lazily by the real RAG
#    path; stub them defensively so nothing drags in heavy deps.
#  * `litert_lm` is imported at the top of `inference.engine`; a tiny fake lets
#    us import the module and inspect its system prompt WITHOUT instantiating
#    the multi-GB engine.
# --------------------------------------------------------------------------
def _install_fake_numpy_if_absent() -> None:
    if "numpy" in sys.modules:
        return
    try:  # prefer a real numpy if one is installed
        import numpy  # noqa: F401

        return
    except Exception:
        pass

    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = list
    fake_np.float32 = "float32"
    fake_np.int16 = "int16"
    fake_np.__path__ = []  # mark as package
    fake_linalg = types.ModuleType("numpy.linalg")
    fake_linalg.norm = lambda *a, **k: 1.0
    fake_np.linalg = fake_linalg
    fake_random = types.ModuleType("numpy.random")
    fake_random.get_state = lambda: ("fake-state",)
    fake_random.set_state = lambda state: None
    fake_random.seed = lambda seed=None: None
    fake_np.random = fake_random
    sys.modules["numpy"] = fake_np
    sys.modules["numpy.linalg"] = fake_linalg
    sys.modules["numpy.random"] = fake_random


def _install_stub_module(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except Exception:
        sys.modules[name] = types.ModuleType(name)


def _install_fake_litert() -> None:
    """Fake `litert_lm` so `inference.engine` imports without the real engine."""
    if "litert_lm" in sys.modules:
        return
    fake = types.ModuleType("litert_lm")

    class _Backend:
        CPU = "cpu"

    class _Engine:  # never instantiated by this test
        def __init__(self, *a, **k):
            raise RuntimeError("litert_lm.Engine must not be constructed in tests")

        def create_conversation(self, *a, **k):
            raise RuntimeError("no conversation in tests")

    fake.Backend = _Backend
    fake.Engine = _Engine
    sys.modules["litert_lm"] = fake


_install_fake_numpy_if_absent()
_install_stub_module("faiss")
_install_stub_module("sentence_transformers")
_install_fake_litert()

import pytest

from config import AppConfig
import factories
from factories import build_retrieval
from inference.prompt_builder import build_prompt
import retrieval.service as service_mod
import retrieval.embedder as embedder_mod
from retrieval.service import NullRetrievalService


# A representative textbook-answerable query (has relevant content in a real
# NCERT corpus) — exactly the kind of turn that must be grounded + cited.
TEXTBOOK_QUERY = "What is sound and how does it travel?"

# Markers that a genuine anti-hallucination / answer-only-from-context guard
# would contain. Their ABSENCE in the system prompt is the defect.
_GUARD_MARKERS = (
    "only from",
    "answer only",
    "do not have",
    "don't have",
    "do not know",
    "don't know",
    "do not invent",
    "don't invent",
    "do not make up",
    "context does not",
    "if the context",
    "not in the context",
)


def _make_cfg(index_dir: str) -> AppConfig:
    """Construct an AppConfig pointed at a throwaway index dir (all fields set)."""
    return AppConfig(
        engine_backend="litert",
        model_path="assets/gemma-4-E4B-it.litertlm",
        n_gpu_layers=0,
        stt_backend="whisper",
        stt_model="base.en",
        stt_compute_type="int8",
        tts_backend="piper",
        tts_voice_path="assets/piper_voices/en_US-lessac-medium.onnx",
        embed_backend="minilm",
        index_dir=index_dir,
        retrieval_k=3,
        min_score=0.25,
    )


# ==========================================================================
# Assertion 1 — Missing index => silent NullRetrievalService + bare uncited prompt.
# EXPECTED OUTCOME ON UNFIXED CODE: PASSES (it confirms the defect).
# ==========================================================================
def test_missing_index_silently_disables_rag_and_returns_null_service(tmp_path):
    """`data/index/` absent -> build_retrieval returns NullRetrievalService.

    **Validates: Requirements 1.7, 1.8**
    """
    index_dir = tmp_path / "index"  # deliberately has NO faiss.index
    cfg = _make_cfg(str(index_dir))

    assert not (index_dir / "faiss.index").exists()

    svc = build_retrieval(cfg)

    # Silent degradation: a no-op service, not a real RAG service.
    assert isinstance(svc, NullRetrievalService), (
        f"Missing index should silently yield NullRetrievalService, got {type(svc).__name__}"
    )
    # retrieve() always returns [] => zero retrieved chunks for ANY query.
    assert svc.retrieve(TEXTBOOK_QUERY, 3) == []
    # cache is a no-op too (no instant replay, nowhere to store).
    assert svc.cache_get(TEXTBOOK_QUERY) is None


def test_no_chunks_yields_bare_uncited_prompt():
    """build_prompt(query, []) returns the BARE question — ungrounded, uncited.

    **Validates: Requirements 1.8**
    """
    prompt = build_prompt(TEXTBOOK_QUERY, [])

    # The defect: the prompt IS just the user's question, verbatim.
    assert prompt == TEXTBOOK_QUERY, (
        f"Expected bare question, got a different prompt: {prompt!r}"
    )
    # No retrieved context block is injected ...
    assert "Context:" not in prompt
    # ... and no citation instruction is present.
    lowered = prompt.lower()
    assert "cite" not in lowered
    assert "page" not in lowered
    assert "textbook context" not in lowered


def test_missing_index_only_signal_is_a_log_warning(tmp_path, caplog):
    """The ONLY signal of disabled RAG is an app.log WARNING (not user-visible).

    **Validates: Requirements 1.7**
    """
    index_dir = tmp_path / "index"
    cfg = _make_cfg(str(index_dir))

    import logging

    with caplog.at_level(logging.WARNING, logger="factories"):
        svc = build_retrieval(cfg)

    assert isinstance(svc, NullRetrievalService)
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, "Expected a WARNING-level log (the only — invisible — signal)"
    assert any("RAG disabled" in r.getMessage() for r in warnings), (
        "The disabled-RAG signal is buried in a log line the user never sees: "
        f"{[r.getMessage() for r in warnings]!r}"
    )


# ==========================================================================
# Assertion 2 — System prompt has NO anti-hallucination / context-only guard.
# EXPECTED OUTCOME ON UNFIXED CODE: PASSES (it confirms the defect).
# ==========================================================================
@pytest.mark.xfail(
    reason="Defect-state snapshot: guard is now present in fixed code (Req 2.10, 2.12). "
           "This exploration test documents the unfixed state.",
    strict=False,
)
def test_system_prompt_has_no_anti_hallucination_guard():
    """LiteRTEngine system prompt never tells the model to answer only from context.

    **Validates: Requirements 1.8**
    """
    # Importing here keeps the fake litert_lm install above effective.
    import inference.engine as engine_mod

    # The prompt is a module-level constant; expose it on the class too if present.
    system_prompt = getattr(engine_mod, "_SYSTEM_PROMPT", None)
    if system_prompt is None:
        system_prompt = getattr(engine_mod.LiteRTEngine, "_SYSTEM_PROMPT", None)
    assert isinstance(system_prompt, str) and system_prompt, "system prompt not found"

    lowered = system_prompt.lower()
    found = [m for m in _GUARD_MARKERS if m in lowered]
    assert found == [], (
        "System prompt unexpectedly contains an anti-hallucination guard "
        f"({found!r}) — on unfixed code there should be none. Prompt: {system_prompt!r}"
    )
    # And it certainly never mentions grounding on a provided Context block.
    assert "context" not in lowered


# ==========================================================================
# Assertion 3 — Build-once gap: a successful rebuild does NOT replace the
# already-running retrieval reference (new index ignored until restart).
# EXPECTED OUTCOME ON UNFIXED CODE: PASSES (it confirms the defect).
# ==========================================================================
class _FakeFreshFaissService:
    """Stand-in for a freshly built FAISSRetrievalService that DOES see chunks."""

    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, query, k=3):
        # A real freshly built service would return relevant chunks here.
        return [f"chunk-for::{query}"]

    def cache_get(self, query):
        return None

    def cache_put(self, query, answer, audio_pcm=None):
        pass


class _FakeEmbedder:
    def __init__(self, *args, **kwargs):
        pass


def test_build_once_gap_running_reference_ignores_new_index(tmp_path, monkeypatch):
    """A startup NullRetrievalService stays stale after an index appears on disk.

    Simulates the running process: `retrieval` is built once at startup and the
    reference is held by `_handle_response`. After a rebuild writes faiss.index,
    the held reference is never replaced, so retrieval stays disabled — while a
    freshly built service WOULD now see chunks.

    **Validates: Requirements 1.9**
    """
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    cfg = _make_cfg(str(index_dir))

    # --- startup: no index yet -> Null service, held by the "running app" ---
    held_retrieval = build_retrieval(cfg)
    assert isinstance(held_retrieval, NullRetrievalService)
    assert held_retrieval.retrieve(TEXTBOOK_QUERY, 3) == []

    # --- a rebuild_index run now writes a fresh index to disk ---
    (index_dir / "faiss.index").write_bytes(b"FRESH-INDEX")

    # Patch the heavy RAG classes so build_retrieval can take the FAISS branch
    # offline (no real faiss / sentence-transformers needed).
    monkeypatch.setattr(service_mod, "FAISSRetrievalService", _FakeFreshFaissService)
    monkeypatch.setattr(embedder_mod, "MiniLMEmbedder", _FakeEmbedder)

    # A FRESHLY built service sees the new index and returns chunks ...
    fresh_retrieval = build_retrieval(cfg)
    assert not isinstance(fresh_retrieval, NullRetrievalService)
    assert fresh_retrieval.retrieve(TEXTBOOK_QUERY, 3) != [], (
        "A freshly built retrieval service should see the new index's chunks"
    )

    # ... but the RUNNING process still holds the stale Null reference, which
    # STILL returns [] => the new index is ignored until restart (the bug).
    assert held_retrieval.retrieve(TEXTBOOK_QUERY, 3) == [], (
        "Build-once gap: the running NullRetrievalService must remain stale"
    )
    assert held_retrieval is not fresh_retrieval


@pytest.mark.xfail(
    reason="Defect-state snapshot: _app_state hot-swap is now implemented in fixed code (Req 1.9). "
           "This exploration test documents the unfixed build-once structure.",
    strict=False,
)
def test_build_once_gap_is_structural_in_main(tmp_path):
    """Structural proof: build_retrieval is called once at startup and never in
    the rebuild_index handler; retrieval is a local passed into _handle_response.

    **Validates: Requirements 1.9**
    """
    main_path = os.path.join(
        os.path.dirname(__file__), "..", "src", "main.py"
    )
    with open(main_path, "r", encoding="utf-8") as f:
        source = f.read()

    # build_retrieval is constructed exactly once (startup in main_loop).
    assert source.count("build_retrieval(cfg)") == 1, (
        "Expected build_retrieval to be called exactly once (startup only)"
    )

    # The rebuild_index handler block must NOT re-run build_retrieval / swap it.
    rebuild_idx = source.index('"rebuild_index"')
    next_branch = source.index('elif data.get("type") == "get_ncert_graph"', rebuild_idx)
    rebuild_block = source[rebuild_idx:next_branch]
    assert "build_retrieval" not in rebuild_block, (
        "rebuild_index handler unexpectedly rebuilds retrieval — the build-once "
        "gap would be closed; on unfixed code it must not."
    )

    # retrieval is created as a local and passed positionally into the handler,
    # not stored in a shared mutable holder that the handler could swap.
    assert "retrieval = build_retrieval(cfg)" in source
    assert "_handle_response(audio_data, engine, tts, stt, retrieval, shutdown_event)" in source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
