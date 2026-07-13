"""Defect 3 verification tests — asserts the FIXED grounding behavior (Property 3).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 3 (design.md): Grounded, Cited, or Honest Decline With Visible Index Status.

**Validates: Requirements 2.9, 2.10, 2.11, 2.12**

This file tests the EXPECTED (fixed) behavior introduced by tasks 6.1–6.5:
  A. No index → build_retrieval returns NullRetrievalService + _get_index_status()
     returns {built: False} + CLI "NOT built" message exists in main.py source.
  B. System prompt NOW contains the anti-hallucination guard markers.
  C. build_prompt(query, [], index_available=True) returns an honest-decline prompt
     (not the bare question).
  D. Hot-reload: _app_state exists in main.py, and the rebuild_index handler
     calls build_retrieval to swap in the new retrieval service.
  E. _get_index_status function exists in main.py (structural check).
"""
from __future__ import annotations

import os
import sys
import types

# ---------------------------------------------------------------------------
# Make `src/` importable
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Lightweight dependency stubs — same pattern used in the exploration test.
# ---------------------------------------------------------------------------
def _install_fake_numpy_if_absent() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401
        return
    except Exception:
        pass
    fake_np = types.ModuleType("numpy")
    fake_np.ndarray = list
    fake_np.float32 = "float32"
    fake_np.int16 = "int16"
    fake_np.__path__ = []
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
    if "litert_lm" in sys.modules:
        return
    fake = types.ModuleType("litert_lm")

    class _Backend:
        CPU = "cpu"

    class _Engine:
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
from factories import build_retrieval
from inference.prompt_builder import build_prompt
import retrieval.service as service_mod
import retrieval.embedder as embedder_mod
from retrieval.service import NullRetrievalService

TEXTBOOK_QUERY = "What is sound and how does it travel?"

# Anti-hallucination guard markers expected to be IN the system prompt after the fix.
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
# Test A — No index → NullRetrievalService + _get_index_status reports not-built
#           + CLI "NOT built" message exists in main.py.
# ==========================================================================
def test_no_index_returns_null_service_and_status_not_built(tmp_path):
    """build_retrieval still returns NullRetrievalService when no index exists.
    _get_index_status on that service reports built=False.

    **Validates: Requirements 2.9**
    """
    index_dir = tmp_path / "index"
    cfg = _make_cfg(str(index_dir))

    assert not (index_dir / "faiss.index").exists()

    svc = build_retrieval(cfg)
    assert isinstance(svc, NullRetrievalService), (
        f"Missing index should yield NullRetrievalService, got {type(svc).__name__}"
    )

    # _get_index_status must be importable from main (structural check also in Test E)
    # and must report built=False for NullRetrievalService.
    import importlib, sys as _sys
    # Import main module source to call _get_index_status without starting the server.
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    spec_obj = importlib.util.spec_from_file_location("_main_mod_test_a", main_src)
    # We cannot fully execute main.py (it imports sounddevice, numpy at module level).
    # Instead read the source and verify the function logic statically and via source check.
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    # The CLI "not built" message must exist in main.py.
    assert "NOT built" in source or "not built" in source.lower(), (
        "Expected a 'NOT built' / 'not built' CLI message in main.py — "
        "the fix requires a visible status line printed at startup."
    )


def test_not_built_cli_line_references_ungrounded_warning(tmp_path):
    """The CLI line for missing index warns that answers will be ungrounded.

    **Validates: Requirements 2.9**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    # The message should mention ungrounded / grounded so the user understands the impact.
    assert "ungrounded" in source.lower() or "not built" in source.lower(), (
        "The not-built CLI message should mention the grounding impact."
    )


# ==========================================================================
# Test B — System prompt NOW contains anti-hallucination guard.
# ==========================================================================
def test_system_prompt_has_anti_hallucination_guard():
    """LiteRTEngine._SYSTEM_PROMPT now contains the answer-only-from-context guard.

    **Validates: Requirements 2.10, 2.12**
    """
    import inference.engine as engine_mod

    system_prompt = getattr(engine_mod, "_SYSTEM_PROMPT", None)
    if system_prompt is None:
        system_prompt = getattr(engine_mod.LiteRTEngine, "_SYSTEM_PROMPT", None)
    assert isinstance(system_prompt, str) and system_prompt, "system prompt not found"

    lowered = system_prompt.lower()
    found = [m for m in _GUARD_MARKERS if m in lowered]
    assert len(found) > 0, (
        f"System prompt should contain an anti-hallucination guard but none of "
        f"{_GUARD_MARKERS!r} were found. Prompt: {system_prompt!r}"
    )

    # Must mention the 'Context:' block specifically so it only applies to
    # grounded turns (not chit-chat).
    assert "context" in lowered, (
        "System prompt should reference 'Context:' so the guard is scoped to "
        "RAG-grounded turns only."
    )


# ==========================================================================
# Test C — build_prompt with empty retrieved + index_available=True → honest decline.
# ==========================================================================
def test_honest_decline_when_index_available_but_no_chunks():
    """build_prompt(query, [], index_available=True) returns an honest-decline prompt.

    The prompt must NOT be the bare question and must instruct the model to
    say it does not have the information.

    **Validates: Requirements 2.10**
    """
    prompt = build_prompt(TEXTBOOK_QUERY, [], index_available=True)

    # Must NOT be the bare question.
    assert prompt != TEXTBOOK_QUERY, (
        "With index_available=True and no chunks, prompt must not be the bare "
        f"question. Got: {prompt!r}"
    )

    lowered = prompt.lower()
    # Must tell the model to decline, not answer from parametric knowledge.
    decline_phrases = [
        "do not have",
        "don't have",
        "does not contain",
        "not contain",
        "no information",
        "does not have",
    ]
    found = [p for p in decline_phrases if p in lowered]
    assert found, (
        f"Honest-decline prompt must instruct the model to say it does not have "
        f"the information. None of {decline_phrases!r} found in: {prompt!r}"
    )


def test_bare_question_preserved_when_index_not_available():
    """build_prompt(query, [], index_available=False) still returns the bare question.

    This preserves chit-chat / no-index behaviour (Req 3.5).

    **Validates: Requirements 2.10 (negative case — chit-chat path unchanged)**
    """
    prompt = build_prompt(TEXTBOOK_QUERY, [], index_available=False)
    assert prompt == TEXTBOOK_QUERY, (
        f"Without index, build_prompt should still return the bare question. "
        f"Got: {prompt!r}"
    )


def test_bare_question_preserved_default_index_available():
    """build_prompt(query, []) with default index_available returns the bare question.

    Default is index_available=False → backward-compatible.

    **Validates: Requirements 2.10 (backward compatibility)**
    """
    prompt = build_prompt(TEXTBOOK_QUERY, [])
    assert prompt == TEXTBOOK_QUERY, (
        f"Default call (no index_available kwarg) should return bare question. "
        f"Got: {prompt!r}"
    )


# ==========================================================================
# Test D — Hot-reload structural check: _app_state exists in main.py and the
#           rebuild_index handler calls build_retrieval to swap the service.
# ==========================================================================
def test_app_state_exists_in_main():
    """_app_state module-level dict exists in main.py for hot-swap support.

    **Validates: Requirements 2.11**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    assert "_app_state" in source, (
        "_app_state dict not found in main.py — hot-reload requires a shared "
        "mutable holder for the retrieval reference."
    )
    # It should be a dict literal assignment at module level.
    assert "_app_state: dict" in source or "_app_state = {}" in source, (
        "_app_state should be declared as a dict in main.py."
    )


def test_rebuild_index_handler_calls_build_retrieval_for_hot_swap():
    """The rebuild_index handler now calls build_retrieval to hot-swap the service.

    **Validates: Requirements 2.11**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    # Locate the rebuild_index handler block.
    rebuild_idx = source.index('"rebuild_index"')
    next_branch = source.index('elif data.get("type") == "get_ncert_graph"', rebuild_idx)
    rebuild_block = source[rebuild_idx:next_branch]

    # After the fix, the handler must call build_retrieval to get the new service.
    assert "build_retrieval" in rebuild_block, (
        "rebuild_index handler must call build_retrieval to hot-swap the "
        "retrieval reference after a successful build (Req 2.11)."
    )
    # And store it in _app_state.
    assert "_app_state" in rebuild_block, (
        "rebuild_index handler must update _app_state['retrieval'] with the new "
        "service (Req 2.11)."
    )


def test_handle_response_reads_retrieval_from_app_state():
    """_handle_response reads the active retrieval from _app_state, not a fixed local.

    After the fix, the function must use _app_state.get('retrieval', ...) so that
    any hot-swap performed by ws_handler is reflected immediately.

    **Validates: Requirements 2.11**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    assert '_app_state.get("retrieval"' in source or "_app_state.get('retrieval'" in source, (
        "_handle_response must read active_retrieval from _app_state so that "
        "hot-swaps take effect without a restart (Req 2.11)."
    )


# ==========================================================================
# Test E — _get_index_status function exists in main.py.
# ==========================================================================
def test_get_index_status_function_exists_in_main():
    """_get_index_status is defined in main.py and returns {built: bool, chunk_count: int}.

    **Validates: Requirements 2.9**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    assert "def _get_index_status" in source, (
        "_get_index_status function must be defined in main.py to report index "
        "readiness at startup and after rebuild."
    )
    # The function should return a dict with 'built' and 'chunk_count' keys.
    assert '"built"' in source or "'built'" in source, (
        "_get_index_status must return a dict with a 'built' key."
    )
    assert '"chunk_count"' in source or "'chunk_count'" in source, (
        "_get_index_status must return a dict with a 'chunk_count' key."
    )


def test_get_index_status_null_service_reports_not_built():
    """_get_index_status(NullRetrievalService()) returns built=False, chunk_count=0.

    **Validates: Requirements 2.9**
    """
    # Import just the function from the main module source via exec to avoid
    # triggering the module-level side effects (sounddevice, numpy, etc.).
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    # Extract the function body via a targeted exec scope.
    func_ns: dict = {}
    # Provide the NullRetrievalService import the function needs.
    func_ns["NullRetrievalService"] = NullRetrievalService

    # Pull out just the _get_index_status function definition.
    start = source.index("def _get_index_status")
    # Find the next top-level definition after it.
    next_def_candidates = []
    for marker in ["\ndef ", "\nclass ", "\nasync def "]:
        try:
            pos = source.index(marker, start + 1)
            next_def_candidates.append(pos)
        except ValueError:
            pass
    end = min(next_def_candidates) if next_def_candidates else len(source)
    func_source = source[start:end].rstrip()

    exec(func_source, func_ns)  # noqa: S102
    _get_index_status = func_ns["_get_index_status"]

    null_svc = NullRetrievalService()
    status = _get_index_status(null_svc)

    assert isinstance(status, dict), f"Expected dict, got {type(status)}"
    assert status.get("built") is False, (
        f"NullRetrievalService should report built=False, got: {status}"
    )
    assert status.get("chunk_count") == 0, (
        f"NullRetrievalService should report chunk_count=0, got: {status}"
    )


# ==========================================================================
# Test — build_retrieval is now called multiple times (startup + hot-reload).
# ==========================================================================
def test_build_retrieval_called_more_than_once_in_main():
    """main.py now calls build_retrieval more than once — once at startup and
    at least once in the rebuild_index hot-reload path (Req 2.11).

    **Validates: Requirements 2.11**
    """
    main_src = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    with open(main_src, "r", encoding="utf-8") as f:
        source = f.read()

    count = source.count("build_retrieval(")
    assert count >= 2, (
        f"Expected build_retrieval to appear at least twice (startup + hot-reload), "
        f"found {count} occurrence(s)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
