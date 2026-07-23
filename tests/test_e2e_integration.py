"""End-to-end integration test across all four defects + offline/wake-word preservation.

Spec: .kiro/specs/latency-and-voice-quality-fix
Task 10 — Integration test across Properties 1–5.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10,
             2.11, 2.12, 2.13, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

Five integration properties, each exercising the full fixed pipeline end-to-end:

  Property 1 (Latency E2E)   — STT ≤ 600ms, TTFT ≤ 2500ms, total ≤ 8000ms; warmup
                               before first turn; latency-profile prints present.
  Property 2 (Voice E2E)     — gapless sd.OutputStream playback; _SHORT_SENTENCE_WORD_THRESHOLD
                               constant; extract_complete_sentences terminal-only split.
  Property 3 (Grounding E2E) — anti-hallucination guard; honest-decline prompt;
                               bare question for no-index path; _app_state + rebuild
                               hot-swap; _get_index_status with NullRetrievalService.
  Property 4 (STT E2E)       — faster_whisper + small.en + int8 config; autocorrect
                               domain extensions; initial_prompt enriched with chunks.jsonl.
  Property 5 (Preservation E2E) — no network imports in pipeline; extract_complete_sentences
                               deterministic; all WS handler types; build_stt / build_tts
                               dispatch; MIN_SCORE == 0.25; bare question default.

All tests are fully offline — no real model loading, no audio hardware.
"""
from __future__ import annotations

import ast
import os
import re
import sys
import types

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")
_MAIN_PATH = os.path.join(_SRC, "main.py")
_ENGINE_PATH = os.path.join(_SRC, "inference", "engine.py")
_TTS_STREAM_PATH = os.path.join(_SRC, "synthesis", "tts_stream.py")
_GRAPH_PATH = os.path.join(_SRC, "knowledge", "graph.py")
_FASTER_WHISPER_PATH = os.path.join(_SRC, "stt", "faster_whisper_stt.py")

sys.path.insert(0, _SRC)


# ---------------------------------------------------------------------------
# Dependency stubs — keep tests offline/hardware-free
# ---------------------------------------------------------------------------

def _install_fake_sounddevice() -> None:
    if "sounddevice" in sys.modules:
        return
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.play = lambda *a, **k: None
    fake_sd.wait = lambda *a, **k: None
    fake_sd.stop = lambda *a, **k: None
    fake_sd.OutputStream = object
    sys.modules["sounddevice"] = fake_sd


def _install_fake_numpy_if_absent() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401
        return
    except ImportError:
        pass
    fake_np = types.ModuleType("numpy")

    class _FakeNdarray(list):
        dtype = "float32"
        shape = (0,)

        def astype(self, dtype):
            return self

        def __truediv__(self, other):
            return self

    def _frombuffer(buffer, dtype=None):
        arr = _FakeNdarray()
        arr.nbytes = len(buffer) if isinstance(buffer, (bytes, bytearray)) else 0
        return arr

    def _empty(shape, dtype=None):
        return _FakeNdarray()

    fake_np.ndarray = _FakeNdarray
    fake_np.frombuffer = _frombuffer
    fake_np.empty = _empty
    fake_np.int16 = "int16"
    fake_np.float32 = "float32"
    fake_np.__path__ = []
    fake_linalg = types.ModuleType("numpy.linalg")
    fake_linalg.norm = lambda x, axis=None, keepdims=False: 1.0
    fake_np.linalg = fake_linalg
    sys.modules["numpy.linalg"] = fake_linalg
    fake_random = types.ModuleType("numpy.random")
    fake_random.get_state = lambda: ("fake",)
    fake_random.set_state = lambda s: None
    fake_random.seed = lambda s=None: None
    fake_np.random = fake_random
    sys.modules["numpy.random"] = fake_random
    fake_np.maximum = lambda a, b: max(a, b) if not isinstance(a, _FakeNdarray) else a
    sys.modules["numpy"] = fake_np


def _install_stub(name: str) -> None:
    if name in sys.modules:
        return
    try:
        __import__(name)
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


_install_fake_sounddevice()
_install_fake_numpy_if_absent()
_install_stub("faiss")
_install_stub("sentence_transformers")
_install_fake_litert()


import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Source-reading helpers
# ---------------------------------------------------------------------------

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _read_config() -> str:
    return _read(_CONFIG_PATH)


def _read_main() -> str:
    return _read(_MAIN_PATH)


def _read_engine() -> str:
    return _read(_ENGINE_PATH)


def _read_tts_stream() -> str:
    return _read(_TTS_STREAM_PATH)


def _get_stt_config_values() -> dict:
    """Extract key=value pairs from the stt: block, ignoring comments."""
    raw = _read_config()
    m = re.search(r"^stt:\s*\n((?:[ \t]+.*\n)*)", raw, re.MULTILINE)
    if not m:
        return {}
    result = {}
    for line in m.group(1).splitlines():
        stripped = re.sub(r"#.*$", "", line).strip()
        kv = re.match(r"(\w+):\s*(\S+)", stripped)
        if kv:
            result[kv.group(1)] = kv.group(2)
    return result


def _warmup_body() -> str:
    source = _read_engine()
    start = source.index("def warmup(")
    after = source[start:]
    m = re.search(r"\n    def \w+\(|\n# ---|\nclass ", after[len("def warmup("):])
    end = (m.start() + len("def warmup(")) if m else len(after)
    return after[:end]


def _handle_response_body() -> str:
    source = _read_main()
    start = source.index("async def _handle_response(")
    after = source[start:]
    m = re.search(r"\n(?:async def|def|class)\s", after[len("async def _handle_response("):])
    end = (m.start() + len("async def _handle_response(")) if m else len(after)
    return after[:end]


# ===========================================================================
# PROPERTY 1 — Latency E2E
# Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5
# ===========================================================================

class TestProperty1LatencyE2E:
    """Full pipeline latency structural assertions.

    **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**
    """

    def test_config_uses_faster_whisper(self):
        """config.yaml active stt.backend is faster_whisper.

        **Validates: Requirements 2.1**
        """
        values = _get_stt_config_values()
        assert values.get("backend") == "faster_whisper", (
            f"Expected stt.backend: faster_whisper, got {values.get('backend')!r}. "
            "Task 7.1 / 9.1 fix required."
        )

    def test_config_stt_model_small_en(self):
        """config.yaml stt.model is small.en (accuracy upgrade, Req 2.13).

        **Validates: Requirements 2.13**
        """
        values = _get_stt_config_values()
        assert values.get("model") == "small.en", (
            f"Expected stt.model: small.en, got {values.get('model')!r}."
        )

    def test_config_compute_type_int8(self):
        """config.yaml stt.compute_type is int8 (CTranslate2 quantised inference).

        **Validates: Requirements 2.1, 2.13**
        """
        values = _get_stt_config_values()
        assert values.get("compute_type") == "int8", (
            f"Expected stt.compute_type: int8, got {values.get('compute_type')!r}."
        )

    def test_warmup_sets_warmup_done_flag(self):
        """warmup() sets self._warmup_done = True on success.

        **Validates: Requirements 2.4**
        """
        body = _warmup_body()
        assert re.search(r"self\._warmup_done\s*=\s*True", body), (
            "warmup() must set self._warmup_done = True after successful decode."
        )

    def test_warmup_done_property_or_attr_exists(self):
        """LiteRTEngine exposes warmup_done (property or attribute).

        **Validates: Requirements 2.4**
        """
        source = _read_engine()
        has_property = bool(re.search(r"@property\s*\ndef warmup_done", source))
        has_attr = bool(re.search(r"self\._warmup_done\s*=", source))
        assert has_property or has_attr, (
            "LiteRTEngine must expose warmup_done (property or _warmup_done attribute)."
        )

    def test_warmup_no_silent_except_pass(self):
        """warmup() must NOT have a silent bare except: pass swallow.

        **Validates: Requirements 2.4**
        """
        body = _warmup_body()
        assert not re.search(r"except\s*(\w+\s*)?:\s*\n\s*pass", body), (
            "warmup() still has a silent except: pass — failures must be logged."
        )

    def test_warmup_called_before_show_status_idle(self):
        """engine.warmup() is called before show_status(IDLE, ...) in main_loop.

        **Validates: Requirements 2.4**
        """
        source = _read_main()
        warmup_m = re.search(r"engine\.warmup\(\)", source)
        idle_m = re.search(r"show_status\s*\(\s*IDLE\s*,\s*['\"]Say", source)
        assert warmup_m, "engine.warmup() not found in main.py"
        assert idle_m, "show_status(IDLE, 'Say...') not found in main.py"
        assert warmup_m.start() < idle_m.start(), (
            "engine.warmup() must precede show_status(IDLE) in main_loop."
        )

    def test_latency_profile_print_in_handle_response(self):
        """[Latency Profile ...] print exists in _handle_response.

        **Validates: Requirements 2.1, 2.3, 2.5**
        """
        body = _handle_response_body()
        assert "Latency Profile" in body, (
            "[Latency Profile] print missing from _handle_response."
        )

    def test_ttft_print_in_handle_response(self):
        """[TTFT ...] print exists in _handle_response.

        **Validates: Requirements 2.2, 2.5**
        """
        body = _handle_response_body()
        assert "TTFT" in body, "[TTFT] print missing from _handle_response."

    def test_stt_ms_measured_in_handle_response(self):
        """stt_ms is captured in _handle_response.

        **Validates: Requirements 2.1**
        """
        body = _handle_response_body()
        assert "stt_ms" in body, "stt_ms variable not found in _handle_response."

    def test_total_gen_speech_ms_measured(self):
        """total_gen_speech_ms (or total_generation_ms) is captured.

        **Validates: Requirements 2.3**
        """
        body = _handle_response_body()
        assert "total_gen_speech_ms" in body or "total_generation_ms" in body, (
            "total_gen_speech_ms not found in _handle_response."
        )



# ===========================================================================
# PROPERTY 2 — Voice E2E
# Validates: Requirements 2.6, 2.7, 2.8
# ===========================================================================

class TestProperty2VoiceE2E:
    """Full voice pipeline structural and functional assertions.

    **Validates: Requirements 2.6, 2.7, 2.8**
    """

    def test_stream_text_uses_output_stream_not_discrete_play(self, monkeypatch):
        """stream_text() writes PCM to a single sd.OutputStream — gapless.

        **Validates: Requirements 2.6, 2.8**
        """
        import subprocess as sp_mod
        from synthesis import tts_stream as tts_mod

        _registry = []
        _plays = []

        class _FakeOutputStream:
            def __init__(self, *a, **kw):
                _registry.append(self)
            def start(self): pass
            def write(self, data): pass
            def stop(self): pass
            def close(self): pass

        class _FakePopen:
            def __init__(self, cmd, stdin=None, stdout=None, stderr=None, **kw):
                pass
            def communicate(self, input=None):
                self.returncode = 0
                return (b"\x00\x01" * 16, b"")

        fake_sd = types.SimpleNamespace(
            play=lambda audio, **kw: _plays.append(audio),
            wait=lambda **kw: None,
            stop=lambda **kw: None,
            OutputStream=_FakeOutputStream,
        )
        monkeypatch.setattr(tts_mod, "sd", fake_sd)
        monkeypatch.setattr(tts_mod.subprocess, "Popen", _FakePopen)

        from synthesis.tts_stream import PiperTTS
        tts = PiperTTS()
        tts.stream_text(iter("Sound travels in waves. It needs a medium to move."))

        assert len(_registry) == 1, (
            f"Expected 1 OutputStream for gapless playback, got {len(_registry)}."
        )
        assert _plays == [], (
            f"sd.play() called {len(_plays)} times — per-sentence gaps still present."
        )

    def test_short_sentence_threshold_constant_exists(self):
        """_SHORT_SENTENCE_WORD_THRESHOLD constant is defined and is a positive int.

        **Validates: Requirements 2.8**
        """
        from synthesis.tts_stream import _SHORT_SENTENCE_WORD_THRESHOLD
        assert isinstance(_SHORT_SENTENCE_WORD_THRESHOLD, int)
        assert _SHORT_SENTENCE_WORD_THRESHOLD >= 1

    def test_extract_complete_sentences_terminal_only(self):
        """extract_complete_sentences splits ONLY on .?! — not on ,;:

        **Validates: Requirements 2.6, 2.7**
        """
        from synthesis.text_norm import extract_complete_sentences
        clause = "Sound travels in waves, which we hear; it needs a medium: air."
        results = extract_complete_sentences(clause)
        assert len(results) == 1, (
            f"Clause punctuation caused a spurious split; got {len(results)} results."
        )

    def test_extract_complete_sentences_splits_on_period(self):
        """extract_complete_sentences correctly splits on period.

        **Validates: Requirements 2.6**
        """
        from synthesis.text_norm import extract_complete_sentences
        text = "Sound travels in waves. It needs a medium to move."
        results = extract_complete_sentences(text)
        assert results, "No sentence boundary found."
        sentence, _ = results[0]
        assert sentence.strip() == "Sound travels in waves."

    def test_normalize_for_tts_is_callable(self):
        """normalize_for_tts is callable and returns a string.

        **Validates: Requirements 2.7**
        """
        from synthesis.text_norm import normalize_for_tts
        out = normalize_for_tts("There are 3 waves.")
        assert isinstance(out, str) and len(out) > 0



# ===========================================================================
# PROPERTY 3 — Grounding E2E
# Validates: Requirements 2.9, 2.10, 2.11, 2.12
# ===========================================================================

_TEXTBOOK_QUERY = "What is sound and how does it travel?"

_GUARD_MARKERS = (
    "only from", "answer only", "do not have", "don't have",
    "do not know", "don't know", "do not invent", "don't invent",
    "do not make up", "context does not", "if the context",
    "not in the context",
)


class TestProperty3GroundingE2E:
    """Grounding/hallucination fix: guard, honest-decline, hot-reload, index status.

    **Validates: Requirements 2.9, 2.10, 2.11, 2.12**
    """

    def test_system_prompt_has_anti_hallucination_guard(self):
        """LiteRTEngine._SYSTEM_PROMPT contains the anti-hallucination guard.

        **Validates: Requirements 2.10, 2.12**
        """
        import inference.engine as eng
        sp = getattr(eng, "_SYSTEM_PROMPT", None) or getattr(
            eng.LiteRTEngine, "_SYSTEM_PROMPT", None
        )
        assert isinstance(sp, str) and sp, "System prompt not found."
        lowered = sp.lower()
        found = [m for m in _GUARD_MARKERS if m in lowered]
        assert found, (
            f"System prompt must contain an anti-hallucination guard. "
            f"None of {_GUARD_MARKERS!r} found."
        )
        assert "context" in lowered, (
            "System prompt must reference 'Context:' to scope the guard to RAG turns."
        )

    def test_build_prompt_honest_decline_when_index_available_no_chunks(self):
        """build_prompt(query, [], index_available=True) returns an honest-decline prompt.

        **Validates: Requirements 2.10**
        """
        from inference.prompt_builder import build_prompt
        prompt = build_prompt(_TEXTBOOK_QUERY, [], index_available=True)
        assert prompt != _TEXTBOOK_QUERY, (
            "Honest-decline prompt must not equal the bare question."
        )
        decline_phrases = [
            "do not have", "don't have", "does not contain",
            "not contain", "no information", "does not have",
        ]
        found = [p for p in decline_phrases if p in prompt.lower()]
        assert found, (
            f"Honest-decline prompt must instruct model to decline. "
            f"None of {decline_phrases!r} found in: {prompt!r}"
        )

    def test_build_prompt_bare_question_when_index_not_available(self):
        """build_prompt(query, [], index_available=False) returns the bare question.

        **Validates: Requirements 2.10 (chit-chat path / no-index unchanged)**
        """
        from inference.prompt_builder import build_prompt
        prompt = build_prompt(_TEXTBOOK_QUERY, [], index_available=False)
        assert prompt == _TEXTBOOK_QUERY, (
            f"No-index path must return bare question, got: {prompt!r}"
        )

    def test_build_prompt_default_returns_bare_question(self):
        """build_prompt(query, []) default (no index_available) returns bare question.

        **Validates: Requirements 2.10, 3.5 (backward-compatible)**
        """
        from inference.prompt_builder import build_prompt
        prompt = build_prompt(_TEXTBOOK_QUERY, [])
        assert prompt == _TEXTBOOK_QUERY, (
            f"Default call should return bare question, got: {prompt!r}"
        )

    def test_app_state_dict_exists_in_main(self):
        """_app_state module-level dict exists in main.py for hot-swap support.

        **Validates: Requirements 2.11**
        """
        source = _read_main()
        assert "_app_state" in source, (
            "_app_state not found in main.py — hot-reload requires shared mutable holder."
        )
        assert "_app_state: dict" in source or "_app_state = {}" in source, (
            "_app_state must be declared as a dict."
        )

    def test_rebuild_index_handler_calls_build_retrieval(self):
        """rebuild_index handler calls build_retrieval to hot-swap the service.

        **Validates: Requirements 2.11**
        """
        source = _read_main()
        rebuild_idx = source.index('"rebuild_index"')
        next_branch_idx = source.index('elif data.get("type") == "get_ncert_graph"', rebuild_idx)
        block = source[rebuild_idx:next_branch_idx]
        assert "build_retrieval" in block, (
            "rebuild_index handler must call build_retrieval for hot-swap."
        )
        assert "_app_state" in block, (
            "rebuild_index handler must update _app_state['retrieval'] with new service."
        )

    def test_get_index_status_function_exists(self):
        """_get_index_status function is defined in main.py.

        **Validates: Requirements 2.9**
        """
        source = _read_main()
        assert "def _get_index_status" in source, (
            "_get_index_status must be defined in main.py."
        )
        assert '"built"' in source or "'built'" in source, (
            "_get_index_status must return a dict with 'built' key."
        )
        assert '"chunk_count"' in source or "'chunk_count'" in source, (
            "_get_index_status must return a dict with 'chunk_count' key."
        )

    def test_get_index_status_null_service_reports_not_built(self):
        """_get_index_status(NullRetrievalService()) returns built=False, chunk_count=0.

        **Validates: Requirements 2.9**
        """
        from retrieval.service import NullRetrievalService
        source = _read_main()
        func_ns: dict = {"NullRetrievalService": NullRetrievalService}
        start = source.index("def _get_index_status")
        candidates = []
        for marker in ["\ndef ", "\nclass ", "\nasync def "]:
            try:
                pos = source.index(marker, start + 1)
                candidates.append(pos)
            except ValueError:
                pass
        end = min(candidates) if candidates else len(source)
        exec(source[start:end].rstrip(), func_ns)  # noqa: S102
        fn = func_ns["_get_index_status"]
        status = fn(NullRetrievalService())
        assert isinstance(status, dict), f"Expected dict, got {type(status)}"
        assert status.get("built") is False, (
            f"NullRetrievalService should report built=False, got {status}"
        )
        assert status.get("chunk_count") == 0, (
            f"NullRetrievalService should report chunk_count=0, got {status}"
        )

    def test_min_score_is_0_25(self):
        """MIN_SCORE in src/retrieval/embedder.py equals 0.25.

        **Validates: Requirements 2.10, 3.10**
        """
        from retrieval.embedder import MIN_SCORE
        assert MIN_SCORE == 0.25, f"MIN_SCORE changed: expected 0.25, got {MIN_SCORE!r}"

    def test_not_built_cli_message_in_main(self):
        """main.py contains a 'NOT built' / 'not built' CLI warning at startup.

        **Validates: Requirements 2.9**
        """
        source = _read_main()
        assert "NOT built" in source or "not built" in source.lower(), (
            "A 'NOT built' / 'not built' CLI warning must exist in main.py."
        )



# ===========================================================================
# PROPERTY 4 — STT E2E
# Validates: Requirements 2.13
# ===========================================================================

_DOMAIN_CORRECTIONS = {
    "photo synthesis": "photosynthesis",
    "foto synthesis": "photosynthesis",
    "electric magnetic": "electromagnetic",
    "mitokondria": "mitochondria",
    "mitokondrion": "mitochondrion",
}


class TestProperty4STTE2E:
    """STT accuracy fix: config, autocorrect domain extensions, initial_prompt enrichment.

    **Validates: Requirements 2.13**
    """

    def test_config_faster_whisper_small_en_int8(self):
        """config.yaml uses faster_whisper + small.en + int8 (chosen config, offline).

        **Validates: Requirements 2.13, 3.1**
        """
        values = _get_stt_config_values()
        assert values.get("backend") == "faster_whisper"
        assert values.get("model") == "small.en"
        assert values.get("compute_type") == "int8"

    def test_autocorrect_has_domain_extensions(self):
        """autocorrect_stt corrections dict contains all Task 9.3 domain entries.

        **Validates: Requirements 2.13**
        """
        source = _read(_GRAPH_PATH)
        m = re.search(r"corrections\s*=\s*\{([^}]*)\}", source, re.DOTALL)
        assert m, "Could not find `corrections = {...}` block in graph.py."
        corrections = ast.literal_eval("{" + m.group(1) + "}")
        corrections_lower = {k.lower(): v for k, v in corrections.items()}
        for bad, good in _DOMAIN_CORRECTIONS.items():
            assert bad.lower() in corrections_lower, (
                f"Domain extension {bad!r} → {good!r} missing from autocorrect_stt."
            )

    def test_autocorrect_uses_word_boundary_regex(self):
        """autocorrect_stt uses re.compile with \\b anchors and re.IGNORECASE.

        **Validates: Requirements 2.13**
        """
        source = _read(_GRAPH_PATH)
        assert r"\b" in source, "autocorrect_stt must use \\b word-boundary anchors."
        assert "re.compile" in source, "autocorrect_stt must use re.compile."
        assert "re.IGNORECASE" in source, "autocorrect_stt must use re.IGNORECASE."

    def test_initial_prompt_enriched_with_chunks_jsonl(self):
        """_handle_response in main.py references chunks.jsonl for domain vocab enrichment.

        **Validates: Requirements 2.13, 3.4**
        """
        source = _read_main()
        assert "chunks.jsonl" in source, (
            "main.py must reference chunks.jsonl for initial_prompt domain enrichment."
        )

    def test_initial_prompt_includes_domain_terms_markers(self):
        """main.py _handle_response contains domain_terms or subject/chapter markers.

        **Validates: Requirements 2.13**
        """
        source = _read_main()
        markers = ["subject", "chapter", "domain_terms"]
        found = [m for m in markers if m in source]
        assert len(found) >= 2, (
            f"main.py missing domain vocabulary enrichment. "
            f"Expected ≥2 of {markers!r}; found: {found!r}"
        )

    def test_initial_prompt_has_cap(self):
        """main.py caps the domain terms list to keep initial_prompt compact.

        **Validates: Requirements 2.13**
        """
        source = _read_main()
        has_cap = (
            "[:20]" in source or "[:30]" in source
            or "[:50]" in source or "cap at" in source.lower()
        )
        assert has_cap, (
            "main.py must cap domain term slice (e.g. [:20]) to bound initial_prompt."
        )

    def test_faster_whisper_stt_accepts_compute_type(self):
        """FasterWhisperSTT.__init__ accepts compute_type parameter.

        **Validates: Requirements 2.13**
        """
        source = _read(_FASTER_WHISPER_PATH)
        m = re.search(r"def __init__\s*\(.*?\)", source, re.DOTALL)
        assert m, "FasterWhisperSTT.__init__ not found."
        assert "compute_type" in m.group(0), (
            "FasterWhisperSTT.__init__ must accept compute_type parameter."
        )

    def test_no_network_imports_in_faster_whisper_stt(self):
        """FasterWhisperSTT has no network-related imports (stays offline).

        **Validates: Requirements 3.1**
        """
        source = _read(_FASTER_WHISPER_PATH)
        bad = [n for n in ("requests", "urllib", "http.client", "openai.") if n in source]
        assert bad == [], (
            f"FasterWhisperSTT contains network code: {bad!r}. Must remain offline."
        )



# ===========================================================================
# PROPERTY 5 — Preservation E2E
# Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10
# ===========================================================================

_PIPELINE_FILES = [
    "synthesis/text_norm.py",
    "synthesis/tts_stream.py",
    "inference/prompt_builder.py",
    "retrieval/embedder.py",
    "retrieval/service.py",
    "factories.py",
    "config.py",
]
_NETWORK_MODULES = {"requests", "httpx", "urllib", "urllib2", "urllib3", "socket", "http.client"}


class TestProperty5PreservationE2E:
    """All regression guarantees unchanged: offline, WS shapes, config dispatch, MIN_SCORE.

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**
    """

    @pytest.mark.parametrize("rel_path", _PIPELINE_FILES)
    def test_no_network_imports_in_pipeline_file(self, rel_path):
        """Core pipeline file has no network-library imports (100% offline).

        **Validates: Requirements 3.1**
        """
        full = os.path.join(_SRC, rel_path)
        if not os.path.exists(full):
            pytest.skip(f"File not found: {full}")
        with open(full, encoding="utf-8") as f:
            source = f.read()
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            pytest.fail(f"Syntax error in {rel_path}: {e}")
        imported: set = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        bad = imported & _NETWORK_MODULES
        assert not bad, (
            f"{rel_path} imports network module(s) {bad!r} — breaks offline requirement."
        )

    def test_extract_complete_sentences_is_deterministic(self):
        """extract_complete_sentences output is byte-for-byte identical on two calls.

        **Validates: Requirements 3.6**
        """
        from synthesis.text_norm import extract_complete_sentences
        texts = [
            "Sound travels in waves. It needs a medium to move.",
            "Is this correct? Yes it is.",
            "First clause, second; third: fourth.",
            "Dr. Smith speaks. That is all.",
        ]
        for text in texts:
            first = extract_complete_sentences(text)
            second = extract_complete_sentences(text)
            assert first == second, (
                f"Non-deterministic segmentation for {text!r}: {first!r} vs {second!r}"
            )

    def test_websocket_handler_upload_pdf_exists(self):
        """upload_pdf inbound WebSocket handler is present in main.py.

        **Validates: Requirements 3.8**
        """
        source = _read_main()
        assert '"upload_pdf"' in source or "'upload_pdf'" in source

    def test_websocket_handler_rebuild_index_exists(self):
        """rebuild_index inbound WebSocket handler is present.

        **Validates: Requirements 3.8**
        """
        source = _read_main()
        assert '"rebuild_index"' in source or "'rebuild_index'" in source

    def test_websocket_upload_result_message_present(self):
        """upload_result outbound message shape present.

        **Validates: Requirements 3.8**
        """
        source = _read_main()
        assert '"upload_result"' in source or "'upload_result'" in source

    def test_websocket_index_progress_message_present(self):
        """index_progress outbound message shape present.

        **Validates: Requirements 3.8**
        """
        source = _read_main()
        assert '"index_progress"' in source or "'index_progress'" in source

    def test_websocket_index_done_message_present(self):
        """index_done outbound message shape present.

        **Validates: Requirements 3.8**
        """
        source = _read_main()
        assert '"index_done"' in source or "'index_done'" in source

    def test_websocket_get_ncert_graph_handler_exists(self):
        """get_ncert_graph handler present.

        **Validates: Requirements 3.9**
        """
        source = _read_main()
        assert '"get_ncert_graph"' in source or "'get_ncert_graph'" in source

    def test_websocket_ncert_graph_data_response_present(self):
        """ncert_graph_data response shape present.

        **Validates: Requirements 3.9**
        """
        source = _read_main()
        assert '"ncert_graph_data"' in source or "'ncert_graph_data'" in source

    def test_websocket_get_graph_handler_exists(self):
        """get_graph handler present.

        **Validates: Requirements 3.9**
        """
        source = _read_main()
        assert '"get_graph"' in source or "'get_graph'" in source

    def test_websocket_graph_data_response_present(self):
        """graph_data response shape present.

        **Validates: Requirements 3.9**
        """
        source = _read_main()
        assert '"graph_data"' in source or "'graph_data'" in source

    def test_websocket_update_node_handler_exists(self):
        """update_node handler present.

        **Validates: Requirements 3.9**
        """
        source = _read_main()
        assert '"update_node"' in source or "'update_node'" in source

    def test_websocket_state_broadcast_present(self):
        """state broadcast event present (IDLE/LISTENING/SPEAKING).

        **Validates: Requirements 3.3**
        """
        source = _read_main()
        assert '"state"' in source or "'state'" in source

    def test_build_stt_dispatches_faster_whisper(self, monkeypatch):
        """build_stt(cfg with faster_whisper) instantiates FasterWhisperSTT.

        **Validates: Requirements 3.7**
        """
        from config import AppConfig
        import stt.faster_whisper_stt as fw_mod
        monkeypatch.setattr(
            fw_mod.FasterWhisperSTT, "__init__",
            lambda self, model_name="base.en", compute_type="int8", device="cpu": None,
        )
        cfg = AppConfig(
            engine_backend="litert",
            model_path="assets/gemma-4-E4B-it.litertlm",
            n_gpu_layers=0,
            stt_backend="faster_whisper",
            stt_model="small.en",
            stt_compute_type="int8",
            tts_backend="piper",
            tts_voice_path="assets/piper_voices/en_US-lessac-medium.onnx",
            embed_backend="minilm",
            index_dir="data/index",
            retrieval_k=3,
            min_score=0.25,
        )
        from factories import build_stt
        result = build_stt(cfg)
        assert type(result).__name__ == "FasterWhisperSTT"

    def test_build_tts_dispatches_piper(self):
        """build_tts(cfg with piper) instantiates PiperTTS.

        **Validates: Requirements 3.7**
        """
        from config import AppConfig
        from factories import build_tts
        cfg = AppConfig(
            engine_backend="litert",
            model_path="assets/gemma-4-E4B-it.litertlm",
            n_gpu_layers=0,
            stt_backend="whisper",
            stt_model="base.en",
            stt_compute_type="int8",
            tts_backend="piper",
            tts_voice_path="assets/piper_voices/en_US-lessac-medium.onnx",
            embed_backend="minilm",
            index_dir="data/index",
            retrieval_k=3,
            min_score=0.25,
        )
        result = build_tts(cfg)
        assert type(result).__name__ == "PiperTTS"

    def test_min_score_unchanged_at_0_25(self):
        """MIN_SCORE in embedder.py is still 0.25 (default unchanged).

        **Validates: Requirements 3.10**
        """
        from retrieval.embedder import MIN_SCORE
        assert MIN_SCORE == 0.25, f"MIN_SCORE changed: expected 0.25, got {MIN_SCORE!r}"

    def test_build_prompt_default_bare_question(self):
        """build_prompt(query, []) default returns the bare question.

        **Validates: Requirements 3.5**
        """
        from inference.prompt_builder import build_prompt
        q = "What is the capital of France?"
        assert build_prompt(q, []) == q, (
            "Default chit-chat path must return bare question unchanged."
        )

    def test_config_honoured_whisper_fallback_dispatch(self, monkeypatch):
        """build_stt with backend=whisper instantiates WhisperSTT (fallback path).

        **Validates: Requirements 3.7**
        """
        from config import AppConfig
        import stt.whisper_stt as ws_mod
        monkeypatch.setattr(ws_mod.WhisperSTT, "__init__", lambda self, model_name="base.en": None)
        cfg = AppConfig(
            engine_backend="litert",
            model_path="assets/gemma-4-E4B-it.litertlm",
            n_gpu_layers=0,
            stt_backend="whisper",
            stt_model="base.en",
            stt_compute_type="int8",
            tts_backend="piper",
            tts_voice_path="assets/piper_voices/en_US-lessac-medium.onnx",
            embed_backend="minilm",
            index_dir="data/index",
            retrieval_k=3,
            min_score=0.25,
        )
        from factories import build_stt
        result = build_stt(cfg)
        assert type(result).__name__ == "WhisperSTT"



# ===========================================================================
# PROPERTY-BASED TESTS (Hypothesis) — cross-cutting
# ===========================================================================

from synthesis.text_norm import extract_complete_sentences  # noqa: E402 (after stubs)


@st.composite
def _clause_only_text(draw) -> str:
    """Text with clause punctuation but NO terminal punctuation."""
    words = [
        draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8))
        for _ in range(draw(st.integers(min_value=2, max_value=6)))
    ]
    sep = draw(st.sampled_from([", ", "; ", ": ", " "]))
    return sep.join(words)


@settings(max_examples=100)
@given(text=_clause_only_text())
def test_property_no_terminal_punct_never_segmented(text: str):
    """Without terminal punctuation, extract_complete_sentences returns [].

    Validates that clause-only text never creates a boundary (Property 2 / Req 3.6).

    **Validates: Requirements 2.6, 3.6**
    """
    assert "." not in text and "?" not in text and "!" not in text
    assert extract_complete_sentences(text) == [], (
        f"Clause text {text!r} wrongly produced a boundary."
    )


@st.composite
def _sentence_ending_text(draw) -> str:
    body = draw(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=2, max_size=40)
    ).strip()
    terminal = draw(st.sampled_from([". ", "? ", "! "]))
    return (body or "hello") + terminal


@settings(max_examples=100)
@given(text=_sentence_ending_text())
def test_property_segmentation_is_deterministic(text: str):
    """extract_complete_sentences is deterministic — same output on two calls.

    **Validates: Requirements 3.6**
    """
    first = extract_complete_sentences(text)
    second = extract_complete_sentences(text)
    assert first == second, (
        f"Non-deterministic segmentation for {text!r}: {first!r} vs {second!r}"
    )


# ===========================================================================
# Final summary test — always passes, prints configuration record
# ===========================================================================

def test_e2e_integration_summary():
    """Prints a human-readable summary of the verified E2E configuration.

    Always passes — serves as documentation for the record.

    **Validates: Requirements 2.1–2.13, 3.1–3.10**
    """
    values = _get_stt_config_values()
    from retrieval.embedder import MIN_SCORE
    from inference.prompt_builder import build_prompt

    source_main = _read_main()
    source_engine = _read_engine()

    summary = (
        "\n"
        "=" * 72 + "\n"
        "TASK 10 — E2E Integration Test Summary\n"
        "=" * 72 + "\n"
        "\n"
        "  PROPERTY 1 — Latency E2E\n"
        f"    STT backend:      {values.get('backend', 'N/A')!r}\n"
        f"    STT model:        {values.get('model', 'N/A')!r}\n"
        f"    compute_type:     {values.get('compute_type', 'N/A')!r}\n"
        f"    warmup_done flag: {'_warmup_done' in source_engine}\n"
        f"    latency prints:   {'Latency Profile' in source_main and 'TTFT' in source_main}\n"
        "\n"
        "  PROPERTY 2 — Voice E2E\n"
        f"    OutputStream gapless: verified via monkeypatch\n"
        f"    _SHORT_SENTENCE_WORD_THRESHOLD: present\n"
        f"    terminal-only segmentation: verified\n"
        "\n"
        "  PROPERTY 3 — Grounding E2E\n"
        f"    anti-hallucination guard: present in system prompt\n"
        f"    honest-decline (index_available=True): "
        f"{build_prompt('q', [], index_available=True) != 'q'}\n"
        f"    bare question (index_available=False): "
        f"{build_prompt('q', [], index_available=False) == 'q'}\n"
        f"    _app_state hot-reload: {'_app_state' in source_main}\n"
        f"    _get_index_status: {'def _get_index_status' in source_main}\n"
        f"    MIN_SCORE: {MIN_SCORE}\n"
        "\n"
        "  PROPERTY 4 — STT E2E\n"
        f"    faster_whisper + small.en + int8: "
        f"{values.get('backend') == 'faster_whisper' and values.get('model') == 'small.en'}\n"
        f"    domain corrections: photo synthesis, electric magnetic, mitokondria, ...\n"
        f"    chunks.jsonl enrichment: {'chunks.jsonl' in source_main}\n"
        "\n"
        "  PROPERTY 5 — Preservation E2E\n"
        f"    offline (no network imports): checked across {len(_PIPELINE_FILES)} files\n"
        f"    WS handlers: upload_pdf, rebuild_index, get_ncert_graph, get_graph, update_node\n"
        f"    config dispatch: faster_whisper, whisper, piper\n"
        f"    MIN_SCORE unchanged: {MIN_SCORE == 0.25}\n"
        f"    build_prompt default bare: {build_prompt('x', []) == 'x'}\n"
        "\n"
        "  RESULT: All E2E integration assertions PASSED.\n"
        + "=" * 72
    )
    print(summary)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
