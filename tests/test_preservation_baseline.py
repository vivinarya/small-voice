"""Preservation baseline tests — Property 5 (Task 5).

Run on UNFIXED code BEFORE any fix.  All tests here MUST PASS — they establish
the oracle that must remain true after every fix group is applied.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 3.10**

Sub-properties:
  P5.1 — Text segmentation is stable (terminal-only splits, deterministic)
  P5.2 — normalize_for_tts is stable (deterministic, consistent)
  P5.3 — Single comma-free sentence goes through unchanged (one segment, in order)
  P5.4 — WebSocket message shapes locked (structural source-code check)
  P5.5 — MIN_SCORE default unchanged (== 0.25)
  P5.6 — Config backend selection honored (build_stt / build_tts dispatch)
  P5.7 — Non-textbook chit-chat path preserved (bare question returned)
  P5.8 — Offline operation: no network imports in core pipeline files
"""
from __future__ import annotations

import os
import sys
import ast
import types

# ---------------------------------------------------------------------------
# Make src/ importable (same convention as all other tests in this project)
# ---------------------------------------------------------------------------
_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Install a fake sounddevice BEFORE importing tts_stream so the module loads
# in a hardware-free test environment.  Mirror the approach used in
# test_defect2_voice_exploration.py.
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
    """Install a minimal numpy stand-in if the real numpy is not available.

    Mirrors the approach used in test_defect2_voice_exploration.py so that
    modules which do ``import numpy as np`` can be imported in the test venv
    even when numpy is not installed.
    """
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
        arr = _FakeNdarray()
        return arr

    fake_np.ndarray = _FakeNdarray
    fake_np.frombuffer = _frombuffer
    fake_np.empty = _empty
    fake_np.int16 = "int16"
    fake_np.float32 = "float32"
    fake_np.__path__ = []  # mark as package

    # Minimal linalg stub (used by embedder._l2_normalize)
    fake_linalg = types.ModuleType("numpy.linalg")
    fake_linalg.norm = lambda x, axis=None, keepdims=False: 1.0
    fake_np.linalg = fake_linalg
    sys.modules["numpy.linalg"] = fake_linalg

    # Minimal random stub (used by hypothesis RNG seeding)
    fake_random = types.ModuleType("numpy.random")
    fake_random.get_state = lambda: ("fake",)
    fake_random.set_state = lambda s: None
    fake_random.seed = lambda s=None: None
    fake_np.random = fake_random
    sys.modules["numpy.random"] = fake_random

    # maximum stub (used by embedder._l2_normalize)
    fake_np.maximum = lambda a, b: max(a, b) if not isinstance(a, _FakeNdarray) else a

    sys.modules["numpy"] = fake_np


_install_fake_sounddevice()
_install_fake_numpy_if_absent()

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


# ============================================================================
# P5.1 — Text segmentation is stable
# ============================================================================

from synthesis.text_norm import extract_complete_sentences, normalize_for_tts


def _full_segment(text: str) -> list[str]:
    """Drain a text string into segments the same way stream_text does."""
    segments: list[str] = []
    buf = text
    while True:
        results = extract_complete_sentences(buf)
        if not results:
            break
        sentence, buf = results[0]
        if sentence.strip():
            segments.append(sentence.strip())
    if buf.strip():
        segments.append(buf.strip())
    return segments


class TestP51TextSegmentationStable:
    """P5.1 — Segmentation splits ONLY on .?! and is byte-for-byte deterministic.

    **Validates: Requirements 3.6**
    """

    def test_multi_sentence_with_commas_not_split_on_commas(self):
        """A comma-containing multi-sentence response splits at . only."""
        text = "Sound travels in waves. It needs a medium to move."
        segs = _full_segment(text)
        assert segs == ["Sound travels in waves.", "It needs a medium to move."]

    def test_comma_semicolon_colon_never_create_boundary(self):
        """Clause punctuation is never a split point."""
        text = "First clause, second; third: fourth."
        segs = _full_segment(text)
        assert len(segs) == 1
        assert "," in segs[0] and ";" in segs[0] and ":" in segs[0]

    def test_question_mark_boundary(self):
        text = "Is this correct? Yes it is."
        segs = _full_segment(text)
        assert segs[0] == "Is this correct?"

    def test_exclamation_mark_boundary(self):
        text = "Excellent! Now let us move on."
        segs = _full_segment(text)
        assert segs[0] == "Excellent!"

    def test_abbreviation_dr_not_split(self):
        text = "Dr. Smith says hello. That is all."
        segs = _full_segment(text)
        assert segs[0].startswith("Dr")

    def test_deterministic_repeated_call(self):
        """Identical input must yield identical output on two calls."""
        text = "The earth orbits the sun. The moon orbits the earth."
        assert _full_segment(text) == _full_segment(text)


# Hypothesis: for any text that ends with a terminal sentence boundary,
# calling extract_complete_sentences twice yields identical results.
_WORD = st.text(alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=1, max_size=40)


@st.composite
def _sentence_ending_text(draw):
    """Generate text that has at least one terminal sentence boundary."""
    body = draw(_WORD).strip()
    terminal = draw(st.sampled_from([". ", "? ", "! "]))
    return body + terminal if body else "hello" + terminal


@settings(max_examples=150)
@given(text=_sentence_ending_text())
def test_p51_segmentation_is_deterministic(text: str):
    """extract_complete_sentences output is byte-for-byte identical on two calls.

    **Validates: Requirements 3.6**
    """
    first = extract_complete_sentences(text)
    second = extract_complete_sentences(text)
    assert first == second, (
        f"Non-deterministic segmentation for {text!r}: "
        f"first={first!r}, second={second!r}"
    )


@st.composite
def _clause_only_text(draw):
    """Text with clause punctuation but NO terminal punctuation."""
    words = [draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8))
             for _ in range(draw(st.integers(min_value=2, max_value=6)))]
    sep = draw(st.sampled_from([", ", "; ", ": ", " "]))
    return sep.join(words)


@settings(max_examples=150)
@given(text=_clause_only_text())
def test_p51_clause_punctuation_never_triggers_boundary(text: str):
    """No terminal punctuation => no boundary found.

    **Validates: Requirements 3.6**
    """
    assert "." not in text and "?" not in text and "!" not in text
    assert extract_complete_sentences(text) == [], (
        f"Clause text '{text}' wrongly produced a boundary"
    )


# ============================================================================
# P5.2 — normalize_for_tts is stable
# ============================================================================


class TestP52NormalizeStable:
    """P5.2 — normalize_for_tts is deterministic and consistent.

    **Validates: Requirements 3.6**
    """

    def test_plain_text_unchanged(self):
        text = "The quick brown fox"
        assert normalize_for_tts(text) == text

    def test_number_expansion_consistent(self):
        assert normalize_for_tts("42") == "forty two"
        assert normalize_for_tts("7") == "seven"
        assert normalize_for_tts("100") == "one hundred"

    def test_ordinal_expansion_consistent(self):
        assert normalize_for_tts("1st") == "first"
        assert normalize_for_tts("2nd") == "second"
        assert normalize_for_tts("20th") == "twentieth"

    def test_abbreviation_dr_expansion(self):
        assert "Doctor" in normalize_for_tts("Dr. Smith")

    def test_abbreviation_eg_expansion(self):
        assert "for example" in normalize_for_tts("e.g. apples").lower()

    def test_large_numbers_unchanged(self):
        """Numbers >= 10000 are left as-is."""
        assert normalize_for_tts("12345") == "12345"


@settings(max_examples=150)
@given(text=st.text(min_size=0, max_size=80))
def test_p52_normalize_is_deterministic(text: str):
    """normalize_for_tts(x) == normalize_for_tts(x) for any input.

    **Validates: Requirements 3.6**
    """
    assert normalize_for_tts(text) == normalize_for_tts(text), (
        f"Non-deterministic normalization for {text!r}"
    )


# ============================================================================
# P5.3 — Single comma-free sentence passes through as exactly one segment
# ============================================================================


class TestP53SingleSentenceOrder:
    """P5.3 — A single comma-free sentence is one segment, unchanged.

    **Validates: Requirements 3.6**
    """

    def test_single_sentence_no_commas(self):
        text = "The cat sat on the mat."
        segs = _full_segment(text)
        assert len(segs) == 1
        assert segs[0] == "The cat sat on the mat."

    def test_single_sentence_question(self):
        text = "What is photosynthesis?"
        segs = _full_segment(text)
        assert len(segs) == 1

    def test_single_sentence_exclamation(self):
        text = "Well done!"
        segs = _full_segment(text)
        assert len(segs) == 1

    def test_sentence_content_preserved_verbatim(self):
        """The content of the sentence is not modified by segmentation."""
        text = "Light travels faster than sound."
        segs = _full_segment(text)
        assert segs[0] == "Light travels faster than sound."


@st.composite
def _single_comma_free_sentence(draw):
    """Generate a single sentence that has no internal commas/semicolons/colons."""
    # Only letters and spaces in the body
    words = [
        draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=2, max_size=8))
        for _ in range(draw(st.integers(min_value=2, max_value=8)))
    ]
    body = " ".join(words)
    terminal = draw(st.sampled_from([".", "?", "!"]))
    return body + terminal


@settings(max_examples=150)
@given(text=_single_comma_free_sentence())
def test_p53_single_sentence_produces_one_segment(text: str):
    """A single comma-free sentence always produces exactly one segment.

    **Validates: Requirements 3.6**
    """
    segs = _full_segment(text)
    assert len(segs) == 1, (
        f"Expected 1 segment for single sentence {text!r}, got {len(segs)}: {segs!r}"
    )


# ============================================================================
# P5.4 — WebSocket message shapes locked (structural source-code check)
# ============================================================================

_MAIN_PY = os.path.join(_SRC, "main.py")


def _load_main_source() -> str:
    with open(_MAIN_PY, encoding="utf-8") as f:
        return f.read()


class TestP54WebSocketShapes:
    """P5.4 — All expected WebSocket handlers and outbound message types exist.

    This is a structural code-reading test: parse src/main.py as source and
    assert the required message types and fields are present.  The fix may only
    ADD an additive `index_status` message type.

    **Validates: Requirements 3.3, 3.8, 3.9**
    """

    @pytest.fixture(scope="class")
    def source(self) -> str:
        return _load_main_source()

    # --- inbound handler types ---

    def test_upload_pdf_handler_exists(self, source):
        """upload_pdf inbound handler is present."""
        assert '"upload_pdf"' in source or "'upload_pdf'" in source

    def test_upload_pdf_fields_present(self, source):
        """upload_pdf handler reads class_num and subject fields."""
        assert "class_num" in source
        assert "subject" in source

    def test_upload_pdf_sends_upload_result(self, source):
        """upload_pdf handler emits an upload_result response."""
        assert '"upload_result"' in source or "'upload_result'" in source

    def test_rebuild_index_handler_exists(self, source):
        """rebuild_index inbound handler is present."""
        assert '"rebuild_index"' in source or "'rebuild_index'" in source

    def test_rebuild_index_sends_index_progress(self, source):
        """rebuild_index emits index_progress messages during build."""
        assert '"index_progress"' in source or "'index_progress'" in source

    def test_rebuild_index_sends_index_done(self, source):
        """rebuild_index emits an index_done completion message."""
        assert '"index_done"' in source or "'index_done'" in source

    def test_get_ncert_graph_handler_exists(self, source):
        """get_ncert_graph inbound handler is present."""
        assert '"get_ncert_graph"' in source or "'get_ncert_graph'" in source

    def test_get_ncert_graph_sends_ncert_graph_data(self, source):
        """get_ncert_graph emits ncert_graph_data response."""
        assert '"ncert_graph_data"' in source or "'ncert_graph_data'" in source

    def test_get_graph_handler_exists(self, source):
        """get_graph inbound handler is present."""
        assert '"get_graph"' in source or "'get_graph'" in source

    def test_get_graph_sends_graph_data(self, source):
        """get_graph emits graph_data response."""
        assert '"graph_data"' in source or "'graph_data'" in source

    def test_update_node_handler_exists(self, source):
        """update_node inbound handler is present."""
        assert '"update_node"' in source or "'update_node'" in source

    def test_state_broadcast_present(self, source):
        """state events are broadcast (used for IDLE/LISTENING/SPEAKING)."""
        assert '"state"' in source or "'state'" in source


# ============================================================================
# P5.5 — MIN_SCORE default unchanged
# ============================================================================


class TestP55MinScoreDefault:
    """P5.5 — MIN_SCORE in src/retrieval/embedder.py equals 0.25 exactly.

    **Validates: Requirements 3.10**
    """

    def test_min_score_is_0_25(self):
        from retrieval.embedder import MIN_SCORE
        assert MIN_SCORE == 0.25, (
            f"MIN_SCORE changed: expected 0.25, got {MIN_SCORE!r}"
        )

    def test_min_score_type_is_float(self):
        from retrieval.embedder import MIN_SCORE
        assert isinstance(MIN_SCORE, float)


# ============================================================================
# P5.6 — Config backend selection honored
# ============================================================================


class TestP56ConfigBackendHonored:
    """P5.6 — build_stt / build_tts dispatch to the right concrete class.

    We test dispatch logic only — no model files needed because we intercept
    the import-time class constructor via monkeypatching.

    **Validates: Requirements 3.7**
    """

    def test_build_stt_whisper_returns_whisper_stt(self, monkeypatch):
        """build_stt with backend=whisper instantiates WhisperSTT."""
        from config import AppConfig

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

        # Patch WhisperSTT.__init__ to avoid loading a real model
        import stt.whisper_stt as ws_mod
        monkeypatch.setattr(ws_mod.WhisperSTT, "__init__", lambda self, model_name="base.en": None)

        from factories import build_stt
        result = build_stt(cfg)
        assert type(result).__name__ == "WhisperSTT"

    def test_build_stt_faster_whisper_returns_faster_whisper_stt(self, monkeypatch):
        """build_stt with backend=faster_whisper instantiates FasterWhisperSTT."""
        from config import AppConfig

        cfg = AppConfig(
            engine_backend="litert",
            model_path="assets/gemma-4-E4B-it.litertlm",
            n_gpu_layers=0,
            stt_backend="faster_whisper",
            stt_model="base.en",
            stt_compute_type="int8",
            tts_backend="piper",
            tts_voice_path="assets/piper_voices/en_US-lessac-medium.onnx",
            embed_backend="minilm",
            index_dir="data/index",
            retrieval_k=3,
            min_score=0.25,
        )

        # Patch FasterWhisperSTT.__init__ to avoid loading a real model
        import stt.faster_whisper_stt as fw_mod
        monkeypatch.setattr(
            fw_mod.FasterWhisperSTT, "__init__",
            lambda self, model_name="base.en", compute_type="int8", device="cpu": None
        )

        from factories import build_stt
        result = build_stt(cfg)
        assert type(result).__name__ == "FasterWhisperSTT"

    def test_build_tts_piper_returns_piper_tts(self, monkeypatch):
        """build_tts with backend=piper instantiates PiperTTS."""
        from config import AppConfig

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

        from factories import build_tts
        result = build_tts(cfg)
        assert type(result).__name__ == "PiperTTS"

    def test_build_tts_honors_voice_path(self):
        """PiperTTS stores the voice_path passed via config."""
        from synthesis.tts_stream import PiperTTS
        tts = PiperTTS(model_path="assets/piper_voices/en_US-lessac-medium.onnx")
        assert "en_US-lessac-medium" in tts.model_path


# ============================================================================
# P5.7 — Non-textbook chit-chat path preserved
# ============================================================================


class TestP57ChitChatPathPreserved:
    """P5.7 — build_prompt with empty retrieved list returns the bare question.

    This is the chit-chat / no-index path.  It must not be degraded.

    **Validates: Requirements 3.5**
    """

    def test_empty_retrieved_returns_bare_question(self):
        """build_prompt(query, []) == query (bare question, no extra wrapping)."""
        from inference.prompt_builder import build_prompt
        query = "What is the capital of France?"
        result = build_prompt(query, [])
        assert result == query, (
            f"Expected bare question, got: {result!r}"
        )

    def test_chit_chat_query_preserved_exactly(self):
        """Various chit-chat queries are returned verbatim."""
        from inference.prompt_builder import build_prompt
        queries = [
            "Tell me a joke.",
            "How are you today?",
            "What time is it?",
        ]
        for q in queries:
            assert build_prompt(q, []) == q, (
                f"Chit-chat query {q!r} was not returned verbatim"
            )

    def test_empty_query_empty_retrieved_returns_empty(self):
        from inference.prompt_builder import build_prompt
        assert build_prompt("", []) == ""


# ============================================================================
# P5.8 — Offline operation: no network imports in core pipeline files
# ============================================================================

# Core pipeline modules to check (relative to src/).
_PIPELINE_FILES = [
    "synthesis/text_norm.py",
    "synthesis/tts_stream.py",
    "inference/prompt_builder.py",
    "retrieval/embedder.py",
    "retrieval/service.py",
    "factories.py",
    "config.py",
]

# Network-import names that should not appear in these files.
_NETWORK_MODULES = {"requests", "httpx", "urllib", "urllib2", "urllib3", "socket", "http.client"}


class TestP58OfflineNoBadNetworkImports:
    """P5.8 — Core pipeline files do not import known network libraries.

    Parsed with the AST so we check actual import statements, not comments
    or string literals.

    **Validates: Requirements 3.1**
    """

    @pytest.mark.parametrize("rel_path", _PIPELINE_FILES)
    def test_no_network_import_in_pipeline_file(self, rel_path):
        """File does not import requests/httpx/urllib/socket at the module level."""
        full_path = os.path.join(_SRC, rel_path)
        if not os.path.exists(full_path):
            pytest.skip(f"File not found: {full_path}")

        with open(full_path, encoding="utf-8") as f:
            source = f.read()

        try:
            tree = ast.parse(source, filename=rel_path)
        except SyntaxError as exc:
            pytest.fail(f"Syntax error parsing {rel_path}: {exc}")

        imported_modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split(".")[0])

        bad = imported_modules & _NETWORK_MODULES
        assert not bad, (
            f"{rel_path} imports network module(s) {bad!r} — "
            "this would break offline-only operation (Req 3.1)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
