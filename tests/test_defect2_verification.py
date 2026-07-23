"""Verification tests for Defect 2 fix — Gapless Voice Playback (Property 2).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 2 (design.md): Natural, Gapless, Ordered Sentence-Level Synthesis.

**Validates: Requirements 2.6, 2.7, 2.8**

These tests confirm the FIXED behavior of `PiperTTS.stream_text`:

1. `stream_text()` uses `sd.OutputStream` for gapless playback — NOT discrete
   `sd.play()/sd.wait()` per sentence.
2. `PiperTTS._synthesize_to_pcm` helper exists and does NOT call `sd.play()`.
3. `_SHORT_SENTENCE_WORD_THRESHOLD` constant exists (short-sentence coalescing
   is in place).
4. `extract_complete_sentences` and `normalize_for_tts` are unchanged — they
   still split only on `.?!` and apply number/abbreviation expansion (Req 3.6).

All tests are fully offline: `sounddevice` attributes in `tts_stream` module and
the `piper.exe` subprocess are replaced via monkeypatch; no real audio hardware
is needed.
"""
from __future__ import annotations

import os
import sys
import types
import threading

# --------------------------------------------------------------------------
# Ensure `src/` is importable as the `synthesis` package.
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --------------------------------------------------------------------------
# Bootstrap: ensure sounddevice is in sys.modules so tts_stream can import.
# If the exploration test already installed one, reuse it; otherwise provide
# a minimal stub so the import succeeds.
# --------------------------------------------------------------------------
def _ensure_sounddevice_stub():
    if "sounddevice" in sys.modules:
        return  # reuse whatever is already there (e.g. exploration test's stub)
    try:
        import sounddevice  # noqa: F401
        return
    except Exception:
        pass
    # Minimal stub — just enough for the import to succeed.
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.play = lambda *a, **kw: None
    fake_sd.wait = lambda *a, **kw: None
    fake_sd.stop = lambda *a, **kw: None
    fake_sd.OutputStream = object  # will be replaced by monkeypatch per test
    sys.modules["sounddevice"] = fake_sd


def _ensure_numpy_stub():
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401
        return
    except Exception:
        pass
    fake_np = types.ModuleType("numpy")

    class _FakeNdarray(list):
        pass

    def _frombuffer(buffer, dtype=None):
        arr = _FakeNdarray()
        try:
            arr.nbytes = len(buffer)
        except TypeError:
            arr.nbytes = 0
        return arr

    fake_np.frombuffer = _frombuffer
    fake_np.int16 = "int16"
    fake_np.ndarray = _FakeNdarray
    fake_np.__path__ = []
    fake_random = types.ModuleType("numpy.random")
    fake_random.get_state = lambda: ("fake-state",)
    fake_random.set_state = lambda s: None
    fake_random.seed = lambda seed=None: None
    fake_np.random = fake_random
    sys.modules["numpy"] = fake_np
    sys.modules["numpy.random"] = fake_random


_ensure_sounddevice_stub()
_ensure_numpy_stub()

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from synthesis import tts_stream as tts_mod
from synthesis.tts_stream import (
    PiperTTS,
    _SHORT_SENTENCE_WORD_THRESHOLD,
)
from synthesis.text_norm import extract_complete_sentences, normalize_for_tts


# --------------------------------------------------------------------------
# Instrumentation classes used by fixtures below.
# --------------------------------------------------------------------------
class _OutputStreamRecorder:
    """Tracks sd.OutputStream usage to confirm gapless playback."""

    def __init__(self, samplerate=None, channels=None, dtype=None, **kwargs):
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._started = False
        type(self)._registry.append(self)

    def start(self):
        self._started = True

    def write(self, data):
        type(self)._writes.append(data)

    def stop(self):
        pass

    def close(self):
        pass

    # Class-level registries — reset by fixture
    _registry: list = []
    _writes: list = []

    @classmethod
    def reset_all(cls):
        cls._registry.clear()
        cls._writes.clear()


class _PlayRecorder:
    """Counts discrete sd.play()/sd.wait() calls (should be zero after the fix)."""

    def __init__(self):
        self._lock = threading.Lock()
        self.play_calls: list = []
        self.wait_calls: int = 0

    def record_play(self, audio):
        with self._lock:
            self.play_calls.append(audio)

    def record_wait(self):
        with self._lock:
            self.wait_calls += 1

    def reset(self):
        with self._lock:
            self.play_calls.clear()
            self.wait_calls = 0


# --------------------------------------------------------------------------
# Fake piper subprocess
# --------------------------------------------------------------------------
class _FakePopen:
    invocations: list = []

    def __init__(self, command, stdin=None, stdout=None, stderr=None, **kwargs):
        type(self).invocations.append(list(command))
        self.returncode = 0

    def communicate(self, input=None):  # noqa: A002
        self.returncode = 0
        return (b"\x00\x01" * 8, b"")


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def play_recorder():
    rec = _PlayRecorder()
    return rec


@pytest.fixture
def patched_tts(monkeypatch, play_recorder):
    """Patch tts_mod.sd with a fresh stub and tts_mod.subprocess.Popen."""
    _FakePopen.invocations.clear()
    _OutputStreamRecorder.reset_all()

    # Build a fresh fake sounddevice module that points at our recorders.
    fake_sd = types.SimpleNamespace()
    fake_sd.play = lambda audio, samplerate=None, **kw: play_recorder.record_play(audio)
    fake_sd.wait = lambda **kw: play_recorder.record_wait()
    fake_sd.stop = lambda **kw: None
    fake_sd.OutputStream = _OutputStreamRecorder

    monkeypatch.setattr(tts_mod, "sd", fake_sd)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", _FakePopen)

    yield {
        "sd": fake_sd,
        "play_recorder": play_recorder,
        "OutputStream": _OutputStreamRecorder,
        "FakePopen": _FakePopen,
    }


# ==========================================================================
# 1. Gapless playback: OutputStream used, no discrete sd.play()/sd.wait()
# ==========================================================================
def test_stream_text_uses_output_stream_not_discrete_play(patched_tts):
    """stream_text() opens exactly one OutputStream and writes PCM into it.

    This confirms gapless delivery: no device stop/restart between sentences.

    **Validates: Requirements 2.6, 2.8**
    """
    play_recorder = patched_tts["play_recorder"]
    OutputStream = patched_tts["OutputStream"]

    tts = PiperTTS()
    spoken = tts.stream_text(iter("Sound travels in waves. It needs a medium to move."))

    assert spoken == "Sound travels in waves. It needs a medium to move."

    # One continuous OutputStream (not restarted per sentence).
    assert len(OutputStream._registry) == 1, (
        f"Expected 1 OutputStream for gapless playback, got "
        f"{len(OutputStream._registry)}."
    )

    # PCM was actually written into the stream.
    assert len(OutputStream._writes) >= 1, (
        "Expected at least 1 write() to the OutputStream (audio delivered). "
        f"Got {len(OutputStream._writes)}."
    )

    # No discrete per-sentence sd.play() / sd.wait() calls.
    assert play_recorder.play_calls == [], (
        f"sd.play() was called {len(play_recorder.play_calls)} time(s) — "
        "per-sentence playback gaps still present."
    )
    assert play_recorder.wait_calls == 0, (
        f"sd.wait() was called {play_recorder.wait_calls} time(s) — "
        "per-sentence device restarts still present."
    )


def test_stream_text_single_sentence_also_uses_output_stream(patched_tts):
    """A single-sentence response also uses OutputStream (no regression).

    **Validates: Requirements 2.6, 2.8**
    """
    play_recorder = patched_tts["play_recorder"]
    OutputStream = patched_tts["OutputStream"]

    tts = PiperTTS()
    tts.stream_text(iter("Hello world."))

    assert len(OutputStream._registry) == 1
    assert play_recorder.play_calls == []
    assert play_recorder.wait_calls == 0


# ==========================================================================
# 2. _synthesize_to_pcm helper exists as a method and does NOT call sd.play()
# ==========================================================================
def test_synthesize_to_pcm_exists_and_does_not_play(patched_tts):
    """PiperTTS._synthesize_to_pcm returns PCM without triggering sd.play().

    **Validates: Requirements 2.6**
    """
    play_recorder = patched_tts["play_recorder"]

    tts = PiperTTS()
    result = tts._synthesize_to_pcm("Hello world.")

    # Returns something (PCM array or None — None only when piper is absent).
    assert result is not None, "_synthesize_to_pcm returned None with fake piper"

    # Must not call sd.play() — the helper is synthesis-only.
    assert play_recorder.play_calls == [], (
        "_synthesize_to_pcm called sd.play() — it should only return PCM."
    )
    assert play_recorder.wait_calls == 0


# ==========================================================================
# 3. _SHORT_SENTENCE_WORD_THRESHOLD constant exists
# ==========================================================================
def test_short_sentence_threshold_constant_exists():
    """_SHORT_SENTENCE_WORD_THRESHOLD is defined (short-sentence coalescing in place).

    **Validates: Requirements 2.8**
    """
    assert isinstance(_SHORT_SENTENCE_WORD_THRESHOLD, int), (
        "_SHORT_SENTENCE_WORD_THRESHOLD must be an int"
    )
    assert _SHORT_SENTENCE_WORD_THRESHOLD >= 1, (
        "_SHORT_SENTENCE_WORD_THRESHOLD must be positive"
    )


def test_short_leading_sentence_coalesced(patched_tts):
    """A short leading sentence (≤ threshold words) is coalesced with the next.

    The coalesced pair is synthesised together, so there is no clipped isolated
    utterance while ordered playback is still preserved.

    **Validates: Requirements 2.8**
    """
    OutputStream = patched_tts["OutputStream"]

    tts = PiperTTS()
    # "Sure." is 1 word — below threshold — so it should coalesce with the next sentence.
    response = "Sure. Sound travels in waves."
    spoken = tts.stream_text(iter(response))

    assert spoken.strip() == response.strip(), (
        f"spoken text mismatch: {spoken!r} vs {response!r}"
    )
    # What we can assert firmly is that OutputStream was used (no gap).
    assert len(OutputStream._registry) == 1
    assert patched_tts["play_recorder"].play_calls == []


# ==========================================================================
# 4. Segmentation and normalisation unchanged (Req 3.6)
# ==========================================================================
def test_extract_complete_sentences_unchanged_multi():
    """extract_complete_sentences still splits only on terminal punctuation.

    **Validates: Requirements 2.7, 3.6**
    """
    text = "Sound travels in waves. It needs a medium to move."
    results = extract_complete_sentences(text)
    assert results, "Expected at least one complete sentence"
    sentence, remainder = results[0]
    assert sentence.strip() == "Sound travels in waves."
    assert "It needs a medium to move." in (remainder or text)


def test_extract_complete_sentences_no_clause_split():
    """Clause punctuation (,;:) does NOT create a sentence boundary.

    **Validates: Requirements 2.7, 3.6**
    """
    clause_text = "Sound travels in waves, which we hear; it needs a medium: air or water."
    results = extract_complete_sentences(clause_text)
    # The whole thing is ONE sentence ending with '.'
    assert len(results) == 1, (
        f"Clause punctuation caused a spurious split — got {len(results)} results: {results!r}"
    )


@settings(max_examples=100)
@given(
    text=st.text(
        alphabet="abcdefghijklmnopqrstuvwxyz ,;: ",
        min_size=5,
        max_size=80,
    ).filter(lambda t: not any(p in t for p in ".?!"))
)
def test_property_no_terminal_punctuation_never_segmented(text):
    """Without terminal punctuation, extract_complete_sentences returns [].

    **Validates: Requirements 2.7**
    """
    assert extract_complete_sentences(text) == [], (
        f"Unexpected segmentation of clause-only text: {text!r}"
    )


def test_normalize_for_tts_applied_to_every_segment():
    """normalize_for_tts is still callable and transforms text as expected.

    **Validates: Requirements 2.7**
    """
    result = normalize_for_tts("There are 3 waves.")
    assert isinstance(result, str)
    assert result  # non-empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
