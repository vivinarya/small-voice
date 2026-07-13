"""Bug-condition exploration test for Defect 2 — Robotic / Choppy Voice (Property 2).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 2 (design.md): Bug Condition — Choppy Inter-Sentence Playback & Voice Timbre.

**Validates: Requirements 1.5, 1.6**

CRITICAL — this is a BUGFIX *exploration* test run on the UNFIXED code. It has
two jobs, and per bugfix exploration-test semantics BOTH outcomes below are the
SUCCESS case (they confirm the real defect). DO NOT "fix" the code or weaken the
test to change these outcomes.

1. REFUTATION assertion (EXPECTED TO PASS on unfixed code):
   Feed a known multi-sentence response through `extract_complete_sentences`
   (src/synthesis/text_norm.py) and assert it NEVER splits on `,` `;` `:`.
   Passing CONFIRMS the documented "comma-split" root cause is already gone — it
   is refuted by the current code.

2. REAL-CAUSE assertion (demonstrates the actual defect on unfixed code):
   Show that `PiperTTS.speak` / `stream_text` (src/synthesis/tts_stream.py)
   performs N separate `piper.exe` subprocess invocations and plays each
   sentence as a SEPARATE `sd.play(); sd.wait()` playback-queue item, producing
   an audible inter-sentence playback gap (device stop/restart between
   sentences). Also note the mid-tier `en_US-lessac-medium` voice as the timbre
   ceiling.

This test is fully OFFLINE and needs no audio hardware or the real `piper.exe`
binary: `sounddevice` and `numpy` are replaced with lightweight instrumentation
stand-ins, and `subprocess.Popen` is mocked to count piper invocations.
"""
from __future__ import annotations

import os
import sys
import types
import threading

# --------------------------------------------------------------------------
# Make `src/` importable as the `synthesis` package (matches test_text_norm.py).
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# --------------------------------------------------------------------------
# Instrumentation stand-ins for `sounddevice` and `numpy`.
#
# These are installed into sys.modules BEFORE importing the TTS module so that
# importing it (which does `import sounddevice as sd` / `import numpy as np`)
# succeeds in this hardware-free, dependency-light environment AND so playback
# calls are recorded. `sounddevice` is FORCED (we never want real audio
# hardware during the test); `numpy` only fills in if a real one is absent.
# --------------------------------------------------------------------------
class _PlaybackRecorder:
    """Records each sd.play()/sd.wait() so we can count discrete playback items.

    Each (play, wait) pair is one isolated playback-queue item. Multiple pairs
    for a single response mean the device is stopped/restarted between
    sentences => the audible inter-sentence gap (the REAL voice defect).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.play_calls: list = []
        self.wait_calls: int = 0

    def record_play(self, audio) -> None:
        with self._lock:
            self.play_calls.append(audio)

    def record_wait(self) -> None:
        with self._lock:
            self.wait_calls += 1

    def reset(self) -> None:
        with self._lock:
            self.play_calls.clear()
            self.wait_calls = 0


PLAYBACK = _PlaybackRecorder()


def _install_fake_sounddevice() -> None:
    fake_sd = types.ModuleType("sounddevice")

    def _play(audio, samplerate=None, *args, **kwargs):
        PLAYBACK.record_play(audio)

    def _wait(*args, **kwargs):
        PLAYBACK.record_wait()

    def _stop(*args, **kwargs):
        pass

    fake_sd.play = _play
    fake_sd.wait = _wait
    fake_sd.stop = _stop

    # Full-featured OutputStream stub: supports the constructor kwargs that
    # stream_text passes (samplerate, channels, dtype) and records write() calls
    # so we can assert gapless playback is used after the fix.
    class _FakeOutputStream:
        write_calls: list = []
        instances: list = []

        def __init__(self, samplerate=None, channels=None, dtype=None, **kwargs):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self._started = False
            type(self).instances.append(self)

        def start(self):
            self._started = True

        def write(self, data):
            type(self).write_calls.append(data)

        def stop(self):
            pass

        def close(self):
            pass

        @classmethod
        def reset_all(cls):
            cls.write_calls.clear()
            cls.instances.clear()

    fake_sd.OutputStream = _FakeOutputStream
    sys.modules["sounddevice"] = fake_sd  # force — never touch real hardware


def _install_fake_numpy_if_absent() -> None:
    if "numpy" in sys.modules:
        return
    try:  # prefer a real numpy if one is installed
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
    # Make the stand-in look like a package with a no-op `random` submodule so
    # tools that detect `numpy` in sys.modules (e.g. hypothesis' RNG seeding)
    # don't crash trying to `import numpy.random`.
    fake_np.__path__ = []  # marks it as a package
    fake_random = types.ModuleType("numpy.random")

    def _get_state():
        return ("fake-state",)

    def _set_state(state):
        return None

    def _seed(seed=None):
        return None

    fake_random.get_state = _get_state
    fake_random.set_state = _set_state
    fake_random.seed = _seed
    fake_np.random = fake_random
    sys.modules["numpy"] = fake_np
    sys.modules["numpy.random"] = fake_random


_install_fake_sounddevice()
_install_fake_numpy_if_absent()

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Now safe to import the code under test and the segmentation helper.
from synthesis import tts_stream as tts_mod
from synthesis.tts_stream import PiperTTS
from synthesis.text_norm import extract_complete_sentences


# The canonical multi-sentence response from the task / design.md.
MULTI_SENTENCE_RESPONSE = "Sound travels in waves. It needs a medium to move."
# A clause-punctuation-heavy single sentence (the kind the *documented* cause
# claimed would be fragmented). The current code must keep it whole.
CLAUSE_HEAVY_RESPONSE = (
    "Sound travels in waves, which we hear; it needs a medium: air or water."
)
EXPECTED_VOICE_MODEL_SUFFIX = "en_US-lessac-medium.onnx"


def _segment_all(text: str) -> list[str]:
    """Fully segment text the same way `stream_text` does (loop + tail)."""
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


# ==========================================================================
# Mocked piper subprocess — counts each `piper.exe` invocation and records the
# command so we can confirm one process PER SENTENCE and the mid-tier voice.
# ==========================================================================
class _FakePopen:
    invocations: list[list[str]] = []

    def __init__(self, command, stdin=None, stdout=None, stderr=None, **kwargs):
        type(self).invocations.append(list(command))
        self.returncode = 0

    def communicate(self, input=None):  # noqa: A002 - mirror subprocess API
        self.returncode = 0
        # Return tiny fake PCM bytes; np.frombuffer turns it into an audio array.
        return (b"\x00\x01" * 8, b"")


@pytest.fixture
def instrumented(monkeypatch):
    """Reset recorders and patch piper subprocess and tts_mod.sd."""
    _FakePopen.invocations.clear()
    PLAYBACK.reset()

    # Rebuild the fake sounddevice module with fresh OutputStream state so
    # assertions in this fixture are independent of import order.
    fake_sd = types.ModuleType("sounddevice")

    def _play(audio, samplerate=None, *args, **kwargs):
        PLAYBACK.record_play(audio)

    def _wait(*args, **kwargs):
        PLAYBACK.record_wait()

    def _stop(*args, **kwargs):
        pass

    fake_sd.play = _play
    fake_sd.wait = _wait
    fake_sd.stop = _stop

    class _FakeOutputStream:
        write_calls: list = []
        instances: list = []

        def __init__(self, samplerate=None, channels=None, dtype=None, **kwargs):
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype
            self._started = False
            type(self).instances.append(self)

        def start(self):
            self._started = True

        def write(self, data):
            type(self).write_calls.append(data)

        def stop(self):
            pass

        def close(self):
            pass

        @classmethod
        def reset_all(cls):
            cls.write_calls.clear()
            cls.instances.clear()

    fake_sd.OutputStream = _FakeOutputStream

    # Patch both sys.modules AND tts_mod.sd so the code under test always
    # uses this instrumented stub regardless of when the module was imported.
    sys.modules["sounddevice"] = fake_sd
    monkeypatch.setattr(tts_mod, "sd", fake_sd)
    monkeypatch.setattr(tts_mod.subprocess, "Popen", _FakePopen)

    yield fake_sd


# ==========================================================================
# Part 1 — REFUTATION: documented comma-split cause is ALREADY GONE.
# EXPECTED OUTCOME ON UNFIXED CODE: these PASS.
# ==========================================================================
def test_refutation_multi_sentence_splits_only_on_terminal_punctuation():
    """The known response splits into whole sentences, never on `,` `;` `:`."""
    segments = _segment_all(MULTI_SENTENCE_RESPONSE)
    assert segments == [
        "Sound travels in waves.",
        "It needs a medium to move.",
    ], f"Unexpected segmentation: {segments!r}"


def test_refutation_clause_punctuation_never_creates_a_boundary():
    """A clause-heavy single sentence stays ONE segment (no `,` `;` `:` split)."""
    segments = _segment_all(CLAUSE_HEAVY_RESPONSE)
    assert len(segments) == 1, (
        f"Comma/semicolon/colon caused a split — documented cause would be "
        f"present. Got {len(segments)} segments: {segments!r}"
    )
    # All clause punctuation is preserved inside the single segment.
    assert "," in segments[0] and ";" in segments[0] and ":" in segments[0]


# Property-based refutation: for ANY text built only from words and clause
# punctuation (no terminal . ? !), `extract_complete_sentences` must yield NO
# complete sentence — i.e. it never treats `,` `;` `:` (or spaces) as a boundary.
_WORDS = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=6)
_CLAUSE_SEPARATORS = st.sampled_from([", ", "; ", ": ", " "])


@st.composite
def _clause_text(draw):
    n = draw(st.integers(min_value=2, max_value=8))
    words = [draw(_WORDS) for _ in range(n)]
    seps = [draw(_CLAUSE_SEPARATORS) for _ in range(n - 1)]
    out = words[0]
    for sep, word in zip(seps, words[1:]):
        out += sep + word
    return out


@settings(max_examples=100)
@given(text=_clause_text())
def test_property2_refutation_clause_text_is_never_segmented(text):
    """No clause punctuation (and no terminal punctuation) => no sentence yet.

    **Validates: Requirements 1.5**
    """
    # Sanity: generator never emits terminal punctuation.
    assert not any(p in text for p in ".?!")
    assert extract_complete_sentences(text) == []


# ==========================================================================
# Part 2 — REAL CAUSE: per-sentence subprocess + per-item playback gaps.
# EXPECTED OUTCOME ON FIXED CODE: this asserts the FIX is in place.
# ==========================================================================
def test_real_cause_per_sentence_subprocess_and_playback_gaps(instrumented):
    """Fixed behavior: gapless OutputStream used instead of discrete sd.play()/sd.wait().

    After the fix (Task 8.1):
    - stream_text() uses a single sd.OutputStream written to per sentence —
      NO discrete sd.play() / sd.wait() calls per sentence (the gap source is gone).
    - Synthesis still fires once per sentence (or coalesced sentence), maintaining
      ordered, complete playback.
    - Full response text is returned correctly.

    **Validates: Requirements 1.5, 1.6**
    """
    # instrumented yields the fake_sd module — get OutputStream from it directly
    # so we use the same class instance that tts_mod.sd points to.
    OutputStream = instrumented.OutputStream

    tts = PiperTTS()  # default model_path => voice in use

    # Stream the known response one character at a time (mimics LLM tokens).
    spoken = tts.stream_text(iter(MULTI_SENTENCE_RESPONSE))

    # The full text is reconstructed and spoken in order (ordering preserved).
    assert spoken == MULTI_SENTENCE_RESPONSE

    # --- FIX CONFIRMED (a): OutputStream was used (gapless playback) ---
    assert len(OutputStream.instances) == 1, (
        "Expected exactly 1 sd.OutputStream to be opened for the entire response "
        f"(gapless single-stream playback). Got {len(OutputStream.instances)} instances."
    )

    # --- FIX CONFIRMED (b): sd.play() NOT called per sentence (gap eliminated) ---
    assert len(PLAYBACK.play_calls) == 0, (
        f"Expected NO discrete sd.play() calls after the gapless fix. "
        f"Got {len(PLAYBACK.play_calls)} call(s) — discrete playback still present."
    )
    assert PLAYBACK.wait_calls == 0, (
        f"Expected NO discrete sd.wait() calls after the gapless fix. "
        f"Got {PLAYBACK.wait_calls} call(s) — per-sentence device gaps still present."
    )

    # --- FIX CONFIRMED (c): PCM written to the stream (sentences synthesised + delivered) ---
    assert len(OutputStream.write_calls) >= 1, (
        "Expected at least 1 write() call to the OutputStream (synthesised PCM delivered). "
        f"Got {len(OutputStream.write_calls)} writes."
    )

    # --- synthesis used the piper subprocess (still synthesises via piper) ---
    assert len(_FakePopen.invocations) >= 1, (
        "Expected at least 1 piper.exe subprocess invocation for synthesis. "
        f"Got {len(_FakePopen.invocations)}."
    )


def test_real_cause_default_voice_is_mid_tier_timbre_ceiling():
    """The default Piper voice is the mid-tier `en_US-lessac-medium` model.

    **Validates: Requirements 1.6**
    """
    tts = PiperTTS()
    assert tts.model_path.endswith(EXPECTED_VOICE_MODEL_SUFFIX), (
        f"Default voice is the timbre ceiling for naturalness: {tts.model_path!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
