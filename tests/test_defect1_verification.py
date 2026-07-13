"""Defect 1 verification tests — asserts the FIXED latency behavior (Property 1).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 1 (design.md): Latency Targets Met With Warmup.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

This file tests the EXPECTED (fixed) behavior introduced by tasks 7.1–7.3:
  A. config.yaml now uses ``stt.backend: faster_whisper`` (task 7.1).
  B. LiteRTEngine.warmup() no longer has a silent ``except: pass`` swallow;
     it sets ``_warmup_done = True`` on success and logs a warning on failure
     (task 7.2).
  C. A ``warmup_done`` property exists on ``LiteRTEngine`` so the caller can
     verify completion (task 7.2).
  D. ``engine.warmup()`` is called BEFORE ``show_status(IDLE, ...)`` in
     ``main_loop`` (task 7.2).
  E. Latency-profile prints (``[Latency Profile -> STT: ...ms | Total
     Gen+Speech: ...ms]`` and ``[TTFT ...]``) exist in ``_handle_response``
     in ``src/main.py`` (task 7.3).
  F. The ``_MEASURED_TURNS`` in the exploration test will include turns from
     the fixed pipeline (parsed from app.log) and every such turn meets the
     latency budgets: stt_ms <= 600, ttft_ms <= 2500,
     total_gen_speech_ms <= 8000.  (Structural + budget assertion.)
"""
from __future__ import annotations

import os
import re

import pytest

# ---------------------------------------------------------------------------
# Repo root helpers
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _src(rel: str) -> str:
    return os.path.join(_REPO_ROOT, "src", rel)

def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Fixed latency budgets (Req 2.1, 2.2, 2.3)
# ---------------------------------------------------------------------------
STT_BUDGET_MS            = 600
TTFT_BUDGET_MS           = 2500
TOTAL_GEN_SPEECH_BUDGET_MS = 8000

_ANSI_RE   = re.compile(r"\x1b\[[0-9;]*m")
_PROFILE_RE = re.compile(
    r"\[Latency Profile -> STT:\s*(\d+)ms\s*\|\s*Total Gen\+Speech:\s*(\d+)ms\]"
)
_TTFT_RE = re.compile(r"\[TTFT \(Time-to-First-Token\):\s*(\d+)ms\]")


# ==========================================================================
# Test A — config.yaml selects faster_whisper backend (Req 2.1)
# ==========================================================================

def test_config_uses_faster_whisper_backend():
    """config.yaml stt.backend is ``faster_whisper`` after the fix.

    **Validates: Requirements 2.1**
    """
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    source = _read(config_path)

    # Must contain ``faster_whisper`` in the stt section.
    assert "faster_whisper" in source, (
        "config.yaml must set stt.backend to 'faster_whisper' (task 7.1 fix). "
        f"Current config:\n{source}"
    )

    # The old unfixed backend must not be the active selection.
    # A commented-out reference is acceptable but an uncommented ``backend: whisper``
    # line (without faster_) must not appear.
    lines = source.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            continue  # ignore comments
        if re.match(r"backend\s*:\s*whisper\b", stripped):
            pytest.fail(
                f"config.yaml still sets backend to the slow 'whisper' path on "
                f"an active (non-commented) line: {line!r}"
            )


def test_config_stt_section_has_model():
    """config.yaml stt section declares a model key (base.en or small.en).

    **Validates: Requirements 2.1**
    """
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    source = _read(config_path)

    assert re.search(r"model\s*:\s*\S+", source), (
        "config.yaml stt section should have a 'model:' key specifying which "
        "model to load (e.g. base.en or small.en)."
    )


# ==========================================================================
# Test B — warmup() no longer has a silent except: pass swallow (Req 2.4)
# ==========================================================================

def _read_warmup_body() -> str:
    engine_src = _src("inference/engine.py")
    source = _read(engine_src)
    start = source.index("def warmup(")
    after = source[start:]
    m = re.search(r"\n    def \w+\(|\n# ---|\nGemmaEngine", after[len("def warmup("):])
    end = (m.start() + len("def warmup(")) if m else len(after)
    return after[:end]


def test_warmup_no_silent_except_pass():
    """warmup() must NOT have a bare ``except Exception: pass`` (or ``except: pass``).

    The silent swallow was the defect: it hid failures and gave no indication
    warmup succeeded.  After the fix a failure is logged, not swallowed.

    **Validates: Requirements 2.4**
    """
    body = _read_warmup_body()
    has_silent_swallow = bool(
        re.search(r"except\s*(\w+\s*)?:\s*\n\s*pass", body)
    )
    assert not has_silent_swallow, (
        "warmup() still has a silent ``except: pass`` swallow. "
        "The fix must log the exception instead."
    )


def test_warmup_logs_on_exception():
    """warmup() logs a warning/error when the decode raises an exception.

    **Validates: Requirements 2.4**
    """
    body = _read_warmup_body()
    has_log = bool(
        re.search(r"log(?:ger)?\.(?:warning|error|exception)\(", body)
    )
    assert has_log, (
        "warmup() should call logger.warning/error/exception() in its except "
        "block so failures are surfaced instead of silently swallowed."
    )


# ==========================================================================
# Test C — warmup_done property exists on LiteRTEngine (Req 2.4)
# ==========================================================================

def test_warmup_done_property_exists():
    """LiteRTEngine exposes a ``warmup_done`` property (or attribute).

    The caller (main_loop) needs this to verify warmup completed before the
    first user turn.

    **Validates: Requirements 2.4**
    """
    engine_src = _src("inference/engine.py")
    source = _read(engine_src)

    has_property = bool(
        re.search(r"@property\s*\ndef warmup_done", source)
        or re.search(r"warmup_done\s*=\s*property\(", source)
    )
    has_attr = bool(
        re.search(r"self\._warmup_done\s*=\s*True", source)
    )
    assert has_property or has_attr, (
        "LiteRTEngine should expose a ``warmup_done`` property or set "
        "``self._warmup_done = True`` so callers can verify completion."
    )


def test_warmup_sets_warmup_done_true_on_success():
    """warmup() sets ``_warmup_done = True`` after a successful decode.

    **Validates: Requirements 2.4**
    """
    body = _read_warmup_body()
    assert re.search(r"self\._warmup_done\s*=\s*True", body), (
        "warmup() must set self._warmup_done = True after a successful decode "
        "so callers can confirm warmup was performed before the first user turn."
    )


# ==========================================================================
# Test D — warmup() called BEFORE show_status(IDLE, ...) in main_loop (Req 2.4)
# ==========================================================================

def test_warmup_called_before_show_status_idle():
    """engine.warmup() is invoked before the first show_status(IDLE, ...) call.

    In the fixed main_loop the warmup block appears before the ``show_status``
    call that transitions the UI into IDLE ready state, ensuring the one-time
    init cost is paid before the first user turn is possible.

    **Validates: Requirements 2.4**
    """
    main_src = _src("main.py")
    source = _read(main_src)

    # Find the positions of key landmarks inside main_loop.
    # We look for the warmup call and the first IDLE status show.
    warmup_match = re.search(r"engine\.warmup\(\)", source)
    # The first show_status(IDLE, ...) call that marks the system as ready.
    idle_ready_match = re.search(r"show_status\s*\(\s*IDLE\s*,\s*['\"]Say", source)

    assert warmup_match, "engine.warmup() not found in main.py"
    assert idle_ready_match, (
        "show_status(IDLE, 'Say Hey Jarvis...') not found in main.py"
    )

    assert warmup_match.start() < idle_ready_match.start(), (
        "engine.warmup() must appear BEFORE show_status(IDLE, ...) in main_loop "
        "so warmup is completed before the system is considered ready for user input. "
        f"warmup at offset {warmup_match.start()}, "
        f"show_status(IDLE) at offset {idle_ready_match.start()}"
    )


def test_warmup_done_check_in_main_loop():
    """main_loop checks engine.warmup_done after calling warmup().

    This check allows the CLI to report whether warmup succeeded or was skipped,
    so the user has visibility into the first-turn latency risk.

    **Validates: Requirements 2.4**
    """
    main_src = _src("main.py")
    source = _read(main_src)

    assert "warmup_done" in source, (
        "main_loop should check engine.warmup_done after engine.warmup() to "
        "report whether warmup completed (Req 2.4)."
    )


# ==========================================================================
# Test E — Latency-profile prints exist in _handle_response (Req 2.5)
# ==========================================================================

def _read_handle_response_body() -> str:
    main_src = _src("main.py")
    source = _read(main_src)
    start = source.index("async def _handle_response(")
    after = source[start:]
    # Ends at the next top-level async def / def / class, or at module end.
    m = re.search(r"\n(?:async def|def|class)\s", after[len("async def _handle_response("):])
    end = (m.start() + len("async def _handle_response(")) if m else len(after)
    return after[:end]


def test_ttft_print_exists_in_handle_response():
    """A ``[TTFT (Time-to-First-Token): ...ms]`` print exists in _handle_response.

    This is the measurement instrument for the TTFT budget assertion.

    **Validates: Requirements 2.2, 2.5**
    """
    body = _read_handle_response_body()
    assert re.search(r"TTFT", body), (
        "[TTFT ...] print not found in _handle_response — required to measure "
        "time-to-first-token for the latency budget (Req 2.2, 2.5)."
    )


def test_latency_profile_print_exists_in_handle_response():
    """A ``[Latency Profile -> STT: ...ms | Total Gen+Speech: ...ms]`` print
    exists in _handle_response.

    **Validates: Requirements 2.1, 2.3, 2.5**
    """
    body = _read_handle_response_body()
    assert re.search(r"Latency Profile", body), (
        "[Latency Profile ...] print not found in _handle_response — required "
        "to measure STT and total latency for the budget assertion (Req 2.1, 2.3, 2.5)."
    )


def test_stt_ms_captured_in_handle_response():
    """stt_ms is measured and printed in _handle_response.

    **Validates: Requirements 2.1**
    """
    body = _read_handle_response_body()
    assert "stt_ms" in body, (
        "stt_ms variable not found in _handle_response — STT latency must be "
        "measured for the budget assertion (Req 2.1)."
    )


def test_total_gen_speech_ms_captured_in_handle_response():
    """total_gen_speech_ms is measured and printed in _handle_response.

    **Validates: Requirements 2.3**
    """
    body = _read_handle_response_body()
    assert "total_gen_speech_ms" in body or "total_generation_ms" in body, (
        "total_gen_speech_ms / total_generation_ms variable not found in "
        "_handle_response — total latency must be measured (Req 2.3)."
    )


# ==========================================================================
# Test F — Budget assertions on any latency-profile lines captured in app.log
# ==========================================================================

def _parse_latency_profile(text: str):
    """Return list of (stt_ms, ttft_ms, total_ms) tuples from log text."""
    clean = _ANSI_RE.sub("", text)
    turns = []
    pending_ttft = None
    for line in clean.splitlines():
        ttft_m = _TTFT_RE.search(line)
        if ttft_m:
            pending_ttft = int(ttft_m.group(1))
            continue
        prof_m = _PROFILE_RE.search(line)
        if prof_m:
            stt_ms   = int(prof_m.group(1))
            total_ms = int(prof_m.group(2))
            ttft_ms  = pending_ttft if pending_ttft is not None else 0
            turns.append((stt_ms, ttft_ms, total_ms))
            pending_ttft = None
    return turns


def test_app_log_fixed_turns_meet_latency_budgets():
    """Any fixed-pipeline turns captured in app.log must meet all latency budgets.

    If app.log is absent or contains no latency-profile lines this test is
    trivially satisfied (no turns measured yet on the fixed pipeline).

    **Validates: Requirements 2.1, 2.2, 2.3**
    """
    log_path = os.path.join(_REPO_ROOT, "app.log")
    if not os.path.exists(log_path):
        pytest.skip("app.log not present — no fixed-pipeline turns measured yet")

    with open(log_path, "r", encoding="utf-8", errors="ignore") as fh:
        content = fh.read()

    turns = _parse_latency_profile(content)
    if not turns:
        pytest.skip("No latency-profile lines found in app.log yet")

    over_budget = []
    for stt_ms, ttft_ms, total_ms in turns:
        if stt_ms > STT_BUDGET_MS:
            over_budget.append(f"STT {stt_ms}>{STT_BUDGET_MS}")
        if ttft_ms > TTFT_BUDGET_MS:
            over_budget.append(f"TTFT {ttft_ms}>{TTFT_BUDGET_MS}")
        if total_ms > TOTAL_GEN_SPEECH_BUDGET_MS:
            over_budget.append(f"total {total_ms}>{TOTAL_GEN_SPEECH_BUDGET_MS}")

    assert not over_budget, (
        "Fixed-pipeline turns in app.log exceed latency budgets: "
        + ", ".join(over_budget)
    )


# ==========================================================================
# Test — no active stt.backend: whisper line in config (belt-and-suspenders)
# ==========================================================================

def test_faster_whisper_is_active_stt_backend():
    """The active stt.backend in config.yaml is faster_whisper, not whisper.

    Complements test_config_uses_faster_whisper_backend with a direct parse.

    **Validates: Requirements 2.1**
    """
    config_path = os.path.join(_REPO_ROOT, "config.yaml")
    source = _read(config_path)

    # Simple YAML parse: find ``backend:`` key inside the stt block.
    # We look for a line matching ``backend: faster_whisper`` (ignoring comments).
    found_faster = False
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if re.match(r"backend\s*:\s*faster_whisper", stripped):
            found_faster = True
            break

    assert found_faster, (
        "config.yaml does not have an active 'backend: faster_whisper' line. "
        "Task 7.1 requires switching the STT backend to faster_whisper to meet "
        "the stt_ms <= 600ms budget."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
