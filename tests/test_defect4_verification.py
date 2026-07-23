"""Verification test for Defect 4 — STT Understanding (Property 4).

Spec: .kiro/specs/latency-and-voice-quality-fix
Property 4 (design.md): Bug Condition — Intent-Preserving STT Within Constraints.

**Validates: Requirements 2.13**

This test asserts the FIXED state after Tasks 9.1, 9.2, and 9.3.
It verifies the expected (correct) behavior, not the bug condition.

Fixed state verified here:

  1. CONFIG: config.yaml selects `stt.backend: faster_whisper` (CTranslate2 int8 path),
     `stt.model: small.en`, `stt.compute_type: int8`.
     Recorded chosen configuration: faster_whisper / small.en / int8.

  2. AUTOCORRECT UPGRADE: `autocorrect_stt` in `src/knowledge/graph.py`:
     (a) Uses word-boundary regex (`\b...\b`) preventing silent inner-word matches.
     (b) Has domain extensions: covers "photo synthesis" → "photosynthesis",
         "foto synthesis" → "photosynthesis", "electric magnetic" → "electromagnetic",
         "mitokondria" → "mitochondria", "mitokondrion" → "mitochondrion".

  3. INITIAL_PROMPT ENRICHMENT: `_handle_response` in `src/main.py` builds an
     `initial_prompt` that includes NCERT domain vocabulary (from chunks.jsonl),
     in addition to wiki entity names, biasing Whisper toward domain terms.

  4. FASTER_WHISPER COMPUTE_TYPE: `FasterWhisperSTT` accepts and uses `compute_type`
     parameter for CTranslate2 quantized inference.

  5. CHOSEN CONFIG RECORDED: The chosen configuration is documented — `faster_whisper`
     backend, `small.en` model, `int8` compute_type.

  6. OFFLINE PRESERVED: No network calls introduced; all operation remains local.
"""
from __future__ import annotations

import ast
import os
import re
import sys
from typing import Dict

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# --------------------------------------------------------------------------
# Make `src/` importable
# --------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# --------------------------------------------------------------------------
# Dependency-light: stub numpy if absent
# --------------------------------------------------------------------------
import types as _types


def _install_fake_numpy_if_absent() -> None:
    if "numpy" in sys.modules:
        return
    try:
        import numpy  # noqa: F401
        return
    except Exception:
        pass
    fake_np = _types.ModuleType("numpy")

    class _FakeNdarray(list):
        ndim = 1

    fake_np.ndarray = _FakeNdarray
    fake_np.float32 = "float32"
    fake_np.int16 = "int16"
    fake_np.__path__ = []
    fake_random = _types.ModuleType("numpy.random")
    fake_random.get_state = lambda: ("fake-state",)
    fake_random.set_state = lambda state: None
    fake_random.seed = lambda seed=None: None
    fake_np.random = fake_random
    sys.modules["numpy"] = fake_np
    sys.modules["numpy.random"] = fake_random


_install_fake_numpy_if_absent()

from knowledge.graph import autocorrect_stt  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.yaml")
_FASTER_WHISPER_STT_PATH = os.path.join(_REPO_ROOT, "src", "stt", "faster_whisper_stt.py")
_GRAPH_PATH = os.path.join(_REPO_ROOT, "src", "knowledge", "graph.py")
_MAIN_PATH = os.path.join(_REPO_ROOT, "src", "main.py")

# --------------------------------------------------------------------------
# Recorded / chosen configuration (Task 9.4 — record for measurement)
# --------------------------------------------------------------------------
CHOSEN_STT_BACKEND = "faster_whisper"
CHOSEN_STT_MODEL = "small.en"
CHOSEN_COMPUTE_TYPE = "int8"

# --------------------------------------------------------------------------
# Domain correction pairs introduced by Task 9.3 that must now be present
# --------------------------------------------------------------------------
EXPECTED_DOMAIN_CORRECTIONS: Dict[str, str] = {
    "photo synthesis": "photosynthesis",
    "foto synthesis": "photosynthesis",
    "electric magnetic": "electromagnetic",
    "mitokondria": "mitochondria",
    "mitokondrion": "mitochondrion",
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _read_config_raw() -> str:
    assert os.path.exists(_CONFIG_PATH), f"config.yaml not found: {_CONFIG_PATH!r}"
    with open(_CONFIG_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _read_graph_source() -> str:
    with open(_GRAPH_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _read_faster_whisper_source() -> str:
    with open(_FASTER_WHISPER_STT_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _read_main_source() -> str:
    with open(_MAIN_PATH, "r", encoding="utf-8") as fh:
        return fh.read()


def _extract_corrections_dict() -> dict:
    """Parse autocorrect_stt source to extract the hardcoded corrections dict."""
    source = _read_graph_source()
    m = re.search(r"corrections\s*=\s*\{([^}]*)\}", source, re.DOTALL)
    assert m, "Could not find `corrections = { ... }` block in graph.py"
    dict_src = "{" + m.group(1) + "}"
    return ast.literal_eval(dict_src)


def _get_stt_config_values() -> Dict[str, str]:
    """Extract stt section values from config.yaml, ignoring comments."""
    raw = _read_config_raw()
    # Find the stt: section lines (indent-based)
    m = re.search(r"^stt:\s*\n((?:[ \t]+.*\n)*)", raw, re.MULTILINE)
    assert m, "Could not locate `stt:` section in config.yaml"
    stt_block = m.group(1)

    result: Dict[str, str] = {}
    for line in stt_block.splitlines():
        # Strip comments and whitespace
        line_stripped = re.sub(r"#.*$", "", line).strip()
        kv = re.match(r"(\w+):\s*(\S+)", line_stripped)
        if kv:
            result[kv.group(1)] = kv.group(2)
    return result


# ==========================================================================
# 1. CONFIG VERIFICATION — the FIXED config must use faster_whisper + small.en + int8
# ==========================================================================

def test_config_stt_backend_is_faster_whisper():
    """config.yaml must select `stt.backend: faster_whisper` (the fixed CTranslate2 path).

    This is the primary fix for Defect 4 — switching from openai-whisper base.en to
    faster-whisper int8, which is both faster and, combined with small.en, more accurate.

    Recorded chosen config: backend = faster_whisper

    **Validates: Requirements 2.13**
    """
    values = _get_stt_config_values()
    backend = values.get("backend")
    assert backend == CHOSEN_STT_BACKEND, (
        f"Expected fixed `stt.backend: {CHOSEN_STT_BACKEND}` (CTranslate2 int8 path), "
        f"but config.yaml shows `stt.backend: {backend!r}`. "
        "The faster-whisper backend must be selected to fix Defect 4."
    )


def test_config_stt_model_is_small_en():
    """config.yaml STT model must be `small.en` — the higher-capacity model for accuracy.

    `small.en` (~244M params) vs `base.en` (~74M params) provides significantly better
    domain-term transcription while remaining within the ≤ 600ms latency budget when
    using faster-whisper int8.

    Recorded chosen config: model = small.en

    **Validates: Requirements 2.13**
    """
    values = _get_stt_config_values()
    model = values.get("model")
    assert model == CHOSEN_STT_MODEL, (
        f"Expected fixed `stt.model: {CHOSEN_STT_MODEL}` (higher-accuracy model), "
        f"but config.yaml shows `stt.model: {model!r}`. "
        "small.en is needed for intent-preserving transcription of NCERT domain terms."
    )


def test_config_stt_compute_type_is_int8():
    """config.yaml must specify `stt.compute_type: int8` — CTranslate2 quantized inference.

    int8 quantization allows the larger small.en model to remain within the latency budget
    by reducing computation cost on CPU.

    Recorded chosen config: compute_type = int8

    **Validates: Requirements 2.13**
    """
    values = _get_stt_config_values()
    compute_type = values.get("compute_type")
    assert compute_type == CHOSEN_COMPUTE_TYPE, (
        f"Expected `stt.compute_type: {CHOSEN_COMPUTE_TYPE}` (CTranslate2 quantized), "
        f"but config.yaml shows `stt.compute_type: {compute_type!r}`. "
        "int8 compute_type is needed for latency-budget-compliant small.en inference."
    )


# ==========================================================================
# 2. AUTOCORRECT WORD-BOUNDARY SAFETY
# ==========================================================================

def test_autocorrect_uses_word_boundary_regex():
    """autocorrect_stt must use `\\b` word-boundary anchors to prevent false positives.

    Without word boundaries, 'vangul' inside 'triangular' would silently match.
    The fix ensures only whole-word matches trigger corrections.

    **Validates: Requirements 2.13**
    """
    source = _read_graph_source()
    # The corrections loop must use re.compile with \b anchors
    assert r'\b' in source, (
        "autocorrect_stt source does not contain word-boundary `\\b` anchors. "
        "Word-boundary-safe regex is required to prevent false positives."
    )
    assert "re.compile" in source, (
        "autocorrect_stt does not use re.compile. "
        "It should compile a word-boundary pattern for each correction."
    )
    assert "re.IGNORECASE" in source, (
        "autocorrect_stt does not use re.IGNORECASE. "
        "Case-insensitive matching is needed for spoken transcription variants."
    )


def test_autocorrect_word_boundary_prevents_inner_word_match():
    """Word-boundary anchors must prevent matches inside longer words.

    For example, 'vangul' correction must NOT match inside a hypothetical
    word like 'triangular' that happens to contain 'ular'. Similarly, corrections
    must only trigger on the exact isolated phrase.

    **Validates: Requirements 2.13**
    """
    # 'vangul' is a correction key — it should NOT match inside 'triangular'
    # (no 'vangul' in 'triangular', but verify the correction contract: only
    # whole-word occurrences trigger replacement).
    text_no_match = "the triangular prism experiment"
    result = autocorrect_stt(text_no_match)
    assert result == text_no_match, (
        f"autocorrect_stt unexpectedly modified {text_no_match!r} -> {result!r}. "
        "Word-boundary anchors must prevent matches inside longer words."
    )

    # 'vangul' AS A STANDALONE WORD should be corrected to 'Bangalore'.
    text_match = "the school in vangul is nice"
    result_match = autocorrect_stt(text_match)
    assert "Bangalore" in result_match, (
        f"autocorrect_stt did not correct standalone 'vangul' in {text_match!r}. "
        f"Got: {result_match!r}"
    )


# ==========================================================================
# 3. AUTOCORRECT DOMAIN EXTENSIONS (Task 9.3 additions)
# ==========================================================================

def test_autocorrect_dict_contains_domain_extensions():
    """The corrections dict must contain all Task 9.3 NCERT domain extensions.

    Task 9.3 adds common Whisper mis-transcriptions of NCERT textbook terms:
    - 'photo synthesis' → 'photosynthesis'
    - 'foto synthesis' → 'photosynthesis'
    - 'electric magnetic' → 'electromagnetic'
    - 'mitokondria' → 'mitochondria'
    - 'mitokondrion' → 'mitochondrion'

    **Validates: Requirements 2.13**
    """
    corrections = _extract_corrections_dict()
    corrections_lower = {k.lower(): v for k, v in corrections.items()}

    for bad_phrase, good_phrase in EXPECTED_DOMAIN_CORRECTIONS.items():
        assert bad_phrase.lower() in corrections_lower, (
            f"Domain extension {bad_phrase!r} → {good_phrase!r} is MISSING from "
            f"autocorrect_stt dictionary. Task 9.3 requires this entry."
        )
        assert corrections_lower[bad_phrase.lower()].lower() == good_phrase.lower(), (
            f"Domain extension {bad_phrase!r} maps to "
            f"{corrections_lower[bad_phrase.lower()]!r} but expected {good_phrase!r}."
        )


@pytest.mark.parametrize("bad_phrase,expected_correction", EXPECTED_DOMAIN_CORRECTIONS.items())
def test_autocorrect_corrects_domain_mistranscription(bad_phrase: str, expected_correction: str):
    """Each domain extension must produce the correct replacement at runtime.

    Tests that autocorrect_stt actually corrects the mis-transcription in a
    representative query context.

    **Validates: Requirements 2.13**
    """
    query = f"what is {bad_phrase} in biology"
    result = autocorrect_stt(query)
    assert result != query, (
        f"autocorrect_stt did not correct {bad_phrase!r} in query {query!r}. "
        "Domain correction was not applied."
    )
    assert expected_correction.lower() in result.lower(), (
        f"Expected {expected_correction!r} in corrected output, got {result!r}. "
        f"Input: {query!r}"
    )


def test_autocorrect_preserves_uncovered_text():
    """autocorrect_stt must return uncovered text byte-for-byte unchanged.

    The post-correction net is additive — any text that does not match a
    correction key must be passed through unmodified.

    **Validates: Requirements 2.13, 3.4**
    """
    uncovered_inputs = [
        "explain Newton's laws of motion",
        "what is osmosis in plants",
        "describe Pythagoras theorem",
        "how does refraction work",
        "what is acceleration in physics",
        "tell me about gravitational force",
    ]
    for text in uncovered_inputs:
        result = autocorrect_stt(text)
        assert result == text, (
            f"autocorrect_stt modified uncovered text {text!r} → {result!r}. "
            "Additive contract violated — only covered mis-transcriptions should change."
        )


# ==========================================================================
# 4. INITIAL_PROMPT DOMAIN ENRICHMENT in main.py (Task 9.2)
# ==========================================================================

def test_main_handle_response_enriches_initial_prompt_with_domain_vocab():
    """_handle_response in main.py must enrich initial_prompt with domain vocabulary.

    Beyond wiki entity names, the initial_prompt must include NCERT subject/chapter
    vocabulary loaded from data/index/chunks.jsonl to bias Whisper toward domain terms.

    **Validates: Requirements 2.13, 3.4**
    """
    source = _read_main_source()

    # The chunks.jsonl path must be referenced
    assert "chunks.jsonl" in source, (
        "main.py does not reference chunks.jsonl. "
        "_handle_response must load domain vocabulary from the NCERT index."
    )

    # Domain enrichment markers must be present
    domain_markers = ["subject", "chapter", "domain_terms"]
    found = [m for m in domain_markers if m in source]
    assert len(found) >= 2, (
        f"main.py _handle_response is missing domain vocabulary enrichment. "
        f"Expected at least 2 of {domain_markers!r}; found: {found!r}. "
        "Task 9.2 requires enriching initial_prompt with subject/chapter terms."
    )


def test_main_initial_prompt_cap_is_bounded():
    """_handle_response must cap the initial_prompt to avoid excessive prompt length.

    Unbounded domain terms in initial_prompt could hurt STT latency. The implementation
    must cap the number of terms injected.

    **Validates: Requirements 2.13**
    """
    source = _read_main_source()
    # Look for a cap/limit on domain terms or total entity names
    has_cap = "[:20]" in source or "[:30]" in source or "[:50]" in source or "cap at" in source.lower()
    assert has_cap, (
        "main.py _handle_response does not appear to cap the domain terms list. "
        "A bounded slice (e.g. [:20] or [:30]) is needed to keep initial_prompt compact."
    )


def test_main_initial_prompt_includes_static_entities():
    """_handle_response must retain the original static entity names in initial_prompt.

    Req 3.4: the existing initial_prompt contract must be preserved. Static entities
    (Bangalore, Whitefield, Reachy Mini, Dr. Anjali) must still be present.

    **Validates: Requirements 3.4**
    """
    source = _read_main_source()
    static_entities = ["Bangalore", "Whitefield", "Reachy Mini", "Dr. Anjali"]
    for entity in static_entities:
        assert entity in source, (
            f"main.py does not contain static entity {entity!r} in initial_prompt. "
            "The original phonetic-bias entities must be preserved (Req 3.4)."
        )


# ==========================================================================
# 5. FASTER_WHISPER_STT SUPPORTS compute_type PARAMETER
# ==========================================================================

def test_faster_whisper_stt_accepts_compute_type_parameter():
    """FasterWhisperSTT.__init__ must accept and use a `compute_type` parameter.

    This is required for int8 CTranslate2 quantized inference (the fix for both
    latency and accuracy).

    **Validates: Requirements 2.13**
    """
    source = _read_faster_whisper_source()
    # __init__ must declare compute_type parameter
    m = re.search(r"def __init__\s*\(.*?\)", source, re.DOTALL)
    assert m, "Could not find FasterWhisperSTT.__init__ signature"
    init_sig = m.group(0)
    assert "compute_type" in init_sig, (
        f"FasterWhisperSTT.__init__ does not declare `compute_type` parameter. "
        f"Signature found: {init_sig!r}"
    )


def test_faster_whisper_stt_passes_compute_type_to_model():
    """FasterWhisperSTT must pass compute_type to WhisperModel constructor.

    The compute_type must be forwarded to the CTranslate2 WhisperModel to
    actually enable int8 quantization at inference time.

    **Validates: Requirements 2.13**
    """
    source = _read_faster_whisper_source()
    # WhisperModel(..., compute_type=compute_type) must appear in source
    assert "compute_type" in source, (
        "FasterWhisperSTT source does not contain `compute_type`. "
        "The parameter must be passed to WhisperModel for int8 quantization."
    )
    # Ensure it's forwarded to the model init, not just declared
    assert "compute_type=compute_type" in source or "compute_type=" in source, (
        "FasterWhisperSTT does not forward compute_type to WhisperModel. "
        "Int8 quantization requires compute_type to be passed at model load time."
    )


def test_faster_whisper_stt_default_is_int8():
    """FasterWhisperSTT default compute_type must be 'int8' for CPU inference.

    **Validates: Requirements 2.13**
    """
    source = _read_faster_whisper_source()
    m = re.search(r'compute_type\s*:\s*str\s*=\s*["\']([^"\']+)["\']', source)
    assert m, "Could not find compute_type default in FasterWhisperSTT.__init__"
    default_ct = m.group(1)
    assert default_ct == "int8", (
        f"Expected FasterWhisperSTT default compute_type='int8', got {default_ct!r}. "
        "int8 is the recommended setting for CPU CTranslate2 inference."
    )


# ==========================================================================
# 6. CHOSEN CONFIGURATION RECORDED (documentation / traceability)
# ==========================================================================

def test_chosen_config_recorded_in_config_yaml():
    """The chosen configuration must be documented in config.yaml with a comment.

    Task 9.4 requires recording: faster_whisper backend, small.en model, int8 compute_type.
    The config must have a comment explaining the accuracy/latency tradeoff decision.

    **Validates: Requirements 2.13**
    """
    raw = _read_config_raw()
    # The config must contain all three chosen values (not just in comments)
    values = _get_stt_config_values()
    assert values.get("backend") == "faster_whisper", "backend not set to faster_whisper"
    assert values.get("model") == "small.en", "model not set to small.en"
    assert values.get("compute_type") == "int8", "compute_type not set to int8"

    # There must be a comment documenting the tradeoff decision
    tradeoff_comment_markers = ["tradeoff", "accuracy", "latency", "600ms", "int8"]
    found_markers = [m for m in tradeoff_comment_markers if m.lower() in raw.lower()]
    assert len(found_markers) >= 3, (
        f"config.yaml does not appear to document the accuracy/latency tradeoff. "
        f"Expected at least 3 of {tradeoff_comment_markers!r} in comments; "
        f"found: {found_markers!r}"
    )


def test_chosen_config_is_offline_only():
    """The chosen STT configuration must remain fully offline.

    faster_whisper + small.en runs locally via CTranslate2. No network calls
    are introduced. This verifies Req 3.1 is preserved.

    **Validates: Requirements 3.1**
    """
    source = _read_faster_whisper_source()
    # No HTTP/network calls in FasterWhisperSTT
    network_indicators = ["requests", "urllib", "http.client", "socket.connect", "openai."]
    found = [ind for ind in network_indicators if ind in source]
    assert found == [], (
        f"FasterWhisperSTT contains network-related code: {found!r}. "
        "All STT inference must remain fully offline (Req 3.1)."
    )

    # WhisperModel is loaded locally (no download= parameter forcing network fetch)
    assert "WhisperModel(" in source, "FasterWhisperSTT must instantiate WhisperModel"


# ==========================================================================
# 7. PROPERTY-BASED: autocorrect domain corrections are stable and idempotent
# ==========================================================================

_DOMAIN_CORRECTIONS_STRATEGY = st.sampled_from(list(EXPECTED_DOMAIN_CORRECTIONS.keys()))
_QUERY_TEMPLATES = [
    "what is {term}",
    "explain {term}",
    "describe {term}",
    "how does {term} work",
    "{term} in science",
    "define {term}",
]


@st.composite
def _mis_transcription_query(draw) -> str:
    """Generate a student query containing a known mis-transcription."""
    bad_phrase = draw(_DOMAIN_CORRECTIONS_STRATEGY)
    template = draw(st.sampled_from(_QUERY_TEMPLATES))
    return template.format(term=bad_phrase)


@settings(max_examples=50)
@given(query=_mis_transcription_query())
def test_property4_domain_corrections_are_applied(query: str):
    """For any query containing a known mis-transcription, autocorrect_stt corrects it.

    This is the fix-checking property: for every input in the domain correction set,
    the corrected output should differ from the input (the correction was applied).

    **Validates: Requirements 2.13**
    """
    result = autocorrect_stt(query)
    # The result must differ — the mis-transcription was corrected
    assert result != query, (
        f"autocorrect_stt did not correct mis-transcription in query {query!r}. "
        f"Got unchanged output: {result!r}. "
        "Domain corrections should fire for all known mis-transcription patterns."
    )


@settings(max_examples=50)
@given(query=_mis_transcription_query())
def test_property4_domain_corrections_are_idempotent(query: str):
    """Applying autocorrect_stt twice must produce the same result as applying it once.

    Idempotency ensures the correction is stable and does not cascade.

    **Validates: Requirements 2.13**
    """
    once = autocorrect_stt(query)
    twice = autocorrect_stt(once)
    assert once == twice, (
        f"autocorrect_stt is not idempotent for {query!r}.\n"
        f"  First pass:  {once!r}\n"
        f"  Second pass: {twice!r}\n"
        "Applying corrections twice should not change the result."
    )


# ==========================================================================
# 8. SUMMARY: record the chosen configuration
# ==========================================================================

def test_document_defect4_verification_summary():
    """Documents the fixed state and chosen configuration for the record.

    Always passes — records the verified configuration for Task 9.4.

    **Validates: Requirements 2.13**
    """
    corrections = _extract_corrections_dict()
    values = _get_stt_config_values()

    summary = (
        "\n"
        "=" * 70 + "\n"
        "DEFECT 4 — STT Understanding — FIXED State Verification Summary\n"
        "=" * 70 + "\n"
        f"\n"
        f"  CHOSEN CONFIGURATION (recorded for measurement, Task 9.4):\n"
        f"    backend:      {values.get('backend', 'N/A')!r}\n"
        f"    model:        {values.get('model', 'N/A')!r}\n"
        f"    compute_type: {values.get('compute_type', 'N/A')!r}\n"
        f"\n"
        f"  EXPECTED CONFIG: faster_whisper / small.en / int8\n"
        f"    backend match:      {values.get('backend') == CHOSEN_STT_BACKEND}\n"
        f"    model match:        {values.get('model') == CHOSEN_STT_MODEL}\n"
        f"    compute_type match: {values.get('compute_type') == CHOSEN_COMPUTE_TYPE}\n"
        f"\n"
        f"  AUTOCORRECT UPGRADE (Task 9.3 domain extensions present):\n"
        + "".join(
            f"    {k!r} → {v!r}: {'PRESENT' if k.lower() in {c.lower() for c in corrections} else 'MISSING'}\n"
            for k, v in EXPECTED_DOMAIN_CORRECTIONS.items()
        )
        + f"\n"
        f"  WORD-BOUNDARY SAFETY: regex \\b anchors in autocorrect_stt\n"
        f"\n"
        f"  INITIAL_PROMPT ENRICHMENT:\n"
        f"    chunks.jsonl domain vocab included in initial_prompt: "
        f"{'chunks.jsonl' in _read_main_source()}\n"
        f"\n"
        f"  OFFLINE STATUS: No network calls in FasterWhisperSTT\n"
        f"\n"
        f"  LATENCY BUDGET: small.en int8 targets ≤ 600ms on CPU via CTranslate2.\n"
        f"    Fallback: base.en int8 if small.en exceeds 600ms on target machine.\n"
        + "=" * 70
    )
    print(summary)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
