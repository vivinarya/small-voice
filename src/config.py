# src/config.py
"""Centralized configuration loader for the pluggable edge assistant.

Priority (highest to lowest):
  1. Environment variables
  2. config.yaml values
  3. Hardcoded defaults (as fallback within the YAML parsing)

Validation rules (fail-fast at startup):
  - engine_backend ∈ {"litert", "llama_cpp", "ollama"}
  - stt_backend ∈ {"whisper", "faster_whisper"}
  - tts_backend ∈ {"piper", "kokoro"}
  - retrieval_backend ∈ {"chroma", "faiss"}
  - embed_backend ∈ {"minilm", "bge_small"}
  - retrieval_k ∈ [1, 5] (clamped)
  - n_gpu_layers is integer
"""
import os
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_VALID_ENGINE  = {"litert", "llama_cpp", "ollama"}
_VALID_STT     = {"whisper", "faster_whisper"}
_VALID_TTS     = {"piper", "kokoro"}
_VALID_EMBED   = {"minilm", "bge_small"}
_VALID_RETRIEVAL = {"chroma", "faiss"}


@dataclass(frozen=True)
class AppConfig:
    engine_backend: str   # "litert" | "llama_cpp"
    model_path: str
    n_gpu_layers: int     # 0 = CPU only, -1 = full GPU offload

    stt_backend: str      # "whisper" | "faster_whisper"
    stt_model: str        # "base.en", "small.en", ...
    stt_compute_type: str # "int8" (CPU quantized) | "float16" (GPU) | "float32"

    tts_backend: str      # "piper" | "kokoro"
    tts_voice_path: str

    embed_backend: str      # "minilm" | "bge_small"
    retrieval_backend: str  # "chroma" | "faiss"  — which vector store to use
    index_dir: str
    retrieval_k: int        # 1..5
    min_score: float        # relevance gate for retrieved chunks (cosine similarity, default 0.25)

    robot_enabled: bool = False   # control Reachy Mini robot
    robot_ip: str = ""            # IP address of the robot Pi Zero


def _load_yaml(path: str) -> dict:
    """Load a YAML config file. Returns empty dict if file missing."""
    try:
        import yaml  # noqa: PLC0415
    except ImportError:
        logger.warning("PyYAML not installed — using env vars / defaults only. pip install PyYAML")
        return {}
    if not os.path.exists(path):
        logger.warning("config.yaml not found at '%s' — using env vars / defaults only.", path)
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _validate(cfg: AppConfig) -> None:
    """Raise ValueError for invalid config values. Fail fast at startup."""
    if cfg.engine_backend not in _VALID_ENGINE:
        raise ValueError(
            f"engine.backend '{cfg.engine_backend}' is invalid. "
            f"Choose one of: {sorted(_VALID_ENGINE)}"
        )
    if cfg.stt_backend not in _VALID_STT:
        raise ValueError(
            f"stt.backend '{cfg.stt_backend}' is invalid. "
            f"Choose one of: {sorted(_VALID_STT)}"
        )
    if cfg.tts_backend not in _VALID_TTS:
        raise ValueError(
            f"tts.backend '{cfg.tts_backend}' is invalid. "
            f"Choose one of: {sorted(_VALID_TTS)}"
        )
    if cfg.embed_backend not in _VALID_EMBED:
        raise ValueError(
            f"retrieval.embed_backend '{cfg.embed_backend}' is invalid. "
            f"Choose one of: {sorted(_VALID_EMBED)}"
        )
    if cfg.retrieval_backend not in _VALID_RETRIEVAL:
        raise ValueError(
            f"retrieval.backend '{cfg.retrieval_backend}' is invalid. "
            f"Choose one of: {sorted(_VALID_RETRIEVAL)}"
        )


def load_config(path: str = "config.yaml") -> AppConfig:
    """Load configuration from YAML file + environment variable overrides.
    
    Environment variables (all optional, override file values):
        ENGINE_BACKEND, MODEL_PATH, N_GPU_LAYERS
        STT_BACKEND, STT_MODEL
        TTS_BACKEND, TTS_VOICE_PATH
        EMBED_BACKEND, INDEX_DIR, RETRIEVAL_K
    """
    raw = _load_yaml(path)
    engine_sect    = raw.get("engine", {})
    stt_sect       = raw.get("stt", {})
    tts_sect       = raw.get("tts", {})
    retrieval_sect = raw.get("retrieval", {})

    # --- Engine ---
    engine_backend = (
        os.environ.get("ENGINE_BACKEND")
        or engine_sect.get("backend", "litert")
    )
    model_path = (
        os.environ.get("MODEL_PATH")
        or engine_sect.get("model_path", "assets/gemma-4-E4B-it.litertlm")
    )
    try:
        n_gpu_layers = int(
            os.environ.get("N_GPU_LAYERS")
            or engine_sect.get("n_gpu_layers", 0)
        )
    except (TypeError, ValueError):
        raise ValueError("n_gpu_layers must be an integer (0 = CPU only, -1 = full GPU offload)")

    # --- STT ---
    stt_backend = (
        os.environ.get("STT_BACKEND")
        or stt_sect.get("backend", "whisper")
    )
    stt_model = (
        os.environ.get("STT_MODEL")
        or stt_sect.get("model", "base.en")
    )
    stt_compute_type = (
        os.environ.get("STT_COMPUTE_TYPE")
        or stt_sect.get("compute_type", "int8")
    )

    # --- TTS ---
    tts_backend = (
        os.environ.get("TTS_BACKEND")
        or tts_sect.get("backend", "piper")
    )
    tts_voice_path = (
        os.environ.get("TTS_VOICE_PATH")
        or tts_sect.get("voice_path", "assets/piper_voices/en_US-lessac-medium.onnx")
    )

    # --- Retrieval ---
    retrieval_backend = (
        os.environ.get("RETRIEVAL_BACKEND")
        or retrieval_sect.get("backend", "chroma")   # default: ChromaDB
    )
    embed_backend = (
        os.environ.get("EMBED_BACKEND")
        or retrieval_sect.get("embed_backend", "minilm")
    )
    index_dir = (
        os.environ.get("INDEX_DIR")
        or retrieval_sect.get("index_dir", "data/chroma")
    )
    try:
        raw_k = int(
            os.environ.get("RETRIEVAL_K")
            or retrieval_sect.get("k", 3)
        )
        retrieval_k = max(1, min(5, raw_k))   # clamp to [1, 5]
        if retrieval_k != raw_k:
            logger.warning("retrieval.k=%d is out of range [1,5]; clamped to %d", raw_k, retrieval_k)
    except (TypeError, ValueError):
        raise ValueError("retrieval.k must be an integer in [1, 5]")

    try:
        min_score = float(
            os.environ.get("MIN_SCORE")
            or retrieval_sect.get("min_score", 0.25)
        )
        if not (0.0 <= min_score <= 1.0):
            logger.warning("retrieval.min_score=%f is outside [0,1]; using 0.25", min_score)
            min_score = 0.25
    except (TypeError, ValueError):
        raise ValueError("retrieval.min_score must be a float in [0.0, 1.0]")

    # --- Robot ---
    robot_sect = raw.get("robot", {})
    robot_enabled = (
        os.environ.get("ROBOT_ENABLED", "").lower() in ("true", "1", "yes")
        or robot_sect.get("enabled", False)
    )
    robot_ip = (
        os.environ.get("ROBOT_IP")
        or robot_sect.get("ip", "")
    )

    cfg = AppConfig(
        engine_backend=engine_backend,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        stt_backend=stt_backend,
        stt_model=stt_model,
        stt_compute_type=stt_compute_type,
        tts_backend=tts_backend,
        tts_voice_path=tts_voice_path,
        embed_backend=embed_backend,
        retrieval_backend=retrieval_backend,
        index_dir=index_dir,
        retrieval_k=retrieval_k,
        min_score=min_score,
        robot_enabled=robot_enabled,
        robot_ip=robot_ip,
    )
    _validate(cfg)
    logger.info(
        "Config loaded: engine=%s stt=%s/%s tts=%s robot=%s(%s)",
        cfg.engine_backend, cfg.stt_backend, cfg.stt_model,
        cfg.tts_backend, cfg.robot_enabled, cfg.robot_ip,
    )
    return cfg
