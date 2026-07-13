# src/factories.py
"""Factory functions that build backend implementations from AppConfig.

These are the ONLY places that import concrete implementation classes.
Everything else depends on the base interfaces only.
"""
import logging
from config import AppConfig
from inference.base import BaseEngine
from stt.base import BaseSTT
from synthesis.base import BaseTTS

logger = logging.getLogger(__name__)


def build_engine(cfg: AppConfig) -> BaseEngine:
    """Instantiate and return the configured LLM engine."""
    if cfg.engine_backend == "litert":
        from inference.engine import LiteRTEngine  # noqa: PLC0415
        logger.info("Building LiteRTEngine from '%s'", cfg.model_path)
        return LiteRTEngine(model_path=cfg.model_path)
    elif cfg.engine_backend == "llama_cpp":
        from inference.llama_cpp_engine import LlamaCppEngine  # noqa: PLC0415
        logger.info(
            "Building LlamaCppEngine from '%s' (n_gpu_layers=%d)",
            cfg.model_path, cfg.n_gpu_layers,
        )
        return LlamaCppEngine(
            model_path=cfg.model_path,
            n_gpu_layers=cfg.n_gpu_layers,
        )
    else:
        # _validate() in config.py should have caught this already
        raise ValueError(f"Unknown engine backend: '{cfg.engine_backend}'")


def build_stt(cfg: AppConfig) -> BaseSTT:
    """Instantiate and return the configured STT model."""
    if cfg.stt_backend == "whisper":
        from stt.whisper_stt import WhisperSTT  # noqa: PLC0415
        logger.info("Building WhisperSTT (model=%s)", cfg.stt_model)
        return WhisperSTT(model_name=cfg.stt_model)
    elif cfg.stt_backend == "faster_whisper":
        from stt.faster_whisper_stt import FasterWhisperSTT  # noqa: PLC0415
        logger.info(
            "Building FasterWhisperSTT (model=%s, compute_type=%s)",
            cfg.stt_model, cfg.stt_compute_type,
        )
        return FasterWhisperSTT(model_name=cfg.stt_model, compute_type=cfg.stt_compute_type)
    else:
        raise ValueError(f"Unknown STT backend: '{cfg.stt_backend}'")


def build_tts(cfg: AppConfig) -> BaseTTS:
    """Instantiate and return the configured TTS engine."""
    if cfg.tts_backend == "piper":
        from synthesis.tts_stream import PiperTTS  # noqa: PLC0415
        logger.info("Building PiperTTS (voice=%s)", cfg.tts_voice_path)
        return PiperTTS(model_path=cfg.tts_voice_path)
    elif cfg.tts_backend == "kokoro":
        from synthesis.kokoro_tts import KokoroTTS  # noqa: PLC0415
        logger.info("Building KokoroTTS (voice=%s)", cfg.tts_voice_path)
        return KokoroTTS(voice_path=cfg.tts_voice_path)
    else:
        raise ValueError(f"Unknown TTS backend: '{cfg.tts_backend}'")


def build_retrieval(cfg: AppConfig):
    """Instantiate and return the configured RetrievalService.
    
    Returns a no-op NullRetrievalService if the FAISS index doesn't exist yet,
    so the assistant still works without the RAG layer built.
    """
    import os  # noqa: PLC0415
    
    index_dir = cfg.index_dir
    faiss_index_path = os.path.join(index_dir, "faiss.index")
    
    if not os.path.exists(faiss_index_path):
        logger.warning(
            "FAISS index not found at '%s'. RAG disabled. "
            "Run: python scripts/build_index.py --src data/docs --out data/index",
            faiss_index_path,
        )
        from retrieval.service import NullRetrievalService  # noqa: PLC0415
        return NullRetrievalService()
    
    from retrieval.service import FAISSRetrievalService  # noqa: PLC0415
    from retrieval.embedder import MiniLMEmbedder  # noqa: PLC0415
    
    logger.info("Building FAISSRetrievalService from '%s'", index_dir)
    embedder = MiniLMEmbedder()
    return FAISSRetrievalService(index_dir=index_dir, embedder=embedder, min_score=cfg.min_score)
