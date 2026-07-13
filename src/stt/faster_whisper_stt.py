# src/stt/faster_whisper_stt.py
"""faster-whisper (CTranslate2) STT backend — used on Orin Nano for speed.

Requires:
    pip install faster-whisper
"""
import logging
import numpy as np
from .base import BaseSTT

logger = logging.getLogger(__name__)


class FasterWhisperSTT(BaseSTT):
    """faster-whisper backend for BaseSTT.
    
    Uses CTranslate2 under the hood; much faster and lighter than openai-whisper
    on the same hardware. Use compute_type="int8" for CPU or "float16" for GPU.
    """

    def __init__(
        self,
        model_name: str = "base.en",
        compute_type: str = "int8",
        device: str = "cpu",
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required for FasterWhisperSTT but is not installed. "
                "Install it with:\n\n"
                "    pip install faster-whisper"
            ) from exc
        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        logger.info("FasterWhisperSTT loaded model '%s' on %s (%s)", model_name, device, compute_type)

    def transcribe(self, audio: np.ndarray, *, initial_prompt: str | None = None) -> str:
        """Transcribe mono float32 audio at 16 kHz.
        
        audio: mono float32 numpy array in [-1, 1] at 16 kHz
        Returns stripped transcript, '' for silence/noise.
        """
        kwargs: dict = {}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        segments, _info = self._model.transcribe(audio, **kwargs)
        text = " ".join(seg.text for seg in segments).strip()
        return text
