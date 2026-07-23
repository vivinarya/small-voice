# src/stt/whisper_stt.py
import numpy as np
from .base import BaseSTT


class WhisperSTT(BaseSTT):
    """openai-whisper backend for BaseSTT."""

    def __init__(self, model_name: str = "base.en") -> None:
        # Deferred import — whisper is only needed when this class is instantiated
        import whisper  # noqa: PLC0415
        self._model = whisper.load_model(model_name)

    def transcribe(self, audio: np.ndarray, *, initial_prompt: str | None = None) -> str:
        """Transcribe mono float32 audio at 16 kHz.

        audio: mono float32 numpy array in [-1, 1] at 16 kHz
        Returns stripped transcript string, '' if empty/noise.
        """
        kwargs: dict = {"fp16": False}
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt
        result = self._model.transcribe(audio, **kwargs)
        return result.get("text", "").strip()
