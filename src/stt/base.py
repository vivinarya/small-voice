# src/stt/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseSTT(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, *, initial_prompt: str | None = None) -> str:
        """audio: mono float32 in [-1, 1] at 16 kHz. Returns stripped transcript ('' if empty)."""
        ...
