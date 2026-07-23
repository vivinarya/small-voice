# src/synthesis/base.py
from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


class BaseTTS(ABC):
    """Pluggable TTS contract.
    
    Implementations own synthesis, text normalization, and audio playback.
    """
    samplerate: int = 22050

    @abstractmethod
    def speak(self, text: str) -> None:
        """Synthesize one complete utterance and enqueue for playback.
        
        Synthesis is synchronous; playback is async (queued to background thread).
        """
        ...

    @abstractmethod
    def stream_text(self, text_iter: Iterator[str]) -> str:
        """Consume an LLM token stream, synthesize at SENTENCE boundaries
        (split ONLY on . ? !), play in order, and return the full spoken text.
        
        Loop invariant: buffer holds only text after the last completed sentence boundary.
        """
        ...

    def _synthesize_to_pcm(self, text: str) -> np.ndarray | None:
        """Synthesize text to raw PCM numpy array without playing.
        
        Default returns None or delegates to synth_to_pcm. Override in implementations.
        """
        try:
            return self.synth_to_pcm(text)
        except NotImplementedError:
            return None

    def synth_to_pcm(self, text: str) -> np.ndarray:
        """Optional: return PCM16 numpy array without playing (used for audio cache).
        
        Default raises NotImplementedError; override in implementations that support it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support synth_to_pcm")
