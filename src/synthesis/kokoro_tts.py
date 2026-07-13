# src/synthesis/kokoro_tts.py
"""Kokoro TTS backend — high-quality voice for Orin Nano (Phase 4+).

Requires:
    pip install kokoro-onnx
    # or: pip install kokoro
"""
import logging
import threading
import queue
from typing import Iterator

import numpy as np
import sounddevice as sd

from .base import BaseTTS
from .text_norm import normalize_for_tts, extract_complete_sentences

logger = logging.getLogger(__name__)


class KokoroTTS(BaseTTS):
    """Kokoro TTS backend (high-quality ONNX voice).
    
    Uses kokoro-onnx for synthesis with the same sentence-level streaming
    and playback queue as PiperTTS, ensuring a natural non-choppy voice.
    
    Install: pip install kokoro-onnx soundfile
    """

    def __init__(
        self,
        voice_path: str = "models/kokoro-v0_19.onnx",
        voice_name: str = "af",
        samplerate: int = 24000,
    ) -> None:
        try:
            from kokoro_onnx import Kokoro  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "kokoro-onnx is required for KokoroTTS but is not installed.\n"
                "Install it with:\n\n"
                "    pip install kokoro-onnx\n"
            ) from exc
        
        self._kokoro = Kokoro(voice_path, voice_name)
        self.samplerate = samplerate
        self._playback_queue: queue.Queue = queue.Queue()
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._playback_thread.start()
        logger.info("KokoroTTS initialized with voice '%s' from '%s'", voice_name, voice_path)

    def _playback_loop(self) -> None:
        while True:
            item = self._playback_queue.get()
            if item is None:
                self._playback_queue.task_done()
                continue
            try:
                sd.play(item, self.samplerate)
                sd.wait()
            except Exception as exc:
                logger.error("KokoroTTS playback error: %s", exc)
            finally:
                self._playback_queue.task_done()

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        normalized = normalize_for_tts(text)
        try:
            samples, sr = self._kokoro.create(normalized, speed=1.0)
            audio = np.array(samples, dtype=np.float32)
            self._playback_queue.put(audio)
        except Exception as exc:
            logger.error("KokoroTTS synthesis error: %s", exc)

    def stream_text(self, text_iter: Iterator[str]) -> str:
        buffer = ""
        full_text = ""

        for chunk in text_iter:
            if not chunk:
                continue
            print(chunk, end="", flush=True)
            buffer += chunk
            full_text += chunk

            results = extract_complete_sentences(buffer)
            if results:
                sentence, buffer = results[0]
                if sentence.strip():
                    self.speak(sentence)

        if buffer.strip():
            self.speak(buffer.strip())
        print()

        self._playback_queue.join()
        return full_text.strip()

    def synth_to_pcm(self, text: str) -> np.ndarray:
        """Return raw PCM float32 samples without playing (for audio cache)."""
        normalized = normalize_for_tts(text)
        samples, _sr = self._kokoro.create(normalized, speed=1.0)
        return np.array(samples, dtype=np.float32)
