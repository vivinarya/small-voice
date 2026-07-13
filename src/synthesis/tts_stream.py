import os
import re
import subprocess
import logging
import sounddevice as sd
import numpy as np
import threading
import queue

from .base import BaseTTS
from .text_norm import normalize_for_tts, extract_complete_sentences

logger = logging.getLogger(__name__)


_MARKDOWN_RE = re.compile(
    r'\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|>\s?|\[([^\]]+)\]\([^)]+\)|\!\[[^\]]*\]\([^)]+\)'
)

# Coalesce a very short leading sentence (≤ this many words) with the next sentence
# before synthesis so isolated short utterances like "Sure." don't sound clipped.
# This is a synthesis-grouping choice only — segmentation is unchanged.
_SHORT_SENTENCE_WORD_THRESHOLD = 3
_MULTI_SPACE_RE = re.compile(r'\s{2,}')

def _clean_for_tts(text: str) -> str:
    """Remove markdown formatting so TTS reads clean prose."""
    # Replace markdown links [label](url) → just label
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # Strip image syntax ![alt](url)
    text = re.sub(r'\!\[[^\]]*\]\([^)]+\)', '', text)
    # Strip bold/italic markers * _ `
    text = re.sub(r'\*{1,3}|_{1,3}|`{1,3}', '', text)
    # Strip heading hashes at start of line
    text = re.sub(r'(?m)^#{1,6}\s?', '', text)
    # Strip block-quote markers
    text = re.sub(r'(?m)^>\s?', '', text)
    # Collapse whitespace
    text = _MULTI_SPACE_RE.sub(' ', text).strip()
    return text


class PiperTTS(BaseTTS):
    def __init__(self, model_path="assets/piper_voices/en_US-lessac-medium.onnx"):
        self.model_path = model_path
        self.piper_path = "assets/piper/piper.exe"
        self.samplerate = 22050
        # playback_queue is kept for the speak() single-utterance path only
        self.playback_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self):
        """Background thread: play single-utterance PCM chunks from the queue."""
        while True:
            item = self.playback_queue.get()
            if item is None:
                self.playback_queue.task_done()
                continue
            try:
                audio_data = item
                sd.play(audio_data, self.samplerate)
                sd.wait()
            except Exception as e:
                logger.error(f"Playback thread error: {e}")
            finally:
                self.playback_queue.task_done()

    def _synthesize_to_pcm(self, text: str) -> "np.ndarray | None":
        """Synthesize text via piper and return raw PCM as a numpy int16 array.

        Returns None if synthesis fails (binary not found, piper error, etc.).
        Does not play anything — the caller decides how to render the audio.
        """
        clean = _clean_for_tts(text)
        if not clean:
            return None

        normalized = normalize_for_tts(clean)
        if not normalized.strip():
            return None

        logger.info(f"Synthesizing: {normalized[:80]}{'...' if len(normalized) > 80 else ''}")

        try:
            command = [
                self.piper_path,
                "--model", self.model_path,
                "--output-raw",
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(input=normalized.encode("utf-8"))

            if process.returncode != 0:
                logger.error(f"Piper error: {stderr.decode()}")
                return None

            return np.frombuffer(stdout, dtype=np.int16)

        except FileNotFoundError:
            logger.error(f"Piper binary not found at '{self.piper_path}'. Is it installed?")
            return None
        except Exception as e:
            logger.error(f"TTS Synthesis Error: {e}")
            return None

    def speak(self, text: str):
        """Synthesize a single utterance and enqueue it for playback.

        Used for standalone one-shot utterances (e.g. the shutdown message).
        Playback is asynchronous — call playback_queue.join() to wait.
        """
        if not text:
            return

        pcm = self._synthesize_to_pcm(text)
        if pcm is not None:
            self.playback_queue.put(pcm)

    def stream_text(self, text_iter):
        """Sentence-level streaming TTS with gapless playback.

        Sentences are synthesised as each complete sentence arrives from the
        LLM token stream, then written into a single open sounddevice
        OutputStream so there is no device stop/restart (and therefore no
        audible gap) between consecutive sentences.

        Ordered playback is preserved: sentences are written to the stream in
        the order they are produced.

        Short-sentence coalescing: if the first complete sentence is very short
        (≤ _SHORT_SENTENCE_WORD_THRESHOLD words), it is held in ``pending_short``
        and prepended to the next sentence before synthesis. This prevents a
        clipped isolated utterance (e.g. "Sure.") without altering
        ``extract_complete_sentences`` or ``normalize_for_tts``.

        Returns the full text spoken.
        """
        buffer = ""
        full_text = ""
        # Holds a short leading sentence waiting to be coalesced with the next one.
        pending_short: str = ""

        # Open a single OutputStream that stays alive across all sentences.
        # Blocking writes mean each sentence's PCM is queued into the device
        # buffer immediately after synthesis, giving early TTFS while remaining
        # completely gapless.
        stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=1,
            dtype="int16",
        )
        stream.start()

        try:
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
                        word_count = len(sentence.split())
                        if not pending_short and word_count <= _SHORT_SENTENCE_WORD_THRESHOLD:
                            # Short leading sentence — hold it for coalescing with the next.
                            pending_short = sentence
                        else:
                            # Prepend any held short sentence and synthesize together.
                            to_speak = (pending_short + " " + sentence).strip() if pending_short else sentence
                            pending_short = ""
                            pcm = self._synthesize_to_pcm(to_speak)
                            if pcm is not None:
                                stream.write(pcm)

            # Flush: combine any held short sentence with whatever remains in the buffer.
            remaining = (pending_short + " " + buffer).strip() if pending_short else buffer.strip()
            if remaining:
                pcm = self._synthesize_to_pcm(remaining)
                if pcm is not None:
                    stream.write(pcm)

            print()

        finally:
            # stop() drains the internal device buffer so all queued audio
            # finishes playing before we return, then releases the device.
            stream.stop()
            stream.close()

        return full_text.strip()


# Backward-compatibility alias
TTSStreamer = PiperTTS
