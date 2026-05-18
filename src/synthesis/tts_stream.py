import os
import re
import subprocess
import logging
import sounddevice as sd
import numpy as np
import threading
import queue

logger = logging.getLogger(__name__)


_MARKDOWN_RE = re.compile(
    r'\*{1,3}|_{1,3}|`{1,3}|#{1,6}\s?|>\s?|\[([^\]]+)\]\([^)]+\)|\!\[[^\]]*\]\([^)]+\)'
)
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


class TTSStreamer:
    def __init__(self, model_path="assets/piper_voices/en_US-lessac-medium.onnx"):
        self.model_path = model_path
        self.piper_path = "assets/piper/piper.exe"
        self.samplerate = 22050   
        self.playback_queue = queue.Queue()
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self):
        """Continuously play audio chunks from the queue in a background thread."""
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

    def speak(self, text: str):
        if not text:
            return

        clean = _clean_for_tts(text)
        if not clean:
            return

        logger.info(f"Speaking: {clean[:80]}{'...' if len(clean) > 80 else ''}")

        try:
            command = [
                self.piper_path,
                "--model", self.model_path,
                "--output-raw"
            ]
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=clean.encode('utf-8'))

            if process.returncode != 0:
                logger.error(f"Piper error: {stderr.decode()}")
                return

            audio_data = np.frombuffer(stdout, dtype=np.int16)
            self.playback_queue.put(audio_data)

        except FileNotFoundError:
            logger.error(f"Piper binary not found at '{self.piper_path}'. Is it installed?")
        except Exception as e:
            logger.error(f"TTS Synthesis Error: {e}")

    def stream_text(self, text_iterator):
        """Sentence-level streaming TTS. Returns the full text spoken."""
        buffer = ""
        full_text = ""
        sentence_endings = {'. ', '? ', '! ', '.\n', '?\n', '!\n'}
        
        for chunk in text_iterator:
            if not chunk: continue
            print(chunk, end="", flush=True)
            buffer += chunk
            full_text += chunk

            # Check for sentence boundary
            for ending in sentence_endings:
                if ending in buffer:
                    parts = buffer.split(ending, 1)
                    sentence = parts[0] + ending.strip()
                    buffer = parts[1] if len(parts) > 1 else ""
                    
                    if sentence.strip():
                        # Synthesizes synchronously, but queues playback asynchronously
                        self.speak(sentence)
                    break
                    
        if buffer.strip():
            self.speak(buffer.strip())
        print()
        
        # Block the main loop until all queued audio finishes playing
        self.playback_queue.join()
        return full_text.strip()
