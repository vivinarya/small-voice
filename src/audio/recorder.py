import sounddevice as sd
import numpy as np
import queue
import logging

logger = logging.getLogger(__name__)

class AudioRecorder:
    def __init__(self, device=None, samplerate=16000, blocksize=512):
        self.device = device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.audio_queue = queue.Queue()
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio status: {status}")
        # Add a copy of the input data to the queue
        self.audio_queue.put(indata.copy())

    def start(self):
        logger.info(f"Starting audio stream on device {self.device}...")
        self.stream = sd.InputStream(
            device=self.device,
            channels=1,
            samplerate=self.samplerate,
            blocksize=self.blocksize,
            callback=self.callback,
            dtype='int16' # PCM 16-bit
        )
        self.stream.start()

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped.")

    def get_audio_chunk(self):
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            return None

    def clear_queue(self):
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
