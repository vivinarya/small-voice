import numpy as np

class AudioProcessor:
    def __init__(self, target_samplerate=16000):
        self.target_samplerate = target_samplerate

    def is_speech(self, audio_chunk, threshold=300):
        """
        Simple energy-based Voice Activity Detection (VAD).
        Calculates the RMS of the chunk and compares against a threshold.
        """
        if audio_chunk.dtype == np.int16:
            rms = np.sqrt(np.mean(audio_chunk.astype(np.float32)**2))
        else:
            rms = np.sqrt(np.mean(audio_chunk**2)) * 32768.0
        return rms > threshold

    def process_for_inference(self, audio_chunk):
        """
        Full processing pipeline for raw audio.
        Returns PCM16 bytes for LiteRT-LM.
        """
        if audio_chunk.dtype != np.int16:
            audio_chunk = (audio_chunk * 32767).astype(np.int16)
        
        return audio_chunk.tobytes()
