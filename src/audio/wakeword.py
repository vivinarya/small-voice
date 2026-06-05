import numpy as np
from openwakeword.model import Model

class WakeWordDetector:
    def __init__(self, model_paths=None):
        """
        Initializes the openWakeWord model using explicit ONNX model paths.
        This avoids the tflite_runtime dependency which is unavailable on Python 3.13/Windows.

        Args:
            model_paths: List of paths to .onnx model files.
                         Defaults to the bundled hey_jarvis ONNX model.
        """
        if model_paths is None:
            model_paths = ["assets/wakeword_models/hey_jarvis_v0.1.onnx"]

        # Pass file paths directly - openWakeWord will use onnxruntime automatically
        self.oww_model = Model(wakeword_models=model_paths, inference_framework="onnx")

        # Extract short names from file paths for prediction_buffer lookup
        self.model_names = [
            p.split("/")[-1].replace(".onnx", "") for p in model_paths
        ]

    def check(self, audio_chunk):
        """
        Check for wake word in the given audio chunk.
        Expects 16kHz int16 audio as a numpy array.
        Returns (detected: bool, model_name: str | None)
        """
        # Ensure input is 1D for openwakeword
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.reshape(-1)

        # predict() handles internal buffering and updates prediction_buffer
        self.oww_model.predict(audio_chunk)

        for name in self.model_names:
            if self.oww_model.prediction_buffer[name][-1] > 0.5:
                return True, name
        return False, None
