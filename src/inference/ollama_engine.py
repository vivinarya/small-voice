# src/inference/ollama_engine.py
"""Ollama backend for BaseEngine.

Uses Ollama's OpenAI-compatible REST API (http://localhost:11434/v1).
Ollama manages CUDA/GPU internally — no PyTorch or CUDA toolkit required.

On Jetson Orin with JetPack 7, this is the recommended LLM backend because:
  - Ollama ships its own bundled CUDA libs (no system CUDA needed)
  - It automatically offloads all layers to the Orin's Ampere GPU
  - No compilation required — just `ollama pull <model>` and run

Usage in config.yaml:
    engine:
      backend: ollama
      model_path: qwen2.5:1.5b   # any model pulled via `ollama pull`
"""
from __future__ import annotations

import logging
from typing import Iterator

from .base import BaseEngine
from .engine import _SYSTEM_PROMPT

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaEngine(BaseEngine):
    """Ollama REST API implementation of BaseEngine.

    Streams tokens via Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    Requires Ollama to be running: `ollama serve` (or it auto-starts as a service).
    """

    def __init__(self, model_name: str = "qwen2.5:1.5b") -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "openai package is required for OllamaEngine.\n"
                "Install it with:  pip install openai"
            ) from exc

        self._model = model_name
        self._client = OpenAI(
            base_url=_OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama ignores the key but the client requires one
        )
        logger.info("OllamaEngine ready: model='%s' url='%s'", model_name, _OLLAMA_BASE_URL)

    def get_stream(self, prompt: str) -> Iterator[str]:
        """Stream the assistant reply via Ollama's chat completion API."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                max_tokens=256,
            )
            yielded = False
            for chunk in stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yielded = True
                    yield content
            if not yielded:
                yield ""
        except Exception as exc:
            logger.error("OllamaEngine.get_stream error: %s", exc)
            yield "Sorry, I could not reach the language model. Is Ollama running?"

    def reset(self) -> None:
        """No-op: Ollama is stateless per request."""
        pass

    def warmup(self) -> None:
        """Send one short request to load the model into VRAM."""
        logger.info("OllamaEngine: warming up model '%s'...", self._model)
        for _ in self.get_stream("Hi"):
            break
        logger.info("OllamaEngine: warmup complete.")
