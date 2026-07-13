# src/inference/llama_cpp_engine.py
"""llama.cpp / GGUF backend for BaseEngine.

The heavy ``llama-cpp-python`` import is deferred to ``__init__`` so that
importing this module never fails when the library is absent — only
instantiation fails, with a clear pip-install message.
"""

import logging
from typing import Iterator

from .base import BaseEngine
from .base import SYSTEM_PROMPT as _SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Fallback ladder for n_gpu_layers when an OOM-like error occurs.
# Values are tried in order; levels >= the failing value are skipped.
_GPU_FALLBACK_LADDER = [-1, 32, 16, 0]

# Keywords that identify an OOM-like exception message (case-insensitive).
_OOM_KEYWORDS = ("out of memory", "cuda", "oom")


def _is_oom_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like an out-of-memory / GPU error."""
    if isinstance(exc, (RuntimeError, MemoryError)):
        return True
    msg = str(exc).lower()
    return any(kw in msg for kw in _OOM_KEYWORDS)


class LlamaCppEngine(BaseEngine):
    """llama.cpp (GGUF) implementation of BaseEngine.

    Loads a GGUF model via ``llama-cpp-python``.  GPU offload layers can be
    configured; if an OOM-like error occurs during loading the engine retries
    with progressively fewer GPU layers before giving up.

    ``llama-cpp-python`` must be installed separately:
        pip install llama-cpp-python
    """

    def __init__(
        self,
        model_path: str,
        n_gpu_layers: int = 0,
        context_size: int = 2048,
        max_new_tokens: int = 256,
    ) -> None:
        # Deferred import — raises a descriptive ImportError if not installed.
        try:
            from llama_cpp import Llama  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppEngine but is not installed. "
                "Install it with:\n\n"
                "    pip install llama-cpp-python\n\n"
                "For CUDA/GPU support see: "
                "https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration"
            ) from exc

        self._max_new_tokens = max_new_tokens
        self._Llama = Llama  # kept for potential future re-loads

        # Build the list of layer counts to attempt, skipping values >= the
        # requested amount (they would be at least as likely to OOM).
        candidates = [n_gpu_layers] + [
            v for v in _GPU_FALLBACK_LADDER if v < n_gpu_layers
        ]

        last_exc: Exception | None = None
        for attempt, layers in enumerate(candidates):
            try:
                if attempt > 0:
                    logger.warning(
                        "LlamaCppEngine: retrying model load with n_gpu_layers=%d "
                        "(previous attempt with %d failed with OOM-like error)",
                        layers,
                        candidates[attempt - 1],
                    )
                self._llm = Llama(
                    model_path=model_path,
                    n_gpu_layers=layers,
                    n_ctx=context_size,
                    verbose=False,
                )
                logger.info(
                    "LlamaCppEngine: model loaded from '%s' with n_gpu_layers=%d",
                    model_path,
                    layers,
                )
                return  # success — stop trying
            except Exception as exc:
                if _is_oom_error(exc):
                    last_exc = exc
                    # Continue down the fallback ladder.
                else:
                    # Non-OOM error (e.g. bad path) — fail immediately.
                    raise

        # All fallback levels exhausted.
        raise RuntimeError(
            f"LlamaCppEngine: failed to load '{model_path}' at all fallback "
            f"GPU-layer levels {candidates}. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # BaseEngine contract
    # ------------------------------------------------------------------

    def get_stream(self, prompt: str) -> Iterator[str]:
        """Stream the assistant reply for *prompt*.

        Calls ``create_chat_completion_openai_v1`` in streaming mode and
        yields delta content strings.  Yields ``""`` as a sentinel if the
        model produces no content at all.
        """
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        yielded = False
        stream = self._llm.create_chat_completion_openai_v1(
            messages=messages,
            stream=True,
            max_tokens=self._max_new_tokens,
        )
        for chunk in stream:
            # chunk is an openai-compat ChatCompletionChunk object.
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yielded = True
                yield content

        if not yielded:
            yield ""

    def reset(self) -> None:
        """No-op: llama.cpp has no server-side conversation state."""
        pass

    def warmup(self) -> None:
        """Pre-load the model into VRAM by consuming one token."""
        for _ in self.get_stream("Hi"):
            break
