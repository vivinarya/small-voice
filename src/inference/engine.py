import logging
from typing import Any, Iterator

from .base import BaseEngine, SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Backward-compat alias — llama_cpp_engine and ollama_engine import this
_SYSTEM_PROMPT = SYSTEM_PROMPT


class LiteRTEngine(BaseEngine):
    """LiteRT-LM (CPU) implementation of BaseEngine.

    Wraps litert_lm.Engine and manages a multi-turn conversation, applying the
    system prompt internally so callers stay backend-neutral.
    litert_lm is imported lazily so this file loads without error on Jetson
    (where litert_lm is not installed and the ollama backend is used instead).
    """

    def __init__(self, model_path: str = "assets/gemma-4-E4B-it.litertlm"):
        try:
            import litert_lm as _litert_lm  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "litert-lm is required for LiteRTEngine but is not installed.\n"
                "On the Jetson Orin, use backend: ollama in config.yaml instead."
            ) from exc
        self.model_path = model_path
        self._engine = _litert_lm.Engine(model_path, backend=_litert_lm.Backend.CPU)
        self.conversation = self._engine.create_conversation()
        self._warmup_done: bool = False
        self._apply_system_prompt()

    def _apply_system_prompt(self) -> None:
        """Inject the system prompt as the first turn of the conversation."""
        try:
            system_msg = {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            }
            self.conversation.send_message(system_msg)
        except Exception:
            pass

    @staticmethod
    def _extract_text(response: Any) -> str:
        contents = response.get("content", [])
        if isinstance(contents, list):
            texts = [
                c.get("text", "")
                for c in contents
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            return " ".join(texts).strip()
        return str(contents).strip()

    def get_stream(self, prompt: str) -> Iterator[str]:
        message = {
            "role": "user",
            "content": [{"type": "text", "text": prompt}],
        }
        if hasattr(self.conversation, "send_message_stream"):
            yielded = False
            for chunk in self.conversation.send_message_stream(message):
                text_chunk = self._extract_text(chunk)
                if text_chunk:
                    yielded = True
                    yield text_chunk
            if not yielded:
                yield ""
        else:
            response = self.conversation.send_message(message)
            yield self._extract_text(response)

    def reset(self) -> None:
        """Clear multi-turn conversation state and re-apply the system prompt."""
        self.conversation = self._engine.create_conversation()
        self._apply_system_prompt()

    def warmup(self) -> None:
        try:
            for _ in self.get_stream("Hi"):
                break
            self.reset()
            self._warmup_done = True
            logger.info("LiteRTEngine.warmup() completed successfully.")
        except Exception as exc:
            logger.warning("LiteRTEngine.warmup() failed: %s", exc)

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done


# Backward-compatibility alias
GemmaEngine = LiteRTEngine
