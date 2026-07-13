import logging
from typing import Any, Iterator

import litert_lm

from .base import BaseEngine

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, helpful voice assistant running on an edge device. "
    "Always reply in 1-3 short sentences. Never use bullet points, markdown, "
    "asterisks, or lists. Speak naturally as if in conversation. "
    "When the user's message includes a 'Context:' block, answer only from that context "
    "and cite the document name and page number. "
    "If the context does not contain the answer, say you do not have that information. "
    "Do not invent facts."
)


class LiteRTEngine(BaseEngine):
    """LiteRT-LM (CPU) implementation of BaseEngine.

    Wraps litert_lm.Engine and manages a multi-turn conversation, applying the
    system prompt internally so callers stay backend-neutral.
    """

    def __init__(self, model_path: str = "assets/gemma-4-E4B-it.litertlm"):
        self.model_path = model_path
        self._engine = litert_lm.Engine(model_path, backend=litert_lm.Backend.CPU)
        self.conversation = self._engine.create_conversation()
        self._warmup_done: bool = False
        self._apply_system_prompt()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_system_prompt(self) -> None:
        """Inject the system prompt as the first turn of the conversation."""
        try:
            system_msg = {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            }
            self.conversation.send_message(system_msg)
        except Exception:
            # Some LiteRT-LM builds don't support the 'system' role — silently skip.
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

    # ------------------------------------------------------------------
    # BaseEngine contract
    # ------------------------------------------------------------------

    def get_stream(self, prompt: str) -> Iterator[str]:
        """Send a user turn and stream the assistant's text response.

        Yields at least one chunk (possibly empty) and does not raise
        mid-stream for normal completion.
        """
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
        """Decode one short token to pay the one-time init cost before the first user turn.

        Sets ``_warmup_done`` to ``True`` on success and logs a warning on failure
        so callers can detect whether the one-time init cost was paid.  The
        ``reset()`` after the decode clears the "Hi" exchange from conversation
        history while keeping the engine's internal graph/weights warm (LiteRT
        JIT-compiles the graph on first decode; subsequent decodes reuse it even
        after a conversation reset).
        """
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
        """True if warmup() completed without raising an exception."""
        return self._warmup_done


# ---------------------------------------------------------------------------
# Backward-compatibility alias — keeps existing callers working during
# the transition.  Remove once main.py is updated to use LiteRTEngine.
# ---------------------------------------------------------------------------
GemmaEngine = LiteRTEngine
