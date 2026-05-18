import base64
import io
import wave
import litert_lm

# System prompt: keeps responses short so fewer tokens are decoded → faster.
_SYSTEM_PROMPT = (
    "You are Jarvis, a fast, helpful voice assistant running on an edge device. "
    "Always reply in 1-3 short sentences. Never use bullet points, markdown, "
    "asterisks, or lists. Speak naturally as if in conversation."
)


class GemmaEngine:
    def __init__(self, model_path: str = "assets/gemma-4-e2b-it.litertlm"):
        self.engine = litert_lm.Engine(
            model_path,
            backend=litert_lm.Backend.CPU,
        )
        # Persist conversation for multi-turn context.
        # Inject system prompt as the first turn so every reply is brief.
        self.conversation = self.engine.create_conversation()
        try:
            system_msg = {
                "role": "system",
                "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
            }
            self.conversation.send_message(system_msg)
        except Exception:
            # Some LiteRT-LM builds don't support 'system' role — silently skip.
            pass

    import typing
    def get_stream(
        self,
        audio_data: bytes | None = None,
        text_data: str | None = None,
    ) -> typing.Iterator[str]:
        """Send a user turn and return the assistant's text response."""

        if text_data:
            message = {
                "role": "user",
                "content": [{"type": "text", "text": text_data}],
            }
            if hasattr(self.conversation, "send_message_stream"):
                for chunk in self.conversation.send_message_stream(message):
                    text_chunk = self._extract_text(chunk)
                    if text_chunk:
                        yield text_chunk
            else:
                response = self.conversation.send_message(message)
                yield self._extract_text(response)

    from typing import Any
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
