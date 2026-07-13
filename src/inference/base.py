# src/inference/base.py
from abc import ABC, abstractmethod
from typing import Iterator

SYSTEM_PROMPT = (
    "You are Jarvis, a fast, helpful voice assistant running on an edge device. "
    "Always reply in 1-3 short sentences. Never use bullet points, markdown, "
    "asterisks, or lists. Speak naturally as if in conversation. "
    "When the user's message includes a 'Context:' block, answer only from that context "
    "and cite the document name and page number. "
    "If the context does not contain the answer, say you do not have that information. "
    "Do not invent facts."
)

class BaseEngine(ABC):
    """Streaming LLM contract. Implementations must be import-safe even if their
    backend library is absent (defer heavy imports to __init__)."""

    @abstractmethod
    def get_stream(self, prompt: str) -> Iterator[str]:
        """Yield response text chunks for a fully-assembled, engine-neutral prompt.
        MUST yield at least once (possibly an empty-string sentinel) and MUST NOT
        raise mid-stream for normal completion."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear multi-turn conversation state (new session)."""
        ...

    def warmup(self) -> None:
        """Optional: decode 1 token to pay one-time init cost before first user turn."""
        return None
