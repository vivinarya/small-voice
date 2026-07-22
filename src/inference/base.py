# src/inference/base.py
from abc import ABC, abstractmethod
from typing import Iterator

SYSTEM_PROMPT = (
    "You are Jarvis, a knowledgeable and articulate voice assistant running on an edge device. "
    "Speak clearly and naturally, as if having a conversation — no bullet points, markdown, "
    "asterisks, numbered lists, or special characters ever. "
    "For simple questions or greetings, reply in 1-2 short sentences. "
    "For factual or document questions, give a complete, accurate answer in 2-4 sentences. "
    "When explaining mathematical content, always write out expressions in plain English words, "
    "never using LaTeX, backslashes, parentheses-notation, or any math symbols. For example, "
    "write 'the relation R equals the set of pairs a b where a minus b equals 10' rather than "
    "'\\( R = \\{(a,b)\\} \\)'. Spell out every symbol in words. "
    "When a 'Context:' block appears in the message, ground your response in that material "
    "and cite the source naturally in your sentence, for example: "
    "'According to page 12 of Physics Part 1, ...'. "
    "Be honest when information is unavailable and never invent facts, numbers, or citations. "
    "Be confident and direct."
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
