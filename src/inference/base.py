# src/inference/base.py
from abc import ABC, abstractmethod
from typing import Iterator

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
