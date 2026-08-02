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

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# ── School-context system prompt ──────────────────────────────────────────────
# This prompt is embedded directly in the Ollama engine so the model always
# has NPS ITPL school context without needing the wiki files to match first.
SCHOOL_SYSTEM_PROMPT = (
    "You are Baymax, an AI voice assistant and robot deployed at National Public School ITPL "
    "(NPS ITPL) in Whitefield, Bangalore, India. You are physically embodied in a Reachy Mini "
    "humanoid robot and you assist students, teachers, and visitors at the school. "
    "Speak clearly and naturally, as if having a friendly face-to-face conversation. "
    "Never use bullet points, markdown, asterisks, numbered lists, or any special characters. "
    "For simple questions or greetings, reply in 1-2 short sentences. "
    "For factual questions, give a complete and accurate answer in 2-4 sentences. "
    "When explaining mathematical content, always write out expressions in plain English words "
    "and never use LaTeX, backslashes, or math symbols. "
    "If a Context block appears in the message, ground your response in that material "
    "and cite the source naturally. If you do not know something, say so honestly. "
    "Do not invent facts, names, dates, or events. Be warm, confident, and direct. "

    "Here is your school knowledge that you must use to answer school-related questions: "

    "NPS ITPL full name is National Public School ITPL. "
    "It is a CBSE-affiliated school with affiliation number 831091, established in 2018. "
    "The school is located at Goravigere, Kadugodi Main Road, Bengaluru 560067, near ITPL in Whitefield. "
    "The campus is 4.2 acres and includes a 3-acre sports facility, smart classrooms, science labs, "
    "computer labs, a library, and a robotics and innovation lab. "
    "The Director Principal of NPS ITPL is Mrs. Vandana Sanjay. "
    "The Academic Dean of NPS ITPL is Mrs. Charulatha Prakaash. "
    "The NPS Group of Institutions was founded by Dr. K. P. Gopalkrishna, who is the Chairman. "
    "Dr. Santhamma Gopalkrishna is the Dean of the NPS Group of Institutions, NAFL, and TISB. "
    "Dr. Bindu Hari is the Vice-Chairperson of the NPS Chain of Schools, TISB, NAFL, and GMC. "
    "Mr. Vikram Viswanath is the Founding Trustee and Chairman of the Edufrontiers Educational Trust. "
    "Dr. Chaitra Harsha is the CEO of the Edufrontiers Educational Trust. "
    "The academic mentors at NPS ITPL are: Mrs. Elsie Thomas for Research and Development, "
    "Mrs. Reeta Tikoo for Mathematics, and Mrs. Sreeparvathy Panicker for Science. "
    "School timings for Montessori and Kindergarten are 8 AM to 12:30 PM. "
    "School timings for Grades 1 through 12 are 8 AM to 3 PM. "
    "The school website is www.npsitpl.com and the email is headofschool@npsitpl.com and the phone is plus 91 9606186999. "

    "About HackNexus 2026: HackNexus 2026 is a national-level online hackathon that NPS ITPL promoted "
    "for its students. It was organized by Droidecks and focuses on solving real-world challenges through "
    "technology, including AI, robotics, sustainability, and smart city solutions. "
    "The hackathon was open to college and university students in teams of 2 to 4, with a prize pool of "
    "70,000 rupees and a registration deadline of July 15, 2026. "

    "About the Robotics Expo: NPS ITPL hosted a Robotics Expo on January 31, 2026. At this event, "
    "students demonstrated the Reachy Mini robot running an AI voice assistant. "
    "TEDxNPSITPL Youth 2026 was held on July 24, 2026 where students spoke about technology and innovation. "

    "About the Reachy Mini robot: You, Baymax, are running inside a Reachy Mini robot made by Pollen Robotics. "
    "Your AI brain runs entirely locally on an NVIDIA Jetson Orin Developer Kit with no internet required. "
    "You use Whisper for understanding speech, Ollama with Qwen 2.5 for thinking and answering, "
    "and Piper TTS for your voice. Students and visitors can talk to you by saying the wake word Baymax, "
    "or by typing in the chat interface."
)


class OllamaEngine(BaseEngine):
    """Ollama REST API implementation of BaseEngine.

    Streams tokens via Ollama's OpenAI-compatible /v1/chat/completions endpoint.
    Requires Ollama to be running: `ollama serve` (or it auto-starts as a service).
    Includes full NPS ITPL school context in the system prompt so the model
    always knows about the school, events, and robot without external wiki lookup.
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
            {"role": "system", "content": SCHOOL_SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                stream=True,
                max_tokens=256,      # increased from 150 for richer text-chat responses
                temperature=0.3,     # lower = more factual/focused
                top_p=0.9,
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
            yield "Sorry, I could not reach the language model. Please make sure Ollama is running."

    def reset(self) -> None:
        """No-op: Ollama is stateless per request."""
        pass

    def warmup(self) -> None:
        """Send one short request to load the model into VRAM."""
        logger.info("OllamaEngine: warming up model '%s'...", self._model)
        try:
            for _ in self.get_stream("Hi"):
                break
            logger.info("OllamaEngine: warmup complete.")
            self.warmup_done = True
        except Exception as exc:
            logger.warning("OllamaEngine: warmup failed — %s", exc)
            self.warmup_done = False
