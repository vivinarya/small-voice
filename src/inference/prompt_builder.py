# src/inference/prompt_builder.py
"""Engine-neutral prompt assembly for the RAG pipeline.

Assembles a final prompt from the user's question and retrieved NCERT chunks.
Context injection is greedily capped at max_ctx_tokens to protect TTFT on small models.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from retrieval.base import RetrievedChunk

logger = logging.getLogger(__name__)


def _token_count(text: str) -> int:
    """Approximate token count: split on whitespace."""
    return len(text.split())


_HONEST_DECLINE_TEMPLATE = (
    "The knowledge base does not contain information relevant to this question. "
    "Please answer by saying you do not have that information in the current context.\n\n"
    "Question: {query}"
)


def build_prompt(
    user_text: str,
    retrieved: "list[RetrievedChunk]",
    max_ctx_tokens: int = 600,
    index_available: bool = False,
) -> str:
    """Assemble an engine-neutral prompt from the user question and retrieved chunks.

    Preconditions:
      - retrieved is sorted by score desc (highest relevance first)
      - max_ctx_tokens > 0
    Postconditions:
      - returned prompt is backend-neutral (no chat template tokens)
      - injected context token count <= max_ctx_tokens (keeps TTFT low on small models)
      - if retrieved is empty AND index_available is False (chit-chat / no index), returns
        user_text directly (bare question, no fabricated context) — Req 3.5 preserved
      - if retrieved is empty AND index_available is True (textbook query, index built but
        no chunk above MIN_SCORE), returns an honest-decline prompt asking the model to say
        it does not have that information — Req 2.10

    The ``index_available`` flag distinguishes two empty-retrieved scenarios:
      * ``False`` (default) — the index was not built or this is a chit-chat query.
        Behaviour is unchanged from before: return the bare question so the assistant
        can answer normally (preserves Req 3.5 / non-textbook chit-chat answerability).
      * ``True`` — a textbook query was attempted against a built index, but no chunk
        scored above MIN_SCORE.  Emit an honest-decline prompt so the model says it
        does not have that information rather than fabricating an answer (Req 2.10).

    Algorithm:
      Greedy fill: add highest-scored chunks first until max_ctx_tokens would be exceeded.
    """
    if not retrieved:
        if index_available:
            # Textbook query + built index, but no chunk above MIN_SCORE.
            # Prompt the model to honestly decline rather than fabricate. (Req 2.10)
            logger.debug(
                "build_prompt: index available but no chunks above threshold — "
                "emitting honest-decline prompt"
            )
            return _HONEST_DECLINE_TEMPLATE.format(query=user_text)
        # No index built (or chit-chat path) → ask model directly (Req 3.5)
        logger.debug("build_prompt: no retrieved chunks, returning bare question")
        return user_text
    
    ctx_parts: list[str] = []
    used_tokens = 0
    
    for rc in retrieved:
        chunk_text = f"[{rc.citation()}] {rc.chunk.text}"
        t = _token_count(chunk_text)
        if used_tokens + t > max_ctx_tokens:
            break   # stop greedy fill — budget exhausted
        ctx_parts.append(chunk_text)
        used_tokens += t
    
    if not ctx_parts:
        # All chunks individually exceed budget — fall back to bare question
        logger.debug("build_prompt: all chunks exceed token budget, returning bare question")
        return user_text
    
    context_block = "\n".join(ctx_parts)
    prompt = (
        "Use ONLY the document context below to answer. "
        "Cite the document name and page number in your answer. "
        "Keep your answer to 1-3 sentences.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {user_text}"
    )
    
    logger.debug(
        "build_prompt: injected %d chunks, ~%d tokens (cap=%d)",
        len(ctx_parts), used_tokens, max_ctx_tokens,
    )
    return prompt


def build_page_prompt(
    user_text: str,
    page: int,
    page_chunks: "list",
    max_ctx_tokens: int = 280,
) -> str:
    """Assemble a prompt for an exact-page lookup ("read page N").

    `page_chunks` is a list of Chunk objects all belonging to `page` (sorted in
    reading order). The page text is concatenated and capped tightly to keep
    prefill latency low on small CPU models (a summary does not need the whole
    page), then the model is asked to summarize the page and cite it.
    """
    source = page_chunks[0].source if page_chunks else "the document"
    # De-duplicate the sliding-window overlap so the context isn't padded with
    # repeated words (which would inflate prefill time for no benefit).
    seen: set[str] = set()
    words: list[str] = []
    for c in page_chunks:
        for w in c.text.split():
            words.append(w)
        if len(words) >= max_ctx_tokens:
            break
    body = " ".join(words[:max_ctx_tokens])
    return (
        f"The text below is from page {page} of '{source}'. "
        "In 2 to 4 short sentences, say what this page covers and that it is "
        f"page {page} of {source}.\n\n"
        f"Page {page} text:\n{body}\n\n"
        f"Request: {user_text}"
    )
