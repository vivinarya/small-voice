# src/retrieval/query_router.py
"""Lightweight intent routing for special, non-semantic queries.

These helpers let the assistant answer meta questions about its knowledge base
deterministically (instead of relying on similarity search), and support exact
page lookup:

  * "What books do you have?"            -> is_book_list_query()
  * "Do you have the maths textbook?"    -> extract_book_name_query()
  * "Read page 65" / "page sixty five"   -> parse_page_number()

All functions are pure and offline (only `re` is used), so they are trivially
unit-testable and safe for the offline pipeline.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Number-word parsing (handles page numbers spoken as words) ──────────────
_UNITS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
_TEENS = {
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}


def _normalize(text: str) -> str:
    """Lowercase and strip punctuation to bare words + digits."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower()).strip()


def words_to_int(tokens: list[str]) -> Optional[int]:
    """Convert a leading run of English number-words to an int (1..999).

    Stops at the first token that is not a number word. Returns None if no
    leading number word is present. Examples:
        ["sixty", "five"]            -> 65
        ["one", "hundred", "twenty"] -> 120
        ["of", "the"]                -> None
    """
    total = 0
    current = 0
    found = False
    for w in tokens:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TEENS:
            current += _TEENS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        else:
            break
        found = True
    return (total + current) if found else None


def parse_page_number(text: str) -> Optional[int]:
    """Extract a page number from a query, supporting digits and number-words.

    Recognizes "page 65", "page number 65", "page no 65", "page sixty five".
    Returns None when no page reference is present.
    """
    t = _normalize(text)
    if "page" not in t:
        return None
    # Digit form: page [number|no] 65
    m = re.search(r"page\s+(?:number\s+|no\s+)?(\d{1,4})", t)
    if m:
        return int(m.group(1))
    # Word form: page [number] sixty five
    m = re.search(r"page\s+(?:number\s+|no\s+)?([a-z ]+)", t)
    if m:
        n = words_to_int(m.group(1).split())
        if n:
            return n
    return None


# ── Book-list intent ────────────────────────────────────────────────────────
_BOOK_LIST_PATTERNS = [
    r"what (?:books|textbooks|documents|files|pdfs)",
    r"which (?:books|textbooks|documents|files|pdfs)",
    r"list (?:the |all )?(?:books|textbooks|documents|files|pdfs)",
    r"what (?:do you have|have you got|is)(?: in your)?(?: knowledge base| library)",
    r"what have i uploaded",
    r"what.* uploaded so far",
    r"do you have any (?:books|textbooks|documents|files)",
]


def is_book_list_query(text: str) -> bool:
    """True if the user is asking what's in the knowledge base as a whole."""
    t = _normalize(text)
    return any(re.search(p, t) for p in _BOOK_LIST_PATTERNS)


# ── Specific-book intent ─────────────────────────────────────────────────────
# Words to strip from an extracted candidate book name.
_STOP_WORDS = {
    "the", "a", "an", "book", "textbook", "pdf", "document", "doc",
    "uploaded", "upload", "just", "my", "that", "this", "i", "of",
    "ncrt", "ncert",  # OCR/STT often mangles these; keep matching loose
    "it", "them", "those", "these", "any", "some", "anything", "one",
    "to", "you", "your", "have", "access", "now", "here",
}

_BOOK_QUERY_PATTERNS = [
    r"do you have (?:access to )?(.+)",
    r"have you got (.+)",
    r"can you access (.+)",
]


def extract_book_name_query(text: str) -> Optional[str]:
    """Extract a candidate book name when the user asks about a specific book.

    Returns the cleaned candidate (stop-words removed), or None if the query
    is not a "do you have <X>" style question. Returns None for book-list
    queries so the caller can route those separately.
    """
    if is_book_list_query(text):
        return None
    t = _normalize(text)
    for pat in _BOOK_QUERY_PATTERNS:
        m = re.search(pat, t)
        if not m:
            continue
        candidate = m.group(1)
        tokens = [w for w in candidate.split() if w not in _STOP_WORDS]
        # Require at least one meaningful token to avoid matching bare
        # "do you have it?" style questions.
        if tokens:
            return " ".join(tokens)
    return None


# ── Answer formatting (voice-friendly, deterministic) ───────────────────────
def format_book_list(sources: list[dict]) -> str:
    """Render a spoken answer listing the indexed documents.

    `sources` is a list of {"source", "page_count", "chunk_count"} dicts.
    """
    if not sources:
        return (
            "I don't have any documents in my knowledge base yet. "
            "Upload a PDF in the Textbooks tab and I'll index it for you."
        )
    names = [s["source"] for s in sources]
    if len(names) == 1:
        s = sources[0]
        return f"I have one document: {s['source']}, with {s['page_count']} pages indexed."
    listing = ", ".join(names[:-1]) + " and " + names[-1]
    return f"I have {len(names)} documents: {listing}."


def _match_score(query_name: str, source: str) -> int:
    """Count how many query tokens appear in the source name (case-insensitive)."""
    q_tokens = [w for w in _normalize(query_name).split() if w not in _STOP_WORDS]
    s_norm = _normalize(source)
    return sum(1 for w in q_tokens if w in s_norm)


def book_query_has_match(query_name: str, sources: list[dict]) -> bool:
    """True if a specific-book query positively matches an indexed document.

    Used by the caller to decide whether to short-circuit with a book-availability
    answer, or fall through to semantic retrieval (for content questions that
    happen to mention a book that isn't indexed under that exact name).
    """
    if not sources:
        return False
    return any(_match_score(query_name, s["source"]) > 0 for s in sources)


def answer_book_query(query_name: str, sources: list[dict]) -> str:
    """Render a spoken yes/no answer about a specific book."""
    if not sources:
        return (
            "I don't have any documents uploaded yet. "
            "Upload the PDF in the Textbooks tab and I'll index it."
        )
    best = None
    best_score = 0
    for s in sources:
        score = _match_score(query_name, s["source"])
        if score > best_score:
            best_score = score
            best = s
    if best and best_score > 0:
        return (
            f"Yes, I have {best['source']} with {best['page_count']} pages indexed. "
            "Ask me about its content and I'll answer with page citations."
        )
    available = ", ".join(s["source"] for s in sources)
    return (
        f"No, I don't have a document matching that. "
        f"I currently have: {available}."
    )
