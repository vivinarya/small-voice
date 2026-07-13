# src/synthesis/text_norm.py
"""Text normalization and sentence segmentation for TTS.

Sentence segmentation splits ONLY on terminal punctuation (. ? !)
and guards common abbreviations to prevent false breaks.
"""
import re
from typing import Generator

# Abbreviations that use a period but are NOT sentence boundaries.
# We look for these immediately before the period.
_ABBREV_PATTERN = re.compile(
    r'\b(?:Dr|Mr|Mrs|Ms|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|No|Vol|pp|Fig|cf|approx|dept|approx)\.'
    r'|(?<!\w)\d+\.',   # "1." in isolation inside a list — guard against "item 1. Next item"
    re.IGNORECASE,
)

# Terminal sentence boundary: . ? ! followed by whitespace (or end), but NOT an abbreviation.
_SENT_BOUNDARY_RE = re.compile(r'([.?!])(\s+|$)')

# Number words for 0-19
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

# Ordinal suffix patterns
_ORDINAL_RE = re.compile(r'\b(\d+)(st|nd|rd|th)\b', re.IGNORECASE)
# Plain integer pattern
_NUMBER_RE = re.compile(r'\b(\d{1,4})\b')

# Abbreviation expansions (applied before TTS)
_ABBREV_EXPAND = [
    (re.compile(r'\bDr\.', re.IGNORECASE), "Doctor"),
    (re.compile(r'\bMr\.', re.IGNORECASE), "Mister"),
    (re.compile(r'\bMrs\.', re.IGNORECASE), "Missus"),
    (re.compile(r'\bMs\.', re.IGNORECASE), "Miss"),
    (re.compile(r'\bProf\.', re.IGNORECASE), "Professor"),
    (re.compile(r'\be\.g\.', re.IGNORECASE), "for example"),
    (re.compile(r'\bi\.e\.', re.IGNORECASE), "that is"),
    (re.compile(r'\bvs\.', re.IGNORECASE), "versus"),
    (re.compile(r'\betc\.', re.IGNORECASE), "and so on"),
    (re.compile(r'\bNo\.', re.IGNORECASE), "number"),
]


def _int_to_words(n: int) -> str:
    """Convert integer 0-9999 to English words."""
    if n < 0:
        return "negative " + _int_to_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (" " + _ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (" " + _int_to_words(rest) if rest else "")
    # 1000-9999
    thousands, rest = divmod(n, 1000)
    return _ONES[thousands] + " thousand" + (" " + _int_to_words(rest) if rest else "")


def _ordinal_to_words(n: int) -> str:
    """Convert integer to ordinal English (1st → first, etc.)."""
    ordinals = {
        1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
        6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
        11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
        15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
        19: "nineteenth", 20: "twentieth",
    }
    if n in ordinals:
        return ordinals[n]
    return _int_to_words(n) + "th"


def normalize_for_tts(text: str) -> str:
    """Normalize text for natural TTS output.

    - Expands common abbreviations (Dr., Mr., e.g., etc.)
    - Converts ordinals (1st → first) and plain numbers (42 → forty two)
    - Does NOT strip sentence-terminal punctuation
    """
    # Expand abbreviations first (before number expansion strips periods)
    for pattern, replacement in _ABBREV_EXPAND:
        text = pattern.sub(replacement, text)

    # Expand ordinals: 1st, 2nd, 3rd, 4th, ...
    def replace_ordinal(m):
        return _ordinal_to_words(int(m.group(1)))
    text = _ORDINAL_RE.sub(replace_ordinal, text)

    # Expand plain integers up to 4 digits
    def replace_number(m):
        n = int(m.group(1))
        if 0 <= n <= 9999:
            return _int_to_words(n)
        return m.group(0)  # leave large numbers unchanged
    text = _NUMBER_RE.sub(replace_number, text)

    return text


def _is_abbreviation_boundary(text: str, match_start: int) -> bool:
    """Return True if the period at match_start is part of a known abbreviation."""
    # Check if any abbreviation pattern ends at match_start+1 (the period position)
    before = text[:match_start + 1]  # up to and including the dot
    return bool(_ABBREV_PATTERN.search(before) and
                _ABBREV_PATTERN.search(before).end() == len(before))


def extract_complete_sentences(buf: str) -> list[tuple[str, str]]:
    """Extract one complete sentence from the buffer.

    Returns a list of at most one (sentence, remainder) tuple.
    Returns [] if no complete sentence boundary found yet.

    Splits ONLY on . ? ! followed by whitespace or end-of-string,
    guarding known abbreviations.
    """
    results = []
    pos = 0
    while pos < len(buf):
        m = _SENT_BOUNDARY_RE.search(buf, pos)
        if not m:
            break

        punct_pos = m.start()
        punct = m.group(1)

        # For periods only: check if it's an abbreviation
        if punct == '.' and _ABBREV_PATTERN.search(buf[:punct_pos + 1]):
            abbrev_m = _ABBREV_PATTERN.search(buf[:punct_pos + 1])
            if abbrev_m and abbrev_m.end() == punct_pos + 1:
                pos = m.end()
                continue

        sentence = buf[:m.end()].rstrip()
        remainder = buf[m.end():]
        if sentence.strip():
            results.append((sentence, remainder))
        return results  # return after first complete sentence

    return results
