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

    - Converts LaTeX math expressions, sets, symbols and markdown to readable English
    - Expands common abbreviations (Dr., Mr., e.g., etc.)
    - Converts ordinals (1st → first) and plain numbers (42 → forty two)
    - Does NOT strip sentence-terminal punctuation
    """
    # 1. LaTeX display and inline math delimiters removal
    text = text.replace(r"\[", " ").replace(r"\]", " ")
    text = text.replace(r"\(", " ").replace(r"\)", " ")

    # 2. LaTeX fractions: \frac{a}{b} -> a over b
    # Loop to support nested fractions if any
    for _ in range(3):
        text = re.sub(r'\\frac\s*\{([^}]+)\}\s*\{([^}]+)\}', r' \1 over \2 ', text)

    # 3. LaTeX square roots: \sqrt{x} -> the square root of x
    text = re.sub(r'\\sqrt\s*\{([^}]+)\}', r' the square root of \1 ', text)

    # 4. LaTeX set braces: \{ and \}
    text = text.replace(r"\{", " ").replace(r"\}", " ")

    # 5. LaTeX standard relational and set operations
    latex_replacements = [
        (r'\\geq\b', " greater than or equal to "),
        (r'\\geq', " greater than or equal to "),
        (r'\\le\b', " less than or equal to "),
        (r'\\le', " less than or equal to "),
        (r'\\leq\b', " less than or equal to "),
        (r'\\leq', " less than or equal to "),
        (r'\\neq\b', " not equal to "),
        (r'\\neq', " not equal to "),
        (r'\\approx\b', " approximately "),
        (r'\\approx', " approximately "),
        (r'\\times\b', " times "),
        (r'\\times', " times "),
        (r'\\div\b', " divided by "),
        (r'\\div', " divided by "),
        (r'\\cdot\b', " times "),
        (r'\\cdot', " times "),
        (r'\\in\b', " in "),
        (r'\\in', " in "),
        (r'\\subset\b', " is a subset of "),
        (r'\\subset', " is a subset of "),
        (r'\\cup\b', " union "),
        (r'\\cup', " union "),
        (r'\\cap\b', " intersection "),
        (r'\\cap', " intersection "),
        (r'\\infty\b', " infinity "),
        (r'\\infty', " infinity "),
        (r'\\sum\b', " sum "),
        (r'\\sum', " sum "),
    ]
    for pattern, rep in latex_replacements:
        text = re.sub(pattern, rep, text)

    # 6. LaTeX exponents and subscripts: x^2 -> x squared, x_i -> x subscript i
    text = re.sub(r'\^2\b', " squared ", text)
    text = re.sub(r'\^3\b', " cubed ", text)
    text = re.sub(r'\^\{?([^}]+)\}?', r' to the power of \1 ', text)
    text = re.sub(r'_\{?([^}]+)\}?', r' subscript \1 ', text)

    # 7. Math symbols and operators translation (only in math context/numbers)
    # Plus, minus, times, divided by, equals, greater than, less than, absolute value
    text = re.sub(r'(?<=\w|\s)\+(?=\w|\s)', " plus ", text)
    text = re.sub(r'(?<=\w|\s)-(?=\w|\s)', " minus ", text)
    text = re.sub(r'(?<=\w|\s)\*(?=\w|\s)', " times ", text)
    text = re.sub(r'(?<=\w|\s)/(?=\w|\s)', " divided by ", text)
    text = re.sub(r'(?<=\w|\s)=(?=\w|\s)', " equals ", text)
    text = re.sub(r'(?<=\w|\s)>(?=\w|\s)', " greater than ", text)
    text = re.sub(r'(?<=\w|\s)<(?=\w|\s)', " less than ", text)
    # Absolute value: |x| -> absolute value of x
    text = re.sub(r'\|([^|]+)\|', r' absolute value of \1 ', text)

    # 8. Clean up brackets/punctuation that TTS shouldn't pronounce literally
    text = text.replace("{", " ").replace("}", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    text = text.replace(":", " ")

    # 9. Markdown clean up
    text = re.sub(r'\*\*|__', "", text)
    text = re.sub(r'\*|_', "", text)
    text = re.sub(r'`', "", text)
    text = re.sub(r'^\s*#+\s+', "", text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[-*+]\s+', "", text, flags=re.MULTILINE)

    # 10. Strip remaining/leftover backslashes and LaTeX commands
    text = re.sub(r'\\[a-zA-Z]+', "", text)
    text = text.replace("\\", "")

    # Expand abbreviations (before number expansion)
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

    # 11. Normalize whitespaces
    text = re.sub(r'\s+', " ", text).strip()

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
