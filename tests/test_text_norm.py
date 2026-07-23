# tests/test_text_norm.py
"""Unit tests for src/synthesis/text_norm.py

Covers the correctness properties from the design spec:
  Property 3: No clause splitting (commas don't cause splits)
  Property 4: TTS reconstruction (sentences cover full text)
  Text normalization: numbers, ordinals, abbreviations
"""
import sys
import os

# Add src to path so imports work without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from synthesis.text_norm import (
    normalize_for_tts,
    extract_complete_sentences,
)


class TestExtractCompleteSentences:
    """Tests for sentence boundary detection."""

    def test_single_sentence_with_period(self):
        """A single sentence followed by a period produces one result."""
        buf = "The cat sat on the mat. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1
        sentence, remainder = results[0]
        assert sentence.strip() == "The cat sat on the mat."
        assert remainder.strip() == ""

    def test_comma_does_not_split(self):
        """Commas inside a sentence do NOT create a boundary (Property 3)."""
        buf = "Hello, world, this is a test. "
        results = extract_complete_sentences(buf)
        # Should produce exactly ONE sentence, not split on commas
        assert len(results) == 1
        sentence, _ = results[0]
        assert "Hello, world, this is a test" in sentence

    def test_semicolon_does_not_split(self):
        """Semicolons do NOT create a boundary."""
        buf = "First clause; second clause. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1
        sentence, _ = results[0]
        assert "First clause; second clause" in sentence

    def test_colon_does_not_split(self):
        """Colons do NOT create a boundary."""
        buf = "Important: remember this. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1

    def test_question_mark_boundary(self):
        """Question marks create boundaries."""
        buf = "Are you there? "
        results = extract_complete_sentences(buf)
        assert len(results) == 1
        sentence, _ = results[0]
        assert sentence.strip() == "Are you there?"

    def test_exclamation_mark_boundary(self):
        """Exclamation marks create boundaries."""
        buf = "Watch out! "
        results = extract_complete_sentences(buf)
        assert len(results) == 1

    def test_incomplete_buffer_no_boundary(self):
        """A buffer without terminal punctuation returns no results."""
        buf = "This is incomplete"
        results = extract_complete_sentences(buf)
        assert results == []

    def test_dr_abbreviation_not_split(self):
        """'Dr.' does NOT trigger a sentence boundary."""
        buf = "Dr. Smith went to the store. "
        results = extract_complete_sentences(buf)
        # Should produce one sentence with Dr. intact
        assert len(results) == 1
        sentence, _ = results[0]
        assert "Dr." in sentence or "Dr" in sentence

    def test_eg_abbreviation_not_split(self):
        """'e.g.' does NOT trigger a sentence boundary."""
        buf = "Some examples, e.g. apples and oranges, are nutritious. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1

    def test_ie_abbreviation_not_split(self):
        """'i.e.' does NOT trigger a sentence boundary."""
        buf = "The smallest unit, i.e. the atom, is discussed in chapter 3. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1

    def test_first_sentence_extracted_with_remainder(self):
        """When buffer has two sentences, only the first is returned with correct remainder."""
        buf = "First sentence. Second sentence. "
        results = extract_complete_sentences(buf)
        assert len(results) == 1
        sentence, remainder = results[0]
        assert "First sentence" in sentence
        assert "Second sentence" in remainder


class TestNormalizeForTTS:
    """Tests for text normalization."""

    def test_integer_expansion(self):
        """Integers are expanded to English words."""
        assert normalize_for_tts("42") == "forty two"
        assert normalize_for_tts("7") == "seven"
        assert normalize_for_tts("100") == "one hundred"

    def test_ordinal_expansion(self):
        """Ordinals (1st, 2nd, 3rd, 4th) are expanded."""
        assert normalize_for_tts("1st") == "first"
        assert normalize_for_tts("2nd") == "second"
        assert normalize_for_tts("3rd") == "third"
        assert normalize_for_tts("4th") == "fourth"
        assert normalize_for_tts("11th") == "eleventh"
        assert normalize_for_tts("20th") == "twentieth"

    def test_dr_expansion(self):
        """'Dr.' is expanded to 'Doctor'."""
        result = normalize_for_tts("Dr. Smith")
        assert "Doctor" in result

    def test_mr_expansion(self):
        """'Mr.' is expanded to 'Mister'."""
        result = normalize_for_tts("Mr. Jones")
        assert "Mister" in result

    def test_eg_expansion(self):
        """'e.g.' is expanded to 'for example'."""
        result = normalize_for_tts("e.g. apples")
        assert "for example" in result.lower()

    def test_ie_expansion(self):
        """'i.e.' is expanded to 'that is'."""
        result = normalize_for_tts("i.e. important")
        assert "that is" in result.lower()

    def test_vs_expansion(self):
        """'vs.' is expanded to 'versus'."""
        result = normalize_for_tts("cats vs. dogs")
        assert "versus" in result.lower()

    def test_etc_expansion(self):
        """'etc.' is expanded to 'and so on'."""
        result = normalize_for_tts("cats, dogs, etc.")
        assert "and so on" in result.lower()

    def test_plain_text_unchanged(self):
        """Plain text without special patterns is returned unchanged."""
        text = "The quick brown fox"
        assert normalize_for_tts(text) == text

    def test_large_number_unchanged(self):
        """Numbers >= 10000 are left unchanged."""
        text = "12345"
        result = normalize_for_tts(text)
        assert result == text  # leave large numbers as-is

    def test_zero(self):
        assert normalize_for_tts("0") == "zero"

    def test_number_in_sentence(self):
        """Numbers embedded in a sentence are expanded correctly."""
        result = normalize_for_tts("There are 3 cats and 12 dogs.")
        assert "three" in result
        assert "twelve" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
