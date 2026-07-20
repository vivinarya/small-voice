import os
import re

def fast_wiki_router(user_speech: str) -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wiki_dir = os.path.join(current_dir, "vault", "wiki", "entities")
    user_speech_lower = user_speech.lower()
    
    if not os.path.exists(wiki_dir):
        return ""
        
    # Check if user speech hits a compiled wiki entity
    for filename in os.listdir(wiki_dir):
        if not filename.endswith(".md"): continue
        entity_name = filename.replace(".md", "").replace("-", " ")
        if entity_name in user_speech_lower:
            print(f"[Graph Hit]: Found compiled node for '{entity_name}'")
            with open(os.path.join(wiki_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                # Strip YAML frontmatter if present
                return re.sub(r'^---.*?---\n', '', content, flags=re.DOTALL).strip()
                
    return ""

def autocorrect_stt(text: str) -> str:
    """Fixes common phonetic transcription errors from the STT model using a domain dictionary.

    Uses word-boundary-safe regex replacement to prevent false positives.
    """
    corrections = {
        # School-specific proper nouns (original entries — unchanged)
        "vangul": "Bangalore",
        "guaidfield": "Whitefield",
        "interest public": "NPS Public",
        "reachy money": "Reachy Mini",
        "reach mini": "Reachy Mini",
        "dr angelo": "Dr. Anjali",
        "dr. angelo": "Dr. Anjali",
        # NPS ITPL and HackNexus specific corrections
        "nps itpl": "NPS ITPL",
        "nps i t p l": "NPS ITPL",
        "nps i.t.p.l.": "NPS ITPL",
        "nps it pl": "NPS ITPL",
        "hacknexus": "HackNexus 2026",
        "hack nexus": "HackNexus 2026",
        "hacknexus 2026": "HackNexus 2026",
        "roopa sridhar": "Mrs. Roopa Sridhar",
        "roopa shridhar": "Mrs. Roopa Sridhar",
        "rupa sridhar": "Mrs. Roopa Sridhar",
        "rupa shridhar": "Mrs. Roopa Sridhar",
        "k g garg": "Mr. K. G. Garg",
        "k. g. garg": "Mr. K. G. Garg",
        "gopalkrishna": "Dr. K. P. Gopalkrishna",
        "kp gopalkrishna": "Dr. K. P. Gopalkrishna",
        "k p gopalkrishna": "Dr. K. P. Gopalkrishna",
        # NCERT science/math domain terms — common Whisper mis-transcriptions
        # (additive: does not change non-buggy transcripts)
        "photo synthesis": "photosynthesis",
        "foto synthesis": "photosynthesis",
        "electric magnetic": "electromagnetic",
        "mitokondria": "mitochondria",
        "mitokondrion": "mitochondrion",
    }

    for bad_phrase, good_phrase in corrections.items():
        # Word-boundary-safe, case-insensitive replacement.
        # \b anchors prevent matching inside longer words (e.g. "vangul" must
        # not match in the middle of "triangular").
        pattern = re.compile(r'\b' + re.escape(bad_phrase) + r'\b', re.IGNORECASE)
        text = pattern.sub(good_phrase, text)

    return text
