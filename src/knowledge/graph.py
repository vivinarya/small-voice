import os

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
                import re
                content = f.read()
                return re.sub(r'^---.*?---\n', '', content, flags=re.DOTALL).strip()
                
    return ""

def autocorrect_stt(text: str) -> str:
    """Fixes common phonetic transcription errors from the STT model using a domain dictionary."""
    corrections = {
        "vangul": "Bangalore",
        "guaidfield": "Whitefield",
        "interest public": "NPS Public",
        "reachy money": "Reachy Mini",
        "reach mini": "Reachy Mini",
        "dr angelo": "Dr. Anjali",
        "dr. angelo": "Dr. Anjali"
    }
    
    # Case-insensitive replacement
    text_lower = text.lower()
    for bad_phrase, good_phrase in corrections.items():
        if bad_phrase in text_lower:
            # We do a simple string replace. For production, regex word-boundaries are better,
            # but this is perfect for the edge showcase.
            import re
            pattern = re.compile(re.escape(bad_phrase), re.IGNORECASE)
            text = pattern.sub(good_phrase, text)
            
    return text
