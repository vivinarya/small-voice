import os

def w(p, c):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(c.strip() + '\n')

b = 'src/knowledge/vault/'

w(b+'SCHEMA.md', '''# Wiki Compilation Schema for Reachy Mini

## Conventions
1. Layer 1 (`raw/`) is IMMUTABLE. Never edit files in `raw/`.
2. Layer 2 (`wiki/`) is managed exclusively by the LLM Compiler.
3. Every page in `wiki/` MUST contain standard YAML frontmatter.
4. Internal connections MUST use Obsidian-style `[[wikilinks]]`.

## Target File Templates

### Entity Template
```markdown
---
type: entity
category: [School / Person / Hardware]
last_updated: YYYY-MM-DD
---
# {Name}
**Summary**: One-sentence conversational overview.

## Quick Facts
- Attribute 1: Value
- Attribute 2: Value

## Detailed Context
{Deep background details compiled from raw sources.}

## Related Nodes
- [[Connected Page 1]]
- [[Connected Page 2]]
```''')

w(b+'index.md', '# Index\nWelcome to Reachy Mini Vault.')
w(b+'log.md', '# Log\n2026-06-05 compiled')
w(b+'raw/school_info.txt', "NPS Public School was founded in 1998. The Principal is Dr. Anjali Sharma. The showcase is happening in the Main Auditorium on June 6th. Reachy Mini will be demoed on a table right next to the principal's seating area.")
w(b+'raw/robot_specs.pdf', 'dummy pdf')
w(b+'raw/schedule.md', '# Schedule')

w(b+'wiki/entities/nps-public-school.md', '''---
type: entity
category: School
last_updated: 2026-06-05
---
# NPS Public School
**Summary**: The school hosting today's interactive robotics exhibition.

## Quick Facts
- Founded: 1998
- Showcase Location: Main Auditorium

## Context
The school is hosting the 2026 hardware showcase. During the event, [[Reachy Mini]] will be stationed actively on a presentation table positioned next to the designated seating area for [[Dr. Anjali Sharma]].''')

w(b+'wiki/entities/dr-anjali-sharma.md', '''---
type: entity
category: Person
last_updated: 2026-06-05
---
# Dr. Anjali Sharma
**Summary**: The Principal of [[NPS Public School]].

## Context
Dr. Sharma oversees the academic and operational leadership of the school. At the upcoming robotics showcase in the Main Auditorium, her primary seating area will be located adjacent to the live demonstration table for [[Reachy Mini]].''')

w(b+'wiki/entities/reachy-mini.md', '''---
type: entity
category: Hardware
last_updated: 2026-06-05
---
# Reachy Mini
**Summary**: The advanced edge-AI interactive robot designed for open-source social robotics.

## Context
Reachy Mini is the featured AI prototype being showcased inside the Main Auditorium at [[NPS Public School]]. The physical unit operates completely offline and is situated directly next to [[Dr. Anjali Sharma]]'s seating space for real-time interaction.''')

w(b+'wiki/concepts/edge-ai-inference.md', '# Edge AI')
w(b+'wiki/concepts/computer-vision.md', '# CV')

w('src/knowledge/graph.py', '''import os

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
                return f.read() # Returns the pre-synthesized markdown directly!
                
    return ""''')

print("Vault created successfully!")
