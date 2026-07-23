import os

def w(p, c):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(c.strip() + '\n')

b = 'src/knowledge/vault/'

# Clean up old entities if they exist
for old_file in ['wiki/entities/nps-public-school.md', 'wiki/entities/dr-anjali-sharma.md']:
    old_path = os.path.join(b, old_file)
    if os.path.exists(old_path):
        os.remove(old_path)

w(b+'SCHEMA.md', '''# Wiki Compilation Schema for Reachy Mini

## Conventions
1. Layer 1 (`raw/`) is IMMUTABLE. Never edit files in `raw/`.
2. Layer 2 (`wiki/`) is managed exclusively by the LLM Compiler.
3. Every page in `wiki/` MUST contain standard YAML frontmatter.
4. Internal connections MUST use Obsidian-style `[[wikilinks]]`.

## Target File Templates

## Entity Template
```markdown
---
type: entity
category: [School / Person / Hardware / Event]
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

w(b+'index.md', '# Index\nWelcome to NPS ITPL Baymax Vault.')
w(b+'log.md', '# Log\n2026-07-20 compiled')
w(b+'raw/school_info.txt', "National Public School ITPL (NPS ITPL) is hosting HackNexus 2026. The Principal is Mrs. Roopa Sridhar. The founder is Mr. K. G. Garg. Chairman is Dr. K. P. Gopalkrishna. Baymax will be demoed next to the principal's seating area.")
w(b+'raw/robot_specs.pdf', 'dummy pdf')
w(b+'raw/schedule.md', '# Schedule')

w(b+'wiki/entities/nps-itpl.md', '''---
type: entity
category: School
last_updated: 2026-07-20
---
# National Public School ITPL
**Summary**: National Public School ITPL (NPS ITPL) is a premier educational institution located in Kadugodi, Whitefield, Bangalore, known for hosting [[HackNexus 2026]].

## Quick Facts
- Affiliation: CBSE (Affiliation No: 831091, CEEB Code: 084677)
- Founder: Mr. K. G. Garg
- Chairman: Dr. K. P. Gopalkrishna
- Principal: [[Mrs. Roopa Sridhar]]
- Contact: +91 96061 86999 or info@npsitpl.com
- Address: Goravigere, Kadugodi Main Road, Bengaluru - 560115

## Context
NPS ITPL is the proud host of the inaugural [[HackNexus 2026]] expo, where students from across schools showcase technology. The offline AI assistant [[Baymax]] will be stationed actively next to Principal [[Mrs. Roopa Sridhar]]'s designated seating area for live demonstration and interaction.

## Related Nodes
- [[HackNexus 2026]]
- [[Mrs. Roopa Sridhar]]
- [[Baymax]]''')

w(b+'wiki/entities/hacknexus.md', '''---
type: entity
category: Event
last_updated: 2026-07-20
---
# HackNexus 2026
**Summary**: The inaugural interschool Model, Simulate, Code and Pitch Expo hosted at [[National Public School ITPL]].

## Quick Facts
- Host Venue: [[National Public School ITPL]]
- Tagline: Where innovation connects with technology
- Objective: Cultivate technological excellence, curiosity, and responsible innovation
- Audience: School students, educators, and trainers

## Event Divisions
- **Smartscape (Grades 5 - 6)**: Design thinking and model building challenge
- **Robo Sweep (Grades 5 - 6)**: Clean-up and sweep robotics challenge
- **RoboCrafter X (Grades 7 - 8)**: Robotics craft and code challenge
- **SafeHaven Nexus (Grades 7 - 8)**: Safety and disaster simulation challenge
- **RoboRush (Grades 7 - 8)**: High-speed robotics racing challenge
- **PyPulse (Grades 9 - 12)**: Python coding and algorithm challenge
- **Innovatrix (Grades 9 - 12)**: Innovation pitch and entrepreneurship expo
- **TechTurf (Grades 9 - 12)**: Advanced tech application expo

## Context
During HackNexus 2026, the offline edge assistant [[Baymax]] is showcased in real-time, helping students and trainers interact with advanced AI without internet dependence.

## Related Nodes
- [[National Public School ITPL]]
- [[Baymax]]''')

w(b+'wiki/entities/roopa-sridhar.md', '''---
type: entity
category: Person
last_updated: 2026-07-20
---
# Mrs. Roopa Sridhar
**Summary**: The Principal of [[National Public School ITPL]].

## Context
Mrs. Roopa Sridhar provides academic and operational leadership for [[National Public School ITPL]]. At the [[HackNexus 2026]] expo, her seating area is situated directly adjacent to the demonstration table for [[Baymax]].

## Related Nodes
- [[National Public School ITPL]]
- [[Baymax]]
- [[HackNexus 2026]]''')

w(b+'wiki/entities/baymax.md', '''---
type: entity
category: Hardware
last_updated: 2026-07-20
---
# Baymax
**Summary**: The advanced offline edge-AI interactive assistant model featured at [[HackNexus 2026]].

## Context
Baymax is the core interactive AI prototype deployed at [[National Public School ITPL]] for the [[HackNexus 2026]] expo. The unit runs 100% offline on a Jetson Orin Nano, and is located directly next to Principal [[Mrs. Roopa Sridhar]]'s seating space.

## Related Nodes
- [[National Public School ITPL]]
- [[HackNexus 2026]]
- [[Mrs. Roopa Sridhar]]''')

w(b+'wiki/concepts/edge-ai-inference.md', '# Edge AI')
w(b+'wiki/concepts/computer-vision.md', '# CV')


print("Vault created successfully!")
