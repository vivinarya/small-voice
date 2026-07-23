# Wiki Compilation Schema for Reachy Mini

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
```
