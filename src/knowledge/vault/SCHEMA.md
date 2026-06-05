# Wiki Compilation Schema for Reachy Mini

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
```

## 🔄 Continuous Update & Mutation Policy

When a new source file in `raw/` contains updates or changes to existing wiki files:

1. **Preserve and Extend**: NEVER delete old context unless it is factually replaced. Append or modify the section dynamically.
2. **Handle Contradictions Explicitly**: If new raw info directly contradicts an existing wiki fact, do not silently overwrite it. Log it structurally:
   - Example: *Note: As of June 2026, the venue was relocated from the Auditorium to the Science Lab.*
3. **Bump Metadata**: Always update the `last_updated:` string in the YAML frontmatter of every modified page.
4. **Maintain Link Integrity**: Do not break existing `[[wikilinks]]`. If an entity page moves or is renamed, update every file that points to it.
