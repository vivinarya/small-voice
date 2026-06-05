import os
import re

def lint_knowledge_graph():
    wiki_dir = "src/knowledge/vault/wiki/entities/"
    compiled_files = {f.replace(".md", "") for f in os.listdir(wiki_dir) if f.endswith(".md")}
    
    print("Checking Knowledge Graph integrity...")
    
    for filename in os.listdir(wiki_dir):
        if not filename.endswith(".md"):
            continue
            
        with open(os.path.join(wiki_dir, filename), "r") as f:
            content = f.read()
            
        # Find all Obsidian styled [[wikilinks]] in the text
        links = re.findall(r"\[\[(.*?)\]\]", content)
        
        for link in links:
            # Normalize link format to match standard filenames
            normalized_link = re.sub(r'[^a-z0-9\s-]', '', link.lower()).replace(" ", "-")
            if normalized_link not in compiled_files:
                print(f"⚠️ [Broken Link Found]: '{filename}' links to [[{link}]], but '{normalized_link}.md' does not exist!")

if __name__ == "__main__":
    lint_knowledge_graph()
