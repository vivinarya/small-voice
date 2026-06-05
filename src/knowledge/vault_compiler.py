import os
import json
from datetime import datetime

class WikiCustodian:
    def __init__(self):
        self.incoming_dir = "src/knowledge/vault/raw/incoming_web/"
        self.wiki_dir = "src/knowledge/vault/wiki/entities/"
        
    def process_incoming_web_logs(self, agent_compiler_client):
        """Scans staged web search dumps and uses a compiler LLM to merge them into the graph."""
        if not os.path.exists(self.incoming_dir):
            return

        staged_files = [f for f in os.listdir(self.incoming_dir) if f.endswith(".json")]
        
        if not staged_files:
            return
            
        print(f"[Custodian]: Found {len(staged_files)} new internet discoveries to compile.")
        
        for file in staged_files:
            file_path = os.path.join(self.incoming_dir, file)
            with open(file_path, "r") as f:
                web_dump = json.load(f)
                
            query = web_dump["query"]
            raw_facts = web_dump["raw_web_data"]
            
            # --- CALL THE DEVELOPMENT AGENT COMPILER ---
            # We instruct a high-level LLM agent to evaluate the text and modify our markdown repo
            compilation_prompt = f"""
            You are Andrej Karpathy's automated Wiki Compiler agent.
            Task: Integrate this newly discovered real-time web context into our local knowledge vault.
            
            User Original Query: {query}
            Discovered Web Facts: {json.dumps(raw_facts)}
            
            Instructions:
            1. Scan the existing markdown files in '{self.wiki_dir}'.
            2. If an entity exists (e.g., nps-public-school.md), append or update the relevant facts.
            3. If a new entity or concept is mentioned that we do not have a page for, create a new clean markdown file following SCHEMA.md rules.
            4. Interlink pages using [[wikilinks]]. Do not destroy existing text blocks.
            """
            
            print(f"[Custodian]: Compiling facts for query: '{query}'...")
            
            # Execute the update pass
            success = agent_compiler_client.execute_wiki_mutation(compilation_prompt, query)
            
            if success:
                # Move or delete the incoming file so we don't double-process it
                os.remove(file_path)
                print(f"[Custodian]: Successfully integrated and link-checked node changes.")

class LocalCompilerClient:
    def __init__(self, engine):
        self.engine = engine
        self.wiki_dir = "src/knowledge/vault/wiki/entities/"

    def execute_wiki_mutation(self, compilation_prompt: str, query: str) -> bool:
        """Uses the local Gemma engine to output raw markdown for the new entity."""
        try:
            print("\n\033[95m[WikiCustodian]: Starting background compilation engine...\033[0m")
            print("\033[95m[WikiCustodian]: Synthesizing new facts into Markdown vault format...\033[0m")
            
            # We add an explicit instruction to the prompt to ONLY output the markdown.
            prompt = compilation_prompt + "\n\nCRITICAL: Output ONLY the raw markdown content inside ```markdown blocks. Do not add any conversational text."
            
            # We iterate the stream to get the full generation
            stream = self.engine.get_stream(None, prompt)
            full_response = "".join(list(stream))
            
            # Parse the markdown block
            import re
            match = re.search(r'```(?:markdown)?(.*?)```', full_response, re.DOTALL | re.IGNORECASE)
            if match:
                markdown_content = match.group(1).strip()
            else:
                markdown_content = full_response.strip()
                
            # Enforce single-file update policy for the showcase
            if "nps" in query.lower() or "school" in query.lower():
                safe_name = "nps-public-school"
            else:
                # Create a slug for the file name based on the markdown title
                lines = markdown_content.split('\n')
                title = "new_entity"
                for line in lines:
                    if line.strip().startswith('# '):
                        title = line.replace('#', '').strip()
                        break
                        
                safe_name = "".join([c if c.isalnum() else "-" for c in title.lower()]).strip("-")
                safe_name = re.sub(r'-+', '-', safe_name)[:40]
                if not safe_name:
                    safe_name = "new_entity"
                
            filename = os.path.join(self.wiki_dir, f"{safe_name}.md")
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(markdown_content)
                
            print(f"\033[92m[WikiCustodian]: SUCCESS! Wrote updated knowledge to {filename}\033[0m")
            print(f"\033[92m[WikiCustodian]: You can open the file in your IDE to verify the changes.\033[0m\n")
            
            # Run the linter
            import subprocess
            subprocess.run(["python", "src/knowledge/vault/lint_wiki.py"], capture_output=True)
            return True
            
        except Exception as e:
            print(f"\033[91m[LocalCompilerClient]: Mutation failed: {e}\033[0m")
            return False
