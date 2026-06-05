import os
import json
import requests
from datetime import datetime

# Lightweight .env parser to avoid extra edge dependencies
def load_env():
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key.strip()] = val.strip()

load_env()

class ActiveWebUpdater:
    def __init__(self):
        self.api_key = os.environ.get("TAVILY_API_KEY", "")
        self.incoming_dir = "src/knowledge/vault/raw/incoming_web/"
        os.makedirs(self.incoming_dir, exist_ok=True)

    def search_and_stage(self, query: str) -> str:
        """Searches the live web, extracts content, and stages it for Vault compilation."""
        print(f"\n\033[96m[WebUpdater]: Initiating internet search for '{query}'...\033[0m")
        
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                ddg_results = list(ddgs.text(query, max_results=3))
                
                if not ddg_results:
                    raise Exception("DuckDuckGo returned empty results.")
                    
                results = []
                for r in ddg_results:
                    results.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "content": r.get("body", "")
                    })
        except ImportError:
            print("\033[91m[WebUpdater]: duckduckgo-search is not installed! Run: pip install duckduckgo-search\033[0m")
            return "Web Search Failed: Package missing."
        except Exception as e:
            print(f"\033[93m[WebUpdater]: DuckDuckGo failed ({str(e)}). Falling back to Tavily API...\033[0m")
            # --- TAVILY FALLBACK ---
            if not self.api_key or self.api_key == "YOUR_TAVILY_API_KEY":
                print("\033[91m[WebUpdater]: Tavily API Key missing! Cannot fallback.\033[0m")
                return "Web Search Failed: DuckDuckGo failed and no Tavily key provided."
            
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 2
            }
            try:
                response = requests.post(url, json=payload, timeout=2.5)
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])
                else:
                    return f"Web Search Failed: Tavily returned status {response.status_code}"
            except Exception as tavily_e:
                return f"Web Search Failed: Both DuckDuckGo and Tavily failed. ({str(tavily_e)})"
                
        # Formulate instant context for the streaming LLM response
        context = "Live Internet Search Results:\n"
        for res in results:
            context += f"- [{res['title']}]({res['url']}): {res['content']}\n"
        
        # --- STAGE FOR PILLAR 2 AUTO-UPDATE ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join([c if c.isalnum() else "_" for c in query.lower()])[:30]
        
        stage_file = os.path.join(self.incoming_dir, f"web_{timestamp}_{safe_query}.json")
        with open(stage_file, "w") as f:
            json.dump({"query": query, "timestamp": timestamp, "raw_web_data": results}, f, indent=2)
            
        print(f"\033[92m[WebUpdater]: Successfully staged web data to {stage_file}\033[0m")
        return context
