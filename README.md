# Silver Lasso – Configure, Play, Benchmark.

*Transform complex data into actionable insights*

Silver Lasso is a lightweight AI platform that lets you assemble specialised "agents", run them through real-time chats, and benchmark their performance – all from a single web interface powered by FastAPI + Tailwind.

---
## 🎯 Why Silver Lasso?
* ✅ **Zero-Friction Deployment** – no cloud accounts or databases required. Runs locally, deploys to Vercel in one click.
* ✅ **Adaptive Intelligence** – plug-and-play tools, configurable LLM back-ends and auto-optimising agent templates.
* ✅ **Privacy by Design** – everything is processed on your machine unless you opt-in to external APIs.
* ✅ **Intuitive Mastery** – visual interface, live progress events and persistent conversation threads.
* ✅ **Scalable Architecture** – GraphRAG-ready knowledge graph layer plus Neo4j / JSON fallback.

---
## 🚀 Quick Start
1. Create a virtual environment (optional but recommended)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
2. Install the dependencies
```bash
pip install -r requirements.txt
```
3. (Optional) add API keys & secrets
Create a `secrets.txt` file in the project root. Any `KEY=value` pair in this file is automatically loaded at runtime:
```text
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
MISTRAL_API_KEY=...
GROQ_API_KEY=...
NEO4J_PASSWORD=...
```
4. Start the development server (hot-reload)
```bash
uvicorn main:app --reload
```
Open http://127.0.0.1:8000 in your browser ➜ the full glass-morphic UI will load.

> In production you can either:
> • `uvicorn main:app --host 0.0.0.0 --port 8000` on your own box, or
> • **Deploy to Vercel** – the provided `vercel.json` is pre-configured.

---
## 🛠️ Available Tools
| Category | Tool | Description |
|----------|------|-------------|
| Core (always on) | `file_operations` | Read, write, append, delete files & list directories |
|  | `data_analysis` | Quick descriptive stats, correlations & basic visualisations for CSV / JSON |
|  | `current_time` | Returns the current date & time (temporal grounding) |
|  | `calculator` | Safe arithmetic expression evaluator |
| Search | `duckduckgo_search` | Privacy-respecting web search (no key) |
|  | `searx_search` | Search via your own or public SearXNG instance |
| Optional | `weather` | Local weather via OpenWeatherMap (uses `OPENWEATHER_API_KEY` if present, otherwise link fallback) |
|  | `arxiv_search` | Academic paper search through arXiv API |
| Interactive | `interactive_canvas` | Opens a collaborative canvas, lets the agent draw / clear shapes |

Retrieve the live tool objects in Python:
```python
from tools import get_tools
for t in get_tools():
    print(t.name, '-', t.description)
```

---
## 🌐 User Interface
* **FastAPI backend** – JSON & SSE endpoints under `/api/*`.
* **Jinja-tailwind frontend** – single-page app in `templates/index.html` with progressive enhancement.
* **Live agent stream** – watch your agent think, use tools and refine answers step-by-step.
* **Persistent SQLite storage** – conversations, agent configurations and benchmark scores live in `agent_log.db`.

---
## 🧠 GraphRAG Engine  
`graphrag_engine.py` provides a pluggable GraphRAG layer.
* **Neo4j mode** – vector & graph indices inside Neo4j 5.x (requires server + credentials).
* **JSON fallback** – zero-dependency in-memory graph for local testing.
* Sentence-Transformers embeddings with optional chunk filtering.

Configure via `config.GRAPHRAG_CONFIG` or override with environment variables.

---
## 🧪 Benchmarking
The `/api/benchmark/*` endpoints let you execute pluggable test harnesses (see `benchmarks.py`).  Results are stored and a live leaderboard is available in the UI.

---
## ⚙️ Configuration Cheatsheet
| File | Purpose |
|------|---------|
| `secrets.txt` | **Recommended** place for keys & passwords (git-ignored) |
| `config.py` | Default model providers, embedding models, GraphRAG & tool settings |
| `database.py` | SQLite schema & helper functions |
| `vercel.json` | Zero-config deployment to Vercel Functions |

Most settings can also be overridden via standard environment variables.

---
## 📦 Key Dependencies
```
langchain-core>=0.3.65
langchain-community>=0.3.25
langchain-openai>=0.3.24
langchain-anthropic>=0.3.15
langchain-mistralai>=0.2.10
langchain-groq>=0.3.2
langgraph>=0.4.8
fastapi>=0.110.0
uvicorn[standard]>=0.32.0
duckduckgo-search>=8.0.4
neo4j>=5.17.0      # optional, for full GraphRAG
```
See `requirements.txt` for the exhaustive list and pinned versions.

---
## 🔧 Troubleshooting
* **DuckDuckGo search not working** ➜ `pip install --upgrade duckduckgo-search`
* **Neo4j connection refused** ➜ ensure Neo4j is running (`docker run -p 7687:7687 neo4j:5`), then set `NEO4J_PASSWORD`.
* **Port already in use** ➜ change the port: `uvicorn agent_setup_landing:app --port 9000 --reload`.

---
## 🤝 Contributing
1. Fork the repo & create a feature branch.
2. Run the unit tests & `black .`.
3. Open a PR – please describe the tool / feature you added and include benchmark results if relevant.

---
## 📄 License
MIT – do what you want, just give credit.

---
## 🙏 Acknowledgements
Silver Lasso stands on the shoulders of giants. Huge thanks to the open-source community and to the projects that made building this platform possible:

* [FastAPI](https://fastapi.tiangolo.com/) & [Starlette](https://www.starlette.io/) – blazing-fast ASGI backend and SSE streaming.
* [LangChain](https://www.langchain.com/) & [LangGraph](https://github.com/langchain-ai/langgraph) – agent orchestration and graph execution.
* [Tailwind CSS](https://tailwindcss.com/) – utility-first styling for the glass-morphic UI.
* [Jinja2](https://palletsprojects.com/p/jinja/) – server-rendered templates.
* [DuckDuckGo Search](https://github.com/deedy5/duckduckgo-search) – privacy-friendly web search API.
* [Sentence-Transformers](https://www.sbert.net/) – lightweight embeddings powering GraphRAG.
* [Neo4j](https://neo4j.com/) – optional vector & graph backend.
* LLM providers – [OpenAI](https://openai.com/), [Anthropic](https://www.anthropic.com/), [Mistral AI](https://mistral.ai/), [Groq](https://groq.com/).
* And the countless maintainers of Python libraries we rely on every day—`requests`, `pydantic`, `pandas`, `beautifulsoup4`, and many more.

Your work and generosity make Silver Lasso possible. Thank you!