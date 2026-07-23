# Agent Lasso

A local-first web app for configuring LLM agents, chatting with them, and scoring them on exam-style benchmarks. FastAPI backend, LangChain/LangGraph agents, Jinja + Tailwind single-page UI, SQLite persistence.

Status: experimental, not maintained since mid-2025. (The UI and some code refer to the app by its working name, "Silver Lasso".)

## What it does

- Assemble agents from configurable LLM backends (OpenAI, Anthropic, Mistral, Groq) and a set of built-in tools: file operations, basic data analysis, calculator, current time, DuckDuckGo/SearXNG search, arXiv search, optional weather, and an interactive canvas (`tools.py`).
- Chat with agents over SSE streaming and keep conversations, agent configs, and scores in a local SQLite database (`agent_log.db`).
- Run YAML-defined benchmark exams (`benchmark_exams/` — MMLU math, ARC, HellaSwag, Winogrande, and others) against agents via `/api/benchmark/*`, with a sortable leaderboard; an `ExamBuilder` agent can scaffold new exam files.
- Optional GraphRAG layer (`graphrag_engine.py`): Neo4j 5.x with vector and graph indices, or a zero-dependency in-memory JSON fallback, using Sentence-Transformers embeddings.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
# open http://127.0.0.1:8000
```

API keys are optional. Any `KEY=value` pairs in a git-ignored `secrets.txt` in the project root are loaded at runtime:

```text
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=...
NEO4J_PASSWORD=...
```

## Configuration

| File | Purpose |
|------|---------|
| `secrets.txt` | API keys and passwords (git-ignored) |
| `config.py` | Model providers, embeddings, GraphRAG and tool settings |
| `database.py` | SQLite schema and helpers |
| `vercel.json` | Vercel Functions deployment config |

Most settings can also be overridden with environment variables.

## License

MIT
