# AI Agent Engineering

My notes and code from a Udemy course on AI agent engineering. Over 21 days it builds AI agents from first principles: it starts with a single call to a local LLM and works up to tools, MCP, RAG, persistent memory, multi-agent systems, LangGraph, CrewAI, LangChain, and agents served over FastAPI with a web frontend.

Everything runs against local [Ollama](https://ollama.com) models, so the examples work offline and need no paid API keys.

Each `DayNN/DayNN_PM` folder is a self-contained, runnable step. Later parts of the same day build on the earlier ones, and each part was added in its own commit with a detailed description of the code changes (`git log --reverse` walks the course in order).

> **Progress:** 1 of 73 lessons committed. Latest: `Day01_P1`, First call to a local LLM (hello_ai.py).

## Overview

- [Setup](#setup)
- [Repository Layout](#repository-layout)
- **Part I: Foundations, built by hand with the OpenAI client**
  - [1. First Calls to a Local LLM - Day01](#1-first-calls-to-a-local-llm---day01)
- [Lesson Index](#lesson-index)
- [Known Issues](#known-issues)
- [Useful Links](#useful-links)

## Setup

The repo is a [uv](https://docs.astral.sh/uv/) project: `pyproject.toml` declares the dependencies, `uv.lock` pins the exact versions they were tested with, and `.python-version` selects Python 3.12.

1. **Install [Ollama](https://ollama.com) and pull the models** the examples use:

   ```bash
   ollama pull qwen3:4b-gpu        # default chat model in the .env files (Days 01-11, 13-14, 16)
   ollama pull qwen3:1.7b          # hard-coded in Days 12, 15, 17, 19, 20
   ollama pull qwen3:4b            # hard-coded in Days 18, 21
   ollama pull nomic-embed-text    # embeddings (Days 06, 11, 12, 17, 19, 20)
   ```

2. **Install uv** (once per machine):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
   # Windows: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Create the environment** from the repo root. This installs Python 3.12 if needed and creates `.venv/` with the exact locked versions:

   ```bash
   uv sync
   ```

4. **Create a `.env` file** in the example folder you want to run (`DayNN/DayNN_PM/.env`). `.env` files are git-ignored, so each clone needs its own. Most folders use:

   ```env
   BASE_URL=http://localhost:11434/v1
   API_KEY=ollama
   MODEL=qwen3:4b-gpu
   ```

   The Day20 FastAPI services read different names instead:

   ```env
   MODEL_NAME=qwen3:4b-gpu
   OLLAMA_BASE_URL=http://localhost:11434
   ```

5. **Run an example from inside its folder** with `uv run`. uv finds `pyproject.toml` in the parent folders, so the shared `.venv` is used from any lesson folder. Paths such as `data/`, `knowledge/` and `server.py` are relative to the current directory.

   ```bash
   cd Day06/Day06_P5
   uv run python rag.py
   ```

   Each section below names the entry point for that day.

**Using pip instead of uv:** `uv export --no-hashes --format requirements-txt > requirements.txt`, then `pip install -r requirements.txt` in a Python 3.12 virtual environment.

**Adding a dependency:** `uv add <package>` updates `pyproject.toml` and `uv.lock` together.

**Why the OpenAI client works with Ollama:** Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`. The `openai` package only needs `base_url` pointed there. `api_key` must be non-empty but Ollama ignores it, which is why it is set to `ollama`.

## Repository Layout

```
ai-agent-engineering-course/
├── Day01/ … Day21/
│   └── DayNN_PM/          # one runnable step per folder (own .env, data, entry point)
├── pyproject.toml         # uv project: dependencies for every example
├── uv.lock                # exact, tested dependency versions
├── .python-version        # Python 3.12
├── .gitignore             # .env, .venv/, venv/, *.db, chroma_db/, __pycache__/
├── .gitattributes         # LF line endings for every file
└── README.md              # these notes
```

---

# Part I: Foundations

## 1. First Calls to a Local LLM - Day01

An LLM call is just an HTTP request: a list of **messages** (each with a `role` and `content`) goes in, and a completion comes out. Everything later in the course builds on this one call.

```python
client = OpenAI(base_url=os.getenv("BASE_URL"), api_key=os.getenv("API_KEY"))

response = client.chat.completions.create(
    model=os.getenv("MODEL"),
    messages=[{"role": "user", "content": "What is Artificial Intelligence?"}],
)
print(response.choices[0].message.content)
```

- [`Day01_P1`](./Day01/Day01_P1/) **Hello AI:** `hello_ai.py` sends one question to the local model and prints the reply.

Run: `uv run python hello_ai.py`

---

## Lesson Index

Every lesson below was added in its own commit; the commit message explains the code changes in detail.

| # | Lesson | What it adds |
|---|---|---|
| 1 | [`Day01_P1`](./Day01/Day01_P1/) | First call to a local LLM (hello_ai.py) |

## Known Issues

Open problems in the course code, kept as-is so each lesson matches the course.

- **Model names vary:** the `.env` template uses `qwen3:4b-gpu`, but several days hard-code `qwen3:1.7b` or `qwen3:4b` (see [Setup](#setup)), so pull all of them.

## Useful Links

- [Ollama](https://ollama.com) · [OpenAI-compatible API in Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [Model Context Protocol](https://modelcontextprotocol.io) · [FastMCP](https://gofastmcp.com)
- [LangChain](https://python.langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [CrewAI](https://docs.crewai.com)
- [Chroma](https://docs.trychroma.com) · [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com) · [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
