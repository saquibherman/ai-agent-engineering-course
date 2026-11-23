# AI Agent Engineering

My notes and code from a Udemy course on AI agent engineering. Over 21 days it builds AI agents from first principles: it starts with a single call to a local LLM and works up to tools, MCP, RAG, persistent memory, multi-agent systems, LangGraph, CrewAI, LangChain, and agents served over FastAPI with a web frontend.

Everything runs against local [Ollama](https://ollama.com) models, so the examples work offline and need no paid API keys.

Each `DayNN/DayNN_PM` folder is a self-contained, runnable step. Later parts of the same day build on the earlier ones, and each part was added in its own commit with a detailed description of the code changes (`git log --reverse` walks the course in order).

> **Progress:** 49 of 73 lessons committed. Latest: `Day17_P3`, Hierarchical process and crew memory.

## Overview

- [Setup](#setup)
- [Repository Layout](#repository-layout)
- **Part I: Foundations, built by hand with the OpenAI client**
  - [1. First Calls to a Local LLM - Day01](#1-first-calls-to-a-local-llm---day01)
  - [2. Chat Loop and Conversation Memory - Day02](#2-chat-loop-and-conversation-memory---day02)
  - [3. System Prompts and Personas - Day03](#3-system-prompts-and-personas---day03)
  - [4. Hand-Written Tools and a Router - Day04](#4-hand-written-tools-and-a-router---day04)
  - [5. File Tools and Grounded Q&A - Day05](#5-file-tools-and-grounded-qa---day05)
  - [6. Retrieval: Keywords to Embeddings (First RAG) - Day06](#6-retrieval-keywords-to-embeddings-first-rag---day06)
  - [7. LLM-Based Tool Planner - Day07](#7-llm-based-tool-planner---day07)
- **Part II: Agent architecture**
  - [8. Model Context Protocol (MCP) - Day08](#8-model-context-protocol-mcp---day08)
  - [9. Agent State and the Plan-Execute-Observe Loop - Day09](#9-agent-state-and-the-plan-execute-observe-loop---day09)
  - [10. Task Decomposition - Day10](#10-task-decomposition---day10)
  - [11. Persistent Semantic Memory - Day11](#11-persistent-semantic-memory---day11)
  - [12. Advanced RAG Pipeline - Day12](#12-advanced-rag-pipeline---day12)
  - [13. Multi-Server MCP and Remote Servers - Day13](#13-multi-server-mcp-and-remote-servers---day13)
  - [14. Multi-Agent Systems - Day14](#14-multi-agent-systems---day14)
  - [15. Production Hardening - Day15](#15-production-hardening---day15)
- **Part III: Agent frameworks**
  - [16. Agents as Graphs with LangGraph - Day16](#16-agents-as-graphs-with-langgraph---day16)
  - [17. Multi-Agent Crews with CrewAI - Day17](#17-multi-agent-crews-with-crewai---day17)
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
- [`Day01_P2`](./Day01/Day01_P2/) **assistant.py:** the same call wrapped in `assistant.py`, the file the next days grow.

Run: `uv run python hello_ai.py`

## 2. Chat Loop and Conversation Memory - Day02

LLMs are **stateless**: each request knows only the messages sent with it. Conversation memory is therefore just a `messages` list that grows every turn and is sent in full with each request.

```python
messages = []
while True:
    user_input = input("\nYou : ")
    messages.append({"role": "user", "content": user_input})
    reply = client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content
    messages.append({"role": "assistant", "content": reply})   # the model "remembers" next turn
```

- [`Day02_P1`](./Day02/Day02_P1/) **Conversation loop:** an `input()` loop that appends both sides of the conversation to `messages`. Type `quit` to exit.
- [`Day02_P2`](./Day02/Day02_P2/) **Print history:** prints the full `messages` list after every turn, so you can watch the context grow.
- [`Day02_P3`](./Day02/Day02_P3/) **Readable history:** prints the history as `Role : content` lines.

Run: `uv run python assistant.py`

**Note:** the whole history is re-sent every turn, so long chats cost more tokens and eventually exceed the model's context window. Later days replace this with retrieval-based memory (Day11).

## 3. System Prompts and Personas - Day03

A **system message** at the start of `messages` sets the model's behavior for the whole conversation: its tone, role and rules. Changing only this one message makes the same model act as a completely different assistant.

```python
messages = [{"role": "system", "content": roles.get(choice, "You are a helpful AI assistant.")}]
```

- [`Day03_P1`](./Day03/Day03_P1/) **System prompt persona:** a fixed "friendly AI teacher" persona.
- [`Day03_P2`](./Day03/Day03_P2/) **Selectable personas:** at startup the user picks one of five personas: teacher, Python expert, travel guide, motivational coach or interviewer.

Run: `uv run python assistant.py`

## 4. Hand-Written Tools and a Router - Day04

LLMs can't check the clock or generate truly random numbers. **Tools** are ordinary functions that do this for them. This day's routing is deliberately naive: keyword matching decides which tool to call, and the LLM is only the fallback.

```python
def execute_tool(user_input):
    text = user_input.lower()
    if "time" in text or "clock" in text:
        return get_current_time()
    if "dice" in text or "die" in text:
        return f"You rolled {roll_dice()}"
    if "password" in text:
        return generate_password()
    return None            # no tool matched → ask the LLM
```

- [`Day04_P1`](./Day04/Day04_P1/) **Standalone tools:** `get_current_time`, `roll_dice` and `generate_password` (which uses `secrets`, not `random`) in `tools.py`, plus `test_tools.py`.
- [`Day04_P2`](./Day04/Day04_P2/) **Tools in the chat loop:** keyword checks inside the loop call a tool before falling back to the LLM.
- [`Day04_P3`](./Day04/Day04_P3/) **Tool manager:** routing moved into `tool_manager.execute_tool()`, tested by `test_tool_manager.py`.
- [`Day04_P4`](./Day04/Day04_P4/) **Wire it together:** every turn goes through `execute_tool()` first.

Run: `uv run python assistant.py`; tests: `uv run python test_tools.py`, `uv run python test_tool_manager.py`

**Limitation:** "what time should I leave?" triggers the time tool. Day07 fixes this by letting the LLM choose the tool.

## 5. File Tools and Grounded Q&A - Day05

**Grounding** means pasting a document's content into the prompt and telling the model to answer *only* from it. This is the manual form of RAG and the main way to reduce hallucinations about your own data.

```python
prompt = f"""Answer the question using only the document below.
Document:
{file_content}
Question: {question}"""
```

- [`Day05_P1`](./Day05/Day05_P1/) **File-reading tool:** `read_text_file()` and sample files in `data/`, tested by `test_file_tool.py`.
- [`Day05_P2`](./Day05/Day05_P2/) **Wire it in:** file reads go through `tool_manager` and the assistant.
- [`Day05_P3`](./Day05/Day05_P3/) **Document commands:** `summarize <file>`, `explain <file>` and `ask <file> <question>`, each building a prompt around the file's contents.

Run: `uv run python assistant.py`, then for example `ask notes.txt What is the project about?`

## 6. Retrieval: Keywords to Embeddings (First RAG) - Day06

You usually can't paste every document into the prompt, so you need to **retrieve** the relevant one first. Keyword search breaks when the wording differs. **Embeddings** turn text into vectors where similar meanings sit close together, and **cosine similarity** measures how close two of them are:

$$\text{cos}(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

```python
question_embedding = create_embedding(user_input)               # nomic-embed-text
best_document = max(document_embeddings,
                    key=lambda f: cosine_similarity(question_embedding, document_embeddings[f]))
context = documents[best_document]                              # → goes into the prompt
```

- [`Day06_P1`](./Day06/Day06_P1/) **Load a knowledge folder:** `load_documents()` reads every `.txt` file in `knowledge/` into a dict.
- [`Day06_P2`](./Day06/Day06_P2/) **Keyword retrieval:** `retrieve(query)` returns the first document that contains the query.
- [`Day06_P3`](./Day06/Day06_P3/) **Embeddings:** `create_embedding()` calls Ollama's `nomic-embed-text`.
- [`Day06_P4`](./Day06/Day06_P4/) **Cosine similarity:** `similarity.cosine_similarity()` compares two embeddings numerically.
- [`Day06_P5`](./Day06/Day06_P5/) **End-to-end RAG:** `rag.py` embeds every document once, embeds each question, picks the closest document and answers from it.

Run: `uv run python rag.py`

## 7. LLM-Based Tool Planner - Day07

Instead of keyword rules, the **LLM chooses the tool**. The planner prompt lists the available tools with a description of when to use each, and asks for *only* a tool name or `none`. This separates *deciding* what to do from *doing* it, which every later agent relies on.

```python
planner_prompt = f"""You are an AI planner.
Available tools:
1. get_current_time - Use when the user asks for the current date or time.
2. roll_dice        - Use when the user asks to roll a dice.
3. generate_password - Use when the user wants a secure password.
If no tool is required, return: none
Return ONLY the tool name.
User Request: {user_request}"""
```

- [`Day07_P1`](./Day07/Day07_P1/) **LLM tool planner:** `planner.choose_tool()`, tested by `test_planner.py`.
- [`Day07_P2`](./Day07/Day07_P2/) **Planner in the assistant:** `assistant.py` routes with `choose_tool()` instead of keywords.

Run: `uv run python assistant.py`

**Note:** small models sometimes add extra words around the tool name, which is why the planner `.strip()`s the reply and the prompt insists on "ONLY". Days 14 and 21 handle this more strictly with structured JSON and Pydantic validation.

---

# Part II: Agent Architecture

## 8. Model Context Protocol (MCP) - Day08

**MCP** is an open protocol that lets an agent **discover** and **call** tools that live in a separate server process. Tools no longer have to be imported functions. Any MCP server, in any language, can plug into any MCP client. This repo uses [FastMCP](https://gofastmcp.com).

```python
# server.py: a tool is a decorated function; its docstring becomes the description
mcp = FastMCP("Time Server")

@mcp.tool()
def current_time():
    """Return the current date and time."""
    return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

# mcp_client.py: Client("server.py") launches the server over stdio
tools  = await client.list_tools()                 # discover
result = await client.call_tool("current_time", {})  # execute
```

- [`Day08_P1`](./Day08/Day08_P1/) **MCP client and server:** a FastMCP time server and a client that connects and lists its tools (`test_client.py`).
- [`Day08_P2`](./Day08/Day08_P2/) **MCP-backed agent:** `agent.py` calls the server's tool from the assistant flow.
- [`Day08_P3`](./Day08/Day08_P3/) **Planner over MCP tools:** builds the planner prompt from the server's *discovered* tool names and descriptions, so adding a tool to the server needs no change in the agent.
- [`Day08_P4`](./Day08/Day08_P4/) **Natural-language answers:** a second LLM call turns the raw tool result into a conversational reply.

Run: `uv run python agent.py`. The client starts `server.py` itself over stdio, so you don't need to run the server separately.

## 9. Agent State and the Plan-Execute-Observe Loop - Day09

An **agent** is a loop, not a single call. Each iteration it **plans** the next action from everything seen so far, **executes** it and **observes** the result, until the planner says `FINISH` or a step limit is reached. All of this is held in an explicit **state** object.

```python
state = create_state(user_request)                 # actions, observations, step counters
while can_continue(state):                         # not finished and step < max_steps
    action = planner(state, tools)                 # LLM sees request + past observations
    if action == "FINISH":
        finish(state, format_answer(state)); break
    observation = await execute_action(action, state, client)
    add_action(state, action)
    record_observation(state, action, observation)
    next_step(state)
```

- [`Day09_P1`](./Day09/Day09_P1/agent/) **Agent state:** `state.py` is a plain dict factory plus helpers (`add_action`, `record_observation`, `can_continue`, `finish`).
- [`Day09_P2`](./Day09/Day09_P2/agent/) **Full loop over MCP:** separate `planner`, `executor`, `loop` and `runner` modules drive the state against the MCP time server.

Run: `cd agent && uv run python runner.py`

**Why `max_steps` matters:** without it, a planner that never returns `FINISH` loops forever. The planner prompt also passes in `Completed Actions` and says "never repeat an action" to stop the model calling the same tool again and again.

## 10. Task Decomposition - Day10

Complex requests ("check the time, then roll a dice and make me a password") work better when they're first **broken into ordered subtasks**. The agent loop then works through them one at a time.

```python
def decompose_task(user_request):
    # prompt rules: only necessary tasks, in execution order, one per line, no numbering
    result = llm(prompt)
    return [line.strip() for line in result.splitlines() if line.strip()]
```

- [`Day10_P1`](./Day10/Day10_P1/) **Task decomposition:** `task_planner.decompose_task()` returns a list of subtasks (`test_task_planner.py`).
- [`Day10_P2`](./Day10/Day10_P2/) **Decomposition and agent loop:** combines it with Day09's loop, plus `answer.py` to format the final answer.

Run: `uv run python runner.py`

## 11. Persistent Semantic Memory - Day11

Day02's `messages` list disappears when the program exits and grows without limit. **Long-term memory** fixes both problems: it saves memories to **SQLite** together with their **embeddings**, then retrieves only the *relevant* ones by cosine similarity. It's the same idea as RAG, applied to the agent's own history.

```sql
CREATE TABLE memories (id INTEGER PRIMARY KEY, user_message TEXT,
                       assistant_message TEXT, embedding TEXT, created_at TIMESTAMP)
```

Not everything is worth remembering. A second LLM call acts as a **memory extractor**. It returns one durable fact (preferences, goals, skills, projects) or `NONE` for one-off things like the current time or a dice roll.

- [`Day11_P1`](./Day11/Day11_P1/) **Persistent semantic memory:** `memory.py` provides `save_memory()` and semantic search over a `memory.db` SQLite file (`test_memory.py`).
- [`Day11_P2`](./Day11/Day11_P2/) **Memory in the agent loop:** past memories are available during each run.
- [`Day11_P3`](./Day11/Day11_P3/) **Memory extraction:** `memory_extractor.py` plus `save_semantic_memory()`, so only useful facts are stored.
- [`Day11_P4`](./Day11/Day11_P4/) **Checkpoint:** a snapshot of the memory-augmented agent.

Run: `uv run python runner.py`. `memory.db` is created in the folder and is git-ignored (`*.db`).

## 12. Advanced RAG Pipeline - Day12

Day06 compared whole documents. Real RAG splits documents into **chunks**, stores them in a **vector database**, combines **semantic** and **keyword** scores (hybrid search), and **reranks** the candidates before generating an answer.

```
document → chunker (200 words, 50 overlap) → embeddings → Chroma (persistent)
query → embed → Chroma top-5 → hybrid score = 0.7·semantic + 0.3·keyword
      → LLM reranker (scores 0-100) → top-3 → generator → answer
```

```python
semantic = 1 / (1 + distance)                     # Chroma distance → similarity
combined = 0.7 * semantic + 0.3 * keyword_score(query, text)
return rerank(query, top_candidates, top_k=3)     # LLM rates each chunk's relevance
```

- [`Day12`](./Day12/Day12/) **Advanced RAG pipeline:**
  - `chunker.py`: overlapping word chunks; the overlap keeps sentences that cross a chunk boundary.
  - `embeddings.py`: embeds the chunks.
  - `chroma_store.py`: a Chroma `PersistentClient` in `./chroma_db`, with `upsert` so re-runs don't duplicate chunks.
  - `retriever.py`: keyword scoring.
  - `reranker.py`: the LLM gives each chunk a relevance score from 0 to 100 (temperature 0, `qwen3:1.7b`).
  - `evaluation.py`: checks whether the expected chunk was retrieved.
  - `generator.py`: writes the answer.
  - `advanced_rag.py` connects these steps, and `main.py` is the interactive entry point.

Run: `uv run python main.py`. It indexes `documents/python.txt` into `./chroma_db` (git-ignored), then answers questions until you type `quit`.

## 13. Multi-Server MCP and Remote Servers - Day13

Real agents connect to **several MCP servers at once**. A **tool registry** maps each tool name to the client of the server that provides it, so the planner sees one combined tool list. MCP servers can also run **remotely over HTTP** instead of as a local stdio subprocess.

```python
for server in servers:                              # "time-mcp-server/server.py", "http://127.0.0.1:8000/mcp"
    client = await connect(server)
    for tool in await discover_tools(client):
        tool_registry[tool.name] = {"client": client, "tool": tool}

# weather-mcp-server/server.py
mcp.run(transport="http", host="127.0.0.1", port=8000)
```

- [`Day13_P1`](./Day13/Day13_P1/) **Multi-server registry:** `mcp_manager.create_tool_registry()` over the time and weather servers (`registry_demo.py`).
- [`Day13_P2`](./Day13/Day13_P2/) **Agent loop over the registry:** the full plan-execute-observe loop over two stdio servers.
- [`Day13_P3`](./Day13/Day13_P3/) **Remote MCP over HTTP:** the weather server is served at `http://127.0.0.1:8000/mcp`.

Run P3 in two terminals:
```bash
uv run python weather-mcp-server/server.py     # terminal 1: HTTP server on :8000
uv run python runner.py                        # terminal 2: agent (launches the time server via stdio)
```

## 14. Multi-Agent Systems - Day14

One prompt that tries to research, code *and* review does each job worse than three **specialized agents**. A **manager** coordinates them through shared state.

```python
state = {"task": task, "research": "", "code": "", "review": "", "history": [], "revision_count": 0}
# manager prompt rules (abridged):
#   research empty → research;  research done, no code → coding;  code, no review → review
#   review "needs_revision" and revisions < MAX → coding;  "approved" or MAX reached → finish
```

- [`Day14_P1`](./Day14/Day14_P1/) **Fixed pipeline:** `manager.py` runs research → coding → review in a fixed order.
- [`Day14_P2`](./Day14/Day14_P2/) **Dynamic manager:** `state.py` plus an LLM that decides which worker runs next.
- [`Day14_P3`](./Day14/Day14_P3/) **Structured review:** the reviewer returns JSON (`status`/`feedback`), and the manager prompt gets strict routing rules.
- [`Day14_P4`](./Day14/Day14_P4/) **Revision limit:** `revision_count` and `MAX_REVISIONS = 3` stop endless code-review cycles.
- [`Day14_P5`](./Day14/Day14_P5/) **Parallel agents:** `parallel_demo.py` uses `asyncio.gather()` to run independent agents concurrently. The total time is the slowest agent's (≈4 s) instead of the sum of all of them (≈9 s).

Run: `uv run python main.py`; P5 also: `uv run python parallel_demo.py`

## 15. Production Hardening - Day15

The same multi-agent system, made robust:

| Concern | File | What it does |
|---|---|---|
| Configuration | `config.py` | Every setting comes from `.env` with a default (`MODEL`, `MAX_REVISIONS`, `MAX_RETRIES`, `TIMEOUT`). |
| Reliability | `llm.py` | `call_llm()` retries with linear backoff (2 s, 4 s, …) and a client timeout. |
| Observability | `logger.py` | Logs as `time \| level \| message` for each attempt and failure. |
| Input safety | `validation.py` | Rejects empty tasks and tasks over 5,000 characters. |
| Least privilege | `permissions.py` | Each worker has a tool allowlist; for example, `coding` can `read_file` but not `get_weather`. |

```python
TOOL_PERMISSIONS = {"research": {"get_weather", "read_file"}, "coding": {"read_file"}, ...}
def is_tool_allowed(agent_name, tool_name):
    return tool_name in TOOL_PERMISSIONS.get(agent_name, set())   # unknown agent → no tools
```

- [`Day15`](./Day15/Day15/): `agent/` holds the manager and state, `workers/` holds the research, coding and reviewer agents, and `main.py` is the entry point.

Run: `uv run python main.py`

---

# Part III: Agent Frameworks

## 16. Agents as Graphs with LangGraph - Day16

[LangGraph](https://langchain-ai.github.io/langgraph/) models the agent loop as a **state graph**. **Nodes** update a typed state, and **conditional edges** decide what runs next. A **checkpointer** saves the state between steps, which makes **human-in-the-loop** pauses possible.

```
START → planner ─FINISH→ answer → END
           │
           └─tool→ approval ─yes→ tool ─step < max→ planner
                       └─no→ answer        └─step ≥ max→ answer
```

```python
def approval_node(state):
    decision = interrupt({"message": "Approve this tool call?", "action": state["action"]})
    return {"approval": decision}

graph = builder.compile(checkpointer=InMemorySaver())
result = await graph.ainvoke(initial_state, {"configurable": {"thread_id": "demo-1"}})
result = await graph.ainvoke(Command(resume="yes"), config)   # continue after the pause
```

- [`Day16_P1`](./Day16/Day16_P1/) **Planner node:** a `StateGraph` with one planner node driven by MCP-discovered tools.
- [`Day16_P2`](./Day16/Day16_P2/) **Plan-execute-answer graph:** adds tool and answer nodes and an `action` state field.
- [`Day16_P3`](./Day16/Day16_P3/) **Human-in-the-loop:** `interrupt()` pauses before each tool call. The `__interrupt__` key in the result signals the pause, and `Command(resume=...)` continues with the user's yes/no answer.

Run: `uv run python main.py`

## 17. Multi-Agent Crews with CrewAI - Day17

[CrewAI](https://docs.crewai.com) is a higher-level take on Day14. You declare **Agents** (role, goal, backstory, tools), **Tasks**, and a **Crew** that runs them, either **sequentially** or **hierarchically**. In a hierarchical crew a manager LLM delegates the work.

```python
researcher = Agent(role="AI Researcher", goal="Research AI agent concepts...",
                   backstory="...", llm=LLM(model="ollama/qwen3:1.7b"), tools=[mcp_current_time])

crew = Crew(agents=[researcher, coder, reviewer],
            tasks=[research_task, coding_task, review_task],
            process=Process.hierarchical, manager_llm=manager_llm,
            memory=True, embedder={"provider": "ollama",
                                   "config": {"model_name": "nomic-embed-text"}})
result = crew.kickoff()
```

- [`Day17_P1`](./Day17/Day17_P1/) **CrewAI crew:** researcher, coder and reviewer with sequential tasks on a local Ollama model.
- [`Day17_P2`](./Day17/Day17_P2/) **MCP tools in a crew:** wraps the Day08 MCP client as a CrewAI `@tool` (`mcp_current_time`). `asyncio.run()` bridges CrewAI's synchronous tools to the asynchronous MCP client.
- [`Day17_P3`](./Day17/Day17_P3/) **Memory and hierarchy:** switches to `Process.hierarchical` with a `manager_llm`, and adds a second `memory_crew` that uses CrewAI's built-in memory (Ollama embeddings) to recall facts across tasks.

Run: `uv run python main.py`

---

## Lesson Index

Every lesson below was added in its own commit; the commit message explains the code changes in detail.

| # | Lesson | What it adds |
|---|---|---|
| 1 | [`Day01_P1`](./Day01/Day01_P1/) | First call to a local LLM (hello_ai.py) |
| 2 | [`Day01_P2`](./Day01/Day01_P2/) | Interactive assistant loop (assistant.py) |
| 3 | [`Day02_P1`](./Day02/Day02_P1/) | Conversation memory with a growing messages list |
| 4 | [`Day02_P2`](./Day02/Day02_P2/) | Print the raw conversation history after each turn |
| 5 | [`Day02_P3`](./Day02/Day02_P3/) | Human-readable conversation history |
| 6 | [`Day03_P1`](./Day03/Day03_P1/) | System prompt persona |
| 7 | [`Day03_P2`](./Day03/Day03_P2/) | Selectable personas at startup |
| 8 | [`Day04_P1`](./Day04/Day04_P1/) | First hand-written tools |
| 9 | [`Day04_P2`](./Day04/Day04_P2/) | Keyword-routed tools inside the chat loop |
| 10 | [`Day04_P3`](./Day04/Day04_P3/) | Extract routing into a tool manager |
| 11 | [`Day04_P4`](./Day04/Day04_P4/) | Wire the tool manager into the assistant |
| 12 | [`Day05_P1`](./Day05/Day05_P1/) | File-reading tool |
| 13 | [`Day05_P2`](./Day05/Day05_P2/) | summarize / explain commands over local files |
| 14 | [`Day05_P3`](./Day05/Day05_P3/) | Grounded question answering ("ask <file> <question>") |
| 15 | [`Day06_P1`](./Day06/Day06_P1/) | Load a knowledge folder |
| 16 | [`Day06_P2`](./Day06/Day06_P2/) | Naive keyword retrieval |
| 17 | [`Day06_P3`](./Day06/Day06_P3/) | Text embeddings with nomic-embed-text |
| 18 | [`Day06_P4`](./Day06/Day06_P4/) | Cosine similarity between embeddings |
| 19 | [`Day06_P5`](./Day06/Day06_P5/) | End-to-end RAG over the knowledge folder |
| 20 | [`Day07_P1`](./Day07/Day07_P1/) | LLM-based tool planner |
| 21 | [`Day07_P2`](./Day07/Day07_P2/) | Planner-driven assistant |
| 22 | [`Day08_P1`](./Day08/Day08_P1/) | MCP server and client with FastMCP |
| 23 | [`Day08_P2`](./Day08/Day08_P2/) | Agent skeleton on top of MCP |
| 24 | [`Day08_P3`](./Day08/Day08_P3/) | Planner built from discovered MCP tools |
| 25 | [`Day08_P4`](./Day08/Day08_P4/) | Execute the chosen MCP tool and answer naturally |
| 26 | [`Day09_P1`](./Day09/Day09_P1/) | Explicit agent state |
| 27 | [`Day09_P2`](./Day09/Day09_P2/) | Plan -> Execute -> Observe loop over MCP |
| 28 | [`Day10_P1`](./Day10/Day10_P1/) | LLM task decomposition |
| 29 | [`Day10_P2`](./Day10/Day10_P2/) | Task-driven agent loop |
| 30 | [`Day11_P1`](./Day11/Day11_P1/) | Persistent semantic memory in SQLite |
| 31 | [`Day11_P2`](./Day11/Day11_P2/) | Memory-augmented agent loop |
| 32 | [`Day11_P3`](./Day11/Day11_P3/) | LLM memory extraction - store only durable facts |
| 33 | [`Day11_P4`](./Day11/Day11_P4/) | Checkpoint of the memory-augmented agent |
| 34 | [`Day12`](./Day12/Day12/) | Advanced RAG pipeline (chunking, Chroma, hybrid search, reranking) |
| 35 | [`Day13_P1`](./Day13/Day13_P1/) | Multi-server MCP tool registry |
| 36 | [`Day13_P2`](./Day13/Day13_P2/) | Agent loop over the multi-server registry with JSON actions |
| 37 | [`Day13_P3`](./Day13/Day13_P3/) | Remote MCP server over HTTP |
| 38 | [`Day14_P1`](./Day14/Day14_P1/) | Multi-agent system with a fixed pipeline |
| 39 | [`Day14_P2`](./Day14/Day14_P2/) | LLM manager that chooses the next worker |
| 40 | [`Day14_P3`](./Day14/Day14_P3/) | Structured JSON review and a revision cycle |
| 41 | [`Day14_P4`](./Day14/Day14_P4/) | Cap the number of revisions |
| 42 | [`Day14_P5`](./Day14/Day14_P5/) | Parallel agents with asyncio.gather |
| 43 | [`Day15`](./Day15/Day15/) | Production hardening of the multi-agent system |
| 44 | [`Day16_P1`](./Day16/Day16_P1/) | First LangGraph StateGraph with a planner node |
| 45 | [`Day16_P2`](./Day16/Day16_P2/) | Plan -> tool -> answer graph with conditional edges |
| 46 | [`Day16_P3`](./Day16/Day16_P3/) | Human-in-the-loop approval with interrupt() |
| 47 | [`Day17_P1`](./Day17/Day17_P1/) | Multi-agent crew with CrewAI |
| 48 | [`Day17_P2`](./Day17/Day17_P2/) | Give a CrewAI agent an MCP tool |
| 49 | [`Day17_P3`](./Day17/Day17_P3/) | Hierarchical process and crew memory |

## Known Issues

Open problems in the course code, kept as-is so each lesson matches the course.

- **Model names vary:** the `.env` template uses `qwen3:4b-gpu`, but several days hard-code `qwen3:1.7b` or `qwen3:4b` (see [Setup](#setup)), so pull all of them.
- **Day09_P2 `test_state.py`:** it imports `from Day09.agent.state import *`, a package path from the course's original layout, so it fails when run from `Day09_P2/agent`. `Day09_P1/agent/test_state.py` is the working version.
- **Day15 `main.py` prints nothing after the run:** a `return` sits at function level right after the `try/except`, so the research, code and review sections and the permission demo are skipped. Indenting the `return` into the `except` block fixes it.

## Useful Links

- [Ollama](https://ollama.com) · [OpenAI-compatible API in Ollama](https://github.com/ollama/ollama/blob/main/docs/openai.md)
- [Model Context Protocol](https://modelcontextprotocol.io) · [FastMCP](https://gofastmcp.com)
- [LangChain](https://python.langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [CrewAI](https://docs.crewai.com)
- [Chroma](https://docs.trychroma.com) · [FAISS](https://github.com/facebookresearch/faiss)
- [FastAPI](https://fastapi.tiangolo.com) · [Server-Sent Events (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
