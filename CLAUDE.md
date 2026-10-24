# CLAUDE.md — AI Assistant Guide for uk-agents

## Project Overview

**uk-agents** is Uzair Kamal's (UK) personal repository of AI agents. There will be many agents serving unrelated purposes (property search, stocks, email, WhatsApp, chat, and more to come). Every agent extends the shared `BaseAgent` in `core/` (Claude LLM, LangGraph pipeline, optional RAG over Pinecone, MCP tools) and is registered in `agents.json`. User-facing clients — REST API, Streamlit web UI, Telegram bot, and a scheduled cron job — share the `OrchestratorAgent`, whose supervisor routes each request to the right agent; agents that need to run on their own have standalone entrypoints (`agents/<name>/main.py`). Do not assume a new agent is property-related.

---

## Repository Structure

```
uk-agents/
├── .env                             # Runtime configuration (project root — always here)
├── pyproject.toml                   # Project metadata and dependencies
├── logging_config.py                # Central logging setup → logs/agent.log
├── README.md
├── CLAUDE.md
├── .gitignore
├── agentP/
│   ├── requirements.txt             # Pinned dependency lockfile
│   ├── urls.txt                     # URLs for web scraping
│   └── src/
│       ├── agent.py                 # LocalAgent — thin facade over LlmModelGraph
│       ├── config/
│       │   └── config.py            # Settings loaded from .env via absolute path
│       ├── model/
│       │   ├── llm_model_graph.py   # LangGraph RAG pipeline (reformulate→retrieve→generate)
│       │   ├── llm_factory.py       # LLM provider factory (Anthropic Claude)
│       │   ├── embedder.py          # SentenceTransformer embeddings + vector search
│       │   ├── rag_context_manager.py  # Retrieves and formats RAG context
│       │   ├── context_builder.py   # Formats property listing dicts as readable text
│       │   └── session_manager.py   # SQLite-backed session history
│       ├── persistence/
│       │   ├── vector_store.py      # Abstract base (add / search)
│       │   ├── factory.py           # Selects the vector store backend (Pinecone)
│       │   └── pinecone_store.py    # Cloud Pinecone implementation
│       ├── housing/
│       │   ├── housing_data_collector.py   # Streams CSV → embedding-ready text
│       │   └── housing_csv_reader.py       # CSV parsing utility
│       ├── gatherers/
│       │   ├── data_collector.py           # Scrapes listing URLs
│       │   └── pl_housing_data_collector.py # Polish housing data ingestion
│       ├── scraping/
│       │   ├── web_scraper.py       # HTTP fetch + HTML cleanup
│       │   ├── url_processor.py     # Regex-based listing extraction
│       │   ├── scrape_se.py         # Selenium-based scraper
│       │   └── utils.py             # Converts listings to LangChain Documents
│       └── prompts/
│           ├── System_Prompt.txt        # Agent persona and response style
│           ├── reformulated_prompt.txt  # Query rewrite template ({user_prompt} placeholder)
│           └── interaction.json         # CLI interaction strings
├── clients/
│   ├── base.py                      # BaseClient ABC (start / stop / context manager)
│   ├── rest/main.py                 # FastAPI REST API (port 8000)
│   ├── streamlit/main.py            # Streamlit web UI (port 8501)
│   ├── telegram/main.py             # Telegram bot (long-polling)
│   ├── cron/main.py                 # Scheduled search (every 30 min)
│   └── tests/                       # Client layer tests
└── logs/
    └── agent.log                    # Unified runtime log (all clients write here)
```

---

## Key Architecture

### Request Flow

```
User (any client)
  │
  ▼
LocalAgent.ask(prompt, stream)
  │
  ▼
LlmModelGraph
  ├── [reformulate]  user_prompt → reformulated_question   (LLM)
  ├── [retrieve]     reformulated_question → context        (vector search)
  └── [generate]     question + context + history → answer  (LLM)
  │
  ▼
Response (blocking string or token stream)
```

### LangGraph Pipeline (`agentP/src/model/llm_model_graph.py`)

`LlmModelGraph` is a compiled `StateGraph` with three nodes:

| Node | Input | Output |
|---|---|---|
| `reformulate` | `user_prompt` | `reformulated_question` |
| `retrieve` | `reformulated_question` | `context` (formatted listings) |
| `generate` | `reformulated_question` + `context` + `history` | `answer` |

**Public methods:**
- `ask(user_query, session_id=None) -> str` — blocking, returns full answer
- `ask_stream(user_query, session_id=None)` — generator, yields tokens from the `generate` node
- `close()` — no-op; present for `LocalAgent` lifecycle compatibility

Both methods are decorated with `@traceable` and produce LangSmith traces.

### Client Layer (`clients/`)

All clients implement `BaseClient`:

```python
class BaseClient(ABC):
    @abstractmethod
    def start(self) -> None: ...  # blocks until stopped
    def stop(self) -> None: ...   # graceful shutdown (override if needed)
    # + context manager (__enter__ / __exit__)
```

### Vector Store Backends (`agentP/src/persistence/`)

- `VectorStore` — abstract base with `add_vectors(vectors)` and `search(query_vector, k)`.
- `PineconeStore` — cloud store; auto-creates index (cosine, serverless AWS us-east-1) if absent.
- `create_vector_store(store_type, dim, config)` in `factory.py` selects the backend at runtime.

### Embeddings (`agentP/src/model/embedder.py`)

- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional float32 vectors)
- Key methods:
  - `embed_documents_to_vectors(docs: dict) -> list[dict]`
  - `embed_texts(texts: list[str]) -> np.ndarray`
  - `embed_query(query: str) -> np.ndarray`
  - `search(query: str, k: int = 5)`
  - `get_store(dim=384) -> VectorStore` — lazy-initializes via factory
  - `save_vectors_in_store(vectors)`

### Logging (`logging_config.py`)

Central `setup_logging()` function — called once at the top of each client entrypoint. Writes to both terminal and `logs/agent.log` at INFO level. All `getLogger(__name__)` calls in the process inherit from it automatically.

```python
from logging_config import setup_logging
setup_logging()
```

---

## Configuration

All settings live in `.env` at the **project root**. `agentP/src/config/config.py` loads it via an absolute path derived from `__file__` — works regardless of where the process is launched from.

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | Vector store backend (`pinecone` only currently) |
| `LLM_MODEL_NAME` | `claude-opus-5-5` | Claude model ID |
| `LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` only currently) |
| `LLM_EFFORT` | `medium` | Thinking depth / token spend: `low`, `medium`, `high`, `xhigh`, `max` |
| `LLM_MAX_TOKENS` | `16000` | Max output tokens per LLM call |
| `ANTHROPIC_API` | `true` | `true` → Anthropic API (needs `ANTHROPIC_API_KEY` + credits); `false` → local Claude Code CLI using the account it is logged in with |
| `ANTHROPIC_API_KEY` | _(required when `ANTHROPIC_API=true`)_ | Anthropic API key from console.anthropic.com |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `URLS_FILE` | `agentP/urls.txt` | URLs for web scraping |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_ENVIRONMENT` | `gcpstart` | Pinecone environment |
| `PINECONE_INDEX_NAME` | `course-ai` | Pinecone index name |
| `PROMPT_FILE` | `agentP/src/prompts/System_Prompt.txt` | Agent system prompt path |
| `INTERACTION_FILE` | `agentP/src/prompts/interaction.json` | CLI interaction strings path |
| `REFORMULATION_PROMPT` | `agentP/src/prompts/reformulated_prompt.txt` | Query rewrite template path |
| `TELEGRAM_BOT_TOKEN` | _(required for Telegram)_ | Token from @BotFather |
| `TELEGRAM_OWNER_ID` | _(empty)_ | Owner's Telegram user id or `@username` (comma-separate several; a channel id works for posts made as that channel). Send `/whoami` to the bot to see yours. Required for agents marked `"owner_only": true` in `agents.json` (the expense tracker) |
| `WHISPER_MODEL` | `small` | Local faster-whisper model size for voice-note transcription |
| `EXPENSES_DIR` | _(per-user data dir)_ | Expense ledger directory: monthly files, `inputs.log` (every message the expense agent received), history and pending item. Empty → `~/.local/share/uk-agents/expenses` (Linux, honours `XDG_DATA_HOME`), `~/Library/Application Support/uk-agents/expenses` (macOS), `%APPDATA%\uk-agents\expenses` (Windows). Relative values resolve from the project root |
| `NORMALISER_PROMPT_FILE` | `orchestrator/prompts/normaliser.txt` | Orchestrator normaliser prompt path |
| `CONFIRMATION_PROMPT_FILE` | `orchestrator/prompts/confirmation.txt` | Prompt that classifies the reply to a pending item (confirm / cancel / correction / unrelated) |
| `CHECKPOINT_DB_FILE` | `checkpoints.db` | SQLite checkpointer for the orchestrator graph; holds only conversations paused for a confirmation (deleted once a run finishes). Relative → from the project root |
| `API_KEY` | _(empty — auth disabled)_ | REST API key; set to enforce `X-API-Key` header |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000)_ | Prompt used by the scheduled cron job |
| `SESSION_DB_FILE` | `sessions.db` | SQLite file for persistent chat session history |
| `SBR_WEBDRIVER` | _(empty)_ | Selenium WebDriver URL for `scrape_se.py` |
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `uk-agents` | LangSmith project name |

---

## Development Workflows

All commands are run from the **project root** (`uk-agents/`).

### Setup

```bash
uv venv && uv sync
# or: pip install -r agentP/requirements.txt

# Edit .env at the project root with real values (ANTHROPIC_API_KEY, PINECONE_API_KEY, …)
```

### Running Clients

```bash
# Telegram bot
python -m clients.telegram.main

# REST API (port 8000)
python -m clients.rest.main

# Streamlit UI (port 8501)
python -m clients.streamlit.main

# Cron job (fires every 30 min)
python -m clients.cron.main

# CLI (interactive)
python -m agentP.src.agent
```

### Running Tests

```bash
# All tests (testpaths in pyproject.toml: core, orchestrator, clients, agents/*/tests)
uv run python -m pytest -q

# With coverage
uv run python -m pytest --cov=core --cov=orchestrator --cov=clients --cov=agents --cov-report=term-missing
```

No `.env` is needed: every setting has a test-safe default. CI (`.github/workflows/tests.yml`) runs the same command on every push and pull request.

All external dependencies (LLM, vector store, HTTP, Telegram) must be mocked. Do not write tests that make real network or model calls. Local stand-ins are fine — e.g. a Python script acting as an MCP server over stdio.

---

## REST API Reference

| Method | Path | Params | Response |
|---|---|---|---|
| GET | `/ask` | `prompt` (str), `stream` (bool, default `false`), `session_id` (str, optional — reuse it to continue a conversation; omitted → fresh session per request) | `{"response": str, "session_id": str}` or SSE stream (session id in the `X-Session-Id` header) |

**Streaming format** (SSE): Each chunk is sent as `data: {chunk}\n\n`.

**Authentication**: When `API_KEY` is set in `.env`, all requests must include `X-API-Key: <key>`. Leave `API_KEY` empty to disable (default).

---

## Code Conventions

### Python Style

- Python 3.13+ with type hints throughout.
- Class-based design; no top-level procedural logic in modules.
- Private methods prefixed with `_`.
- Config values always accessed through `Config` in `config.py` — never hardcode paths or keys.
- All imports within `agentP/src/` use **relative imports** (`from .model.x import Y`, `from ..config.config import Config`).
- Context managers (`__enter__`/`__exit__`/`close()`) used for resource-owning classes.

### Adding a New Client

1. Create `clients/<name>/main.py`.
2. Add `clients/<name>/__init__.py`.
3. Implement `BaseClient` from `clients/base.py`.
4. Call `setup_logging()` from `logging_config` at the top of the entrypoint.
5. Run with `python -m clients.<name>.main` from the project root.

### Adding a New LLM Provider

1. Add a branch in `agentP/src/model/llm_factory.py` matching the provider name string.
2. Read required keys from the `Config` object.
3. Return a LangChain-compatible `BaseLLM` or `BaseChatModel`.

### Adding a New Vector Store

1. Create a file under `agentP/src/persistence/`.
2. Implement `VectorStore` abstract class — `add_vectors(vectors)` and `search(query_vector, k)`.
3. Register the `store_type` string in `agentP/src/persistence/factory.py`.

### Modifying Prompts

- Prompts live in `agentP/src/prompts/`.
- `reformulated_prompt.txt` must retain the `{user_prompt}` placeholder.
- Do not embed conditional business logic inside prompt strings.

---

## Important File Locations

| Purpose | Path |
|---|---|
| Runtime environment config | `.env` (project root) |
| Central logging setup | `logging_config.py` (project root) |
| Project dependencies | `pyproject.toml` (project root) |
| Agent system prompt | `agentP/src/prompts/System_Prompt.txt` |
| Query reformulation prompt | `agentP/src/prompts/reformulated_prompt.txt` |
| CLI interaction messages | `agentP/src/prompts/interaction.json` |
| URLs for web scraping | `agentP/urls.txt` |
| Runtime log | `logs/agent.log` |
| Agent registry (orchestrator) | `agents.json` — `"owner_only": true` means only the bot owner (`TELEGRAM_OWNER_ID`) may use the agent; others are refused, and a non-owner message in a chat paused on its confirmation runs as a separate turn |
| MCP server registry | `mcp.json` — `"{python}"` in a command = current interpreter, launched from the project root; `"owner_only": true` keeps the server out of the default tool set (`langchain_tools()` with no `server_name`), so only an agent that asks for it by name gets its tools; `"timeout": <seconds>` (default 60) caps each stdio call — a server that doesn't answer in time is killed and restarted on the next call |

---

## Common Gotchas

- **Always run from the project root**: `python -m clients.telegram.main`, not `python clients/telegram/main.py`. The `-m` flag sets `sys.path` to the project root, making `agentP` and `clients` importable.
- **Relative imports inside `agentP/src/`**: Use `from .model.x import Y` (same package) or `from ..config.config import Config` (parent package). Bare imports like `from config.config import Config` will fail when running from the project root.
- **`.env` is at the project root**: `config.py` loads it via an absolute path — no need to `cd` anywhere first.
- **Pinecone required**: `STORE_TYPE=pinecone` needs a valid `PINECONE_API_KEY` and network access.
- **Two Claude backends** (`ANTHROPIC_API` flag): `true` uses the Anthropic API with `ANTHROPIC_API_KEY` (billed per token); `false` uses `ChatClaudeCode` (`core/src/model/claude_code_chat.py`), which drives the local `claude` CLI via the Claude Agent SDK with its logged-in subscription. In Claude Code mode the agent's bound tools are exposed as an in-process MCP server and Claude Code runs the tool loop itself, so responses never contain `tool_calls`; built-in Claude Code tools and user settings are disabled.
- **Read `.text`, not `.content`**: Claude responses include thinking blocks, so `AIMessage.content` is a list. Use `response.text` to get the answer string.
- **Confirmations are LangGraph interrupts**: an agent that leaves a pending item (`BaseAgent.pending_summary`) makes the orchestrator pause in the `confirmation` node via `interrupt()`; the next message on that session resumes it with `Command(resume=...)` (`MultiAgentGraph._start`). Routing, cleanup and classification use structured output (`llm.with_structured_output(Model, method="json_schema")`), which `ChatClaudeCode` also implements — test fakes for orchestrator LLMs must provide `with_structured_output`.
- **Session persistence**: Chat history is stored in `SESSION_DB_FILE` (SQLite). The file is created automatically on first run at the project root.
