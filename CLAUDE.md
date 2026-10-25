# CLAUDE.md — AI Assistant Guide for uk-agents

## Project Overview

**uk-agents** is Uzair Kamal's (UK) personal repository of AI agents. There will be many agents serving unrelated purposes (property search, stocks, email, WhatsApp, expenses, tax filing, chat, and more to come). Every agent extends the shared `BaseAgent` in `core/` (Claude LLM, LangGraph pipeline, optional RAG over Pinecone, MCP tools) and is registered in `agents.json`. User-facing clients — REST API, Streamlit web UI, Telegram bot and a scheduled cron job — share the `OrchestratorAgent`, whose router plans which agent(s) handle each request; agents that need to run on their own have standalone entrypoints (`agents/<name>/main.py`). Do not assume a new agent is property-related.

`README.md` is the user-facing guide; `docs/expenditure-tracker.md` and `docs/fbr-tax-filing.md` cover those two agents in depth. `docs/superpowers/` holds dated design notes — historical, not kept up to date.

---

## Repository Structure

```
uk-agents/
├── .env / .env-template        # Runtime configuration (project root — always here)
├── agents.json                 # Agent registry (orchestrator reads it at startup)
├── mcp.json                    # MCP server registry
├── pyproject.toml / uv.lock    # Dependencies (uv); pytest config (testpaths)
├── logging_config.py           # setup_logging() → terminal + logs/agent.log (rotating)
├── core/src/                   # Shared infrastructure
│   ├── base_agent.py           # BaseAgent ABC (Template Method hooks) + NullRagContextManager
│   ├── aio.py                  # BackgroundLoop — a dedicated event loop on its own thread
│   ├── output_guard.py         # scrub() / StreamScrubber — strips tool-call markup from answers
│   ├── utils.py                # load_prompt(), run_service()
│   ├── config/config.py        # Config — every setting, loaded from .env
│   ├── model/
│   │   ├── llm_model_graph.py  # Per-agent LangGraph pipeline: retrieve → generate (tool loop)
│   │   ├── llm_factory.py      # create_llm(): ChatAnthropic or ChatClaudeCode (ANTHROPIC_API)
│   │   ├── claude_code_chat.py # ChatClaudeCode — LangChain chat model over the local claude CLI
│   │   ├── session_manager.py  # HistoryStore: SessionManager (SQLite) / InMemoryHistoryStore
│   │   ├── mcp_registry.py     # MCPRegistry on the official mcp SDK client
│   │   ├── mcp_tools.py        # Loads mcp.json → shared `_mcp` registry
│   │   ├── embedder.py, rag_context_manager.py, context_builder.py   # RAG
│   ├── persistence/            # VectorStore ABC, factory, PineconeStore
│   └── speech/transcriber.py   # Local faster-whisper (Telegram voice notes)
├── agents/
│   ├── conversational/         # Greetings, small talk; lists the other agents
│   ├── property/               # RAG property search + ingestion (housing/, gatherers/, scraping/)
│   ├── finance/                # Stocks via the `stocks` MCP server
│   ├── whatsapp/               # WhatsAppAgent + standalone group listener (main.py, Green API)
│   ├── mails/                  # MailAgent + standalone IMAP digest monitor (main.py)
│   ├── expenditure/            # Expense tracker: SQLite ledger + in-process LangChain tools
│   └── fbr_tax/                # FBR IRIS filing: Claude Code session + Chrome (runner.py)
├── orchestrator/
│   ├── agent_interface.py      # OrchestratorAgent — the facade clients use
│   ├── graph.py                # MultiAgentGraph — router → agents → synthesiser (+ confirmation)
│   ├── state.py                # GraphState, MAX_AGENT_TURNS
│   ├── agent_registry.py       # agents.json → AgentEntry list
│   ├── conversation.py         # ConversationMemory — recent turns for the router
│   ├── tracing.py              # LangSmith tagging helpers
│   ├── nodes/                  # router, agent, synthesiser, confirmation
│   └── prompts/                # router.txt, synthesiser.txt, confirmation.txt
├── clients/                    # base.py (BaseClient) + rest/, streamlit/, telegram/, cron/
└── logs/agent.log              # Unified runtime log
```

---

## Key Architecture

### Request Flow

```
Client (REST / Streamlit→REST / Telegram / cron)
  │
  ▼
OrchestratorAgent.aask() / ask() / astream()
  │
  ▼
MultiAgentGraph (on its own BackgroundLoop; SQLite checkpointer)
  ├── router        ONE structured LLM call: clean the message + ordered plan of agents
  │                 (sees the session's last 3 turns from ConversationMemory)
  ├── agent nodes   each planned agent's aask(), in order — no LLM call between agents
  ├── synthesiser   one output → passthrough; several → LLM merge
  └── confirmation  an agent left a pending item → interrupt() until the next message
  │
  ▼
Answer (string or tokens) — every agent output and the merge pass through output_guard
```

### Per-agent pipeline (`core/src/model/llm_model_graph.py`)

`LlmModelGraph` is a two-node `StateGraph`:

| Node | Does |
|---|---|
| `retrieve` | `rag_context_manager.get_context(prompt)` — vector search for RAG agents, `""` for `NullRagContextManager` |
| `generate` | system prompt + context + session history + prompt → LLM; runs tool calls in a loop (max `MAX_TOOL_CALLS = 5`) |

Public methods: `ask()`, `aask()` (async), `ask_stream()`, `close()`. `ask`/`aask` and each tool call are `@traceable` (`agent.<name>`, `tool.<name>`). History comes from the `HistoryStore` (`SESSION_DB_FILE`), last `MAX_HISTORY_TURNS * 2` messages.

### BaseAgent (`core/src/base_agent.py`)

Subclasses implement three hooks — `get_system_prompt()`, `get_rag_context_manager()`, `get_mcp_tools()` (`None` = every MCP tool, `[]` = none, or a list). They inherit `ask()`, `aask()`, `close()` and the pending-confirmation hooks `pending_summary()` / `confirm_pending()` / `discard_pending()` (no-ops unless overridden). An agent that answers some other way (e.g. `FbrTaxAgent` runs a Claude Code session) overrides `ask()` **and** `aask()`; if only `ask()` is overridden, `BaseAgent.aask()` runs that `ask()` in a worker thread.

### Client Layer (`clients/`)

All clients implement `BaseClient` (`start()` blocks until stopped, `stop()`, context manager).

| Client | Backend call | Notes |
|---|---|---|
| REST (`:8000`) | `aask` / `astream` | `session_id` param continues a conversation; SSE streaming |
| Streamlit (`:8501`) | HTTP → REST `/ask` | Needs the REST API running; no session id, so no memory between messages |
| Telegram | `aask(..., on_route=…)` | Per-chat session; voice notes (faster-whisper); shows "🤖 <Agent> is working…" |
| Cron | `ask` | Sends `CRON_SEARCH_PROMPT` every 30 minutes |

### Logging (`logging_config.py`)

`setup_logging()` is called once at the top of each entrypoint. It writes to the terminal and `logs/agent.log` (rotating) at INFO level.

---

## Configuration

All settings live in `.env` at the **project root**; `core/src/config/config.py` loads it by absolute path. Path settings resolve relative values from the project root.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | LLM provider (`anthropic` only) |
| `LLM_MODEL_NAME` | `claude-opus-5-5` | Claude model ID (per-agent override: `"llm_model"` in `agents.json`) |
| `LLM_EFFORT` | `medium` | Thinking depth: `low`, `medium`, `high`, `xhigh`, `max` |
| `LLM_MAX_TOKENS` | `16000` | Max output tokens per LLM call |
| `ANTHROPIC_API` | `true` | `true` → Anthropic API (`ANTHROPIC_API_KEY`); `false` → local Claude Code CLI and its login |
| `ANTHROPIC_API_KEY` | — | Needed when `ANTHROPIC_API=true` |
| `STORE_TYPE` | `pinecone` | Vector store backend (`pinecone` only) |
| `SENTENCE_TRANSFORMER_MODEL` | — | Embedding model, e.g. `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` / `PINECONE_ENVIRONMENT` | — | Pinecone credentials and index (index auto-created: cosine, AWS us-east-1) |
| `PROMPT_FILE` | `agents/property/prompts/System_Prompt.txt` | Property agent prompt (also `LlmModelGraph`'s fallback prompt) |
| `URLS_FILE` | `agents/property/urls.txt` | URL list for `agents.property.gatherers.data_collector` (create it) |
| `MAX_HISTORY_TURNS` | `10` | Agent history window (turns) |
| `SESSION_DB_FILE` | `sessions.db` | Agent chat history, the router's conversation memory, FBR session ids |
| `CHECKPOINT_DB_FILE` | `checkpoints.db` | Orchestrator checkpointer — only conversations paused for a confirmation |
| `ROUTER_PROMPT_FILE` / `SYNTHESISER_PROMPT_FILE` / `CONFIRMATION_PROMPT_FILE` | `orchestrator/prompts/*.txt` | Orchestrator prompts |
| `AGENTS_FILE` / `MCP_FILE` | `agents.json` / `mcp.json` | Registries |
| `TELEGRAM_BOT_TOKEN` | — | From @BotFather |
| `WHISPER_MODEL` | `small` | faster-whisper size for voice notes |
| `API_KEY` | _(empty — auth off)_ | REST API key (`X-API-Key` header) |
| `CRON_SEARCH_PROMPT` | 2-bed Poland < 1000 | Cron job prompt |
| `EXPENSES_DIR` | per-user data dir | Folder for `expenses.db`. Empty → `~/Library/Application Support/uk-agents/expenses` (macOS), `~/.local/share/uk-agents/expenses` (Linux, honours `XDG_DATA_HOME`), `%APPDATA%\uk-agents\expenses` (Windows) |
| `FBR_TAX_DIR` | `~/work/fbr-tax-filing` | Project holding the `fbr-iris-filing` skill |
| `FBR_TAX_DATA_DIR` | per-user data dir `/fbr_tax` | The FBR agent's `filings.db` |
| `FBR_TAX_MAX_TURNS` | `200` | Tool calls one FBR step may take |
| `WHATSAPP_*` | — | Green API instance, group id, user name, away message, poll interval |
| `MAIL_*` | — | IMAP accounts (JSON), check/summary intervals, digest group, `MAIL_SEEN_DB` |
| `SBR_WEBDRIVER` | — | Selenium WebDriver URL, read directly by `agents/property/scraping/scrape_se.py` |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | `false` / — / `uk-agents` | LangSmith tracing |

`FINANCE_PROMPT_FILE` and `FINANCE_PINECONE_INDEX_NAME` live in `agents/finance/config.py`. `MEMORY_FILE`, `INTERACTION_FILE`, `REFORMULATION_PROMPT` and `RAG_K` exist in `Config` but nothing reads them (RAG uses k=5).

---

## Development Workflows

All commands run from the **project root**.

```bash
uv sync --group dev          # install (Python 3.13+)
cp .env-template .env        # then fill in real values

python -m clients.telegram.main     # Telegram bot
python -m clients.rest.main         # REST API, port 8000
python -m clients.streamlit.main    # Streamlit UI, port 8501 (start REST first)
python -m clients.cron.main         # cron job, every 30 min
python -m agents.whatsapp.main      # WhatsApp group listener (standalone)
python -m agents.mails.main         # mail digest monitor (standalone)
```

### Running Tests

```bash
uv run python -m pytest -q
uv run python -m pytest --cov=core --cov=orchestrator --cov=clients --cov=agents --cov-report=term-missing
```

`testpaths` in `pyproject.toml`: `core/tests`, `orchestrator/tests`, `clients/tests`, `agents` (each agent's `tests/`). No `.env` is needed — every setting has a test-safe default. CI (`.github/workflows/tests.yml`) runs `uv run python -m pytest -q` on every push and pull request.

External services (LLM, Pinecone, HTTP, Telegram, IMAP, Claude Code) are always mocked — no test makes a real network or model call. Local stand-ins are fine and preferred over mocks where cheap: real SQLite on `tmp_path`, real LangGraph wiring with scripted fake LLMs, real MCP servers (in-process `MCPServer` or `core/tests/model/mcp_test_server.py` over stdio). Test functions are named `test_<behaviour>`.

---

## REST API Reference

| Method | Path | Params | Response |
|---|---|---|---|
| GET | `/ask` | `prompt` (str), `stream` (bool, default `false`), `session_id` (str, optional — reuse it to continue a conversation; omitted → a fresh session per request) | `{"response": str, "session_id": str}`, or an SSE stream with the session id in the `X-Session-Id` header |

Streaming format (SSE): each chunk is sent as `data: {chunk}\n\n`. When `API_KEY` is set, every request needs `X-API-Key: <key>`.

---

## Code Conventions

- Python 3.13+, type hints throughout; class-based design; private methods prefixed with `_`.
- Settings always come from `Config` (or an agent's own config module) — never hardcode paths or keys.
- Packages are imported absolutely from the project root (`from core.src.config.config import Config`); inside a package, relative imports are fine (`from .ledger import ExpenseLedger`).
- Resource-owning classes provide `close()` and a context manager.
- Prompts live next to their code (`agents/<name>/prompts/`, `orchestrator/prompts/`). Never hardcode MCP tool names in a prompt — the model learns them from the tool schemas. No conditional business logic in prompt text; the router prompt's `{agents}` / `{today}` and the confirmation prompt's `{pending}` / `{today}` placeholders must stay.

### Adding a New Agent

1. `agents/<name>/agent.py`: subclass `BaseAgent`, implement the three hooks; prompt in `agents/<name>/prompts/System_Prompt.txt`.
2. Register it in `agents.json`: `name`, `class`, `description` (what the router routes on), `enabled`, `rag`, optional `display_name`, `summary` (shown by the conversational agent), `remarks` (Telegram waiting lines), `llm_model` / `llm_provider`.
3. Add tests under `agents/<name>/tests/`.
4. If it answers outside `LlmModelGraph`, override both `ask()` and `aask()`.
5. If it must run on its own, add `agents/<name>/main.py` (call `setup_logging()` first; `run_service()` in `core/src/utils.py`).

### Adding a New Client

Create `clients/<name>/main.py` (+ `__init__.py`), implement `BaseClient`, call `setup_logging()` first, use `OrchestratorAgent`, run with `python -m clients.<name>.main`.

### Adding an LLM Provider / Vector Store

- Provider: a branch in `core/src/model/llm_factory.py` returning a LangChain `BaseChatModel` that also supports `with_structured_output(..., method="json_schema")` (the orchestrator needs it).
- Vector store: implement `VectorStore` (`add_vectors`, `search`) under `core/src/persistence/` and register its `store_type` in `factory.py`.

---

## Common Gotchas

- **Always run from the project root** with `-m` (`python -m clients.telegram.main`), so `core`, `agents`, `orchestrator` and `clients` are importable.
- **Two Claude backends** (`ANTHROPIC_API`): `true` → `ChatAnthropic` with `ANTHROPIC_API_KEY` (billed per token); `false` → `ChatClaudeCode`, which drives the local `claude` CLI through the Claude Agent SDK with its logged-in subscription. In Claude Code mode the agent's tools are exposed as an in-process MCP server and Claude Code runs the tool loop itself, so responses never contain `tool_calls`; Claude Code's built-in tools and user/project settings are disabled. The FBR agent always uses the Claude Code login, whatever `ANTHROPIC_API` says.
- **Read `.text`, not `.content`**: Claude responses include thinking blocks, so `AIMessage.content` is a list.
- **Async first, sync wrappers**: the orchestrator graph runs on its own `BackgroundLoop` (`core/src/aio.py`) with an aiosqlite checkpointer. Clients with an event loop call `OrchestratorAgent.aask()` / `astream()`; sync callers (cron, scripts) use `ask()`. `AgentNode` awaits an agent's `aask()` (or runs a sync-only `ask()` in a thread); blocking work (SQLite, ledger, RAG, Claude Code sessions) runs via `asyncio.to_thread`. Messages on one session are processed one at a time; sessions run concurrently.
- **Override `ask()` → override `aask()` too.** The orchestrator calls `aask()`. An agent that only overrode `ask()` used to fall through to the plain LLM pipeline (that is how the FBR agent stopped opening Chrome); `BaseAgent.aask()` now falls back to an overridden `ask()`, but write the `aask()` explicitly.
- **Confirmations are LangGraph interrupts**: an agent that leaves a pending item (`pending_summary()` not `None`) makes the orchestrator pause in the `confirmation` node via `interrupt()`; the next message on that session resumes it with `Command(resume=...)` (`MultiAgentGraph._start`) and is classified confirm / cancel / correction / unrelated. Routing and classification use `llm.with_structured_output(Model, method="json_schema")` — test fakes for orchestrator LLMs must provide `with_structured_output`.
- **Conversation memory**: `ConversationMemory` stores each finished turn under the pseudo-agent `orchestrator` in `SESSION_DB_FILE` and gives the router the last 3 turns, as a human message before the user's message (never in the system prompt). That is how "yes" or "and for last month?" reach the right agent.
- **Output guard** (`core/src/output_guard.py`): answers never show tool-call markup (`<invoke>`, `<parameter>`, `<function_calls>`, thinking, system reminders). `AgentNode`, the synthesiser, the token stream, `ConversationMemory` and `GreenAPIClient.send_message` scrub every answer and log a warning naming the agent; an answer that was only markup becomes `WITHHELD_REPLY`. A new outbound path that bypasses the orchestrator must call `scrub()` (or `StreamScrubber`) itself. A leak usually means a model was told about a tool it cannot call.
- **MCP**: `MCPRegistry` keeps one persistent `mcp.Client` connection per server on its own background loop. In `mcp.json`, `"{python}"` in a command means the current interpreter, launched from the project root with the full environment; other stdio servers get a minimal environment plus the variables listed in `"env"`. `"timeout": <seconds>` (default 60) caps each call — a server that times out is disconnected and reconnects on the next call.
- **The expense tracker is SQLite + in-process tools** (`agents/expenditure/ledger.py`, `tools.py`), not an MCP server — deliberately (see `docs/expenditure-tracker.md`).
- **Anyone who can message the Telegram bot can use every agent**, including the expense tracker and FBR filing — there is no owner check.
- **Tests never stub `sys.modules` for project code**: patch the names a test uses (`patch("clients.telegram.main.OrchestratorAgent")`). `core/tests/conftest.py` stubs only heavy third-party ML packages (sentence-transformers, pinecone).
- **Pinned dependency**: `av<19` in `pyproject.toml` — faster-whisper 1.2.1 passes `av.open(metadata_errors=…)`, which PyAV 19 removed. Lift it when faster-whisper supports PyAV 19 (`core/tests/speech/test_transcriber.py` decodes a real voice note to catch this).
