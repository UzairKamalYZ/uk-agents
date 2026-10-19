# CLAUDE.md — AI Assistant Guide for uk-agents

## Project Overview

**uk-agents** is Uzair Kamal's (UK) personal repository of AI agents. It holds many agents serving unrelated purposes (property search, stocks, email, WhatsApp, small talk, expense tracking, and more to come). Do not assume a new agent is property-related.

- Every agent extends the shared `BaseAgent` in `core/` (Claude LLM, a small LangGraph pipeline, optional RAG over Pinecone, MCP tools) and is registered in `agents.json`.
- User-facing clients (Telegram bot, REST API, Streamlit UI, cron job) share one `OrchestratorAgent`. Its graph cleans each message, routes it to the right agent, and formats the answer.
- Agents that must run on their own (WhatsApp listener, mail monitor) have standalone entrypoints in `agents/<name>/main.py`.

See `README.md` for the user-facing tour and `docs/expenditure-tracker.md` for the expense tracker's design and roadmap.

---

## Repository Structure

```
uk-agents/
├── .env / .env-template        # Runtime configuration (project root — always here)
├── pyproject.toml              # Dependencies + pytest testpaths
├── agents.json                 # Agent registry read by the orchestrator
├── mcp.json                    # MCP server registry (tools for agents)
├── logging_config.py           # setup_logging() → rotating logs/agent.log
├── core/                       # Shared framework — no agent-specific code
│   ├── src/base_agent.py       # BaseAgent (Template Method) + pending-item hooks
│   ├── src/config/config.py    # Config: every setting, loaded from <root>/.env
│   ├── src/model/
│   │   ├── llm_factory.py      # create_llm(provider, model) → ChatAnthropic | ChatClaudeCode
│   │   ├── claude_code_chat.py # ChatClaudeCode — drives the local `claude` CLI (Agent SDK)
│   │   ├── llm_model_graph.py  # Per-agent LangGraph: retrieve → generate (+ tool loop)
│   │   ├── mcp_registry.py     # stdio/HTTP MCP clients → LangChain StructuredTools
│   │   ├── mcp_tools.py        # _mcp registry singleton built from mcp.json
│   │   ├── embedder.py, rag_context_manager.py, context_builder.py
│   │   └── session_manager.py  # SessionManager — SQLite chat history per (agent, session)
│   ├── src/persistence/        # VectorStore ABC, factory, PineconeStore
│   ├── src/speech/transcriber.py  # Local faster-whisper speech-to-text
│   └── tests/
├── orchestrator/
│   ├── agent_interface.py      # OrchestratorAgent — the facade every client uses
│   ├── graph.py                # MultiAgentGraph: builds/compiles the StateGraph, on_route
│   ├── agent_registry.py       # load_agents(): agents.json → AgentEntry list
│   ├── state.py                # GraphState TypedDict, MAX_AGENT_TURNS
│   ├── tracing.py              # record_route / tag_current_trace (logs + LangSmith)
│   ├── conversation.py         # ConversationMemory — last turns per session for follow-ups
│   ├── nodes/                  # normaliser, supervisor, agent, synthesiser, _json helper
│   ├── prompts/                # normaliser.txt, supervisor.txt, synthesiser.txt
│   └── tests/
├── agents/
│   ├── conversational/         # small talk
│   ├── property/               # property search (RAG) + scraping/ingestion
│   ├── finance/                # stocks (Yahoo Finance MCP)
│   ├── whatsapp/               # WhatsApp agent + standalone group listener
│   ├── mails/                  # mail agent + standalone IMAP monitor
│   └── expenditure/            # expense tracker: SQLite ledger + in-process tools
│       └── (each agent: agent.py, prompts/System_Prompt.txt, optional tests/, main.py)
├── clients/
│   ├── base.py                 # BaseClient ABC (start / stop / context manager)
│   ├── telegram/main.py        # Telegram bot: text + voice notes, agent announcement
│   ├── rest/main.py            # FastAPI (port 8000)
│   ├── streamlit/main.py       # Streamlit UI (port 8501)
│   ├── cron/main.py            # Scheduled prompt every 30 min
│   └── tests/
├── docs/                       # Design docs (docs/superpowers/ = specs and plans)
└── logs/                       # agent.log (git-ignored)
```

`agentP/` is a leftover from before the `core/` + `agents/` split, and only an empty test package remains. Don't add code there.

---

## Key Architecture

### Request flow

```
Client (Telegram / REST / Streamlit / cron)
  │   Telegram voice note → Transcriber (local faster-whisper) → text
  ▼
OrchestratorAgent.ask(prompt, stream=False, session_id=..., *, is_owner=False, on_route=None)
  ▼
MultiAgentGraph (orchestrator/graph.py)
  START → normaliser ─┬─► supervisor ⇄ agent_X (loop, ≤ MAX_AGENT_TURNS=6) → synthesiser → END
                      ├─► agent_X       (correction to a pending item, "direct")
                      └─► synthesiser   (pending item confirmed / discarded / re-asked)
```

| Node | Job |
|---|---|
| `normaliser` | One LLM call. It cleans the text (transcription fixes, "ninety euros" → `€90`, relative dates → `YYYY-MM-DD`; no currency means euros, no date means today). If an agent has a pending item, it classifies the reply as `confirm`, `cancel`, `correction` or `unrelated`. Confirm and cancel are resolved in code. |
| `supervisor` | LLM returns `{"next": "<agent>", "reason": "..."}` or `{"next": "FINISH"}`. The result is validated and fuzzy-matched against the agent names. |
| `agent_X` (`AgentNode`) | Calls `agent.ask(user_input, session_id=...)`. Refuses `owner_only` agents when `is_owner` is false. |
| `synthesiser` | One output: passed through unchanged, with no LLM call. Several outputs: an LLM merge. The `Last month:` / `This month:` lines are always kept word for word. |

`GraphState` fields: `user_input` (cleaned), `raw_input`, `session_id`, `request_id`, `history`, `is_owner`, `messages`, `agent_outputs`, `next`, `direct`, `turns`.

### Follow-ups (`orchestrator/conversation.py`)

`MultiAgentGraph` keeps the last `HISTORY_TURNS` (3) user/assistant turns per session in `sessions.db` (pseudo-agent `"orchestrator"`, answers labelled with the agent name). `_initial_state` loads them into `state["history"]`, and `ask()` records each finished turn.
- The normaliser rewrites follow-ups ("yes" after "Want the breakdown?") into self-contained requests.
- Supervisor rule 0 routes a continuation to the agent that asked.
- Without this, a reply to the bot's own question is routed as a brand-new conversation.

Pass `MultiAgentGraph(history_store=...)` to inject a store; tests use an in-memory fake.

### Agent announcement (`on_route`)

`OrchestratorAgent.ask(..., on_route=cb)` calls `cb(AgentEntry)` as soon as the supervisor (or the normaliser) picks an agent, before that agent runs. Telegram uses it to change "🔀 Finding the right agent…" into "🤖 *<display_name>* is working on your query…" plus a random line from the entry's `remarks`. The text comes from `agents.json`. A failing callback is logged and never breaks the request. Only the blocking path supports `on_route`, not `stream=True`.

### Routing logs and LangSmith tracing

- Each turn gets a `request_id` (`tracing.new_request_id()`).
- `record_route()` logs `[route] request=<id> <node> → <agent> (<reason>)`. It also emits a LangSmith run named `route → <agent>`, tagged `agent:<name>` and `routed_by:<node>`.
- `OrchestratorAgent.ask` is `@traceable("orchestrator.request")` and is tagged `agent:<name>` for each agent used. The graph runs with `run_name="orchestrator.graph"` and `request_id`/`session_id` metadata (see `MultiAgentGraph._run_config`).
- With `LANGCHAIN_TRACING_V2=false`, only the log lines are written.

### BaseAgent (`core/src/base_agent.py`)

Subclasses implement three hooks. `BaseAgent.__init__` calls them, then builds the LLM and an `LlmModelGraph`.

```python
def get_system_prompt(self) -> str: ...            # usually load_prompt(<agent>/prompts/System_Prompt.txt)
def get_rag_context_manager(self): ...             # RagContextManager(Embedder()) or NullRagContextManager()
def get_mcp_tools(self) -> list | None: ...        # _mcp.langchain_tools(server_name="x"); [] = no tools
```

Optional hooks for agents that keep a "pending item awaiting yes/no" (the normaliser calls them): `pending_summary(session_id) -> str | None`, `confirm_pending(session_id) -> str`, `discard_pending(session_id) -> None`. The defaults do nothing.

`LlmModelGraph` (per agent) is `START → retrieve → generate → END`. `generate` runs the tool-call loop. `ask()` blocks and `ask_stream()` yields tokens. Both are `@traceable`.

### MCP tools

- `mcp.json` declares servers: stdio `command` or HTTP `url`.
  - `"{python}"` in a command means the current interpreter, launched from the project root.
  - `"owner_only": true` keeps that server out of the default tool set.
- `_mcp.langchain_tools(server_name="x")` returns that server's tools, or `[]` (with a warning) if the server is unknown. `langchain_tools()` with no name returns every server **except** owner-only ones.
- Prompts must never hardcode tool names. Tools are discovered, and prompts describe intents.

### Expense tracker (`agents/expenditure/`)

- **Storage.** SQLite `expenses.db` in `EXPENSES_DIR`, holding:
  - `expenses`: amounts in integer cents; undo sets `deleted_at` and never deletes the row;
  - `inputs`: every message received;
  - `pending`: one slot;
  - `monthly_totals`: a view.
- **Code layout.** `ExpenseLedger` holds all the SQL. `replies.py` holds all the user-facing text. `tools.py` builds plain LangChain tools that run in-process and are bound only to this agent (no MCP, no subprocess). The tools are `add_expense`, `propose_expense`, `undo_last_expense`, `get_expense_totals`, `get_expense_summary` and `get_expense_breakdown`.
- **The agent.** `ExpenditureAgent.ask` puts `Today is …` in front of the prompt. If the tools didn't load, it refuses instead of letting Claude make up a save.
- **Rules.** Totals are always computed in SQL or Python, and Claude returns the tool's text word for word.
- Full design and roadmap: `docs/expenditure-tracker.md`.

### Clients (`clients/`)

All clients implement `BaseClient` (`start()` blocks; `stop()`; context manager) and call `setup_logging()` first.
- **Telegram** supports text and voice notes. For voice it shows "🎤 You said: …" and loads the Whisper model at startup. It passes `is_owner`, based on `TELEGRAM_OWNER_ID`.
- **REST** has a single `GET /ask?prompt=…&stream=false` endpoint. With `stream=true` it streams SSE `data: {chunk}\n\n`. If `API_KEY` is set, the `X-API-Key` header is required.

### Logging (`logging_config.py`)

`setup_logging(console=True, log_file="agent.log")` writes `logs/agent.log`, anchored at the project root (rotating, 5 × 5 MB, level from `LOG_LEVEL`). It uses `force=True`, so it wins over libraries that configure logging first.

The chatty loggers are held at WARNING: `httpx` (whose request URLs contain the Telegram bot token), `httpcore`, `telegram.ext`, `mcp` and `anthropic`. A process whose stdout carries a protocol (e.g. a stdio MCP server) should call `setup_logging(console=False, log_file="<own>.log")`.

---

## Configuration

All settings live in `.env` at the project root. `core/src/config/config.py` loads it via an absolute path, and code reads settings only through `Config`. Agent-specific config may live in an agent module (e.g. `agents/finance/config.py`).

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Only `anthropic` today |
| `LLM_MODEL_NAME` | `claude-opus-5-5` | Default model; per-agent override with `llm_model` in `agents.json` |
| `ANTHROPIC_API` | `true` | `true` → Anthropic API (`ANTHROPIC_API_KEY`, billed per token); `false` → local Claude Code CLI using its logged-in account |
| `ANTHROPIC_API_KEY` | _(required when `ANTHROPIC_API=true`)_ | From console.anthropic.com |
| `LLM_EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max` |
| `LLM_MAX_TOKENS` | `16000` | Max output tokens per call |
| `STORE_TYPE` | `pinecone` | Vector store backend (Pinecone only) |
| `PINECONE_API_KEY` / `PINECONE_ENVIRONMENT` / `PINECONE_INDEX_NAME` | — | Pinecone connection |
| `SENTENCE_TRANSFORMER_MODEL` | — | Embedding model, e.g. `sentence-transformers/all-MiniLM-L6-v2` (384-d) |
| `PROMPT_FILE` | `agents/property/prompts/System_Prompt.txt` | Property agent system prompt (resolved from the project root) |
| `INTERACTION_FILE` / `REFORMULATION_PROMPT` | `agents/property/prompts/interaction.json` / `…/reformulated_prompt.txt` | Property agent extras |
| `URLS_FILE` | `agents/property/urls.txt` | URL list for property scraping |
| `NORMALISER_PROMPT_FILE` / `SUPERVISOR_PROMPT_FILE` / `SYNTHESISER_PROMPT_FILE` | `orchestrator/prompts/*.txt` | Orchestrator prompts |
| `AGENTS_FILE` / `MCP_FILE` | `agents.json` / `mcp.json` | Registries (resolved from the project root) |
| `TELEGRAM_BOT_TOKEN` | _(required for Telegram)_ | From @BotFather |
| `TELEGRAM_OWNER_ID` | _(empty)_ | Owner's Telegram user id; only matters for agents marked `"owner_only": true` (none today) |
| `WHISPER_MODEL` | `small` | faster-whisper size: `tiny` / `base` / `small` / `medium` / `large-v3` |
| `EXPENSES_DIR` | _(per-user data dir)_ | Folder for `expenses.db`. Empty → `~/Library/Application Support/uk-agents/expenses` (macOS), `~/.local/share/uk-agents/expenses` (Linux, honours `XDG_DATA_HOME`), `%APPDATA%\uk-agents\expenses` (Windows). Relative paths resolve from the project root |
| `LOG_LEVEL` | `INFO` | Root log level |
| `API_KEY` | _(empty — auth off)_ | REST `X-API-Key` |
| `MAX_HISTORY_TURNS` | `10` | Turns of chat history sent to the LLM per agent and session |
| `SESSION_DB_FILE` | `sessions.db` | SQLite chat history (project root; git-ignored); survives restarts |
| `CRON_SEARCH_PROMPT` | _(2-bed apartments in Poland < 1000)_ | Prompt the cron client sends every run |
| `SBR_WEBDRIVER` | _(empty)_ | Selenium WebDriver URL for property scraping |
| `WHATSAPP_*`, `MAIL_*` | — | Green API and IMAP settings for the standalone services (see `config.py`) |
| `LANGCHAIN_TRACING_V2` / `LANGCHAIN_API_KEY` / `LANGCHAIN_PROJECT` | `false` / — / `uk-agents` | LangSmith tracing |

---

## Development Workflows

Run every command from the **project root**.

```bash
uv venv && uv sync                       # install (uv.lock may be excluded locally)
cp .env-template .env                    # then fill in keys

python -m clients.telegram.main          # Telegram bot
python -m clients.rest.main              # REST API on :8000
streamlit run clients/streamlit/main.py  # Streamlit UI on :8501
python -m clients.cron.main              # cron job
python -m agents.whatsapp.main           # standalone WhatsApp listener
python -m agents.mails.main              # standalone mail monitor

python -m pytest -q                      # full suite (testpaths in pyproject.toml)
python -m pytest orchestrator/tests -q   # one area
```

### Testing rules

- Mock every external dependency: LLM, Pinecone, HTTP, Telegram, Whisper. Never make real network or model calls. Local subprocesses are fine.
- Filesystem and database tests use `tmp_path`. Never touch the real `EXPENSES_DIR` or `logs/`.
- `clients/tests/conftest.py` replaces `core.src.config.config`, `orchestrator.agent_interface` and `telegram` with MagicMocks **for the whole pytest process**. Tests elsewhere that need real config values must patch them locally (see `orchestrator/tests/test_graph.py` `_CONFIG`), or resolve them at call time instead of import time.
- When adding a new test directory, add it to `testpaths` in `pyproject.toml`.

---

## Code Conventions

- Python 3.13+, type hints throughout, class-based modules, private helpers prefixed `_`.
- Inside `core/src/` use relative imports (`from .model.x import Y`). Elsewhere, absolute imports from the project root (`from core.src.config.config import Config`) plus relative imports within the same package.
- Settings come through `Config`. Never hardcode keys or absolute paths.
- Context managers (`__enter__` / `__exit__` / `close()`) for classes that own resources.
- User-facing text that must be exact (expense replies, refusals) lives in code. Claude may not reword it.
- Match the surrounding style. Orchestrator modules carry explanatory module docstrings; keep them up to date.

### Adding a new agent

1. Create `agents/<name>/agent.py` subclassing `BaseAgent`, with `prompts/System_Prompt.txt`.
2. Register it in `agents.json`:
   ```json
   {"name": "x_agent", "class": "agents.x.agent.XAgent", "description": "<when to route here>",
    "enabled": true, "rag": false, "display_name": "X Agent", "remarks": ["one-liner shown while it works"]}
   ```
   Optional keys: `owner_only`, `llm_provider`, `llm_model`. The supervisor routes on `description`, so make it precise.
3. Tools: add a server to `mcp.json` and return `_mcp.langchain_tools(server_name="…")`. Return `[]` for no tools. Never return `None` (it means "every non-owner-only server").
4. Add tests under `agents/<name>/tests/` and register the directory in `testpaths`.

### Adding a new client

Create `clients/<name>/main.py` and `__init__.py`, implement `BaseClient`, call `setup_logging()` at the top, and run it with `python -m clients.<name>.main`.

### Prompts

- Orchestrator prompts keep their placeholders: normaliser `{today}` `{history}` `{pending}`; supervisor `{agents}` `{history}` `{user_input}` `{agent_outputs}`.
- The property agent's `reformulated_prompt.txt` keeps `{user_prompt}`.
- Prompts describe intents, never tool names, and never hold conditional business logic. Put that in code.

---

## Common Gotchas

- **Run from the project root with `-m`.** `python -m clients.telegram.main` works; `python clients/telegram/main.py` breaks imports.
- **Read `.text`, not `.content`.** Claude responses include thinking blocks, so `AIMessage.content` is a list.
- **Two Claude backends.**
  - `ANTHROPIC_API=false` uses `ChatClaudeCode`, which runs the local `claude` CLI through the Agent SDK. Bound tools are exposed as an in-process MCP server and Claude Code runs the tool loop itself, so responses never contain `tool_calls`. Built-in Claude Code tools and user settings are disabled.
  - `core/tests/model/test_llm_factory.py` fails when `core/tests` runs alone with `ANTHROPIC_API=false` in `.env`. It passes in the full suite.
- **Chat history is persistent.** `BaseAgent` gives each `LlmModelGraph` a `SessionManager(Config.SESSION_DB_FILE)`, so history is kept per (agent class name, session_id) in `sessions.db` and survives restarts. Only the last `MAX_HISTORY_TURNS` turns go to the LLM. An `LlmModelGraph` built without `history_store` (as in tests) keeps history in memory. Expenses live separately in `expenses.db`; mail state in `MAIL_SEEN_DB`.
- **Owner lock.** `owner_only` exists in `agents.json` (agent refusal plus pending isolation) and in `mcp.json` (tool scoping). Nothing uses it today. The expense tools are not in `mcp.json` at all; only `ExpenditureAgent` binds them, so other agents can never write the ledger.
- **Never log secrets.** Telegram URLs contain the bot token. Keep `httpx` at WARNING in `logging_config.py`.
- **Pinecone required for RAG agents.** `property_agent` and `whatsapp_agent` need `PINECONE_API_KEY` and network access at startup.
