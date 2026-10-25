# UK Agents

**UK Agents** is Uzair Kamal's (UK) personal repository of AI agents. Each agent does one job for Uzair: finding property, analysing stocks, watching his inboxes, covering for him in WhatsApp groups, or just chatting. New agents are added all the time, and they don't have to be related to each other.

All agents share the same foundation:

- **One base class**: every agent extends `BaseAgent` (`core/`), so the LLM (Claude), session history, optional RAG and MCP tools come for free. A new agent is a prompt plus a few hook methods.
- **One registry**: agents are listed in `agents.json`. Adding, disabling or re-modelling an agent is a config change.
- **One front door**: the REST API, Streamlit UI, Telegram bot and cron job all talk to a single `OrchestratorAgent`. Its router reads each request and plans which agent(s) handle it, by their `description`.
- **Background services**: agents that need to run on their own (e.g. the WhatsApp listener and the mail monitor) have standalone entrypoints that run alongside the orchestrator.

### Agent catalog

| Agent | Purpose | Data / tools | Runs as |
|---|---|---|---|
| `conversational_agent` | Greetings, small talk, general questions | — | Orchestrator |
| `property_agent` | Property search over indexed listings | RAG (Pinecone) | Orchestrator |
| `finance_agent` | Stock analysis and investment questions | Yahoo Finance MCP | Orchestrator |
| `whatsapp_agent` | Short, mobile-friendly property answers | RAG (Pinecone) | Orchestrator |
| `mail_agent` | Questions about emails, drafting replies | — | Orchestrator |
| WhatsApp listener | Detects when Uzair is mentioned in a group and auto-replies | Green API | Standalone (`agents.whatsapp.main`) |
| Mail monitor | Summarises new emails across inboxes and sends a digest | IMAP + Green API | Standalone (`agents.mails.main`) |

To add your own, see [Adding a New Agent](#adding-a-new-agent).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Package Layout](#package-layout)
- [Multi-Agent Orchestration](#multi-agent-orchestration)
- [Specialist Agents](#specialist-agents)
- [Core Infrastructure](#core-infrastructure)
- [MCP Tool Integration](#mcp-tool-integration)
- [Client Layer](#client-layer)
- [Data Ingestion](#data-ingestion)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Running the Clients](#running-the-clients)
- [Adding an MCP Server](#adding-an-mcp-server)
- [Adding a New Agent](#adding-a-new-agent)
- [Testing](#testing)
- [LangSmith Observability](#langsmith-observability)

---

## Architecture Overview

```
User (REST / Streamlit / Telegram / Cron)
  │
  ▼
OrchestratorAgent           ← single facade — all clients talk to this
  │
  ▼
MultiAgentGraph             ← LangGraph multi-agent graph (async, checkpointed)
  │
  ├─► router                ← ONE LLM call: clean the message + plan the agents
  │       │
  │       └─► planned agents, in order (no LLM call between them):
  │             conversational_agent  ← greetings, small talk
  │             property_agent        ← RAG: vector search → listings
  │             finance_agent         ← live stock data via Yahoo Finance MCP
  │             whatsapp_agent        ← RAG property search, mobile-friendly replies
  │             mail_agent            ← email questions from conversation history
  │             expenditure_tracker   ← owner's expenses (owner_only)
  │
  ├─► synthesiser           ← one agent: its answer as is; several: LLM merge
  │       │
  │       ├─► END
  │       └─► confirmation  ← an agent left a pending item: interrupt() until the
  │                           user's next message (yes / no / correction / other)
  ▼
 END

Standalone background services (run independently):

WhatsAppListener  ← polls Green API → LLM mention check → auto-reply
MailMonitor       ← polls IMAP inboxes → LLM summarise → periodic delivery
```

Every user message enters at the router, which cleans it and plans the agents in one structured LLM call. The plan runs without further routing calls, and the synthesiser combines the outputs into one reply — so a single-agent message costs one orchestration call on top of the agent's own work.

---

## Package Layout

```
uk-agents/
├── .env                              # Runtime configuration (project root)
├── .env-template                     # Template — copy to .env and fill in keys
├── mcp.json                          # MCP server registry
├── agents.json                       # Agent registry (name, class, model, RAG flag)
├── pyproject.toml                    # Project metadata and dependencies
├── logging_config.py                 # Central logging → logs/agent.log
│
├── core/                             # Shared AI infrastructure (all agents import from here)
│   └── src/
│       ├── base_agent.py             # BaseAgent ABC — Template Method pattern
│       ├── utils.py                  # load_prompt() file loader
│       ├── config/
│       │   └── config.py             # Settings loaded from .env (absolute path)
│       ├── model/
│       │   ├── llm_model_graph.py    # LangGraph pipeline (retrieve → generate)
│       │   ├── llm_factory.py        # LLM provider factory (Anthropic API or Claude Code)
│       │   ├── claude_code_chat.py   # ChatClaudeCode — LangChain model over the local Claude Code CLI
│       │   ├── embedder.py           # SentenceTransformer + vector search
│       │   ├── rag_context_manager.py# Vector search → formatted context string
│       │   ├── context_builder.py    # Formats listing dicts as readable text
│       │   ├── session_manager.py    # SQLite-backed session history
│       │   ├── mcp_registry.py       # MCPRegistry on the official mcp SDK client
│       │   └── mcp_tools.py          # Loads mcp.json → shared _mcp registry
│       └── persistence/
│           ├── vector_store.py       # Abstract base (add_vectors / search)
│           ├── factory.py            # Selects the vector store backend (Pinecone)
│           └── pinecone_store.py     # Cloud Pinecone store
│
├── agents/                           # Domain-specific agent implementations
│   ├── conversational/
│   │   ├── agent.py                  # ConversationalAgent(BaseAgent)
│   │   └── prompts/System_Prompt.txt
│   ├── property/
│   │   ├── agent.py                  # PropertySearchAgent(BaseAgent) — RAG
│   │   ├── housing/                  # CSV → embedding-ready text pipeline
│   │   ├── gatherers/                # URL + Polish housing data ingestion
│   │   ├── scraping/                 # HTTP (BeautifulSoup) + Selenium scrapers
│   │   └── prompts/
│   │       ├── System_Prompt.txt
│   │       ├── reformulated_prompt.txt
│   │       └── interaction.json
│   ├── finance/
│   │   ├── agent.py                  # FinanceAgent(BaseAgent) — stocks MCP only
│   │   ├── config.py                 # FinanceConfig (prompt path)
│   │   └── prompts/System_Prompt.txt
│   ├── whatsapp/                     # WhatsApp integration
│   │   ├── agent.py                  # WhatsAppAgent(BaseAgent) — RAG, mobile replies
│   │   ├── listener.py               # WhatsAppListener — polls Green API, auto-replies
│   │   ├── green_api_client.py       # Green API HTTP client
│   │   ├── main.py                   # Standalone entrypoint for the listener
│   │   └── prompts/
│   │       ├── System_Prompt.txt
│   │       └── mention_check.txt     # LLM prompt: "is this message for {user_name}?"
│   └── mails/                        # Email monitor
│       ├── agent.py                  # MailAgent(BaseAgent) — email Q&A in orchestrator
│       ├── monitor.py                # MailMonitor — polls IMAP, summarises, delivers
│       ├── imap_client.py            # IMAPAccount — fetch messages since a timestamp
│       ├── email_store.py            # SQLite-backed dedup store (Message-ID tracking)
│       ├── main.py                   # Standalone entrypoint for the monitor
│       └── prompts/
│           ├── System_Prompt.txt
│           └── summarise_email.txt   # LLM prompt: summarise one email
│
├── orchestrator/                     # Multi-agent graph
│   ├── agent_interface.py            # OrchestratorAgent — public facade
│   ├── agent_registry.py             # Reads agents.json → live agent instances
│   ├── graph.py                      # MultiAgentGraph — builds + compiles graph
│   ├── state.py                      # GraphState TypedDict, MAX_AGENT_TURNS
│   ├── nodes/
│   │   ├── router.py                 # RouterNode — cleans the message + plans agents
│   │   ├── agent.py                  # AgentNode — wraps a BaseAgent instance
│   │   ├── synthesiser.py            # SynthesiserNode — merges outputs
│   │   └── confirmation.py           # ConfirmationNode — interrupt() for pending items
│   └── prompts/
│       ├── router.txt                # Cleaning + planning rules, agent list placeholder
│       ├── synthesiser.txt           # Merge/deduplicate instructions
│       └── confirmation.txt          # Classify a reply to a pending item
│
├── clients/                          # Delivery interfaces
│   ├── base.py                       # BaseClient ABC (start / stop)
│   ├── rest/main.py                  # FastAPI REST API (port 8000)
│   ├── streamlit/main.py             # Streamlit web UI (port 8501)
│   ├── telegram/main.py              # Telegram long-polling bot
│   └── cron/main.py                  # Scheduled search (every 30 min)
│
└── logs/
    └── agent.log                     # Unified runtime log (all clients write here)
```

---

## Multi-Agent Orchestration

### How the graph works

The graph is a LangGraph `StateGraph` compiled once at startup inside `MultiAgentGraph.__init__`, with a SQLite checkpointer so a run can pause for a confirmation and resume on the next message.

**State** (`orchestrator/state.py`):

```python
class GraphState(TypedDict):
    user_input: str                          # cleaned message the agents see
    raw_input: str                           # message as received
    session_id: str
    messages: Annotated[list, add_messages]
    agent_outputs: dict[str, str]            # name → answer this turn
    plan: list[str]                          # agents to run, in order (router)
    next: str                                # where a resumed confirmation goes
    turns: int                               # agent calls this turn
    is_owner: bool
    awaiting: str | None                     # agent waiting for a yes/no
```

**Graph topology** (`orchestrator/graph.py`):

```
START → router → agent_A → agent_B … → synthesiser → END
                                                    └→ confirmation (interrupt) → …
```

After the router and after each agent, `_route_plan` picks the first planned agent that has not answered yet, or the synthesiser when none is left (or when `turns` reaches `MAX_AGENT_TURNS`).

### Router node (`orchestrator/nodes/router.py`)

One LLM call with structured output (`RoutePlan`): the cleaned `text` (transcription fixes, spoken amounts → digits, relative dates → YYYY-MM-DD) and the ordered `agents` list. The list items are an enum of the enabled agent names, so the model cannot name an agent that does not exist. Rules live in `orchestrator/prompts/router.txt`: greetings → the small-talk agent only; one topic → one agent; several topics → each needed agent, ordered by dependency.

### Agent node (`orchestrator/nodes/agent.py`)

Wraps a `BaseAgent` instance. When invoked:
1. Awaits `agent.aask(user_input, session_id=session_id)` (or runs `agent.ask` in a worker thread)
2. Stores result in `state["agent_outputs"][agent_name]`
3. Increments `state["turns"]`

### Synthesiser node (`orchestrator/nodes/synthesiser.py`)

Merges all agent outputs into one clean answer.

- **Single-agent passthrough**: if only one agent ran, the synthesiser returns that output directly without an extra LLM call.
- **Multi-agent synthesis**: injects all collected outputs as context and calls the LLM to write a fresh, deduplicated, markdown-formatted reply.

---

## Specialist Agents

The agents below are the ones in the repository today; the list is expected to keep growing. All agents extend `BaseAgent` (`core/src/base_agent.py`) using the **Template Method** pattern. `BaseAgent.__init__` calls three hook methods that each subclass overrides:

```python
class BaseAgent(ABC):
    def get_system_prompt(self) -> str: ...      # load your prompt file
    def get_rag_context_manager(self): ...       # RAG manager or NullRagContextManager
    def get_mcp_tools(self) -> list | None: ...  # None=all tools, []=none, or filtered list
```

Agents are registered in **`agents.json`** at the project root. The orchestrator reads this file at startup — no code changes are needed to add, remove, or reconfigure an agent.

---

### Conversational Agent

**Class**: `agents.conversational.agent.ConversationalAgent`

**Role**: Handles greetings, small talk, and off-topic questions. Keeps the conversation friendly without invoking RAG or tools.

**Configuration**:
- RAG: disabled (`NullRagContextManager`)
- MCP tools: none (returns `[]`)
- System prompt: warm, brief responses; redirects property/finance questions to the right agent

**Flow**:
```
User: "Hi there!"
  │
  ▼
router → plan [conversational_agent]
  │
  ▼
LlmModelGraph (retrieve skips vector search, generate answers directly)
  │
  ▼
"Hello! Great to have you here. How can I help you find your perfect property today?"
  │
  ▼
plan done → synthesiser (passthrough)
```

---

### Property Search Agent

**Class**: `agents.property.agent.PropertySearchAgent`

**Role**: Finds property listings that match the user's natural-language criteria. Only presents data that exists in the vector store — never invents or estimates.

**Configuration**:
- RAG: enabled (`RagContextManager` + `Embedder`)
- MCP tools: default (all registered MCP tools — e.g. currency conversion if available)
- Vector store: Pinecone (cloud), configured via `STORE_TYPE`
- System prompt: strict data scope — if no matching listings in context, says so honestly

**Internal pipeline** (`core/src/model/llm_model_graph.py`):

```
user_prompt
  │
  ▼
[retrieve node]
  Embedder.embed_query(user_prompt)
  → VectorStore.search(query_vector, k=RAG_K)
  → RagContextManager formats results as readable text
  → state["context"] = "Property 1: ...\nProperty 2: ..."
  │
  ▼
[generate node]
  SystemMessage(system_prompt)
  SystemMessage("Relevant listings:\n{context}")    ← only when context is non-empty
  SystemMessage("No matching listings found...")    ← only when context empty AND rag_enabled
  + session_history + HumanMessage(user_prompt)
  │
  LLM invoked → may call MCP tools (loop up to MAX_TOOL_CALLS=5)
  │
  ▼
answer string
```

**Tool call loop** (inside `generate` node):

```
LLM response
  │
  ├─ has tool_calls? ──► invoke each tool → append ToolMessage → re-invoke LLM
  │                                                    ↑
  │                       (repeat up to MAX_TOOL_CALLS=5 times)
  │
  └─ no tool_calls → break → return response.text as answer
```

**Data ingestion**:

Properties are pre-indexed into the vector store offline. Two ingestion methods are provided:

```bash
# From CSV files (Polish housing data)
python -m agents.property.housing.housing_data_collector

# From URLs (web scraping)
python -m agents.property.gatherers.data_collector
```

---

### Finance Agent

**Class**: `agents.finance.agent.FinanceAgent`

**Role**: Stock analysis and investment advice using live market data fetched via MCP tools. Strictly scoped to stocks and equities — will not handle currency conversion or mortgages.

**Configuration**:
- RAG: disabled (`NullRagContextManager`)
- MCP tools: `stocks` server only (`_mcp.langchain_tools(server_name="stocks")`)
- MCP server: `@fre4x/yahoo-finance` — Yahoo Finance via stdio subprocess
- System prompt: always fetch live data first; present price, P/E, EPS, news sentiment, recommendation

**Available tools** (from Yahoo Finance MCP):

| Tool | Description |
|---|---|
| `yfin_get_quotes` | Real-time price quotes for one or more tickers |
| `yfin_get_historical_prices` | OHLCV data for a ticker over a time period |
| `yfin_get_stock_info` | Full profile: price, market cap, fundamentals |
| `yfin_get_news` | Latest news articles for a ticker |
| `yfin_get_financial_statement` | Income statement, balance sheet, or cashflow |
| `yfin_get_holder_info` | Major, institutional, mutual fund, or insider holders |
| `yfin_get_option_dates` | Available option expiration dates |
| `yfin_get_option_chain` | Calls or puts chain for a given expiry |
| `yfin_get_recommendations` | Analyst recommendations and upgrades/downgrades |
| `yfin_get_stock_actions` | Dividend and split history |

**Flow**:
```
User: "What's the outlook for AAPL?"
  │
  ▼
router → plan [finance_agent]
  │
  ▼
LlmModelGraph (retrieve skips — NullRagContextManager returns "")
  │
  ▼
generate node: LLM sees system prompt + user question
  LLM calls yfin_get_stock_info(ticker="AAPL")
  → MCP subprocess returns price, P/E, EPS, market cap
  LLM calls yfin_get_news(ticker="AAPL")
  → MCP subprocess returns recent headlines + sentiment
  LLM synthesises: "AAPL is trading at $X. P/E is Y. Recent news is..."
  │
  ▼
plan done → synthesiser (passthrough)
```

---

### WhatsApp Agent

**Class**: `agents.whatsapp.agent.WhatsAppAgent`

**Role**: Handles property search questions arriving via WhatsApp. Produces short, mobile-friendly replies without markdown tables or heavy formatting.

**Configuration**:
- RAG: enabled (same vector store as `PropertySearchAgent`)
- MCP tools: none (returns `[]`)
- System prompt: brief bullet points, no markdown tables, conversational tone

**Standalone listener** (`agents/whatsapp/listener.py`):

`WhatsAppListener` is a separate background service that runs independently of the orchestrator. It polls the Green API for new group messages, uses an LLM to detect whether the owner is mentioned, and sends a configurable away message if they are.

```
WhatsApp group message received
  │
  ▼
Green API notification queue
  │
  ▼
WhatsAppListener._extract_group_text()   ← filter to target group, text-only
  │
  ▼
_is_asking_for_user()                    ← LLM checks mention_check.txt prompt
  │
  ├─ YES → GreenAPIClient.send_message(group_id, WHATSAPP_AWAY_MESSAGE)
  └─ NO  → skip
  │
  ▼
GreenAPIClient.delete_notification()     ← always delete to prevent reprocessing
```

Run the listener standalone:
```bash
python -m agents.whatsapp.main
```

---

### Mail Agent

**Class**: `agents.mails.agent.MailAgent`

**Role**: Answers conversational questions about emails within the orchestrator (e.g. "do I have emails from X?", "draft a reply to the invoice"). Works solely from conversation history — no RAG or MCP tools.

**Configuration**:
- RAG: disabled (`NullRagContextManager`)
- MCP tools: none (returns `[]`)

**Standalone monitor** (`agents/mails/monitor.py`):

`MailMonitor` is a separate background service that runs independently of the orchestrator. It periodically polls configured IMAP inboxes, summarises new emails with the LLM, and delivers a formatted digest on a separate schedule.

```
Every MAIL_CHECK_INTERVAL_HOURS (default 2 h):
  IMAPAccount.fetch_since(last_check_wall)
    → for each new email not in EmailStore (SQLite dedup):
        LLM summarises email → appended to pending list
        EmailStore.mark_seen(message_id)

Every MAIL_SUMMARY_INTERVAL_HOURS (default 5 h):
  pending list → formatted digest
    → GreenAPIClient.send_message(MAIL_SUMMARY_WHATSAPP_GROUP_ID, digest)
       (or logged if MAIL_SUMMARY_WHATSAPP_GROUP_ID is empty)
  pending list cleared
```

Supports multiple IMAP accounts simultaneously (JSON array in `MAIL_ACCOUNTS`). A SQLite store (`MAIL_SEEN_DB`) guarantees each email is summarised at most once, even across restarts.

Run the monitor standalone:
```bash
python -m agents.mails.main
```

---

## Core Infrastructure

### BaseAgent (`core/src/base_agent.py`)

```python
class BaseAgent(ABC):
    def __init__(self, session_id=None, rag_enabled=False,
                 llm_provider=None, llm_model=None):
        # Calls get_system_prompt(), get_rag_context_manager(), get_mcp_tools()
        # Builds LlmModelGraph with the returned values
        ...

    # --- Hooks — override in every subclass ---
    @abstractmethod
    def get_system_prompt(self) -> str: ...

    @abstractmethod
    def get_rag_context_manager(self): ...

    @abstractmethod
    def get_mcp_tools(self) -> list | None: ...

    # --- Public interface — inherited as-is ---
    def ask(self, prompt, stream=False, session_id=None): ...
    def close(self): ...
```

`NullRagContextManager` (also in `base_agent.py`) is the no-op placeholder returned by agents that don't need vector search. It always returns `""` from `get_context()`.

### LangGraph pipeline (`core/src/model/llm_model_graph.py`)

Two-node graph: `retrieve → generate`.

**Node: retrieve**
- Calls `rag_context_manager.get_context(user_prompt)`
- For RAG-enabled agents: embeds the query, searches the vector store, returns formatted listings
- For non-RAG agents (`NullRagContextManager`): returns `""` instantly — no vector search

**Node: generate**
- Builds the message list: system prompt → (context if present) → session history → user message
- Calls `llm.invoke(messages)` — LLM may request tool calls
- Loops up to `MAX_TOOL_CALLS = 5`, executing each tool and appending `ToolMessage` results
- Returns `response.text` (text blocks only, no thinking blocks) as `state["answer"]`

**Session history**: kept per `session_id` in memory (dict of lists). Sliding window: last `MAX_HISTORY_TURNS * 2` messages.

**LangSmith tracing**: every `ask()` call is wrapped in `@traceable(name=f"agent.{agent_name}")`. Tool invocations are wrapped in `@traceable(name=f"tool.{tool_name}")`.

### Embedder (`core/src/model/embedder.py`)

Uses `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional float32 vectors).

| Method | Purpose |
|---|---|
| `embed_query(text)` | Single query → numpy vector |
| `embed_texts(texts)` | Batch of texts → numpy array |
| `embed_documents_to_vectors(docs)` | Dict of documents → list of vector dicts |
| `search(query, k=5)` | End-to-end: embed + vector store search |
| `save_vectors_in_store(vectors)` | Persist vectors to the configured store |

### Vector stores (`core/src/persistence/`)

| Backend | Class | When to use |
|---|---|---|
| Pinecone | `PineconeStore` | Production; auto-creates index if absent |

Selected with `STORE_TYPE=pinecone` in `.env`.

---

## MCP Tool Integration

Tools are declared in **`mcp.json`** at the project root. No Python changes are needed to add a server.

```json
{
  "servers": [
    {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]},
    {"name": "stocks",  "command": ["npx", "--yes", "@fre4x/yahoo-finance"]}
  ]
}
```

At startup, `mcp_tools.py` reads this file and registers each server in the `MCPRegistry`, which talks to them through the official `mcp` SDK client (`mcp.Client`). Two transport types are supported:

| Type | Config key | Notes |
|---|---|---|
| stdio subprocess | `"command": [...]` | Minimal environment unless listed in `"env": [...]`; `{python}` servers get the full environment |
| Streamable HTTP | `"url": "..."` | Optional `"api_key_env"` / `"api_key_param"` add a key as a query parameter |

Both accept `"owner_only": true` and `"timeout": <seconds>` (default 60).

**How it works**:
1. `MCPRegistry` keeps one persistent `MCPServerConnection` per server, owned by a task on a shared background event loop.
2. `langchain_tools(server_name=None)` lists each server's tools and generates LangChain `StructuredTool`s from their JSON schemas — usable with `.invoke()` and `.ainvoke()`.
3. The LLM is bound to these tools via `llm.bind_tools(tools)`.
4. The `generate` node inspects `response.tool_calls` and routes execution to the right tool.
5. A call that times out or fails disconnects that server (stopping a stdio subprocess); the next call reconnects.

**Per-agent tool filtering**: agents can restrict which tools they receive:

```python
# FinanceAgent only gets tools from the "stocks" server
def get_mcp_tools(self) -> list:
    return _mcp.langchain_tools(server_name="stocks")

# PropertySearchAgent gets all tools (or pass [] for none)
def get_mcp_tools(self) -> list | None:
    return None   # None = all registered tools
```

**JSON-encoded argument coercion**: some LLMs emit array/object parameters as JSON strings (e.g. `tickers: "[\"AAPL\"]"` instead of `tickers: ["AAPL"]`). `LlmModelGraph._coerce_args()` detects and deserializes these before invoking the tool.

---

## Client Layer

All clients implement `BaseClient` (`clients/base.py`):

```python
class BaseClient(ABC):
    @abstractmethod
    def start(self) -> None: ...   # blocks until stopped
    def stop(self) -> None: ...    # graceful shutdown
    # + __enter__ / __exit__ context manager
```

| Client | Entry point | Transport | Notes |
|---|---|---|---|
| `RestClient` | `clients/rest/main.py` | FastAPI + SSE (port 8000) | Optional `X-API-Key` auth |
| `StreamlitClient` | `clients/streamlit/main.py` | Web UI (port 8501) | Requires REST API running first |
| `TelegramClient` | `clients/telegram/main.py` | Telegram long-polling | Per-chat session IDs |
| `CronClient` | `clients/cron/main.py` | Scheduled loop | Fires `CRON_SEARCH_PROMPT` every 30 min |
| `WhatsAppListener` | `agents/whatsapp/main.py` | Green API polling | Standalone — does not use OrchestratorAgent |
| `MailMonitor` | `agents/mails/main.py` | IMAP polling | Standalone — does not use OrchestratorAgent |

All four use `OrchestratorAgent` as their backend. The Telegram client runs blocking `agent.ask()` calls in a thread executor to avoid blocking the async event loop.

### REST API endpoints

| Method | Path | Params | Response |
|---|---|---|---|
| `GET` | `/ask` | `prompt` (str), `stream` (bool, default `false`) | `{"response": str}` or SSE stream |

Streaming format (SSE): `data: {chunk}\n\n` with 10 ms delay between chunks.

---

## Data Ingestion

### CSV (Polish housing data)

```bash
# Place CSV files in agents/property/datasets/pl-housing/
python -m agents.property.housing.housing_data_collector
```

Streams the CSV in 5 000-row chunks, converts each row to a semantic text document, embeds in batches of 100, and persists to the configured vector store.

### From URLs

```bash
# Edit agents/property/urls.txt — one URL per line
python -m agents.property.gatherers.data_collector
```

Fetches each URL with BeautifulSoup (or Selenium for anti-bot sites), extracts listing data, and indexes it into the vector store.

---

## Configuration

All settings live in `.env` at the **project root**. `core/src/config/config.py` loads it via an absolute path derived from `__file__` — works regardless of working directory.

### Core LLM

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | LLM backend (`anthropic` is the supported provider) |
| `LLM_MODEL_NAME` | `claude-opus-5-5` | Claude model ID |
| `LLM_EFFORT` | `medium` | Thinking depth / token spend: `low`, `medium`, `high`, `xhigh`, `max` |
| `LLM_MAX_TOKENS` | `16000` | Max output tokens per LLM call |
| `ANTHROPIC_API` | `true` | `true` → Anthropic API (needs `ANTHROPIC_API_KEY` + credits); `false` → local Claude Code CLI using the account it is logged in with |
| `ANTHROPIC_API_KEY` | _(required when `ANTHROPIC_API=true`)_ | Anthropic API key from console.anthropic.com |

### Vector store

| Variable | Default | Description |
|---|---|---|
| `STORE_TYPE` | `pinecone` | Vector store backend (`pinecone` only) |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `RAG_K` | `5` | Listings returned per vector search |
| `PINECONE_API_KEY` | _(required for cloud)_ | Pinecone API key |
| `PINECONE_INDEX_NAME` | `property-agent` | Pinecone index name |

### Paths

| Variable | Default | Description |
|---|---|---|
| `PROMPT_FILE` | `agents/property/prompts/System_Prompt.txt` | Property agent system prompt |
| `ROUTER_PROMPT_FILE` | `orchestrator/prompts/router.txt` | Router prompt (cleanup + agent plan) |
| `CONFIRMATION_PROMPT_FILE` | `orchestrator/prompts/confirmation.txt` | Classifies replies to a pending item |
| `CHECKPOINT_DB_FILE` | `checkpoints.db` | Orchestrator checkpointer (paused confirmations only) |
| `SYNTHESISER_PROMPT_FILE` | `orchestrator/prompts/synthesiser.txt` | Synthesiser merge prompt |
| `AGENTS_FILE` | `agents.json` | Agent registry |
| `MCP_FILE` | `mcp.json` | MCP server registry |
| `SESSION_DB_FILE` | `sessions.db` | SQLite session history |
| `MAX_HISTORY_TURNS` | `10` | Sliding window depth |

### Clients

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | _(required)_ | Token from [@BotFather](https://t.me/BotFather) |
| `API_KEY` | _(empty — auth disabled)_ | REST API key; set to enforce `X-API-Key` header |
| `CRON_SEARCH_PROMPT` | _(2-bed Poland <1000 PLN)_ | Prompt used by the scheduled job |

### WhatsApp (Green API)

Register at [green-api.com](https://green-api.com), create a free instance, scan the QR code with your phone, then enable "Incoming webhooks" in the instance settings.

| Variable | Default | Description |
|---|---|---|
| `WHATSAPP_INSTANCE_ID` | _(required)_ | Green API instance ID |
| `WHATSAPP_INSTANCE_TOKEN` | _(required)_ | Green API instance token |
| `WHATSAPP_GROUP_ID` | _(required)_ | Target group chat ID (format: `120363xxx@g.us`) |
| `WHATSAPP_USER_NAME` | `Uzair` | Name the LLM checks for in mention detection |
| `WHATSAPP_AWAY_MESSAGE` | _(built-in)_ | Auto-reply sent when the user is mentioned |
| `WHATSAPP_POLL_INTERVAL` | `2` | Seconds between polls when the queue is empty |

To find the group chat ID:
```bash
python -c "from agents.whatsapp.green_api_client import GreenAPIClient; [print(c['id'], c.get('name')) for c in GreenAPIClient().get_contacts() if c.get('type')=='group']"
```

### Mail Monitor

| Variable | Default | Description |
|---|---|---|
| `MAIL_ACCOUNTS` | `[]` | JSON array of IMAP account configs (see `.env-template` for examples) |
| `MAIL_CHECK_INTERVAL_HOURS` | `2` | How often to fetch new emails (hours) |
| `MAIL_SUMMARY_INTERVAL_HOURS` | `5` | How often to send the accumulated summary (hours) |
| `MAIL_SUMMARY_WHATSAPP_GROUP_ID` | _(empty — log only)_ | WhatsApp group to deliver the digest to |
| `MAIL_SEEN_DB` | `mail_seen.db` | SQLite file tracking already-summarised Message-IDs |

Example `MAIL_ACCOUNTS` value:
```json
[
  {"email": "you@gmail.com", "imap_host": "imap.gmail.com", "imap_port": 993, "password": "app-password"},
  {"email": "you@outlook.com", "imap_host": "outlook.office365.com", "password": "app-password"}
]
```

### Observability

| Variable | Default | Description |
|---|---|---|
| `LANGCHAIN_TRACING_V2` | `false` | Set `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | _(empty)_ | LangSmith API key |
| `LANGCHAIN_PROJECT` | `uk-agents` | LangSmith project name |

---

## Getting Started

### Prerequisites

- Python 3.13+
- Either Claude Code installed and logged in (`ANTHROPIC_API=false`, uses your Claude subscription — personal use only), or an [Anthropic API key](https://console.anthropic.com) with credits (`ANTHROPIC_API=true`)
- Node.js + `npx` (for MCP servers)
- Pinecone account

### Installation

```bash
git clone https://github.com/UzairKamalYZ/uk-agents.git
cd uk-agents
uv venv && uv sync
```

Configure the environment:

```bash
cp .env-template .env
# Edit .env with your Anthropic key, Pinecone key, Telegram token, etc.
```

---

## Running the Clients

All commands are run from the **project root** using the `-m` flag so that all packages are importable.

### REST API

```bash
python -m clients.rest.main
# Listening on http://localhost:8000
```

```bash
# Blocking
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw"

# Streaming (SSE)
curl "http://localhost:8000/ask?prompt=3+bed+Warsaw&stream=true"
```

### Streamlit UI

Requires the REST API to be running first.

```bash
python -m clients.streamlit.main
# Opens http://localhost:8501
```

### Telegram bot

```bash
python -m clients.telegram.main
```

### Cron job

```bash
python -m clients.cron.main
# Fires CRON_SEARCH_PROMPT every 30 minutes
```

### WhatsApp listener (standalone)

Polls the configured WhatsApp group for new messages. When the owner is mentioned, sends the configured away message.

```bash
python -m agents.whatsapp.main
```

Requires `WHATSAPP_INSTANCE_ID`, `WHATSAPP_INSTANCE_TOKEN`, and `WHATSAPP_GROUP_ID` to be set in `.env`.

### Mail monitor (standalone)

Polls IMAP inboxes, summarises new emails with the LLM, and delivers a periodic digest.

```bash
python -m agents.mails.main
```

Requires `MAIL_ACCOUNTS` to be set in `.env`. Set `MAIL_SUMMARY_WHATSAPP_GROUP_ID` to deliver the digest to a WhatsApp group; leave it empty to write to the log instead.

---

## Adding an MCP Server

Edit `mcp.json` and restart. No Python changes required.

**stdio server** (subprocess):
```json
{"name": "my_server", "command": ["npx", "--yes", "@acme/my-mcp-server"]}
```

**HTTP server** (remote JSON-RPC):
```json
{
  "name": "my_api",
  "url": "https://api.example.com/mcp",
  "api_key_env": "MY_API_KEY",
  "api_key_param": "apikey"
}
```

Tools are auto-discovered via `tools/list` and bound to the LLM at startup. System prompts never hardcode tool names — the LLM learns them from the schema.

---

## Adding a New Agent

1. Create the agent class:

```python
# agents/myagent/agent.py
from core.src.base_agent import BaseAgent, NullRagContextManager
from core.src.utils import load_prompt

class MyAgent(BaseAgent):
    def get_system_prompt(self) -> str:
        return load_prompt("agents/myagent/prompts/System_Prompt.txt")

    def get_rag_context_manager(self):
        return NullRagContextManager()   # or RagContextManager(Embedder())

    def get_mcp_tools(self) -> list | None:
        return []   # [], None (all tools), or filtered list
```

2. Add a system prompt at `agents/myagent/prompts/System_Prompt.txt`.

3. Register in `agents.json`:

```json
{
  "name": "my_agent",
  "class": "agents.myagent.agent.MyAgent",
  "description": "Handles questions about X. Route here when the user asks about X.",
  "enabled": true,
  "rag": false
}
```

Every agent uses `LLM_PROVIDER` / `LLM_MODEL_NAME` from `.env`. To give one agent a different model, add `"llm_model": "claude-haiku-5-5"` (and optionally `"llm_provider"`) to its entry.

The router will automatically plan your agent in based on its description. No changes to `core/`, `orchestrator/`, or any client are needed.

An agent doesn't have to sit behind the orchestrator. If it needs to run on its own (polling, schedules, webhooks), give it a `main.py` entrypoint like `agents/whatsapp/main.py` or `agents/mails/main.py`, reuse `create_llm()` from `core/`, and run it with `python -m agents.<name>.main`.

Add a row for it to the [Agent catalog](#agent-catalog).

---

## Testing

```bash
# All tests
python -m pytest -v

# By layer
python -m pytest core/tests/ -v
python -m pytest clients/tests/ -v

# With coverage
python -m pytest --cov=core/src --cov=agents --cov=orchestrator --cov=clients --cov-report=term-missing
```

All external dependencies (LLM calls, vector store, network, Telegram API, MCP subprocesses) are mocked. No real services are required to run the test suite.

---

## LangSmith Observability

Set in `.env`:

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your-key>
LANGCHAIN_PROJECT=uk-agents
```

**Trace hierarchy** for a typical property search:

```
orchestrator.request
  └─ agent.PropertySearchAgent        ← @traceable on LlmModelGraph.ask()
       ├─ retrieve                    ← LangGraph node
       └─ generate                    ← LangGraph node
            └─ tool.yfin_get_quotes   ← @traceable on each tool invocation
```

Traces are visible at [smith.langchain.com](https://smith.langchain.com) under the configured project name.
