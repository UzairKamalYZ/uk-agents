# Expenditure Tracker + Voice Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log expenses by typed or voice Telegram messages into monthly text files, with a cleanup step, confirm-or-correct flow, undo, owner lock, and `prev_month / current_month` totals on every expense reply.

**Architecture:** A pure-Python `ExpenseLedger` owns the files; a local stdio MCP server exposes it as tools to a new `ExpenditureAgent`. The orchestrator gains a `normaliser` node before the supervisor (cleans text, resolves pending confirmations deterministically) and an owner flag in graph state. Telegram transcribes voice locally with faster-whisper and feeds the transcript in like typed text.

**Tech Stack:** Python 3.13, LangGraph, langchain-core, `mcp` 2.x (`MCPServer`), faster-whisper, python-telegram-bot 21, pytest/unittest.

**Spec:** `docs/superpowers/specs/2026-10-18-expenditure-tracker-design.md`

## Global Constraints

- Agent routing name `expenditure_tracker`; files `expenses/YYYY-MM.txt`, line format `2026-10-18 | Colruyt | 90.00`; dir from `Config.EXPENSES_DIR` (default `expenses`, resolved from project root); `expenses/` in `.gitignore`.
- Totals line, exact shape: `prev_month: €1000, current_month: €120` (whole euros drop `.00`; otherwise 2 dp, e.g. `€12.50`). Totals are computed in code, relative to today.
- Euros only. No categories, budgets, scheduled messages.
- Expense logging only when `Config.TELEGRAM_OWNER_ID` is set and equals the Telegram user id; REST/cron are never owner. Other agents stay usable by everyone.
- Transcription local only: faster-whisper, `Config.WHISPER_MODEL` default `small`, `device="cpu"`, `compute_type="int8"`, language auto-detected. Audio never leaves the machine.
- Prompts must not hardcode MCP tool names (project standard — tools are discovered).
- All tests mock LLM, Whisper, Telegram; filesystem tests use `tmp_path`/`tempfile`. No network/model calls.
- Keep `OrchestratorAgent.ask(prompt, stream=False, session_id="default")` and `BaseAgent.ask` working unchanged for existing callers; new params are keyword-only with safe defaults.
- Follow `CLAUDE.md`: type hints, class-based modules, config via `Config`, deps in `pyproject.toml`.

## Review Focus

1. A bare "yes" with **no** pending expense must route normally (supervisor → conversational), not error or log anything. → Task 5 test `test_yes_without_pending_goes_to_supervisor`.
2. Back-dated expense ("yesterday" on the 1st) lands in the previous month's file, moves `prev_month`, and "undo last" removes it from that file. → Task 1 test `test_backdated_expense_and_undo_hit_previous_month_file`.
3. Totals on 1 January must use December of the previous year. → Task 1 test `test_totals_across_year_boundary`.
4. Silent/empty voice note (empty transcript) must not reach the orchestrator; user is asked to retry or type. → Task 7 test `test_empty_transcript_asks_to_retry`.
5. Float noise / comma decimals from the LLM (`12.499999`, `"12,5"`) must store as `12.50`; a `|` in a place name must not corrupt the line. → Task 1 tests `test_amount_rounding_and_comma`, `test_pipe_in_place_is_sanitised`.

---

### Task 1: Expense ledger

**Files:**
- Create: `agents/expenditure/__init__.py`, `agents/expenditure/ledger.py`, `agents/expenditure/replies.py`
- Create: `agents/expenditure/tests/__init__.py`, `agents/expenditure/tests/test_ledger.py`
- Modify: `core/src/config/config.py` (add `EXPENSES_DIR`), `.gitignore` (add `expenses/`), `pyproject.toml` (`testpaths` += `"agents/expenditure/tests"`, `"orchestrator/tests"`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Expense: date: datetime.date; place: str; amount: Decimal` with `def line(self) -> str` → `"2026-10-18 | Colruyt | 90.00"`.
  - `class ExpenseLedger(root: Path)`:
    - `add(date: date, place: str, amount: Decimal | float | str) -> Expense` — appends to `root/YYYY-MM.txt` (by expense date), pushes onto `root/history.json`, clears nothing else.
    - `undo_last() -> Expense | None` — pops `history.json`, removes the **last** matching line from that month's file; `None` if history empty.
    - `totals(today: date) -> tuple[Decimal, Decimal]` — `(prev_month, current_month)`, summed from files.
    - `set_pending(expense: Expense, problem: str) -> None`, `get_pending() -> tuple[Expense, str] | None`, `clear_pending() -> None`, `confirm_pending() -> Expense | None` (adds pending via `add`, clears it). Stored in `root/pending.json`.
  - `replies.py`: `format_eur(amount: Decimal) -> str`; `format_totals(prev: Decimal, cur: Decimal) -> str`; `format_logged(e: Expense, prev, cur) -> str` → `"Logged €90 at Colruyt (2026-10-18).\nprev_month: €1000, current_month: €120"`; `format_undone(e, prev, cur) -> str` → `"Removed €90 at Colruyt (2026-10-18).\n<totals>"`; `format_question(e: Expense) -> str` → `"I understood: Colruyt, €19, 2026-10-18. Reply yes to save, or send the corrected version."`.
  - `Config.EXPENSES_DIR = str(_PROJECT_ROOT / os.getenv("EXPENSES_DIR", "expenses"))`.

Amount normalisation: `Decimal(str(x).replace(",", "."))` quantised to `0.01` with `ROUND_HALF_UP`. Place: strip, replace `|` with `/`. Create `root` lazily on first write.

> Note (deviation from spec, builder's call): pending is stored in `root/pending.json`, not a SQLite table, because MCP tools don't receive `session_id` and there is a single owner — one pending slot is sufficient and keeps the MCP server and agent sharing one store.

- [ ] **Step 1: Write failing tests** in `agents/expenditure/tests/test_ledger.py` (use `tmp_path`):
  - `test_add_writes_month_file_line`: `add(date(2026,10,18), "Colruyt", 90)` → `(tmp/"2026-10.txt").read_text() == "2026-10-18 | Colruyt | 90.00\n"`.
  - `test_totals_prev_and_current`: add 1000 on 2026-09-05, 30 on 2026-10-02 → `totals(date(2026,10,18)) == (Decimal("1000.00"), Decimal("30.00"))`; empty ledger → `(Decimal("0"), Decimal("0"))` (compare numerically).
  - `test_totals_across_year_boundary`: add 50 on 2026-12-31 → `totals(date(2027,1,1))[0] == Decimal("50.00")`.
  - `test_backdated_expense_and_undo_hit_previous_month_file`: add 10 on 2026-10-01, then 20 on 2026-09-30; `undo_last()` returns the 20 expense; `2026-09.txt` has no `20.00` line; `2026-10.txt` unchanged.
  - `test_undo_on_empty_returns_none`.
  - `test_undo_removes_only_last_duplicate`: two identical adds, one undo → exactly one line remains.
  - `test_amount_rounding_and_comma`: `add(..., 12.499999).amount == Decimal("12.50")`; `add(..., "12,5").amount == Decimal("12.50")`.
  - `test_pipe_in_place_is_sanitised`: place `"A|B"` → line contains `"A/B"`, file has exactly 3 ` | `-separated fields.
  - `test_pending_roundtrip_and_confirm`: `set_pending(e, "amount unclear")`; `get_pending() == (e, "amount unclear")`; `confirm_pending() == e`; `get_pending() is None`; month file contains the line. `confirm_pending()` with nothing pending → `None`.
  - `test_replies_format`: `format_totals(Decimal("1000"), Decimal("12.5")) == "prev_month: €1000, current_month: €12.50"`; `format_question(Expense(date(2026,10,18),"Colruyt",Decimal("19.00"))) == "I understood: Colruyt, €19, 2026-10-18. Reply yes to save, or send the corrected version."`.
- [ ] **Step 2: Run** `python -m pytest agents/expenditure/tests/test_ledger.py -v` → FAIL (module not found).
- [ ] **Step 3: Implement** `ledger.py`, `replies.py`, the `Config`, `.gitignore`, and `testpaths` changes.
- [ ] **Step 4: Run** the same command → all PASS.
- [ ] **Step 5: Commit** `feat(expenditure): add expense ledger and reply formatting`

---

### Task 2: Expenses MCP server

**Files:**
- Create: `agents/expenditure/mcp_server.py`, `agents/expenditure/tests/test_mcp_server.py`
- Modify: `mcp.json` (add `{"name": "expenses", "command": ["{python}", "-m", "agents.expenditure.mcp_server"]}`), `core/src/model/mcp_tools.py` (replace a `"{python}"` command element with `sys.executable`), `core/tests/model/test_mcp_tools.py`, `pyproject.toml` (add `"mcp>=2.0"` explicitly — already present transitively)

**Interfaces:**
- Consumes: `ExpenseLedger`, `replies.*` from Task 1; `Config.EXPENSES_DIR`.
- Produces: an MCP server named `expenses` with four tools. Each is also a plain module-level function (so it's unit-testable) taking an optional `ledger: ExpenseLedger | None = None` keyword that defaults to `ExpenseLedger(Path(Config.EXPENSES_DIR))`; only the LLM-facing params are exposed in the tool schema:
  - `add_expense(date: str, place: str, amount: float) -> str` → `format_logged(...)` with today's totals. Docstring: save a clearly understood expense; date `YYYY-MM-DD`.
  - `propose_expense(date: str, place: str, amount: float, problem: str) -> str` → stores pending, returns `format_question(...)`. Docstring: use instead of saving when anything about the input looks wrong or uncertain.
  - `undo_last_expense() -> str` → `format_undone(...)` or `"Nothing to undo."`.
  - `get_expense_totals() -> str` → `format_totals(...)`.
  - `add_expense` clears any pending entry (a correction that is now clear gets saved and the old pending must not linger); `propose_expense` replaces it.
  - Invalid date string → returns `"Invalid date '<value>', expected YYYY-MM-DD."` (no exception).

Use the mcp 2.x API: `from mcp.server.mcpserver import MCPServer`; `server = MCPServer("expenses")`; register each function as a tool; `if __name__ == "__main__": server.run()` (stdio). Verify the decorator/registration call and `run()` transport default against the installed package (`.venv/lib/python3.13/site-packages/mcp/server/mcpserver/`) before writing — FastMCP was renamed in 2.x and APIs changed. Make sure the `ledger` kwarg is excluded from the schema (register thin wrappers without it if needed).

- [ ] **Step 1: Write failing tests** in `test_mcp_server.py`:
  - `test_add_expense_returns_logged_reply` (ledger on `tmp_path`, patch `agents.expenditure.mcp_server.date` or pass through a `today` seam you add) → reply starts with `"Logged €90 at Colruyt (2026-10-18)."` and contains `"prev_month: "`.
  - `test_propose_expense_stores_pending_and_writes_nothing` → returns the question text; `ledger.get_pending()` not None; no month file exists.
  - `test_add_expense_clears_pending`.
  - `test_undo_last_expense_empty` → `"Nothing to undo."`.
  - `test_invalid_date` → `"Invalid date 'yesterday', expected YYYY-MM-DD."`.
  - `test_server_lists_four_tools`: spawn `[sys.executable, "-m", "agents.expenditure.mcp_server"]` through `MCPProcess` from `core/src/model/mcp_registry.py` with env `EXPENSES_DIR=<tmp>`, `list_tools()` names == `{"add_expense","propose_expense","undo_last_expense","get_expense_totals"}`, and no tool schema contains `ledger`. (Local subprocess, no network.)
  - In `core/tests/model/test_mcp_tools.py`: `test_python_placeholder_resolves_to_sys_executable` — `_load_registry` on a tmp json with `["{python}", "-m", "x"]` → registered command `[sys.executable, "-m", "x"]`.
- [ ] **Step 2: Run** `python -m pytest agents/expenditure/tests/test_mcp_server.py core/tests/model/test_mcp_tools.py -v` → new tests FAIL.
- [ ] **Step 3: Implement** server, placeholder substitution, `mcp.json` entry, dependency.
- [ ] **Step 4: Run** same command → PASS (existing mcp_tools tests still pass).
- [ ] **Step 5: Commit** `feat(expenditure): expose ledger as local expenses MCP server`

---

### Task 3: ExpenditureAgent + pending hooks on BaseAgent

**Files:**
- Create: `agents/expenditure/agent.py`, `agents/expenditure/prompts/System_Prompt.txt`, `agents/expenditure/tests/test_agent.py`
- Modify: `core/src/base_agent.py`, `core/tests/test_base_agent.py`, `agents.json`

**Interfaces:**
- Consumes: `_mcp.langchain_tools(server_name="expenses")`; `ExpenseLedger`, `replies` (Task 1).
- Produces:
  - On `BaseAgent` (defaults, non-abstract): `pending_summary(self, session_id: str) -> str | None` → `None`; `confirm_pending(self, session_id: str) -> str` → `""`; `discard_pending(self, session_id: str) -> None`.
  - `class ExpenditureAgent(BaseAgent)` in `agents/expenditure/agent.py`, same shape as `agents/finance/agent.py` (`NullRagContextManager`, prompt via `load_prompt`). Overrides (ignore `session_id` — single owner; ledger from `Config.EXPENSES_DIR`, created in `__init__` after `super().__init__`, or lazily):
    - `pending_summary` → `"Colruyt, €19, 2026-10-18 (problem: amount unclear)"` or `None`.
    - `confirm_pending` → `format_logged(e, *totals(today))`, or `"Nothing to confirm."`.
    - `discard_pending` → `ledger.clear_pending()`.
  - `agents.json` entry: `{"name": "expenditure_tracker", "class": "agents.expenditure.agent.ExpenditureAgent", "description": "Logs the owner's personal spending (e.g. 'Colruyt 90 euros today'), undoes the last logged expense, and answers how much was spent this or last month. Route here for any message about money spent, purchases, expenses or spending totals. Does not handle stocks, investments or property.", "enabled": true, "rag": false, "owner_only": true}`.

System prompt content (wording is the implementer's): today's date is in the user message (the normaliser resolves dates); save clear expenses; when amount/place/date is missing, unrecognisable or implausible, propose instead of saving; for "undo"/"delete last" undo; for spending questions get totals; **reply with exactly the text the tool returned, nothing added**; never compute totals yourself; never name tools.

- [ ] **Step 1: Write failing tests**:
  - `core/tests/test_base_agent.py::test_pending_hooks_default_noop` → a minimal concrete subclass returns `None`, `""`, `None`.
  - `test_agent.py` (patch `agents.expenditure.agent.ExpenseLedger` or set `Config.EXPENSES_DIR` to `tmp_path`; patch LLM creation the way `core/tests/test_base_agent.py` does): `test_confirm_pending_logs_and_returns_totals` → after `ledger.set_pending(...)`, `confirm_pending("s")` starts with `"Logged €19 at Colruyt"` and file has the line; `test_confirm_without_pending` → `"Nothing to confirm."`; `test_pending_summary_and_discard`; `test_uses_expenses_mcp_server` → `get_mcp_tools()` calls `_mcp.langchain_tools(server_name="expenses")`.
- [ ] **Step 2: Run** `python -m pytest core/tests/test_base_agent.py agents/expenditure/tests/test_agent.py -v` → FAIL.
- [ ] **Step 3: Implement** hooks, agent, prompt, `agents.json` entry.
- [ ] **Step 4: Run** same → PASS.
- [ ] **Step 5: Commit** `feat(expenditure): add ExpenditureAgent and pending hooks`

---

### Task 4: Owner flag through the orchestrator

**Files:**
- Modify: `orchestrator/state.py`, `orchestrator/agent_registry.py`, `orchestrator/nodes/agent.py`, `orchestrator/graph.py`, `orchestrator/agent_interface.py`
- Create: `orchestrator/tests/__init__.py`, `orchestrator/tests/conftest.py`, `orchestrator/tests/test_agent_node.py`, `orchestrator/tests/test_agent_registry.py`

**Interfaces:**
- Produces:
  - `GraphState` new fields: `is_owner: bool`, `raw_input: str`, `direct: bool`.
  - `AgentEntry.owner_only: bool = False` (from `spec.get("owner_only", False)`).
  - `AgentNode(agent_name: str, agent_instance: Any, owner_only: bool = False)`; when `owner_only and not state.get("is_owner", False)` it does **not** call the agent and outputs `"Sorry, this is only available to the bot's owner."` (still increments `turns`).
  - `MultiAgentGraph.ask(user_input, session_id="default", *, is_owner: bool = False)`, same kwarg on `stream` and `stream_tokens`; initial state sets `is_owner`, `raw_input=user_input`, `direct=False`. Factor the duplicated initial-state dict into `_initial_state(user_input, session_id, is_owner) -> GraphState`.
  - `OrchestratorAgent.ask(prompt, stream=False, session_id="default", *, is_owner: bool = False)` → passed through both paths.

`orchestrator/tests/conftest.py`: set the env vars `core/tests/conftest.py` sets and stub `langchain_anthropic` and `claude_agent_sdk` like `clients/tests/conftest.py`, so `orchestrator.graph` imports.

- [ ] **Step 1: Write failing tests**:
  - `test_agent_node.py`: `test_owner_only_refuses_non_owner` (agent mock not called; output text exact); `test_owner_only_allows_owner`; `test_regular_agent_ignores_owner_flag`.
  - `test_agent_registry.py`: tmp agents.json with a mock-able class path (e.g. a tiny class in the test module) → `owner_only` read, default `False`.
- [ ] **Step 2: Run** `python -m pytest orchestrator/tests -v` → FAIL.
- [ ] **Step 3: Implement**; in `graph.py` pass `agent.owner_only` into `AgentNode`.
- [ ] **Step 4: Run** `python -m pytest orchestrator/tests clients/tests -v` → PASS.
- [ ] **Step 5: Commit** `feat(orchestrator): carry owner flag and gate owner-only agents`

---

### Task 5: Normaliser node, pending resolution, graph wiring

**Files:**
- Create: `orchestrator/nodes/normaliser.py`, `orchestrator/prompts/normaliser.txt`, `orchestrator/tests/test_normaliser.py`, `orchestrator/tests/test_graph.py`
- Modify: `orchestrator/nodes/__init__.py`, `orchestrator/graph.py`, `core/src/config/config.py` (`NORMALISER_PROMPT_FILE`, default `orchestrator/prompts/normaliser.txt`), `orchestrator/prompts/synthesiser.txt`

**Interfaces:**
- Consumes: `AgentEntry` (with `owner_only`, `instance` exposing the Task 3 hooks); `GraphState` fields from Task 4.
- Produces: `NormaliserNode(llm: BaseChatModel, prompt_template: str, agents: list[AgentEntry], today: Callable[[], date] = date.today)`; `__call__(state) -> dict`.

Behaviour:
1. Find the first agent with `pending_summary(session_id)` not `None`, skipping `owner_only` agents when `not is_owner`.
2. One LLM call: prompt template with `{today}` (ISO date + weekday) and `{pending}` (`summary` or `None`) replaced; human message = raw input. Expected reply JSON `{"text": "...", "pending": "confirm" | "cancel" | "correction" | "unrelated" | null}` (parse like `SupervisorNode._parse_next`: strip fences, json, regex fallback).
3. Results:
   - Parse failure → `{"user_input": raw}`; pending untouched; go to supervisor.
   - No pending → `{"user_input": text}` → supervisor.
   - `confirm` → `{"agent_outputs": {name: agent.confirm_pending(sid)}, "next": "synthesiser"}`.
   - `cancel` → `discard_pending`; `{"agent_outputs": {name: "Discarded."}, "next": "synthesiser"}`.
   - `correction` → `{"user_input": f"{text}\n\n(Correction to the pending expense: {summary})", "next": name, "direct": True}`.
   - `unrelated` (or null while pending) → `discard_pending`; `{"user_input": text}` → supervisor.

Prompt rules (wording the implementer's): fix transcription/spelling errors, spoken numbers → digits with `€`, relative dates → `YYYY-MM-DD`; for messages not about spending, only fix obvious errors — never reword or change meaning; classify against pending only when one exists.

Graph: `START → normaliser`; conditional edge from normaliser: `state["next"]` in agent names → that agent; `"synthesiser"` → synthesiser; else `"supervisor"`. In `_make_router`, if `state.get("direct")` and `agent_outputs` is non-empty → `"synthesiser"`. Normaliser uses the shared LLM.

Synthesiser prompt: add a rule that any line starting with `prev_month:` is copied verbatim; and generalise "property-search assistant" to "personal assistant".

- [ ] **Step 1: Write failing tests** (`FakeLLM` returning a fixed `.text`; agents as `AgentEntry` with `MagicMock` instances):
  - `test_normaliser.py`: `test_cleans_text_without_pending` (`user_input` == LLM text, LLM got `{today}` filled: prompt contains `"2026-10-18"`); `test_confirm_calls_confirm_pending`; `test_cancel_discards`; `test_correction_routes_direct_with_summary`; `test_unrelated_discards_and_goes_to_supervisor`; `test_parse_failure_keeps_raw_and_pending` (`discard_pending` not called); `test_owner_only_pending_ignored_for_non_owner` (`pending_summary` not consulted, LLM sees `None`).
  - `test_graph.py` (patch `orchestrator.graph.create_llm`, `orchestrator.graph.load_agents`, `orchestrator.graph.load_prompt`; a scripted fake LLM that answers normaliser/supervisor calls by inspecting the system prompt):
    - `test_yes_without_pending_goes_to_supervisor`: "yes", no pending → supervisor routes to `conversational_agent`; expenditure agent `ask` and `confirm_pending` not called.
    - `test_confirm_skips_supervisor`: pending present, normaliser says `confirm` → answer == `confirm_pending` return; supervisor LLM never invoked.
    - `test_correction_runs_agent_once_then_synthesiser`: `direct` path → expenditure `ask` called once, no second supervisor decision.
- [ ] **Step 2: Run** `python -m pytest orchestrator/tests -v` → new tests FAIL.
- [ ] **Step 3: Implement** node, prompt, config, wiring, synthesiser prompt.
- [ ] **Step 4: Run** `python -m pytest orchestrator/tests core/tests clients/tests -v` → PASS.
- [ ] **Step 5: Commit** `feat(orchestrator): add normaliser node with pending confirmation routing`

---

### Task 6: Local transcriber

**Files:**
- Create: `core/src/speech/__init__.py`, `core/src/speech/transcriber.py`, `core/tests/speech/__init__.py`, `core/tests/speech/test_transcriber.py`
- Modify: `core/src/config/config.py` (`WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")`), `pyproject.toml` (`"faster-whisper>=1.1.0"`), `core/tests/conftest.py` (stub `faster_whisper`)

**Interfaces:**
- Produces: `class Transcriber(model_size: str = Config.WHISPER_MODEL)`; `transcribe(self, audio_path: str | Path) -> str`. `WhisperModel(model_size, device="cpu", compute_type="int8")` created lazily on first `transcribe` and reused; `model.transcribe(str(path), vad_filter=True)`; join `segment.text.strip()` with spaces; return `""` when no segments.

- [ ] **Step 1: Write failing tests**: `test_joins_segments` (stub returns segments `" Colruyt"`, `" ninety euros "` → `"Colruyt ninety euros"`); `test_empty_audio_returns_empty_string`; `test_model_loaded_once_with_cpu_int8` (two calls → `WhisperModel` constructed once with `("small", device="cpu", compute_type="int8")`).
- [ ] **Step 2: Run** `python -m pytest core/tests/speech -v` → FAIL.
- [ ] **Step 3: Implement**; run `uv add faster-whisper` (updates `pyproject.toml` + `uv.lock`).
- [ ] **Step 4: Run** same → PASS.
- [ ] **Step 5: Commit** `feat(core): add local faster-whisper transcriber`

---

### Task 7: Telegram voice notes + owner id

**Files:**
- Modify: `clients/telegram/main.py`, `clients/tests/test_telegram_client.py`, `clients/tests/conftest.py` (stub `core.src.speech.transcriber` if needed), `core/src/config/config.py` (`TELEGRAM_OWNER_ID = os.getenv("TELEGRAM_OWNER_ID", "")`), `.env-template` (add `TELEGRAM_OWNER_ID=`, `WHISPER_MODEL=small`, `EXPENSES_DIR=expenses`)

**Interfaces:**
- Consumes: `Transcriber` (Task 6); `OrchestratorAgent.ask(..., is_owner=...)` (Task 4).
- Produces in `_Bot`:
  - `_is_owner(user_id: int) -> bool` → `bool(Config.TELEGRAM_OWNER_ID) and str(user_id) == Config.TELEGRAM_OWNER_ID`.
  - `_answer(update, text: str, header: str = "") -> None` — the existing placeholder/joke + executor + `edit_text` flow, now passing `is_owner`, with `header` prefixed to both placeholder and final text. `_handle_message` becomes a call to it.
  - `_handle_voice(update, context)` registered with `MessageHandler(filters.VOICE, ...)`: download via `await (await context.bot.get_file(update.message.voice.file_id)).download_to_drive(path)` into a `tempfile.TemporaryDirectory()`, transcribe in the executor, then:
    - empty transcript → reply `"🎤 I couldn't hear anything — please try again or type it."`, orchestrator not called;
    - else `_answer(update, transcript, header=f"🎤 You said: {transcript}\n\n")`.
    - transcription exception → log and reply `"Sorry, I couldn't transcribe that voice note. Please try again or type it."`.
  - `Transcriber()` constructed once in `_Bot.__init__`.
  - `/start` text: `"Hello! I'm your personal assistant. Send me a message or a voice note."`
  - Waiting placeholder text: drop "property" wording → `"Thinking...\n\n💡 While you wait: {joke}"`.

- [ ] **Step 1: Write failing tests** (run handlers with `asyncio.run`, `AsyncMock` update/context, patched `Transcriber` and `Config`):
  - `test_voice_shows_transcript_then_answer`: transcript `"Colruyt 90 euros today"`, agent returns `"Logged €90…"` → final `edit_text` arg == `"🎤 You said: Colruyt 90 euros today\n\nLogged €90…"`; `ask` called with `"Colruyt 90 euros today"`.
  - `test_empty_transcript_asks_to_retry`: agent `ask` not called; reply text exact.
  - `test_transcription_error_replies_sorry`.
  - `test_owner_flag_true_for_owner_id` / `test_owner_flag_false_when_unset_or_other_user`: `ask` received `is_owner=True` / `False`.
- [ ] **Step 2: Run** `python -m pytest clients/tests/test_telegram_client.py -v` → new tests FAIL.
- [ ] **Step 3: Implement**.
- [ ] **Step 4: Run** full suite `python -m pytest -v` → PASS.
- [ ] **Step 5: Commit** `feat(telegram): voice notes via local transcription and owner-gated expenses`

---

### Task 8: End-to-end check (manual, with UK)

- [ ] Set `TELEGRAM_OWNER_ID` in `.env` (UK's id from @userinfobot), start `python -m clients.telegram.main`.
- [ ] Voice "Colruyt 90 euros today" → transcript shown, `Logged €90 at Colruyt (…)` + totals; `expenses/2026-10.txt` has the line.
- [ ] Typed "spent something at the shop" → confirmation question; "yes" → logged; repeat and send "no" → `Discarded.`; repeat and ask a stock question → pending dropped, stock answer.
- [ ] "undo last" → `Removed …` + updated totals.
- [ ] From a second Telegram account, an expense message → owner-only refusal; a greeting still works.
