# Expenditure Tracker + Voice Input — Design

Date: 2026-10-18
Status: Draft, awaiting approval (builder check done)

## Intent

UK wants to log day-to-day spending by just telling the bot, e.g. "Colruyt 90 euros today",
either typed or as a Telegram voice note. Each log is written to a monthly text file and the
reply always shows how much was spent last month and so far this month. The aim is
effortless capture: talk into the phone, see the running total, move on.

Voice is a general enhancement to the main (orchestrator-driven) bot, not only for expenses:
any agent can be talked to by voice.

## Goals

1. **Voice input in Telegram.** Voice notes are transcribed locally (free, offline) and the
   transcript is shown to the user immediately; the transcript is then handled exactly like a
   typed message.
2. **Cleanup step in the main agent.** Every incoming message (voice or typed, any agent) is
   normalised before routing: misheard/misspelled words fixed, spoken numbers turned into
   amounts, relative dates ("today", "yesterday") resolved to real dates.
3. **New `expenditure_tracker` agent.** Records expenses into `expenses/YYYY-MM.txt` and ends
   every expense reply with both totals:
   ```
   Logged €90 at Colruyt (2026-10-18).
   prev_month: €1000, current_month: €120
   ```
4. **Confirm-or-correct.** If the agent sees a problem with the input (missing amount,
   unrecognisable place, implausible date/amount), it saves nothing and replies with what it
   understood, asking the user to reply **yes** to save or send a corrected version. "yes"
   saves it; anything else is treated as the corrected input and goes through the same check.

## Non-goals

- No scheduled / start-of-month message (explicitly dropped by UK).
- No categories, budgets, charts, or per-store breakdowns — only the two totals.
- No multi-currency; amounts are always euros.
- No paid / cloud speech-to-text.
- Voice for REST / Streamlit / WhatsApp clients is not in scope (Telegram only for now). The
  transcription component should be reusable so other clients can adopt it later.

## Anti-goals (would make this a failure even if it "works")

- Saving a wrong amount silently. When in doubt, confirm.
- A bare "yes" reply getting routed to the conversational agent as small talk instead of
  confirming the pending expense.
- The cleanup step rewording non-expense messages (property/stock questions) so their meaning
  changes. For those it should only fix obvious transcription/spelling errors.
- Audio leaving the machine.

## Current state (from the repo)

- `clients/telegram/main.py` only handles `filters.TEXT`; voice notes are ignored. It already
  posts a placeholder message and later `edit_text`s it with the answer.
- `orchestrator/graph.py`: START → supervisor → agent(s) → synthesiser → END. The supervisor
  sees only the current `user_input` and this turn's agent outputs — **no chat history** —
  so it cannot know a "yes" refers to a pending expense.
- Agents extend `BaseAgent` (`core/src/base_agent.py`) and are registered in `agents.json`.
  Tools come from MCP servers declared in `mcp.json`; prompts must not hardcode tool names
  (project standard — tools are discovered via MCP).
- Session history is SQLite (`SESSION_DB_FILE`, `core/src/model/session_manager.py`).
- No speech-to-text dependency exists.

## Decisions

### Decided by UK
- Agent name: expenditure tracker (`expenditure_tracker` in `agents.json`).
- Channel: Telegram first, typed or voice.
- Transcription: free, local.
- Monthly text file per month; every expense reply shows `prev_month` and `current_month`.
- A correction/cleanup step reformulates the incoming message.
- Single user: the bot is only for UK. Expense logging is locked to UK's Telegram user id
  (new `.env` setting `TELEGRAM_OWNER_ID`). Messages about expenses from any other user get a
  short refusal and nothing is written. One ledger, no per-user files.
- "Undo last" is in scope: the user can say e.g. "undo last" / "delete the last one"; the
  most recent line in the ledger is removed and the reply shows what was removed plus the
  updated totals.
- On a problem: show what was understood, "yes" to confirm, otherwise the reply is the
  corrected input.

### Claude's calls (UK may override)
- **Transcription engine:** `faster-whisper`, multilingual model (auto language detection —
  UK may speak English, Dutch or French). Default model size `small`, configurable via
  `.env` (`WHISPER_MODEL`, e.g. `base`/`small`/`medium`), CPU, int8. Model loaded once at bot
  startup. Telegram voice is OGG/Opus; faster-whisper decodes it via PyAV, so no ffmpeg
  binary required.
- **Transcript display:** bot first replies `🎤 You said: <transcript>` plus the thinking
  placeholder, then edits that same message to append the answer (keeping the transcript
  line visible).
- **Cleanup step placement:** a new `normaliser` node in the orchestrator graph, between
  START and the supervisor. It rewrites `user_input` (keeps the raw text in a new
  `raw_input` state field). It's given today's date. Prompt in
  `orchestrator/prompts/normaliser.txt`. Applies to all messages regardless of channel.
- **File format:** `expenses/YYYY-MM.txt` at the project root (dir configurable via
  `EXPENSES_DIR`), one line per expense, tab- or pipe-separated:
  `2026-10-18 | Colruyt | 90.00`. Month file is chosen by the expense date, not the
  logging date. `expenses/` added to `.gitignore`.
- **Tooling:** a small local Python MCP server (stdio) registered in `mcp.json` as
  `expenses`, exposing three tools: add an expense (date, place, amount), undo the last
  added expense (most recently written line across month files — track it so a back-dated
  entry is still the one undone), and get totals (previous- and current-month totals
  relative to today). The agent's
  `get_mcp_tools()` returns `_mcp.langchain_tools(server_name="expenses")`, same as
  `FinanceAgent`. Totals are computed by code, never by the LLM.
- **Pending confirmation / "yes" routing:** a generic "sticky route" in the orchestrator.
  When the expenditure agent asks for confirmation, it records a pending entry for the
  session (session_id → agent name + the understood expense) in a new SQLite table in
  `SESSION_DB_FILE` (note: agent chat history is currently in memory in
  `LlmModelGraph._histories`; `SessionManager` is unused, so this is a new table). On the
  next message for that session, the normaliser also classifies the message against the
  pending entry as one of: `confirm` ("yes", "y", "ja", "oui", "ok"…), `cancel` ("no",
  "cancel", "laat maar"…), `correction` (about the same expense), or `unrelated`.
  - `confirm` → route straight to the expenditure agent, which saves the pending expense.
  - `correction` → route straight to the expenditure agent with the pending expense plus the
    new message; it is re-checked like a new input.
  - `cancel` → pending cleared, short "Discarded." reply.
  - `unrelated` (property/stock question, a different expense…) → pending cleared, message
    goes through the supervisor as normal.
  The pending entry is also cleared once saved. No timeout. How the agent signals
  "pending" (e.g. a tool call that writes the pending record vs. a structured marker in its
  output) is left to the builder, but it must be decided in code, not by fragile text
  parsing of the reply.
- **Confirmation is triggered only when the agent sees a problem** (UK's rule), for voice and
  typed alike — no extra always-confirm for voice. To make plausible-but-wrong numbers less
  likely to slip through, the expense reply always echoes the saved date, place and amount.
- **Totals line must survive verbatim.** If the synthesiser merges outputs from several
  agents, its prompt must keep the `prev_month / current_month` line unchanged.
- **Non-expense questions to this agent** ("how much did I spend this month?") are answered
  with the two totals, using the same totals tool.
- **Owner lock placement:** enforced in the Telegram client before the orchestrator is
  called — non-owner chats can still use the other agents but `expenditure_tracker` must
  refuse. Simplest robust approach: the client passes an owner flag into the graph state
  (via an optional kwarg on `OrchestratorAgent.ask`, default safe = not owner), and the
  expenditure agent node refuses when it's false. REST/cron callers are non-owner by
  default. If `TELEGRAM_OWNER_ID` is unset, expense logging is disabled for everyone.
- Telegram `/start` greeting updated to stop saying "property agent" only.

## Constraints

- Follow `CLAUDE.md` conventions: Python 3.13, type hints, class-based modules, config via
  `Config`, `setup_logging()` in entrypoints, new deps in `pyproject.toml`.
- Tests mock everything external: Whisper model, LLM, Telegram, filesystem (use tmp dirs).
  No real model or network calls.
- Existing tests and public signatures (`OrchestratorAgent.ask`, `BaseAgent.ask`) keep
  working. Adding the `normaliser` node must not break current routing tests.
- Works in both LLM backends (`ANTHROPIC_API=true/false`); in Claude Code mode tools run via
  the in-process MCP bridge, so the expense tools must work there too.

## Left to the builder

- Exact normaliser prompt wording and expenditure agent system prompt.
- Exact line format/separator and how amounts with cents are parsed/rounded (store 2 dp).
- Whether `prev_month` / `current_month` are relative to today or to the expense date
  (default: today).
- The mechanism by which the agent signals a pending confirmation (see above).
- Where the transcription helper lives (suggest `core/src/speech/transcriber.py` so other
  clients can reuse it).
- Handling of empty/failed transcriptions (reply asking the user to try again or type it).
