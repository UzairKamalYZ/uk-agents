# Expenditure Tracker

The single place where UK's spending is recorded and, over time, analysed.
This document describes how the tracker works today and where it is going. Keep it
up to date whenever the tracker changes: it is the reference for every future feature.

---

## 1. Purpose

- **Capture with no effort.** Say or type "Aldi 10" in Telegram and it's saved. No
  forms, no categories to pick.
- **Never lose a record.** Every expense and every message received is kept for the long run.
- **Answer spending questions.** Totals, biggest expenses, where the money went.
- **Grow into analysis.** Trends, categories, budgets and insights build on the same data
  (see [Roadmap](#7-roadmap)).

## 2. How a message flows

```
Telegram (text or voice note)
  │  voice → local faster-whisper transcript ("🎤 You said: …")
  ▼
OrchestratorAgent.ask(..., on_route=…)
  │
  ├─ normaliser   cleans text: "aldi ten" → "Aldi €10 2026-10-19"; resolves yes/no on a pending item
  ├─ supervisor   picks expenditure_tracker  → on_route → "🤖 Expenditure Tracker is working on your query…"
  ├─ expenditure_tracker
  │     ExpenditureAgent.ask prefixes "Today is …", Claude picks ONE of the agent's in-process tools
  │     and returns the tool's text verbatim
  └─ synthesiser  single agent → passthrough (the reply is already formatted)
  ▼
Telegram edits the status message into the final answer
```

Routing is logged (`[route] request=… supervisor → expenditure_tracker (…)`) and traced in
LangSmith (`route → expenditure_tracker`, tag `agent:expenditure_tracker`).

## 3. Behaviour

### Defaults

| Missing in the message | Assumed |
|---|---|
| currency | euros (the only currency) |
| date | today |

A missing currency or date **never** triggers a confirmation.

### What the user can say

| Intent | Example | Tool | Reply |
|---|---|---|---|
| Log an expense | "Aldi 10", "Colruyt 90 yesterday" | `add_expense` | ✅ saved + this/last month totals |
| Something's unclear | "spent something at the shop", "Colruyt 9000" | `propose_expense` | ⚠️ "Please check…" (saved only after "yes") |
| Confirm / discard / correct | "yes" / "no" / "it was 19" | normaliser → `confirm_pending` / `discard_pending` / agent | saved / "Discarded." / re-checked |
| Undo | "undo last", "delete the last one" | `undo_last_expense` | ↩️ removed + totals |
| How much did I spend? | "how much are my expenses", "what did I spend this month" | `get_expense_summary` | totals for this month and last month, plus the 5 largest expenses this month |
| Where did it go? | "spending per shop last month" | `get_expense_breakdown` | per-place totals for a month, largest first |
| Just the totals | (tool kept for compatibility) | `get_expense_totals` | this/last month totals |

Confirmation is asked **only** when the amount or place is missing or unrecognisable, or
when the amount or date looks implausible. When the amount is still unknown, "yes" doesn't
save; the bot asks for the amount again.

### Formatting rules

- Every reply is built in code (`agents/expenditure/replies.py`) and ends with
  `— Expenditure Tracker`. Claude must return it word for word.
- Money is always `€1,234.56`.
- Totals are **always computed by code** (SQL), never by Claude.

## 4. Data model

One SQLite file: `expenses.db` in `EXPENSES_DIR`. The default is the per-user data folder,
e.g. `~/Library/Application Support/uk-agents/expenses/` on macOS. It sits outside the
project, so restarts, `git clean` and redeploys don't touch it.

| Table / view | Holds | Notes |
|---|---|---|
| `expenses` | id, `spent_on` (YYYY-MM-DD), `month` (YYYY-MM), `place`, `amount_cents`, `created_at`, `deleted_at` | Amounts are integer cents. **Rows are never deleted**: undo sets `deleted_at`. |
| `inputs` | every message the agent received, with a timestamp, plus confirm/discard markers | The audit trail of what was said. |
| `pending` | at most one row: the expense awaiting "yes" and its problem | Single user, so one slot. |
| `monthly_totals` (view) | month, entries, total_eur | Per-month record for browsing (`sqlite3 expenses.db 'select * from monthly_totals'`). |

Each operation opens a
short-lived connection, with WAL mode and a 10 s busy timeout.

## 5. Components

| File | Responsibility |
|---|---|
| `agents/expenditure/ledger.py` | `ExpenseLedger`: all SQL; the only code that touches the database |
| `agents/expenditure/replies.py` | All user-facing text |
| `agents/expenditure/tools.py` | `build_tools()`: the six LangChain tools, run in-process. Validates input (positive amounts, ISO dates) and logs every tool call. Each tool's description is its function's docstring |
| `agents/expenditure/agent.py` | `ExpenditureAgent`: date prefix, pending hooks for the normaliser, refuses with "unavailable" if the tools didn't load |
| `agents/expenditure/prompts/System_Prompt.txt` | Which tool for which intent; reply-verbatim rule |
| `agents.json` → `expenditure_tracker` | Routing description, display name, waiting remarks |

## 6. Operations

| What | Where |
|---|---|
| Data | `$EXPENSES_DIR/expenses.db` |
| Bot / orchestrator log | `logs/agent.log` (rotating, 5 × 5 MB) |
| Tool-call log | `logs/agent.log` (logger `agents.expenditure.tools`): every tool call, its arguments and the first line of its reply |
| Traces | LangSmith project `LANGCHAIN_PROJECT`, tag `agent:expenditure_tracker` |
| Backup | Copy `expenses.db` (and `-wal` / `-shm` files if the bot is running), or `sqlite3 expenses.db ".backup backup.db"` |

## 7. Roadmap

Ordered roughly by value. Each item builds on the existing data, and **no stored data is
ever rewritten** for a new feature.

1. **Categories.** Add a `category` column (groceries, transport, rent, eating out…).
   Claude suggests one on save ("Aldi" → groceries), and the user can correct it. Then add
   per-category summaries.
2. **Trends.** Month-over-month change, a 3/6/12-month average, and "you spent 20% more
   on groceries than usual".
3. **Budgets.** A monthly budget overall or per category, with an alert in the save reply
   when it's at 80% or over.
4. **Search and edit.** "What did I spend at IKEA this year?" and "change yesterday's
   Aldi to 12". Edits keep the old row (soft-delete plus new row) so history stays complete.
5. **Recurring expenses.** Rent and subscriptions logged automatically each month, with a
   reminder if one is missing.
6. **Reports.** A monthly digest message on the 1st (opt-in), and CSV/Excel export.
7. **Receipts.** Send a photo and Claude reads the place, amount and date, then saves
   through the same propose/confirm flow.
8. **Multi-currency.** Store the original currency and amount next to the euro amount.

Design rules for all of the above:
- Computation stays in SQL or Python. Claude picks the tool and wording only.
- New behaviour comes in as a new tool plus a prompt line. The prompt never hardcodes
  tool names.
- Every new tool logs its calls and gets tests on a temporary database.
