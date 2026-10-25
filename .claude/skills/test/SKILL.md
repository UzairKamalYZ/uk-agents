Run the full test suite for this project.

```bash
uv run python -m pytest -q
```

Run it from the project root; no `.env` is needed (every setting has a test-safe default). CI runs the same command.

Test locations (see `testpaths` in `pyproject.toml`):
- `core/tests/` — BaseAgent, LlmModelGraph, MCP registry, Claude Code backend, history store, embedder, transcriber, output guard, config defaults
- `orchestrator/tests/` — graph routing, router, confirmations (interrupt/resume), conversation memory, agent nodes, registry, facade
- `clients/tests/` — REST, Telegram, cron, Streamlit and BaseClient
- `agents/<name>/tests/` — per-agent tests (expenditure ledger / tools / agent, FBR agent and tracker, property tools, conversational catalog, WhatsApp client)

Report:
1. How many tests passed, failed and were skipped
2. Full output for any failure, with the error message and traceback
3. Coverage, if asked for:
   ```bash
   uv run python -m pytest --cov=core --cov=orchestrator --cov=clients --cov=agents --cov-report=term-missing
   ```

---

## Testing Principles

### Naming
Test functions are named `test_<behaviour>` — pytest only collects names starting with `test`. Describe the behaviour, e.g. `test_yes_resumes_and_confirms_without_routing`, `test_stale_session_is_not_resumed`.

### Structure
Arrange, act, assert — in that order, without interleaving. A blank line between the phases is enough; comments are not required.

### What to test
- Behaviour through the public interface (`ask` / `aask`, graph runs, tool functions), not private helpers.
- Boundary conditions: empty strings, `None`, zero results, corrupt files, timeouts.
- Error paths: exceptions from external calls must be caught and their handling tested.
- Do not test framework internals (FastAPI routing, Streamlit rendering) or constructors with no logic.

### External services are never called
No test makes a real network or model call: LLMs, Pinecone, HTTP, Telegram, IMAP, Green API and Claude Code sessions are mocked or faked. Patch the name where it is used (`patch("clients.telegram.main.OrchestratorAgent")`), never `sys.modules` for project code; `core/tests/conftest.py` stubs only heavy third-party ML packages. Use `AsyncMock` for async methods.

### Prefer real local stand-ins
Where it is cheap, run the real thing locally instead of a mock:
- SQLite (session history, expense ledger, FBR tracker) on `tmp_path`
- the real LangGraph wiring with a scripted fake LLM (`orchestrator/tests/test_graph.py`; fakes must implement `with_structured_output`)
- real MCP servers — in-process `MCPServer` via `MCPRegistry.register_server()`, or `core/tests/model/mcp_test_server.py` over stdio
- real decoders (e.g. faster-whisper decoding a generated voice note)

### Independence
Tests must not share mutable state or depend on execution order. Use function-scoped fixtures and `tmp_path`; never touch the real `EXPENSES_DIR`, `sessions.db` or `logs/`. Shared fixtures go in the relevant `conftest.py`.

### Parametrize input variants
Use `@pytest.mark.parametrize` instead of copying a test for each input.

### Bug fixes
Write the failing test first, watch it fail for the right reason, then fix.
