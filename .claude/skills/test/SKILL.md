Run the full test suite for this project.

```bash
uv run python -m pytest -v --tb=short
```

Run it from the project root; no `.env` is needed.

Test locations (see `testpaths` in `pyproject.toml`):
- `core/tests/` — BaseAgent, LlmModelGraph, MCP client, Claude Code backend, history store, embedder, transcriber
- `orchestrator/tests/` — graph routing, confirmations (interrupt/resume), normaliser, agent nodes, registry
- `clients/tests/` — REST, Telegram, cron, Streamlit and BaseClient
- `agents/<name>/tests/` — per-agent tests (expenditure ledger / MCP server / agent, property tools)

Run the bash command above and report:
1. How many tests passed, failed, and were skipped
2. All tests follow the naming pattern `should_<expected_outcome>_when_<condition>`
3. Full output for any failures with the error message and traceback
4. Overall coverage if available

---

## Testing Principles

### Test Naming
All test functions must follow the pattern:
```
should_<expected_outcome>_when_<condition>
```
Examples:
- `should_return_listings_when_valid_query_provided`
- `should_raise_value_error_when_empty_prompt_given`
- `should_skip_embedding_when_vector_store_unavailable`

### Test Structure — Arrange / Act / Assert (AAA)
Every test must have three clearly separated phases:
```python
def should_return_answer_when_valid_prompt_provided():
    # Arrange
    agent = LocalAgent(model=mock_model)
    prompt = "2 bed flat in Warsaw"

    # Act
    result = agent.ask(prompt)

    # Assert
    assert result == "Here are some listings..."
```
Never mix setup, execution, and assertions into a single block.

### What to Test
- **Unit tests**: one class or function in isolation — mock all collaborators (LLM, vector store, HTTP, Telegram API).
- **Boundary conditions**: empty strings, `None`, max-length inputs, zero results, network timeouts.
- **Error paths**: exceptions raised from external calls must be caught and tested explicitly.
- **Do not test**: framework internals (FastAPI routing machinery, Streamlit rendering), `__init__` constructors with no logic, or trivial property accessors.

### Mocking Rules
- All external I/O **must** be mocked: LLM calls, vector store calls, HTTP requests, Telegram API, file reads.
- Use `unittest.mock.patch` or `pytest-mock`'s `mocker` fixture.
- Never instantiate real `LlmModelGraph`, `FAISSStore`, or `PineconeStore` in tests.
- Prefer `MagicMock` for collaborators; use `AsyncMock` for `async` methods.
- Mock at the **import boundary** — patch where the name is used, not where it is defined.

### Coverage
Run with coverage to get a term-missing report:
```bash
python -m pytest agentP/tests/ clients/tests/ \
  --cov=agentP/src --cov=clients \
  --cov-report=term-missing \
  --cov-fail-under=80
```
- Minimum threshold: **80%** overall.
- Every new module added to `agentP/src/` or `clients/` must have a corresponding test file.
- Uncovered lines in critical paths (RAG pipeline, vector store, session manager) must be justified with a comment.

### One Assertion Focus Per Test
Each test should verify **one behaviour**. If you need to assert multiple things about the same result, that is acceptable — but do not test two unrelated behaviours in a single test function.

### Test Independence
- Tests must not share mutable state; use `setUp` / teardown or pytest fixtures with function scope.
- Do not rely on test execution order.
- Temporary files or SQLite DBs created during tests must be cleaned up in a fixture or `finally` block.

### Fixtures
Prefer pytest fixtures over repeated setup code:
```python
@pytest.fixture
def mock_model():
    model = MagicMock()
    model.ask.return_value = "mocked answer"
    return model
```
Place shared fixtures in `conftest.py` at the relevant test directory level.

### Parametrize for Input Variants
Use `@pytest.mark.parametrize` instead of duplicating test functions for different inputs:
```python
@pytest.mark.parametrize("prompt,expected", [
    ("", ValueError),
    (None, TypeError),
    ("valid query", "some answer"),
])
def should_handle_prompt_variants_when_different_inputs_given(prompt, expected, mock_model):
    ...
```
