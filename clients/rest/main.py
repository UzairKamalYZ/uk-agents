from logging_config import setup_logging
setup_logging()

import asyncio
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from orchestrator.agent_interface import OrchestratorAgent
from core.src.config.config import Config
from clients.base import BaseClient

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validates the API key header when API_KEY is configured."""
    expected = Config.API_KEY
    if expected and api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent = OrchestratorAgent()
    app.state.agent = agent
    print("✅ Agent started.")
    yield
    agent.close()
    print("🧹 Agent closed successfully.")


app = FastAPI(lifespan=lifespan)


async def sse_formatter(stream):
    """Formats a generator of strings into Server-Sent Events.

    The agent's generator is blocking, so each chunk is pulled in a worker thread.
    """
    try:
        async for chunk in iterate_in_threadpool(stream):
            yield f"data: {chunk}\n\n"
    except asyncio.CancelledError:
        print("Client disconnected.")


@app.get("/ask")
async def ask(
    prompt: str,
    stream: bool = False,
    session_id: str | None = None,
    _: None = Depends(verify_api_key),
):
    """Endpoint to ask the agent a question.

    Pass the same session_id on each call to keep a conversation; without one,
    every request starts a fresh session instead of sharing a global history.
    """
    return await _ask(app.state.agent, prompt, stream, session_id)


async def _ask(agent, prompt: str, stream: bool, session_id: str | None):
    session_id = session_id or str(uuid.uuid4())
    if stream:
        return StreamingResponse(
            sse_formatter(agent.ask(prompt, stream=True, session_id=session_id)),
            media_type="text/event-stream",
            headers={"X-Session-Id": session_id},
        )
    # The pipeline is blocking — run it in a worker thread so other requests keep being served
    response = await run_in_threadpool(agent.ask, prompt, stream=False, session_id=session_id)
    return {"response": response, "session_id": session_id}


class RestClient(BaseClient):
    """Serves the uk-agents REST API via uvicorn."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port

    def start(self) -> None:
        uvicorn.run(app, host=self.host, port=self.port)


if __name__ == "__main__":
    RestClient().start()
