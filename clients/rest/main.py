from logging_config import setup_logging
setup_logging()

import asyncio
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

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
    """Formats a generator of strings into Server-Sent Events."""
    try:
        for chunk in stream:
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.01)
    except asyncio.CancelledError:
        print("Client disconnected.")


@app.get("/ask")
async def ask(prompt: str, stream: bool = False, _: None = Depends(verify_api_key)):
    """Endpoint to ask the agent a question."""
    agent = app.state.agent
    if stream:
        return StreamingResponse(
            sse_formatter(agent.ask(prompt, stream=True)),
            media_type="text/event-stream",
        )
    return {"response": agent.ask(prompt, stream=False)}


class RestClient(BaseClient):
    """Serves the uk-agents REST API via uvicorn."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        self.host = host
        self.port = port

    def start(self) -> None:
        uvicorn.run(app, host=self.host, port=self.port)


if __name__ == "__main__":
    RestClient().start()
