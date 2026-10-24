"""A real MCP server for the client tests — run over stdio, or imported for in-process use."""
import asyncio
import os
import sys

from mcp.server.mcpserver import MCPServer

server = MCPServer("test-server")


@server.tool(name="echo", description="Echo the text back.")
def echo(text: str, times: int = 1) -> str:
    return " ".join([text] * times)


@server.tool(name="cwd", description="The server's working directory.")
def cwd() -> str:
    return os.getcwd()


@server.tool(name="pid", description="The server's process id.")
def pid() -> str:
    return str(os.getpid())


@server.tool(name="hang", description="Sleep for a while.")
async def hang(seconds: float = 30) -> str:
    await asyncio.sleep(seconds)
    return "woke up"


@server.tool(name="boom", description="Always fails.")
def boom() -> str:
    raise ValueError("bad args")


if __name__ == "__main__":
    if "--slow-start" in sys.argv:
        import time
        time.sleep(30)
    server.run()
