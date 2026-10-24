"""
MCP tool registry for the property agent.

MCP servers are declared in ``mcp.json`` at the project root.  Add a server
by appending an entry — no code changes required:

.. code-block:: json

    {
      "servers": [
        {"name": "finance", "command": ["npx", "--yes", "@easysolutions906/mcp-finance"]},
        {"name": "maps",    "command": ["npx", "--yes", "@acme/mcp-maps"]}
      ]
    }

``"{python}"`` in a command is replaced by the running interpreter, and such a
server is launched from the project root so ``-m package.module`` resolves.
``"owner_only": true`` keeps a server out of the default tool set
(``langchain_tools()`` with no ``server_name``); request it by name instead.
Stdio servers start with a minimal environment (HOME, PATH, …) so secrets in
``.env`` don't leak to third-party packages; list the variables a server needs
in ``"env": ["SOME_API_KEY"]``.  ``{python}`` servers (our own code) get the
full environment.
``"timeout": <seconds>`` overrides how long a server may take to answer one
call (default 60) before it is disconnected (a stdio subprocess is stopped)
and reconnected on the next call.  ``"url"`` entries are Streamable HTTP
servers; ``"api_key_env"`` / ``"api_key_param"`` add a key from the
environment as a query parameter.

The registry is built automatically at import time from that file.
"""

import json
import logging
import os
import sys
from pathlib import Path

from ..config.config import Config
from .mcp_registry import DEFAULT_TIMEOUT, MCPRegistry

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(Config.MCP_FILE)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_registry(config_path: Path) -> MCPRegistry:
    registry = MCPRegistry()

    if not config_path.exists():
        logger.warning("mcp-tools: config not found at %s — no servers registered", config_path)
        return registry

    try:
        data = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        logger.exception("mcp-tools: failed to parse %s — no servers registered", config_path)
        return registry

    for entry in data.get("servers", []):
        name = entry.get("name")
        command = entry.get("command")
        url = entry.get("url")
        owner_only = bool(entry.get("owner_only", False))

        if name and command:
            own_code = "{python}" in command
            # A {python} module only resolves when launched from the project root.
            cwd = str(_PROJECT_ROOT) if own_code else None
            command = [sys.executable if part == "{python}" else part for part in command]
            # Our own servers get the full environment; third-party ones only what they list
            if own_code:
                env = dict(os.environ)
            else:
                env = {k: os.environ[k] for k in entry.get("env", []) if k in os.environ}
            logger.info("mcp-tools: adding stdio server '%s'", name)
            timeout = float(entry.get("timeout", DEFAULT_TIMEOUT))
            registry.register(name, command, cwd=cwd, owner_only=owner_only, timeout=timeout, env=env)

        elif name and url:
            # Build query params: resolve any api_key_env reference from environment
            params: dict[str, str] = {}
            api_key_env = entry.get("api_key_env")
            api_key_param = entry.get("api_key_param", "apikey")
            if api_key_env:
                api_key = os.getenv(api_key_env)
                if api_key:
                    params[api_key_param] = api_key
                else:
                    logger.warning(
                        "mcp-tools: env var '%s' not set for server '%s' — requests may fail",
                        api_key_env, name,
                    )
            logger.info("mcp-tools: adding HTTP server '%s' → %s", name, url)
            timeout = float(entry.get("timeout", DEFAULT_TIMEOUT))
            registry.register_http(name, url, params=params, owner_only=owner_only, timeout=timeout)

        else:
            logger.warning("mcp-tools: skipping invalid entry %s", entry)

    return registry


_mcp = _load_registry(_CONFIG_PATH)


def close_mcp() -> None:
    """Shut down all registered MCP subprocesses. Call during application teardown."""
    _mcp.close()
