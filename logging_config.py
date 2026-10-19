import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Anchored to the project root so logs land in one place whatever the working directory.
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "agent.log"

_FORMAT = "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUPS = 5

# Chatty third-party loggers. httpx logs every request URL at INFO, and Telegram's URLs
# contain the bot token — keep them at WARNING so the token never reaches the log files.
_QUIET = ("httpx", "httpcore", "urllib3", "telegram.ext", "mcp", "anthropic")


def setup_logging(console: bool = True, log_file: str = LOG_FILE.name) -> None:
    """Configure the root logger once per process: rotating file in logs/, plus the terminal.

    LOG_LEVEL in the environment (DEBUG, INFO, WARNING…) overrides the INFO default.
    Processes that must keep stdout/stderr clean (e.g. stdio MCP servers) pass console=False
    and their own log_file, so two processes never rotate the same file.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(LOG_DIR / log_file, maxBytes=_MAX_BYTES, backupCount=_BACKUPS, encoding="utf-8"),
    ]
    if console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format=_FORMAT,
        handlers=handlers,
        force=True,  # libraries (e.g. the MCP server) may install their own root handlers first
    )
    for name in _QUIET:
        logging.getLogger(name).setLevel(logging.WARNING)
