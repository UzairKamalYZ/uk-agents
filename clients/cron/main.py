from logging_config import setup_logging
setup_logging()

import logging
import time

from orchestrator.agent_interface import OrchestratorAgent
from core.src.config.config import Config
from core.src.utils import run_service
from clients.base import BaseClient

logger = logging.getLogger(__name__)


def _run_job() -> None:
    """Send CRON_SEARCH_PROMPT through the orchestrator once and log the answer."""
    with OrchestratorAgent() as agent:
        logger.info("Cron job asking: %r", Config.CRON_SEARCH_PROMPT)
        logger.info("Cron job answer: %s", agent.ask(Config.CRON_SEARCH_PROMPT))


class CronClient(BaseClient):
    """Sends CRON_SEARCH_PROMPT on a fixed interval."""

    def __init__(self, interval_minutes: int = 30):
        self.interval_minutes = interval_minutes
        self._running = False

    def start(self) -> None:
        self._running = True
        logger.info("Cron client started — running every %d minutes", self.interval_minutes)
        while self._running:
            try:
                _run_job()
            except Exception:
                logger.exception("Cron job failed")
            # Sleep in 1-second ticks so stop() can interrupt promptly.
            for _ in range(self.interval_minutes * 60):
                if not self._running:
                    break
                time.sleep(1)

    def stop(self) -> None:
        self._running = False


if __name__ == "__main__":
    run_service(CronClient(), "cron")
