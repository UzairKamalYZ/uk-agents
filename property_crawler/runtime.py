"""Logging and the run-until-stopped loop for the crawler service."""
import logging
import logging.handlers
import signal
import sys

from .settings import ROOT

_FORMAT = "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Terminal + logs/crawler.log (rotating, 5 × 5 MB) under the repository root."""
    root = logging.getLogger()
    if any(getattr(h, "_property_crawler", False) for h in root.handlers):
        return   # already set up
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stderr),
        logging.handlers.RotatingFileHandler(log_dir / "crawler.log", maxBytes=5_000_000,
                                             backupCount=5, encoding="utf-8"),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._property_crawler = True
        root.addHandler(handler)
    root.setLevel(level)


def run_service(service, name: str) -> None:
    """Run service.start() until Ctrl-C or SIGTERM, which call service.stop()."""
    logger = logging.getLogger(name)

    def _handle_sigterm(signum, frame):
        logger.info("[%s] SIGTERM — shutting down", name)
        service.stop()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        service.start()
    except KeyboardInterrupt:
        logger.info("[%s] keyboard interrupt — shutting down", name)
        service.stop()
