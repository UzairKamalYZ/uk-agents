"""
agents/property/crawler/main.py — the property listing crawler.

Run from the project root:
    python -m agents.property.crawler.main          # every CRAWLER_INTERVAL_MINUTES (default 60)
    python -m agents.property.crawler.main --once   # one pass, then exit (for a system crontab)

Each pass visits the sites in CRAWLER_SITES_FILE one by one with headless Chromium
(Playwright) and upserts new or changed listings into the Pinecone index the property
agent searches. One-time setup: `uv run playwright install chromium`.
"""
from logging_config import setup_logging

setup_logging()

import argparse
import logging
import time
from datetime import timedelta

from core.src.config.config import Config
from core.src.utils import run_service
from .crawler import CrawlReport, PropertyCrawler
from .fetcher import PlaywrightFetcher, RobotsChecker
from .indexer import ListingIndexer
from .seen_store import SeenStore
from .sites import load_sites

logger = logging.getLogger(__name__)


class CrawlerService:
    """Runs a crawl pass, then waits interval_minutes; start() blocks until stop()."""

    def __init__(self, interval_minutes: int | None = None, indexer: ListingIndexer | None = None,
                 fetcher_factory=None):
        self.interval_minutes = interval_minutes or Config.CRAWLER_INTERVAL_MINUTES
        self._indexer = indexer or ListingIndexer()   # kept across passes: the model loads once
        self._fetcher_factory = fetcher_factory or (lambda: PlaywrightFetcher(
            headless=Config.CRAWLER_HEADLESS,
            timeout_seconds=Config.CRAWLER_TIMEOUT_SECONDS,
            executable_path=Config.CRAWLER_BROWSER_PATH,
        ))
        self._stop_requested = False

    def run_once(self) -> CrawlReport:
        # Re-read the site list every pass, so edits to sites.json need no restart.
        sites = load_sites(Config.CRAWLER_SITES_FILE)
        with SeenStore(Config.CRAWLER_DB_FILE) as store:
            crawler = PropertyCrawler(
                sites=sites,
                fetcher=self._fetcher_factory(),
                store=store,
                indexer=self._indexer,
                robots=RobotsChecker() if Config.CRAWLER_RESPECT_ROBOTS else None,
                page_delay_seconds=Config.CRAWLER_PAGE_DELAY_SECONDS,
                back_off=timedelta(minutes=Config.CRAWLER_BACK_OFF_MINUTES),
                max_back_off=timedelta(hours=Config.CRAWLER_MAX_BACK_OFF_HOURS),
                should_stop=lambda: self._stop_requested,
            )
            return crawler.run_once()

    def start(self) -> None:
        self._stop_requested = False
        logger.info("[crawler] started — a pass every %d minutes", self.interval_minutes)
        while not self._stop_requested:
            try:
                self.run_once()
            except Exception:
                logger.exception("[crawler] pass failed")
            # Sleep in 1-second ticks so stop() can interrupt promptly.
            for _ in range(self.interval_minutes * 60):
                if self._stop_requested:
                    break
                time.sleep(1)

    def stop(self) -> None:
        self._stop_requested = True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crawl property sites into the vector index.")
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    args = parser.parse_args(argv)
    service = CrawlerService()
    if args.once:
        service.run_once()
    else:
        run_service(service, "crawler")


if __name__ == "__main__":
    main()
