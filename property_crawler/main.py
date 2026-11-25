"""
property_crawler/main.py — the property listing crawler.

Run from the repository root:
    python -m property_crawler.main          # every CRAWLER_INTERVAL_MINUTES (default 60)
    python -m property_crawler.main --once   # one pass, then exit

Each pass visits the sites in CRAWLER_SITES_FILE one by one with headless Chromium
(Playwright) and upserts new or changed listings into the Pinecone index
PINECONE_INDEX_NAME. One-time setup: `playwright install chromium`. See README.md.
"""
import argparse
import logging
import time
from datetime import timedelta

from .crawler import CrawlReport, PropertyCrawler
from .fetcher import PlaywrightFetcher, RobotsChecker
from .indexer import ListingIndexer
from .runtime import run_service, setup_logging
from .seen_store import SeenStore
from .settings import Settings
from .sites import load_sites
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class CrawlerService:
    """Runs a crawl pass, then waits settings.interval_minutes; start() blocks until stop()."""

    def __init__(self, settings: Settings, indexer: ListingIndexer | None = None, fetcher_factory=None):
        self.settings = settings
        # Kept across passes: the embedding model loads once
        self._indexer = indexer or ListingIndexer(VectorIndex(
            settings.pinecone_api_key, settings.pinecone_index_name, settings.embedding_model,
        ))
        self._fetcher_factory = fetcher_factory or (lambda: PlaywrightFetcher(
            headless=settings.headless,
            timeout_seconds=settings.timeout_seconds,
            executable_path=settings.browser_path,
        ))
        self._stop_requested = False

    def run_once(self) -> CrawlReport:
        s = self.settings
        # Re-read the site list every pass, so edits to sites.json need no restart.
        sites = load_sites(s.sites_file)
        with SeenStore(s.db_file) as store:
            crawler = PropertyCrawler(
                sites=sites,
                fetcher=self._fetcher_factory(),
                store=store,
                indexer=self._indexer,
                robots=RobotsChecker() if s.respect_robots else None,
                page_delay_seconds=s.page_delay_seconds,
                back_off=timedelta(minutes=s.back_off_minutes),
                max_back_off=timedelta(hours=s.max_back_off_hours),
                should_stop=lambda: self._stop_requested,
            )
            return crawler.run_once()

    def start(self) -> None:
        self._stop_requested = False
        logger.info("[crawler] started — a pass every %d minutes", self.settings.interval_minutes)
        while not self._stop_requested:
            try:
                self.run_once()
            except Exception:
                logger.exception("[crawler] pass failed")
            # Sleep in 1-second ticks so stop() can interrupt promptly.
            for _ in range(self.settings.interval_minutes * 60):
                if self._stop_requested:
                    break
                time.sleep(1)

    def stop(self) -> None:
        self._stop_requested = True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Crawl property sites into the vector index.")
    parser.add_argument("--once", action="store_true", help="run one pass and exit")
    args = parser.parse_args(argv)
    setup_logging()
    try:
        service = CrawlerService(Settings.from_env())
    except ValueError as exc:
        logger.error("[crawler] configuration error: %s", exc)
        raise SystemExit(1)
    if args.once:
        service.run_once()
    else:
        run_service(service, "crawler")


if __name__ == "__main__":
    main()
