"""
property_crawler/main.py — the property listing crawler.

Run from the repository root:
    python -m property_crawler.main          # at CRAWLER_RUN_AT (default 04:30 and 13:00, Europe/Brussels)
    python -m property_crawler.main --once   # one pass now, then exit

Each pass visits the sites in CRAWLER_SITES_FILE one by one with headless Chromium
(Playwright) and upserts new or changed listings into the Pinecone index
PINECONE_INDEX_NAME. One-time setup: `playwright install chromium`. See README.md.
"""
import argparse
import logging
import time
from datetime import datetime, timedelta, timezone

from .crawler import CrawlReport, PropertyCrawler
from .fetcher import PlaywrightFetcher, RobotsChecker
from .indexer import ListingIndexer
from .runtime import run_service, setup_logging
from .schedule import Schedule, parse_times
from .seen_store import SeenStore
from .settings import Settings
from .sites import load_sites
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class CrawlerService:
    """Runs crawl passes on the schedule (fixed times of day, or an interval); start() blocks until stop()."""

    def __init__(self, settings: Settings, indexer: ListingIndexer | None = None, fetcher_factory=None,
                 clock=lambda: datetime.now(timezone.utc), sleep=time.sleep):
        self.settings = settings
        self.schedule = Schedule(tuple(parse_times(settings.run_at)), settings.timezone, settings.interval_minutes)
        self._clock = clock
        self._sleep = sleep
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

    def run_once(self, trigger: str = "once") -> CrawlReport:
        """One pass now. trigger (recorded in the crawl history): schedule | once | request."""
        with SeenStore(self.settings.db_file) as store:
            # Recorded first, so a pass that fails still counts: the next one waits for its turn
            started = self._clock()
            store.record_pass(started)
            run_id = store.start_run(started, trigger)
            try:
                report = self._crawl(store)
            except Exception as exc:
                store.finish_run(run_id, self._clock(), "failed", error=f"{type(exc).__name__}: {exc}")
                raise
            store.finish_run(run_id, self._clock(), "stopped" if self._stop_requested else "ok", report.sites)
            return report

    def _crawl(self, store: SeenStore) -> CrawlReport:
        s = self.settings
        # Re-read the site list every pass, so edits to sites.json need no restart.
        sites = load_sites(s.sites_file)
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
        logger.info("[crawler] started — %s", self.schedule.describe())
        while not self._stop_requested:
            with SeenStore(self.settings.db_file) as store:
                last_pass = store.last_pass()
            due = self.schedule.next_run(self._clock(), last_pass)
            if due > self._clock():
                logger.info("[crawler] next pass at %s UTC", f"{due:%Y-%m-%d %H:%M}")
            # Sleep in 1-second ticks so stop() and a "run now" request are noticed promptly; compare
            # with the clock, so a suspended machine still runs the pass as soon as it wakes up.
            trigger = "schedule"
            while not self._stop_requested and self._clock() < due:
                if self._take_request(last_pass):
                    trigger = "request"
                    break
                self._sleep(1)
            if self._stop_requested:
                break
            try:
                self.run_once(trigger)
            except Exception:
                logger.exception("[crawler] pass failed")

    def _take_request(self, last_pass: datetime | None) -> bool:
        """A "run now" request (the request file, written by my-stats): consume it, and say whether
        to start a pass — not if the last pass started less than request_cooldown_minutes ago."""
        path = self.settings.run_request_file
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            logger.exception("[crawler] could not remove the run request %s — ignored", path)
            return False
        cooldown = timedelta(minutes=self.settings.request_cooldown_minutes)
        if last_pass is not None and self._clock() - last_pass < cooldown:
            logger.info("[crawler] run request ignored — the last pass started at %s UTC, less than %d min ago",
                        f"{last_pass:%H:%M}", self.settings.request_cooldown_minutes)
            return False
        logger.info("[crawler] run requested — starting a pass now")
        return True

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
        service.run_once("once")
    else:
        run_service(service, "crawler")


if __name__ == "__main__":
    main()
