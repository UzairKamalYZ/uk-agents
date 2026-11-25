"""
One crawl pass: visit each site in turn, page by page, and index the listings that
are new or changed since the last pass.
"""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .extractor import Listing, extract_listings, next_page_url
from .seen_store import SeenStore
from .sites import Site

logger = logging.getLogger(__name__)


@dataclass
class SiteResult:
    site: str
    pages: int = 0
    found: int = 0
    indexed: int = 0
    error: str | None = None


@dataclass
class CrawlReport:
    sites: list[SiteResult] = field(default_factory=list)

    @property
    def indexed(self) -> int:
        return sum(s.indexed for s in self.sites)

    def summary(self) -> str:
        lines = [f"{s.site}: {s.pages} page(s), {s.found} listing(s), {s.indexed} new/changed"
                 + (f", error: {s.error}" if s.error else "") for s in self.sites]
        return "; ".join(lines) or "no sites"


class PropertyCrawler:
    """
    fetcher: open() / new_site() / fetch(url, wait_for, scrolls) / close() — PlaywrightFetcher in production.
    robots: allowed(url) -> bool, or None to skip robots.txt checks.
    """

    def __init__(self, sites: list[Site], fetcher, store: SeenStore, indexer,
                 robots=None, page_delay_seconds: float = 3,
                 should_stop: Callable[[], bool] = lambda: False,
                 sleep: Callable[[float], None] = time.sleep):
        self._sites = sites
        self._fetcher = fetcher
        self._store = store
        self._indexer = indexer
        self._robots = robots
        self._delay = page_delay_seconds
        self._should_stop = should_stop
        self._sleep = sleep

    def run_once(self) -> CrawlReport:
        report = CrawlReport()
        if not self._sites:
            logger.warning("[crawler] no enabled sites — nothing to crawl")
            return report
        self._fetcher.open()
        try:
            for site in self._sites:
                if self._should_stop():
                    break
                report.sites.append(self._crawl_site(site))
        finally:
            self._fetcher.close()
        logger.info("[crawler] pass finished — %d listing(s) indexed (%s)", report.indexed, report.summary())
        return report

    def _crawl_site(self, site: Site) -> SiteResult:
        """One site; any failure is logged and recorded, and the pass moves on to the next site."""
        result = SiteResult(site.name)
        try:
            self._fetcher.new_site()
            for start_url in site.start_urls:
                self._crawl_from(site, start_url, result)
        except Exception as exc:   # one broken site must not stop the others
            logger.exception("[crawler] %s failed", site.name)
            result.error = f"{type(exc).__name__}: {exc}"
        logger.info("[crawler] %s: %d page(s), %d listing(s), %d new/changed",
                    site.name, result.pages, result.found, result.indexed)
        return result

    def _crawl_from(self, site: Site, url: str | None, result: SiteResult) -> None:
        visited: set[str] = set()
        pages = 0
        while url and url not in visited and pages < site.max_pages and not self._should_stop():
            if self._robots is not None and not self._robots.allowed(url):
                logger.warning("[crawler] %s: robots.txt disallows %s — skipped", site.name, url)
                return
            if result.pages:
                self._sleep(self._delay)   # be polite between page loads
            visited.add(url)
            html = self._fetcher.fetch(url, wait_for=site.wait_for, scrolls=site.scrolls)
            pages += 1
            result.pages += 1

            listings = extract_listings(html, site, url)
            result.found += len(listings)
            if not listings:
                logger.warning("[crawler] %s: no listings matched %r on %s — check the selectors",
                               site.name, site.listing_selector, url)
            fresh = self._index(listings)
            result.indexed += len(fresh)
            if site.stop_when_seen and listings and not fresh:
                logger.info("[crawler] %s: nothing new on %s — not paging further", site.name, url)
                return
            url = next_page_url(html, site, url)

    def _index(self, listings: list[Listing]) -> list[Listing]:
        fresh = self._store.unseen(listings)
        if fresh:
            self._indexer.index(fresh)
            self._store.touch(fresh)   # only after the upsert succeeded, so a failure is retried next pass
        return fresh
