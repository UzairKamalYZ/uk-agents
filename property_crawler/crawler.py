"""
One crawl pass: visit each site in turn, page by page, and index the listings that
are new or changed since the last pass.

Pacing ("wait, stop and see"):
  wait  — a random pause between page loads (page_delay_seconds … 2 × page_delay_seconds);
  stop  — the first refusal from a site (HTTP 403/429/503, a bot-check page) ends that
          site for this pass;
  see   — the site is skipped until its back-off ends (back_off, then doubled per repeated
          block up to max_back_off, or longer if the site sent Retry-After); a normal page
          clears it.
"""
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta

from .blocking import block_reason, parse_retry_after
from .extractor import Listing, extract_listings, next_page_url
from .seen_store import SeenStore
from .sites import Site

logger = logging.getLogger(__name__)


class SiteBlocked(Exception):
    def __init__(self, reason: str, retry_after: timedelta | None = None):
        super().__init__(reason)
        self.reason = reason
        self.retry_after = retry_after


@dataclass
class SiteResult:
    site: str
    pages: int = 0
    found: int = 0
    indexed: int = 0
    error: str | None = None
    blocked: str | None = None      # why the site refused us this pass
    skipped: str | None = None      # still backing off from an earlier block

    def describe(self) -> str:
        if self.skipped:
            return f"{self.site}: skipped ({self.skipped})"
        text = f"{self.site}: {self.pages} page(s), {self.found} listing(s), {self.indexed} new/changed"
        if self.blocked:
            text += f", blocked: {self.blocked}"
        if self.error:
            text += f", error: {self.error}"
        return text


@dataclass
class CrawlReport:
    sites: list[SiteResult] = field(default_factory=list)

    @property
    def indexed(self) -> int:
        return sum(s.indexed for s in self.sites)

    def summary(self) -> str:
        return "; ".join(s.describe() for s in self.sites) or "no sites"


def _jitter(delay: float) -> float:
    return random.uniform(delay, 2 * delay)


class PropertyCrawler:
    """
    fetcher: open() / new_site() / fetch(url, wait_for, scrolls) -> FetchedPage / close()
             — PlaywrightFetcher in production.
    robots: allowed(url) -> bool, or None to skip robots.txt checks.
    """

    def __init__(self, sites: list[Site], fetcher, store: SeenStore, indexer,
                 robots=None, page_delay_seconds: float = 8,
                 back_off: timedelta = timedelta(hours=1),
                 max_back_off: timedelta = timedelta(hours=24),
                 should_stop: Callable[[], bool] = lambda: False,
                 sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[float], float] = _jitter):
        self._sites = sites
        self._fetcher = fetcher
        self._store = store
        self._indexer = indexer
        self._robots = robots
        self._delay = page_delay_seconds
        self._back_off = back_off
        self._max_back_off = max_back_off
        self._should_stop = should_stop
        self._sleep = sleep
        self._jitter = jitter
        self._loads = 0   # page loads this pass — every load after the first waits first

    def run_once(self) -> CrawlReport:
        report = CrawlReport()
        if not self._sites:
            logger.warning("[crawler] no enabled sites — nothing to crawl")
            return report
        due = []
        for site in self._sites:
            until = self._store.blocked_until(site.name)
            if until is None:
                due.append(site)
            else:
                skipped = f"backing off after a block until {until:%Y-%m-%d %H:%M} UTC"
                logger.info("[crawler] %s: %s", site.name, skipped)
                report.sites.append(SiteResult(site.name, skipped=skipped))
        if due:
            self._fetcher.open()
            try:
                for site in due:
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
            self._store.clear_block(site.name)
        except SiteBlocked as block:
            until = self._store.record_block(site.name, block.reason, self._back_off,
                                             self._max_back_off, block.retry_after)
            result.blocked = block.reason
            logger.warning("[crawler] %s refused the crawler (%s) — stopped; not trying again before %s UTC",
                           site.name, block.reason, f"{until:%Y-%m-%d %H:%M}")
        except Exception as exc:   # one broken site must not stop the others
            logger.exception("[crawler] %s failed", site.name)
            result.error = f"{type(exc).__name__}: {exc}"
        logger.info("[crawler] %s", result.describe())
        return result

    def _crawl_from(self, site: Site, url: str | None, result: SiteResult) -> None:
        visited: set[str] = set()
        pages = 0
        while url and url not in visited and pages < site.max_pages and not self._should_stop():
            if self._robots is not None and not self._robots.allowed(url):
                logger.warning("[crawler] %s: robots.txt disallows %s — skipped", site.name, url)
                return
            self._wait()
            visited.add(url)
            page = self._fetcher.fetch(url, wait_for=site.wait_for, scrolls=site.scrolls)
            pages += 1
            result.pages += 1

            listings = extract_listings(page.html, site, url)
            reason = block_reason(page, len(listings))
            if reason:
                raise SiteBlocked(reason, parse_retry_after(page.retry_after))
            result.found += len(listings)
            if not listings:
                logger.warning("[crawler] %s: no listings matched %r on %s — check the selectors",
                               site.name, site.listing_selector, url)
            fresh = self._index(listings)
            result.indexed += len(fresh)
            if site.stop_when_seen and listings and not fresh:
                logger.info("[crawler] %s: nothing new on %s — not paging further", site.name, url)
                return
            url = next_page_url(page.html, site, url)

    def _wait(self) -> None:
        """A random pause before every page load but the first of the pass."""
        if self._loads:
            self._sleep(self._jitter(self._delay))
        self._loads += 1

    def _index(self, listings: list[Listing]) -> list[Listing]:
        fresh = self._store.unseen(listings)
        if fresh:
            self._indexer.index(fresh)
            self._store.touch(fresh)   # only after the upsert succeeded, so a failure is retried next pass
        return fresh
