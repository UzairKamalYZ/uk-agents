"""
Renders pages in headless Chromium with Playwright (JavaScript-heavy listing sites
need a real browser), and checks robots.txt before visiting a URL.
"""
import logging
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class FetchedPage:
    html: str
    status: int | None = None          # HTTP status of the page itself
    retry_after: str | None = None     # the Retry-After header, if the site sent one


class PlaywrightFetcher:
    """One Chromium for a crawl pass; each site gets its own browser context (cookies)."""

    def __init__(self, headless: bool = True, timeout_seconds: float = 30,
                 executable_path: str | None = None):
        self._headless = headless
        self._timeout_ms = int(timeout_seconds * 1000)
        self._executable_path = executable_path or None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def open(self) -> "PlaywrightFetcher":
        from playwright.sync_api import sync_playwright   # lazy: importing it starts nothing, but keep tests light
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless, executable_path=self._executable_path,
        )
        return self

    def new_site(self) -> None:
        """Start a fresh browser context, so one site's cookies never reach the next."""
        self._close_context()
        self._context = self._browser.new_context(user_agent=USER_AGENT, locale="en-US")
        self._context.set_default_timeout(self._timeout_ms)
        self._page = self._context.new_page()

    def fetch(self, url: str, wait_for: str | None = None, scrolls: int = 0) -> FetchedPage:
        """The page's HTML after JavaScript has run, with its HTTP status. Raises on navigation errors."""
        from playwright.sync_api import TimeoutError as PlaywrightTimeout
        if self._page is None:
            self.new_site()
        page = self._page
        response = page.goto(url, wait_until="domcontentloaded", timeout=self._timeout_ms)
        status = response.status if response is not None else None
        headers = response.headers if response is not None else {}
        if status is not None and status >= 400:
            # A refusal (block, rate limit): don't wait for listings that will never come.
            return FetchedPage(page.content(), status, headers.get("retry-after"))
        if wait_for:
            try:
                page.wait_for_selector(wait_for, timeout=self._timeout_ms)
            except PlaywrightTimeout:
                logger.warning("[crawler] %s: %r did not appear within %ss",
                               url, wait_for, self._timeout_ms // 1000)
        for _ in range(scrolls):
            page.mouse.wheel(0, 20000)
            page.wait_for_timeout(1000)
        return FetchedPage(page.content(), status, headers.get("retry-after"))

    def _close_context(self) -> None:
        if self._context is not None:
            self._context.close()
        self._context = self._page = None

    def close(self) -> None:
        try:
            self._close_context()
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._playwright is not None:
                self._playwright.stop()
            self._browser = self._playwright = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class RobotsChecker:
    """Caches each host's robots.txt for one crawl pass. Unreachable robots.txt → allowed."""

    def __init__(self, user_agent: str = "*", timeout_seconds: float = 10):
        self._user_agent = user_agent
        self._timeout = timeout_seconds
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}

    def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._parsers:
            self._parsers[origin] = self._load(origin)
        parser = self._parsers[origin]
        return parser is None or parser.can_fetch(self._user_agent, url)

    def _load(self, origin: str) -> urllib.robotparser.RobotFileParser | None:
        try:
            response = requests.get(f"{origin}/robots.txt", timeout=self._timeout,
                                    headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            logger.info("[crawler] %s/robots.txt unreachable (%s) — treating as allowed", origin, exc)
            return None
        if response.status_code >= 400:
            return None
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser
