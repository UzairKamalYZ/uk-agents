"""
The crawler's site list (CRAWLER_SITES_FILE, default sites.json next to this module).

Each site says where its listing pages are and how to read a listing card with CSS
selectors. See README.md for every key.
"""
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldSelector:
    """
    Where one listing field lives inside a card: the selector's text, or one of its
    attributes; `pattern` (a regex) then keeps its first group, e.g. "(\\d+) bdr".
    """
    selector: str
    attr: str | None = None
    pattern: str | None = None


@dataclass(frozen=True)
class Site:
    name: str
    start_urls: list[str]
    listing_selector: str
    fields: dict[str, FieldSelector] = field(default_factory=dict)
    link_selector: str | None = "a[href]"
    next_page_selector: str | None = None
    max_pages: int = 1
    wait_for: str | None = None        # defaults to listing_selector
    scrolls: int = 0                   # scroll to the bottom this many times (lazy-loaded cards)
    stop_when_seen: bool = True        # stop paging once a page holds nothing new
    city: str | None = None            # used when a card names no city
    currency: str | None = None       # e.g. "EUR"; shown with the price
    property_type: str | None = None   # used when a card names no type
    enabled: bool = True


def _field(spec) -> FieldSelector:
    if isinstance(spec, str):
        return FieldSelector(spec)
    return FieldSelector(spec["selector"], spec.get("attr"), spec.get("pattern"))


def _site(raw: dict) -> Site:
    start_urls = raw.get("start_urls") or ([raw["url"]] if raw.get("url") else [])
    if not raw.get("name") or not start_urls or not raw.get("listing_selector"):
        raise ValueError("a site needs name, start_urls (or url) and listing_selector")
    return Site(
        name=raw["name"],
        start_urls=list(start_urls),
        listing_selector=raw["listing_selector"],
        fields={k: _field(v) for k, v in (raw.get("fields") or {}).items()},
        link_selector=raw.get("link_selector", "a[href]"),
        next_page_selector=raw.get("next_page_selector"),
        max_pages=max(1, int(raw.get("max_pages", 1))),
        wait_for=raw.get("wait_for") or raw["listing_selector"],
        scrolls=int(raw.get("scrolls", 0)),
        stop_when_seen=bool(raw.get("stop_when_seen", True)),
        city=raw.get("city"),
        currency=raw.get("currency"),
        property_type=raw.get("property_type"),
        enabled=bool(raw.get("enabled", True)),
    )


def load_sites(path: str | Path) -> list[Site]:
    """The enabled sites in the file, in file order. A malformed site is logged and skipped."""
    path = Path(path)
    if not path.exists():
        logger.warning("[crawler] %s not found — no sites to crawl (set CRAWLER_SITES_FILE)", path)
        return []
    raw_sites = json.loads(path.read_text(encoding="utf-8")).get("sites", [])
    sites = []
    for i, raw in enumerate(raw_sites):
        try:
            site = _site(raw)
        except (KeyError, TypeError, ValueError) as exc:
            logger.error("[crawler] site #%d in %s skipped: %s", i + 1, path, exc)
            continue
        if site.enabled:
            sites.append(site)
    return sites
