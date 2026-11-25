"""
Recognises a site refusing the crawler (rate limit, firewall, bot check) so the crawler
stops that site and backs off instead of hammering it. The crawler never tries to get
past a block.
"""
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from .fetcher import FetchedPage

BLOCK_STATUSES = {401, 403, 429, 503}

# Shown instead of the page by common bot-protection services
_CHALLENGE_TITLES = re.compile(
    r"just a moment|attention required|access denied|are you a robot|"
    r"verify you are human|security check|too many requests|pardon our interruption",
    re.IGNORECASE,
)
_CHALLENGE_MARKERS = ("captcha-delivery.com", "cf-challenge", "cf-turnstile", "g-recaptcha", "h-captcha")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def block_reason(page: FetchedPage, found_listings: int) -> str | None:
    """Why this page is a refusal, or None if it is a normal page."""
    if page.status in BLOCK_STATUSES:
        return f"HTTP {page.status}"
    if found_listings:
        return None   # real content: a bot-check script in the page is not a block
    title = _TITLE.search(page.html or "")
    if title and _CHALLENGE_TITLES.search(title.group(1)):
        return f"challenge page ({title.group(1).strip()[:60]})"
    lowered = (page.html or "").lower()
    for marker in _CHALLENGE_MARKERS:
        if marker in lowered:
            return f"challenge page ({marker})"
    return None


def parse_retry_after(value: str | None, now: datetime | None = None) -> timedelta | None:
    """Retry-After as seconds ("120") or an HTTP date."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return timedelta(seconds=int(value))
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return max(when - (now or datetime.now(timezone.utc)), timedelta(0))
