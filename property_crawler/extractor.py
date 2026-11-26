"""
Turns a rendered listing page into Listing records using a site's CSS selectors.
Pure HTML in, data out — no browser, so it is tested on plain HTML strings.
"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urldefrag, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from .classify import (AVAILABLE, deal_from_label, detect_availability, detect_deal, detect_type,
                       split_locality)
from .sites import FieldSelector, Site

MAX_DESCRIPTION_CHARS = 1000

ID_PREFIX = "crawl-"   # every vector id the crawler writes starts with this

# "1 250", "1.250,50", "1,250.50", "48,5", "1250": a thousands separator only before an exact
# group of three digits, so "€1,250 1250€" (a price shown twice) reads as 1250, not 12501250.
_SEP = r"[\s\u00a0\u202f.,']"
_NUMBER = re.compile(
    rf"\d{{1,3}}(?!\d)(?:{_SEP}\d{{3}}(?!\d))*(?:[.,]\d{{1,2}}(?!\d))?"
    rf"|\d+(?:[.,]\d{{1,2}}(?!\d))?"
)


@dataclass
class Listing:
    site: str
    url: str
    title: str = ""
    description: str = ""
    city: str | None = None
    price: float | None = None
    price_text: str = ""
    currency: str | None = None
    rooms: float | None = None
    area: float | None = None
    property_type: str | None = None
    postcode: str | None = None
    deal: str | None = None              # "rent" or "sale"
    availability: str = AVAILABLE        # "available", "under offer", "sold" or "rented"
    photo: str | None = None             # absolute image URL
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Stable vector id: from the listing's URL, so re-crawling updates the same vector."""
        return ID_PREFIX + hashlib.sha1(self.url.encode("utf-8")).hexdigest()[:32]

    @property
    def text(self) -> str:
        """The document that is embedded."""
        parts = [self.title.rstrip(".") + "." if self.title else ""]
        where = f" in {self.city}" if self.city else ""
        deal = {"rent": " for rent", "sale": " for sale"}.get(self.deal or "", "")
        parts.append(f"{self.property_type or 'Property'}{deal}{where}.")
        if self.postcode:
            parts.append(f"Postcode {self.postcode}.")
        if self.availability != AVAILABLE:
            parts.append(f"Status: {self.availability}.")
        if self.price is not None:
            parts.append(f"Price {_fmt(self.price)}{' ' + self.currency if self.currency else ''}.")
        if self.rooms is not None:
            parts.append(f"{_fmt(self.rooms)} rooms.")
        if self.area is not None:
            parts.append(f"{_fmt(self.area)} square meters.")
        parts.extend(f"{k.replace('_', ' ').capitalize()}: {v}." for k, v in self.extra.items())
        if self.description:
            parts.append(self.description)
        return " ".join(p for p in parts if p)

    @property
    def fingerprint(self) -> str | None:
        """
        The property itself, whichever site lists it: rent/sale, where, price, size, type. The same
        flat on an agency's site, a portal and an aggregator gets the same fingerprint, so it is
        stored once. None when too little is known to tell two properties apart safely.
        """
        where = self.postcode or _fold(self.city)
        if self.price is None or not where or (self.area is None and self.rooms is None):
            return None
        parts = [self.deal or "?", where.replace(" ", ""), _fmt(round(self.price)),
                 _fmt(round(self.area)) if self.area is not None else "?",
                 _fmt(self.rooms) if self.rooms is not None else "?", _fold(self.property_type) or "?"]
        return "|".join(parts)

    def content_hash(self) -> str:
        """Changes when anything that is indexed changes (e.g. a price drop)."""
        return hashlib.sha1(self.text.encode("utf-8")).hexdigest()

    def metadata(self, crawled_at: str) -> dict:
        """Pinecone metadata. The searching app reads these keys (see README.md → What lands in Pinecone)."""
        meta = {
            "source": "crawler",
            "site": self.site,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "city": self.city,
            "price": self.price,
            "currency": self.currency,
            "rooms": self.rooms,
            "squareMeters": self.area,
            "type": self.property_type,
            "postcode": self.postcode,
            "deal": self.deal,
            "availability": self.availability,
            "photo": self.photo,
            "crawled_at": crawled_at,
            **self.extra,
        }
        return {k: v for k, v in meta.items() if v not in (None, "")}


def _fmt(n: float) -> str:
    return str(int(n)) if float(n).is_integer() else str(n)


def parse_number(text: str | None) -> float | None:
    """First number in text: '3 500 €' → 3500, '€1,250.50' → 1250.5, '1.200 EUR' → 1200, '48,5 m²' → 48.5."""
    if not text:
        return None
    match = _NUMBER.search(text)
    if not match:
        return None
    raw = re.sub(r"[\s  ']", "", match.group()).rstrip(".,")
    # A final separator followed by 1–2 digits is a decimal point; every other separator groups thousands.
    decimal = re.search(r"[.,](\d{1,2})$", raw)
    if decimal:
        whole = re.sub(r"[.,]", "", raw[: decimal.start()])
        raw = f"{whole}.{decimal.group(1)}"
    else:
        raw = re.sub(r"[.,]", "", raw)
    try:
        return float(raw)
    except ValueError:
        return None


def _fold(text: str | None) -> str:
    """'Liège ' → 'liege': lower case, no accents, single spaces."""
    text = unicodedata.normalize("NFKD", text or "")
    return re.sub(r"\s+", " ", "".join(c for c in text if not unicodedata.combining(c))).strip().lower()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _read(card: Tag, spec: FieldSelector) -> str:
    node = card.select_one(spec.selector)
    if node is None:
        return ""
    text = _clean(str(node.get(spec.attr) or "")) if spec.attr else _clean(node.get_text(" "))
    if spec.pattern:
        match = re.search(spec.pattern, text, re.IGNORECASE)
        if not match:
            return ""
        return _clean(match.group(1) if match.groups() else match.group(0))
    return text


# Query parameters that only track the visit; the same ad with and without them is one listing
_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|gclsrc|dclid|msclkid|mc_[ce]id|_ga|_gl|ref|referrer|source|"
                       r"searchid|search_id|position|pos|rank|from|origin|xtor|trk|tracking\w*)$", re.IGNORECASE)


def _absolute(href: str, page_url: str) -> str:
    url = urldefrag(urljoin(page_url, href.strip()))[0]
    parts = urlsplit(url)
    if not parts.query:
        return url
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _TRACKING.match(k)]
    return urlunsplit(parts._replace(query=urlencode(query)))


_KNOWN_FIELDS = {"title", "price", "city", "rooms", "area", "type", "description", "url",
                 "deal", "availability", "photo"}


def extract_listings(html: str, site: Site, page_url: str) -> list[Listing]:
    """Every listing card on the page; a card without a link gets a URL from its text."""
    soup = BeautifulSoup(html, "html.parser")
    listings: dict[str, Listing] = {}
    for card in soup.select(site.listing_selector):
        values = {name: _read(card, spec) for name, spec in site.fields.items()}
        card_text = _clean(card.get_text(" "))
        if not card_text:
            continue

        href = values.get("url")
        if not href and site.link_selector:
            link = card if card.name == "a" and card.get("href") else card.select_one(site.link_selector)
            href = link.get("href") if link is not None else None
        if href:
            url = _absolute(href, page_url)
        else:
            digest = hashlib.sha1(card_text.encode("utf-8")).hexdigest()[:16]
            url = f"{page_url}#listing-{digest}"

        price_text = values.get("price", "")
        title = values.get("title", "")
        city, postcode = split_locality(values.get("city") or site.city)
        photo = values.get("photo")
        listing = Listing(
            site=site.name,
            url=url,
            title=title,
            description=(values.get("description") or card_text)[:MAX_DESCRIPTION_CHARS],
            city=city,
            price=parse_number(price_text),
            price_text=price_text,
            currency=site.currency,
            rooms=parse_number(values.get("rooms")),
            area=parse_number(values.get("area")),
            property_type=(values.get("type") or site.property_type
                           or detect_type(title) or detect_type(card_text)),
            postcode=postcode,
            # The site's own label, then the page URL ("/for-rent/"), then the card's words ("€950 / month")
            deal=(deal_from_label(values.get("deal", "")) or site.deal
                  or detect_deal(page_url, f"{title} {price_text} {card_text}")),
            availability=detect_availability(values.get("availability", ""), card_text),
            photo=_absolute(photo, page_url) if photo and not photo.startswith("data:") else None,
            extra={k: v for k, v in values.items() if k not in _KNOWN_FIELDS and v},
        )
        listings.setdefault(listing.url, listing)   # the same ad shown twice on a page counts once
    return list(listings.values())


def next_page_url(html: str, site: Site, page_url: str) -> str | None:
    """The absolute URL of the next results page, or None on the last page."""
    if not site.next_page_selector:
        return None
    node = BeautifulSoup(html, "html.parser").select_one(site.next_page_selector)
    href = node.get("href") if node is not None else None
    if not href or str(href).startswith(("javascript:", "#")):
        return None
    return _absolute(str(href), page_url)
