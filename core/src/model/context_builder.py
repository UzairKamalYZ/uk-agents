_FIELDS = [
    # (metadata key, label, suffix)
    ("title", "Title", ""),
    ("city", "Location", ""),
    ("type", "Type", ""),
    ("rooms", "Rooms", ""),
    ("squareMeters", "Surface", " m²"),
]


def _present(value) -> bool:
    return value is not None and str(value).strip().lower() not in {"", "nan", "none"}


def _fmt(value) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_context_from_listings(listings: list[dict]) -> str:
    """Formats retrieved listings for the prompt; a field a listing doesn't have is left out."""
    if not listings:
        return "No listings found."

    blocks = []
    for i, l in enumerate(listings, start=1):
        lines = [f"Listing {i}"]
        for key, label, suffix in _FIELDS:
            if _present(l.get(key)):
                lines.append(f"{label}: {_fmt(l[key])}{suffix}")
        if _present(l.get("price")):
            currency = f" {l['currency']}" if _present(l.get("currency")) else ""
            lines.append(f"Price: {_fmt(l['price'])}{currency}")
        if _present(l.get("text")):
            lines.append(f"Description:\n{l['text']}")
        if _present(l.get("url")):
            lines.append(f"Link: {l['url']}")
        if _present(l.get("crawled_at")):
            lines.append(f"Last seen: {l['crawled_at']}")
        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)
