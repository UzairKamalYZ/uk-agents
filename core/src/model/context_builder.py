
def _format_amenities(listing: dict) -> str:
    """Formats a listing's boolean amenities into a comma-separated string."""
    amenities = [
        "Parking" if listing.get("hasParkingSpace") else None,
        "Balcony" if listing.get("hasBalcony") else None,
        "Elevator" if listing.get("hasElevator") else None,
        "Security" if listing.get("hasSecurity") else None,
        "Storage" if listing.get("hasStorageRoom") else None,
    ]
    return ", ".join([a for a in amenities if a]) or "None"


def build_context_from_listings(listings: list[dict]) -> str:
    """Builds a formatted string context from a list of listing dictionaries."""
    if not listings:
        return "No listings found."

    def safe(v, suffix=""):
        if v is None or str(v) == "nan" or v == "":
            return "N/A"
        return f"{v}{suffix}"

    blocks = []
    for i, l in enumerate(listings, start=1):
        blocks.append(
            f"""
Listing {i}
Location: {safe(l.get("city"))}
Price: {safe(l.get("price"), " " + (l.get("currency") or "PLN"))}
Rooms: {safe(l.get("rooms"))}
Surface: {safe(l.get("squareMeters"), " m²")}
Floor: {safe(l.get("floor"))} / {safe(l.get("floorCount"))}
Type: {safe(l.get("type"))}
Ownership: {safe(l.get("ownership"))}
Building material: {safe(l.get("buildingMaterial"))}
Condition: {safe(l.get("condition"))}
Amenities: {_format_amenities(l)}
Description:
{safe(l.get("text"))}
""".strip()
            + (f"\nLink: {l['url']}" if l.get("url") else "")
        )

    return "\n\n".join(blocks)
