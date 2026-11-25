from core.src.model.context_builder import build_context_from_listings


def test_context_shows_the_listing_link_and_its_currency():
    context = build_context_from_listings([{"city": "Berlin", "price": 1200.0, "currency": "EUR",
                                            "url": "https://portal.example/offer/1", "text": "Flat",
                                            "crawled_at": "2026-11-25T10:00:00+00:00"}])
    assert "Location: Berlin" in context
    assert "Price: 1200 EUR" in context
    assert "Link: https://portal.example/offer/1" in context
    assert "Last seen: 2026-11-25T10:00:00+00:00" in context


def test_context_leaves_out_fields_a_listing_does_not_have():
    context = build_context_from_listings([{"price": 900, "rooms": float("nan")}])
    assert context == "Listing 1\nPrice: 900"
    assert build_context_from_listings([]) == "No listings found."
