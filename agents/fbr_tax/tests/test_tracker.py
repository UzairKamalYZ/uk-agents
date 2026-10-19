import asyncio
from datetime import datetime, timedelta

import pytest

import agents.fbr_tax.tracker as tracker_mod
from agents.fbr_tax.tracker import (
    ALREADY_SUBMITTED, ASK_CONTINUE_OR_FRESH, CHECK_FIRST, CHECK_TTL, CREATE_DRAFT,
    PREVIOUS_NOT_SUBMITTED, FilingTracker, build_server,
)

NOW = datetime(2026, 10, 19, 20, 0)


class Clock:
    def __init__(self):
        self.t = NOW

    def __call__(self):
        return self.t


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def tracker(tmp_path, clock):
    return FilingTracker(tmp_path / "data" / "filings.db", now=clock)


def test_nothing_recorded_means_check_iris(tracker):
    action, why = tracker.decide(2027)
    assert action == CHECK_FIRST
    assert "TY2027 and TY2026" in why


def test_submitted_year_is_never_redrafted(tracker):
    tracker.record(2027, "submitted")
    assert tracker.decide(2027)[0] == ALREADY_SUBMITTED


def test_old_submitted_record_still_counts(tracker, clock):
    tracker.record(2027, "submitted")
    clock.t = NOW + timedelta(days=60)
    assert tracker.decide(2027)[0] == ALREADY_SUBMITTED


def test_previous_year_not_submitted_stops(tracker):
    tracker.record(2026, "draft_saved", 1)
    tracker.record(2027, "not_started")
    assert tracker.decide(2027)[0] == PREVIOUS_NOT_SUBMITTED


def test_existing_draft_asks_continue_or_fresh(tracker):
    tracker.record(2026, "submitted")
    tracker.record(2027, "draft_saved", 2)
    action, why = tracker.decide(2027)
    assert action == ASK_CONTINUE_OR_FRESH
    assert "2 draft(s)" in why


def test_no_draft_and_not_submitted_creates(tracker):
    tracker.record(2026, "submitted")
    tracker.record(2027, "not_started", 0)
    assert tracker.decide(2027)[0] == CREATE_DRAFT


def test_stale_unsubmitted_record_must_be_rechecked(tracker, clock):
    tracker.record(2026, "submitted")
    tracker.record(2027, "not_started")
    clock.t = NOW + CHECK_TTL + timedelta(minutes=1)
    action, why = tracker.decide(2027)
    assert action == CHECK_FIRST and "TY2027" in why and "TY2026" not in why


def test_record_overwrites_and_validates(tracker):
    tracker.record(2027, "not_started")
    tracker.record(2027, "draft_saved", 1, "saved after checkpoint")
    f = tracker.get(2027)
    assert (f.status, f.draft_count, f.note) == ("draft_saved", 1, "saved after checkpoint")
    with pytest.raises(ValueError):
        tracker.record(2027, "filed")
    with pytest.raises(ValueError):
        tracker.record(2027, "draft_saved", -1)


def test_server_tools(tracker, monkeypatch):
    monkeypatch.setattr(tracker_mod, "create_sdk_mcp_server", lambda name, tools: {t.name: t.handler for t in tools})
    handlers = build_server(tracker)
    run = lambda name, args: asyncio.run(handlers[name](args))["content"][0]["text"]
    assert run("show_filing_records", {}) == "Nothing recorded yet."
    assert run("record_filing_status", {"tax_year": 2026, "status": "submitted"}).startswith("Recorded TY2026: submitted")
    bad = asyncio.run(handlers["record_filing_status"]({"tax_year": 2027, "status": "filed"}))
    assert bad["is_error"] is True
    run("record_filing_status", {"tax_year": 2027, "status": "not_started", "draft_count": 0})
    assert run("decide_filing_action", {"tax_year": 2027}).startswith("create_draft:")
    assert "TY2026: submitted" in run("show_filing_records", {})
