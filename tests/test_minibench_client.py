"""Unit tests for MetaculusClient construction (no network)."""

import pytest

from metaculus_bot.minibench_analysis.client import MetaculusClient


def test_token_trailing_newline_is_stripped_from_header():
    """A secret stored with a trailing newline must not poison the auth header.

    Regression: an unstripped "Token <...>\\n" made requests raise
    "ValueError: Invalid header value" before any call, so the whole run
    crashed and produced no report files.
    """
    client = MetaculusClient(token="abc123\n")
    assert client.token == "abc123"
    auth = client._headers["Authorization"]
    assert auth == "Token abc123"
    assert "\n" not in auth and "\r" not in auth


def test_token_surrounding_whitespace_is_stripped():
    client = MetaculusClient(token="  abc123  ")
    assert client.token == "abc123"


def test_token_from_env_is_stripped(monkeypatch):
    monkeypatch.setenv("METACULUS_TOKEN", "envtoken\n")
    client = MetaculusClient()
    assert client.token == "envtoken"


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MetaculusClient()


def test_whitespace_only_token_raises(monkeypatch):
    monkeypatch.delenv("METACULUS_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MetaculusClient(token="   \n")


def _client(_get):
    c = MetaculusClient(token="t")
    c._get = _get  # type: ignore[assignment]
    return c


def test_list_tournaments_matches_minibench_in_paginated_results():
    calls = []

    def fake_get(path, params=None):
        calls.append(path)
        return {
            "results": [
                {"id": 1, "slug": "minibench", "name": "MiniBench", "start_date": "2026-06-01"},
                {"id": 2, "slug": "some-other-cup", "name": "Other", "start_date": "2026-05-01"},
            ]
        }

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench"]
    assert "/projects/tournaments/minibench/" not in calls  # no fallback needed


def test_list_tournaments_tolerates_bare_list_response():
    def fake_get(path, params=None):
        # No "results" wrapper — a bare list.
        return [{"id": 3, "slug": "minibench-archive", "name": "Mini Bench May"}]

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench-archive"]


def test_list_tournaments_falls_back_to_slug_when_none_match():
    def fake_get(path, params=None):
        if path == "/projects/tournaments/minibench/":
            return {"id": 42, "slug": "minibench", "name": "MiniBench"}
        # Listing returns projects, but none are MiniBench.
        return {"results": [{"id": 9, "slug": "unrelated", "name": "Unrelated"}]}

    got = _client(fake_get).list_minibench_tournaments()
    assert [t["slug"] for t in got] == ["minibench"]


def test_list_tournaments_empty_when_listing_and_fallback_both_empty():
    def fake_get(path, params=None):
        return None  # nothing from the listing, nothing from the slug fetch

    assert _client(fake_get).list_minibench_tournaments() == []
