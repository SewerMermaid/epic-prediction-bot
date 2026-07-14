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
