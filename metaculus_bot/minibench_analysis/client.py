"""Thin Metaculus API client for MiniBench analysis.

ISOLATION NOTE: every Metaculus HTTP call lives here so the rest of the package
stays pure/testable. The endpoints below match the API that ``forecasting_tools``
already uses for ``/posts/`` and token auth (verified), plus a few endpoints
(``/users/me/``, tournament listing, leaderboard) that are Metaculus-standard but
were NOT exercised from the build sandbox (its egress to metaculus.com is
blocked). They run for real on the GitHub runner. If Metaculus has renamed one,
fix it here — nothing else imports ``requests``. Each method degrades gracefully
(logs + returns empty/None) rather than crashing the whole run.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

_BASE = "https://www.metaculus.com/api"
_PAGE = 100  # Metaculus caps the posts API at 100 per request.
# Slug of the current MiniBench tournament (matches forecasting_tools'
# MetaculusApi.CURRENT_MINIBENCH_ID); used as a fallback when the tournament
# listing surfaces nothing.
_CURRENT_MINIBENCH_SLUG = "minibench"


class MetaculusClient:
    def __init__(self, token: str | None = None, *, pace_seconds: float = 0.4) -> None:
        raw = token if token is not None else os.getenv("METACULUS_TOKEN")
        # Secrets are routinely stored with a trailing newline. Left unstripped it
        # lands in the "Authorization: Token <...>" header, which requests/urllib3
        # rejects with "ValueError: Invalid header value" *before* any call is made
        # — so strip surrounding whitespace to stay robust to how the secret was set.
        self.token = raw.strip() if isinstance(raw, str) else raw
        if not self.token:
            raise ValueError("METACULUS_TOKEN must be set to query Metaculus")
        self._pace = pace_seconds

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.token}", "Accept-Language": "en"}

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
        url = f"{_BASE}{path}"
        try:
            time.sleep(self._pace)
            resp = requests.get(url, params=params, headers=self._headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError:
            logger.warning("GET %s -> HTTP %s", url, getattr(resp, "status_code", "?"))
        except requests.RequestException as exc:
            logger.warning("GET %s failed: %s", url, type(exc).__name__)
        return None

    # -- identity -----------------------------------------------------------

    def get_me(self) -> dict[str, Any] | None:
        """The authenticated account (our bot). {id, username, ...}."""
        return self._get("/users/me/")

    # -- tournaments --------------------------------------------------------

    def list_minibench_tournaments(self) -> list[dict[str, Any]]:
        """All MiniBench tournaments, oldest-first by start date.

        MiniBench is a rolling series (a new ~2-week tournament every fortnight);
        the current one carries slug ``minibench`` and past ones are archived as
        separate projects. We list tournament projects, keep those whose slug or
        name looks like MiniBench, and sort chronologically so callers can index
        (current == last, "two sessions ago" == last but two).

        The ``/projects/tournaments/`` listing was never exercised offline, so it
        may not surface MiniBench at all. When the scan finds nothing we fall back
        to fetching the current tournament by its known slug (the same path
        ``forecasting_tools`` uses) so callers still get at least one tournament.
        """
        found: dict[Any, dict[str, Any]] = {}
        scanned = 0
        sample: list[str] = []
        offset = 0
        while True:
            page = self._get("/projects/tournaments/", {"limit": _PAGE, "offset": offset})
            results = _results(page)
            if not results:
                break
            for proj in results:
                scanned += 1
                slug = (proj.get("slug") or "").lower()
                name = (proj.get("name") or "").lower()
                if len(sample) < 25:
                    sample.append(slug or name)
                if "minibench" in slug or "minibench" in name or "mini bench" in name:
                    found[proj.get("id", slug)] = proj
            # Bare-list responses aren't paginated; stop after a single pass.
            if not isinstance(page, dict) or len(results) < _PAGE:
                break
            offset += _PAGE

        if not found:
            # Diagnostic: surface what the listing actually returned so a zero
            # result is debuggable from the job log (we can't reach the API offline).
            logger.warning(
                "No MiniBench tournament matched in %d scanned project(s); sample slugs: %s. "
                "Falling back to slug %r.",
                scanned,
                sample,
                _CURRENT_MINIBENCH_SLUG,
            )
            current = self._get(f"/projects/tournaments/{_CURRENT_MINIBENCH_SLUG}/")
            if isinstance(current, dict) and (current.get("id") or current.get("slug")):
                found[current.get("id", _CURRENT_MINIBENCH_SLUG)] = current

        tournaments = sorted(found.values(), key=_tournament_sort_key)
        logger.info("Found %d MiniBench tournament(s): %s", len(tournaments), [t.get("slug") for t in tournaments])
        return tournaments

    def get_leaderboard(self, project_id: int | str) -> list[dict[str, Any]]:
        """Leaderboard entries for a tournament, best rank first (best-effort).

        Returns [] if the endpoint shape differs; the caller then reports my-bot
        results only. Entries are normalized to expose ``rank``, ``username``,
        ``user_id``, ``score``, ``take``, ``peer_score`` where present.
        """
        data = self._get(f"/projects/{project_id}/leaderboard/")
        entries = data.get("entries") if isinstance(data, dict) else (data if isinstance(data, list) else None)
        if not entries:
            logger.warning("No leaderboard entries for project %s (endpoint shape may differ)", project_id)
            return []
        normalized = [_normalize_leaderboard_entry(e) for e in entries]
        normalized.sort(key=lambda e: (e["rank"] is None, e["rank"] if e["rank"] is not None else 0))
        return normalized

    # -- posts / questions --------------------------------------------------

    def get_resolved_posts(self, tournament: int | str) -> list[dict[str, Any]]:
        """Every resolved post in a tournament, with ``question`` payloads.

        Requests ``with_cp`` so aggregation (community) data is included, and
        pages through the full tournament. ``my_forecasts`` is populated for the
        authenticated bot automatically.
        """
        posts: list[dict[str, Any]] = []
        offset = 0
        while True:
            page = self._get(
                "/posts/",
                {
                    "tournaments": tournament,
                    "statuses": "resolved",
                    "limit": _PAGE,
                    "offset": offset,
                    "with_cp": "true",
                },
            )
            results = _results(page)
            if not results:
                break
            posts.extend(results)
            if len(results) < _PAGE:
                break
            offset += _PAGE
        logger.info("Tournament %s: %d resolved posts", tournament, len(posts))
        return posts

    def get_post_with_forecasts(self, post_id: int) -> dict[str, Any] | None:
        """Full detail for one post, used to try to recover other bots' forecasts.

        Best-effort: many tournaments do not expose other users' individual
        forecasts via the API. Returns None on failure; caller degrades to
        leaderboard aggregates for that bot.
        """
        return self._get(f"/posts/{post_id}/")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _results(page: dict[str, Any] | list | None) -> list[dict[str, Any]]:
    # Metaculus list endpoints normally return {"results": [...]}, but tolerate a
    # bare list too (the tournaments listing shape was never verified offline).
    if isinstance(page, list):
        return page
    if not isinstance(page, dict):
        return []
    results = page.get("results")
    return results if isinstance(results, list) else []


def _tournament_sort_key(proj: dict[str, Any]) -> str:
    """Sort MiniBench tournaments chronologically using whatever date is present."""
    for key in ("start_date", "open_time", "created_at", "forecasting_start_time"):
        val = proj.get(key)
        if val:
            return str(val)
    # Fall back to id so ordering is at least stable/monotonic with creation.
    return str(proj.get("id", ""))


def _normalize_leaderboard_entry(entry: dict[str, Any]) -> dict[str, Any]:
    user = entry.get("user") if isinstance(entry.get("user"), dict) else {}
    return {
        "rank": entry.get("rank") or entry.get("medal_rank"),
        "username": entry.get("username") or user.get("username") or entry.get("aggregation_method"),
        "user_id": entry.get("user_id") or user.get("id"),
        "score": entry.get("score") or entry.get("total_score"),
        "take": entry.get("take") or entry.get("prize"),
        "peer_score": entry.get("peer_score") or entry.get("spot_peer_score"),
    }
