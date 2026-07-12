"""Parse raw Metaculus post/question JSON into typed forecasts, resolutions, verdicts.

Pure (no network) so it is unit-testable against synthetic JSON. Shapes follow
the Metaculus posts API (the same payloads ``forecasting_tools`` consumes).
"""

from __future__ import annotations

import logging
from typing import Any

from metaculus_bot.minibench_analysis.aggregate import QuestionVerdict
from metaculus_bot.minibench_analysis.scoring import (
    NumericCdf,
    argmax_multiple_choice,
    beat_chance_binary,
    beat_chance_multiple_choice,
    beat_chance_numeric,
    directional_binary,
    within_iqr_numeric,
)

logger = logging.getLogger(__name__)

METACULUS_BASE = "https://www.metaculus.com"

# Resolution strings that mean "no ground truth" -> excluded from denominators.
_NON_SCORABLE = {"annulled", "ambiguous", "above_upper_bound", "below_lower_bound"}


def iter_question_jsons(post: dict[str, Any]) -> list[dict[str, Any]]:
    """A post is either a single question or a group of sub-questions."""
    if isinstance(post.get("question"), dict):
        return [post["question"]]
    group = post.get("group_of_questions")
    if isinstance(group, dict) and isinstance(group.get("questions"), list):
        return [q for q in group["questions"] if isinstance(q, dict)]
    return []


def type_bucket(qjson: dict[str, Any]) -> str | None:
    """Map the Metaculus question type into one of our three report buckets.

    ``discrete`` (a numeric variant with a coarser CDF) is bucketed with numeric.
    Unknown types return None and are skipped.
    """
    t = qjson.get("type")
    if t == "binary":
        return "binary"
    if t == "multiple_choice":
        return "multiple_choice"
    if t in ("numeric", "discrete", "date"):
        return "numeric"
    return None


def parse_resolution(qjson: dict[str, Any], bucket: str) -> tuple[bool, Any]:
    """Return (scorable, value). scorable=False for annulled/ambiguous/out-of-bounds."""
    res = qjson.get("resolution")
    if res is None or (isinstance(res, str) and res.lower() in _NON_SCORABLE):
        return False, None
    if bucket == "binary":
        r = str(res).lower()
        if r in ("yes", "no"):
            return True, (r == "yes")
        return False, None
    if bucket == "multiple_choice":
        options = qjson.get("options") or []
        if res in options:
            return True, options.index(res)
        return False, None
    # numeric
    try:
        return True, float(res)
    except (TypeError, ValueError):
        return False, None


def _latest_forecast_values(qjson: dict[str, Any]) -> list[float] | None:
    """The authenticated bot's most recent forecast_values for this question."""
    try:
        values = qjson["my_forecasts"]["latest"]["forecast_values"]
    except (KeyError, TypeError):
        return None
    return list(values) if values else None


def _numeric_cdf(qjson: dict[str, Any], forecast_values: list[float]) -> NumericCdf | None:
    scaling = qjson.get("scaling") or {}
    range_min = scaling.get("range_min")
    range_max = scaling.get("range_max")
    if range_min is None or range_max is None:
        return None
    return NumericCdf.from_forecast_values(
        forecast_values,
        range_min=float(range_min),
        range_max=float(range_max),
        zero_point=scaling.get("zero_point"),
    )


def _peer_score(qjson: dict[str, Any]) -> float | None:
    """Best-effort per-question peer score for the authenticated bot, if present."""
    try:
        scores = qjson["my_forecasts"]["latest"].get("score_data") or {}
    except (KeyError, TypeError):
        return None
    for key in ("peer_score", "spot_peer_score"):
        if isinstance(scores.get(key), (int, float)):
            return float(scores[key])
    return None


def verdict_from_question(
    qjson: dict[str, Any],
    *,
    is_my_bot: bool,
    forecast_values: list[float] | None = None,
) -> QuestionVerdict | None:
    """Classify one question for one bot into a QuestionVerdict.

    ``forecast_values`` defaults to the authenticated bot's ``my_forecasts`` when
    None; pass another bot's values explicitly for the top-10 best-effort path.
    Tier-2 (directional/argmax/IQR) is computed only when ``is_my_bot`` is True.
    """
    bucket = type_bucket(qjson)
    if bucket is None:
        return None
    qid = qjson.get("id", -1)
    values = forecast_values if forecast_values is not None else _latest_forecast_values(qjson)
    answered = values is not None
    scorable, resolved = parse_resolution(qjson, bucket)
    peer = _peer_score(qjson) if is_my_bot else None

    verdict = QuestionVerdict(
        question_id=qid, question_type=bucket, answered=answered, scorable=scorable, peer_score=peer
    )
    if not (answered and scorable):
        return verdict

    try:
        if bucket == "binary":
            p_yes = float(values[-1])  # [p_no, p_yes] or [p_yes]; last element is P(yes)
            verdict.beat_chance = beat_chance_binary(p_yes, resolved)
            if is_my_bot:
                verdict.tier2_correct = directional_binary(p_yes, resolved)
        elif bucket == "multiple_choice":
            probs = [float(v) for v in values]
            verdict.beat_chance = beat_chance_multiple_choice(probs, resolved)
            if is_my_bot:
                verdict.tier2_correct = argmax_multiple_choice(probs, resolved)
        else:  # numeric
            cdf = _numeric_cdf(qjson, [float(v) for v in values])
            if cdf is None:
                verdict.scorable = False
                return verdict
            bc = beat_chance_numeric(cdf, resolved)
            if bc is None:  # resolved outside the in-range grid
                verdict.scorable = False
                return verdict
            verdict.beat_chance = bc
            if is_my_bot:
                verdict.tier2_correct = within_iqr_numeric(cdf, resolved)
    except (ValueError, IndexError, TypeError) as exc:
        logger.warning("Could not classify question %s (%s): %s", qid, bucket, exc)
        verdict.scorable = False
    return verdict


def build_question_url(post: dict[str, Any], qjson: dict[str, Any]) -> str | None:
    """Canonical Metaculus question URL from the post id (+ slug when present)."""
    post_id = post.get("id") or qjson.get("post_id")
    if post_id is None:
        return None
    slug = post.get("slug") or qjson.get("slug")
    tail = f"{slug}/" if slug else ""
    return f"{METACULUS_BASE}/questions/{post_id}/{tail}"


def format_my_prediction(bucket: str, values: list[float] | None, qjson: dict[str, Any]) -> str:
    """Human-readable rendering of the bot's own forecast for the per-question list."""
    if not values:
        return ""
    try:
        if bucket == "binary":
            return f"{float(values[-1]) * 100:.0f}% yes"
        if bucket == "multiple_choice":
            probs = [float(v) for v in values]
            options = qjson.get("options") or []
            i = max(range(len(probs)), key=lambda k: probs[k])
            label = options[i] if i < len(options) else f"option {i}"
            return f"{label} ({probs[i] * 100:.0f}%)"
        cdf = _numeric_cdf(qjson, [float(v) for v in values])
        if cdf is None:
            return ""
        p25, p50, p75 = (cdf.value_at_percentile(q) for q in (0.25, 0.5, 0.75))
        return f"P25={p25:.4g}, P50={p50:.4g}, P75={p75:.4g}"
    except (ValueError, IndexError, TypeError):
        return ""


def my_bot_question_detail(
    post: dict[str, Any], qjson: dict[str, Any], *, minibench_label: str | None = None
) -> dict[str, Any] | None:
    """One per-question row for my bot: title, links, forecast text, verdict.

    All fields come from the already-fetched post/question JSON — no extra API
    call. ``my_answer_url`` is the question URL (Metaculus renders the bot's own
    forecast there; there is no separate per-forecast permalink).
    """
    bucket = type_bucket(qjson)
    if bucket is None:
        return None
    verdict = verdict_from_question(qjson, is_my_bot=True)
    if verdict is None:
        return None
    values = _latest_forecast_values(qjson)
    url = build_question_url(post, qjson)
    row: dict[str, Any] = {}
    if minibench_label is not None:
        row["minibench"] = minibench_label
    row.update(
        {
            "question_id": qjson.get("id"),
            "question_type": bucket,
            "title": post.get("title") or qjson.get("title"),
            "question_url": url,
            "my_answer_url": url,
            "my_prediction": format_my_prediction(bucket, values, qjson),
            "resolution": qjson.get("resolution"),
            "answered": verdict.answered,
            "scorable": verdict.scorable,
            "beat_chance": verdict.beat_chance,
            "tier2_correct": verdict.tier2_correct,
            "peer_score": verdict.peer_score,
        }
    )
    return row
