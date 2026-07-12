"""Render MiniBench analysis into records, CSV/XLSX files, and markdown summaries.

Kept free of network + Metaculus specifics so record-building is testable. CSV is
always written; XLSX is written when ``openpyxl`` is importable (the workflow
installs it) and skipped with a warning otherwise.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from metaculus_bot.minibench_analysis.aggregate import QUESTION_TYPES, BotSummary, TypeSummary

logger = logging.getLogger(__name__)

_TYPE_LABEL = {"binary": "binary", "multiple_choice": "mc", "numeric": "numeric"}


def _type_cells(prefix: str, ts: TypeSummary, *, include_tier2: bool) -> dict[str, Any]:
    cells = {
        f"{prefix}_scored": ts.scorable,
        f"{prefix}_beatchance_n": ts.beat_chance_hits,
        f"{prefix}_beatchance_pct": ts.beat_chance_pct,
    }
    if include_tier2:
        cells[f"{prefix}_tier2_n"] = ts.tier2_hits
        cells[f"{prefix}_tier2_pct"] = ts.tier2_pct
    return cells


def top10_records(summaries: list[BotSummary], aggregates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per top-10 bot: leaderboard aggregate + Tier-1 accuracy by type.

    ``aggregates`` maps bot_name -> {leaderboard_score, take, peer_score,
    per_question_available}. Per-type beat-chance cells are populated when
    per-question forecasts were retrievable, else left None (marked in the flag).
    """
    rows: list[dict[str, Any]] = []
    for s in summaries:
        agg = aggregates.get(s.bot_name, {})
        row: dict[str, Any] = {
            "rank": s.rank,
            "bot": s.bot_name,
            "leaderboard_score": agg.get("leaderboard_score"),
            "take": agg.get("take"),
            "peer_score": agg.get("peer_score"),
            "per_question_available": agg.get("per_question_available", False),
            "overall_scored": s.overall.scorable,
            "overall_beatchance_n": s.overall.beat_chance_hits,
            "overall_beatchance_pct": s.overall.beat_chance_pct,
        }
        for t in QUESTION_TYPES:
            row.update(_type_cells(_TYPE_LABEL[t], s.by_type[t], include_tier2=False))
        rows.append(row)
    return rows


def my_bot_answered_records(summary: BotSummary, *, label: str | None = None) -> dict[str, Any]:
    """A single record: how many questions my bot answered, by type + total."""
    row: dict[str, Any] = {}
    if label is not None:
        row["minibench"] = label
    for t in QUESTION_TYPES:
        row[f"{_TYPE_LABEL[t]}_answered"] = summary.by_type[t].answered
    row["total_answered"] = summary.overall.answered
    return row


def my_bot_accuracy_records(summary: BotSummary, *, label: str | None = None) -> dict[str, Any]:
    """A single record: my bot's Tier-1 + Tier-2 accuracy and peer score by type."""
    row: dict[str, Any] = {}
    if label is not None:
        row["minibench"] = label
    for t in QUESTION_TYPES:
        row.update(_type_cells(_TYPE_LABEL[t], summary.by_type[t], include_tier2=True))
        row[f"{_TYPE_LABEL[t]}_peer_avg"] = summary.by_type[t].avg_peer_score
    row.update(_type_cells("overall", summary.overall, include_tier2=True))
    row["overall_peer_avg"] = summary.overall.avg_peer_score
    return row


def write_csv(records: list[dict[str, Any]], path: str) -> None:
    pd.DataFrame(records).to_csv(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(records))


def write_xlsx(sheets: dict[str, list[dict[str, Any]]], path: str) -> bool:
    """Write a multi-sheet workbook. Returns False (with a warning) if openpyxl absent."""
    try:
        import openpyxl  # noqa: F401,PLC0415
    except ImportError:
        logger.warning("openpyxl not installed — skipping XLSX (%s); CSV still written", path)
        return False
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, records in sheets.items():
            pd.DataFrame(records).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    logger.info("Wrote %s (%d sheet(s))", path, len(sheets))
    return True


def _fmt_pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.1f}%"


def render_top10_markdown(rows: list[dict[str, Any]], tournament_label: str) -> str:
    lines = [
        f"### MiniBench top-10 accuracy — {tournament_label}",
        "",
        'Tier-1 "beat chance" (baseline score > 0): >50% on the resolved binary side, '
        ">1/N on the resolved MC option, above-uniform density on the resolved numeric bin.",
        "",
        "| Rank | Bot | Overall beat-chance | Binary | MC | Numeric | Leaderboard | Per-Q data |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        overall = f"{r['overall_beatchance_n']}/{r['overall_scored']} ({_fmt_pct(r['overall_beatchance_pct'])})"

        def cell(pfx: str) -> str:
            n, d, p = r[f"{pfx}_beatchance_n"], r[f"{pfx}_scored"], r[f"{pfx}_beatchance_pct"]
            return "n/a" if not r["per_question_available"] else f"{n}/{d} ({_fmt_pct(p)})"

        lines.append(
            f"| {r['rank']} | {r['bot']} | {overall} | {cell('binary')} | {cell('mc')} | "
            f"{cell('numeric')} | {r['leaderboard_score']} | {'yes' if r['per_question_available'] else 'no'} |"
        )
    return "\n".join(lines)


def render_my_bot_markdown(answered: dict[str, Any], accuracy: dict[str, Any], label: str) -> str:
    total = answered.get("total_answered", 0)
    lines = [
        f"### My bot — {label}",
        "",
        f"Answered **{total}** questions "
        f"(binary {answered.get('binary_answered', 0)}, mc {answered.get('mc_answered', 0)}, "
        f"numeric {answered.get('numeric_answered', 0)}).",
        "",
        "| Type | Beat-chance | Tier-2 (directional/argmax/IQR) | Avg peer |",
        "|---|---|---|---|",
    ]
    for t in ("binary", "mc", "numeric", "overall"):
        bc = f"{accuracy.get(f'{t}_beatchance_n')}/{accuracy.get(f'{t}_scored')} ({_fmt_pct(accuracy.get(f'{t}_beatchance_pct'))})"
        t2 = (
            f"{accuracy.get(f'{t}_tier2_n')}/{accuracy.get(f'{t}_scored')} ({_fmt_pct(accuracy.get(f'{t}_tier2_pct'))})"
        )
        peer = accuracy.get(f"{t}_peer_avg")
        peer_s = "n/a" if peer is None else f"{peer:.1f}"
        lines.append(f"| {t} | {bc} | {t2} | {peer_s} |")
    return "\n".join(lines)
