"""CLI entry point for MiniBench analysis (run by the two GitHub Actions).

Modes:
  two-sessions-ago   Top-N bots + my bot for the MiniBench ``--session-offset``
                     back (default 2, because the most recent MiniBench often
                     still has questions that close/resolve later).
  all-except-current My bot only, one row per resolved MiniBench, current excluded.

Outputs land in ``--output-dir`` (CSV always; XLSX when openpyxl is available)
and a markdown summary is printed to stdout (the workflow tees it to the job
summary / Discord).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from metaculus_bot.minibench_analysis.aggregate import BotSummary, QuestionVerdict, summarize_bot
from metaculus_bot.minibench_analysis.client import MetaculusClient
from metaculus_bot.minibench_analysis.parse import iter_question_jsons, verdict_from_question
from metaculus_bot.minibench_analysis.report import (
    my_bot_accuracy_records,
    my_bot_answered_records,
    render_my_bot_markdown,
    render_top10_markdown,
    top10_records,
    write_csv,
    write_xlsx,
)

logger = logging.getLogger(__name__)


def _my_bot_verdicts(posts: list[dict]) -> list[QuestionVerdict]:
    verdicts: list[QuestionVerdict] = []
    for post in posts:
        for qjson in iter_question_jsons(post):
            v = verdict_from_question(qjson, is_my_bot=True)
            if v is not None:
                verdicts.append(v)
    return verdicts


def _try_other_bot_forecast(qjson: dict, user_id: int | None) -> list[float] | None:
    """Best-effort hook to recover another bot's forecast_values for a question.

    Most AIB tournaments do NOT expose other users' individual forecasts via the
    API, so this returns None by default and the top-N table falls back to
    leaderboard aggregates (per-type accuracy marked unavailable). If a tournament
    DOES expose them, implement the extraction here — nothing else changes.
    """
    return None


def _top_n_summaries(posts: list[dict], leaderboard: list[dict], top_n: int):
    summaries: list[BotSummary] = []
    aggregates: dict[str, dict] = {}
    for entry in leaderboard[:top_n]:
        name = entry.get("username") or f"user_{entry.get('user_id')}"
        verdicts: list[QuestionVerdict] = []
        per_q_available = False
        for post in posts:
            for qjson in iter_question_jsons(post):
                fv = _try_other_bot_forecast(qjson, entry.get("user_id"))
                if fv is None:
                    continue
                per_q_available = True
                v = verdict_from_question(qjson, is_my_bot=False, forecast_values=fv)
                if v is not None:
                    verdicts.append(v)
        summaries.append(summarize_bot(name, verdicts, rank=entry.get("rank")))
        aggregates[name] = {
            "leaderboard_score": entry.get("score"),
            "take": entry.get("take"),
            "peer_score": entry.get("peer_score"),
            "per_question_available": per_q_available,
        }
    return summaries, aggregates


def _pick_tournament(client: MetaculusClient, offset: int) -> dict | None:
    tournaments = client.list_minibench_tournaments()
    if not tournaments:
        logger.error("No MiniBench tournaments found.")
        return None
    if len(tournaments) <= offset:
        logger.warning(
            "Only %d MiniBench tournament(s) found; need %d back. Using the oldest available.",
            len(tournaments),
            offset + 1,
        )
        return tournaments[0]
    return tournaments[-(offset + 1)]


def run_two_sessions_ago(client: MetaculusClient, output_dir: str, *, offset: int, top_n: int) -> str:
    target = _pick_tournament(client, offset)
    if target is None:
        return "No MiniBench tournament available to analyze."
    label = target.get("name") or target.get("slug") or str(target.get("id"))
    tid = target.get("id") or target.get("slug")
    logger.info("Analyzing MiniBench: %s (id=%s)", label, tid)

    posts = client.get_resolved_posts(tid)
    leaderboard = client.get_leaderboard(tid)

    top_summaries, aggregates = _top_n_summaries(posts, leaderboard, top_n)
    top_rows = top10_records(top_summaries, aggregates)
    write_csv(top_rows, os.path.join(output_dir, "top_bots_accuracy.csv"))

    my_summary = summarize_bot("my-bot", _my_bot_verdicts(posts))
    answered = my_bot_answered_records(my_summary, label=label)
    accuracy = my_bot_accuracy_records(my_summary, label=label)
    write_csv([answered], os.path.join(output_dir, "my_bot_answered.csv"))
    write_csv([accuracy], os.path.join(output_dir, "my_bot_accuracy.csv"))
    write_xlsx(
        {"answered": [answered], "accuracy": [accuracy], "top_bots": top_rows},
        os.path.join(output_dir, "my_bot_report.xlsx"),
    )

    return "\n\n".join(
        [
            render_top10_markdown(top_rows, label),
            render_my_bot_markdown(answered, accuracy, label),
            _top_n_caveat(aggregates),
        ]
    )


def run_all_except_current(client: MetaculusClient, output_dir: str) -> str:
    tournaments = client.list_minibench_tournaments()
    if len(tournaments) < 2:
        return "Fewer than 2 MiniBench tournaments found; nothing to analyze after excluding the current one."
    past = tournaments[:-1]  # drop the current (latest) one

    answered_rows: list[dict] = []
    accuracy_rows: list[dict] = []
    for t in past:
        label = t.get("name") or t.get("slug") or str(t.get("id"))
        posts = client.get_resolved_posts(t.get("id") or t.get("slug"))
        summary = summarize_bot("my-bot", _my_bot_verdicts(posts))
        answered_rows.append(my_bot_answered_records(summary, label=label))
        accuracy_rows.append(my_bot_accuracy_records(summary, label=label))

    write_csv(answered_rows, os.path.join(output_dir, "my_bot_history_answered.csv"))
    write_csv(accuracy_rows, os.path.join(output_dir, "my_bot_history_accuracy.csv"))
    write_xlsx(
        {"answered": answered_rows, "accuracy": accuracy_rows},
        os.path.join(output_dir, "my_bot_history.xlsx"),
    )

    total_answered = sum(r.get("total_answered", 0) for r in answered_rows)
    lines = [
        f"### My bot — MiniBench history ({len(past)} tournaments, current excluded)",
        "",
        f"Answered **{total_answered}** questions across {len(past)} past MiniBench(es).",
        "",
        "| MiniBench | Answered | Overall beat-chance | Overall Tier-2 |",
        "|---|---|---|---|",
    ]
    for a, acc in zip(answered_rows, accuracy_rows):
        bc = acc.get("overall_beatchance_pct")
        t2 = acc.get("overall_tier2_pct")
        bc_s = "n/a" if bc is None else f"{bc:.1f}%"
        t2_s = "n/a" if t2 is None else f"{t2:.1f}%"
        lines.append(f"| {a.get('minibench')} | {a.get('total_answered')} | {bc_s} | {t2_s} |")
    return "\n".join(lines)


def _top_n_caveat(aggregates: dict[str, dict]) -> str:
    any_per_q = any(a.get("per_question_available") for a in aggregates.values())
    if any_per_q:
        return ""
    return (
        "> Note: Metaculus did not expose other bots' per-question forecasts, so the top-N "
        "per-type accuracy columns are leaderboard aggregates only (marked n/a). My-bot "
        "figures above are computed from full forecast data."
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Analyze MiniBench results.")
    parser.add_argument("--mode", choices=["two-sessions-ago", "all-except-current"], required=True)
    parser.add_argument("--output-dir", default="minibench_reports")
    parser.add_argument("--session-offset", type=int, default=2, help="How many MiniBenches back (default 2).")
    parser.add_argument("--top-n", type=int, default=10, help="How many top bots to analyze (default 10).")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    client = MetaculusClient()

    me = client.get_me()
    if me:
        logger.info("Authenticated as: %s (id=%s)", me.get("username"), me.get("id"))

    if args.mode == "two-sessions-ago":
        summary = run_two_sessions_ago(client, args.output_dir, offset=args.session_offset, top_n=args.top_n)
    else:
        summary = run_all_except_current(client, args.output_dir)

    print(summary)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as fh:
            fh.write(summary + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
