"""Aggregate per-question verdicts into per-bot, per-type accuracy summaries.

Pure/testable: takes already-classified per-question records and rolls them up
into counts and percentages. No network.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

QUESTION_TYPES = ("binary", "multiple_choice", "numeric")


@dataclass
class QuestionVerdict:
    """One bot's outcome on one question.

    ``beat_chance`` is the Tier-1 verdict (all bots). ``tier2_correct`` is the
    intuitive-breakdown verdict (my bot only; None for other bots or when not
    applicable). ``peer_score`` is Metaculus's per-question peer score when the
    API exposed it, else None. ``scorable`` is False for annulled/ambiguous/
    out-of-bounds resolutions, which are excluded from every denominator.
    """

    question_id: int
    question_type: str
    answered: bool
    scorable: bool
    beat_chance: bool | None = None
    tier2_correct: bool | None = None
    peer_score: float | None = None


@dataclass
class TypeSummary:
    answered: int = 0
    scorable: int = 0
    beat_chance_hits: int = 0
    tier2_applicable: int = 0
    tier2_hits: int = 0
    peer_scores: list[float] = field(default_factory=list)

    @property
    def beat_chance_pct(self) -> float | None:
        return _pct(self.beat_chance_hits, self.scorable)

    @property
    def tier2_pct(self) -> float | None:
        return _pct(self.tier2_hits, self.tier2_applicable)

    @property
    def avg_peer_score(self) -> float | None:
        return sum(self.peer_scores) / len(self.peer_scores) if self.peer_scores else None


@dataclass
class BotSummary:
    """A bot's full roll-up: per-type plus an ``overall`` bucket across all types."""

    bot_name: str
    rank: int | None = None
    by_type: dict[str, TypeSummary] = field(default_factory=lambda: {t: TypeSummary() for t in QUESTION_TYPES})
    overall: TypeSummary = field(default_factory=TypeSummary)

    def add(self, v: QuestionVerdict) -> None:
        buckets = (self.by_type.setdefault(v.question_type, TypeSummary()), self.overall)
        for b in buckets:
            if v.answered:
                b.answered += 1
            if not v.scorable:
                continue
            b.scorable += 1
            if v.beat_chance:
                b.beat_chance_hits += 1
            if v.tier2_correct is not None:
                b.tier2_applicable += 1
                if v.tier2_correct:
                    b.tier2_hits += 1
            if v.peer_score is not None:
                b.peer_scores.append(v.peer_score)


def _pct(num: int, den: int) -> float | None:
    return round(100.0 * num / den, 1) if den else None


def summarize_bot(bot_name: str, verdicts: list[QuestionVerdict], rank: int | None = None) -> BotSummary:
    summary = BotSummary(bot_name=bot_name, rank=rank)
    for v in verdicts:
        summary.add(v)
    return summary


def count_answered_by_type(verdicts: list[QuestionVerdict]) -> dict[str, int]:
    """How many questions the bot actually answered, per type (the 'my bot' CSV)."""
    counts: dict[str, int] = defaultdict(int)
    for v in verdicts:
        if v.answered:
            counts[v.question_type] += 1
    return dict(counts)
