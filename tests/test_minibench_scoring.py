"""Tests for the pure MiniBench accuracy logic (scoring + aggregation)."""

import math

import pytest

from metaculus_bot.minibench_analysis.aggregate import (
    QuestionVerdict,
    count_answered_by_type,
    summarize_bot,
)
from metaculus_bot.minibench_analysis.scoring import (
    NumericCdf,
    argmax_multiple_choice,
    beat_chance_binary,
    beat_chance_multiple_choice,
    beat_chance_numeric,
    directional_binary,
    generate_cdf_locations,
    within_iqr_numeric,
)


def _linear_cdf(range_min: float, range_max: float, cdf_size: int = 201) -> NumericCdf:
    """A perfectly uniform CDF over [range_min, range_max] (cdf[i] == i/(n-1))."""
    values = [i / (cdf_size - 1) for i in range(cdf_size)]
    return NumericCdf.from_forecast_values(values, range_min=range_min, range_max=range_max, zero_point=None)


def _cdf_from_percentiles(points: list[tuple[float, float]], range_min: float, range_max: float) -> NumericCdf:
    """Build a CDF on the standard grid by linearly interpolating (value, cumprob) points."""
    cdf_size = 201
    x_axis = generate_cdf_locations(range_min, range_max, None, cdf_size)
    xs = [p[0] for p in points]
    cs = [p[1] for p in points]
    cdf = []
    for x in x_axis:
        if x <= xs[0]:
            cdf.append(cs[0])
        elif x >= xs[-1]:
            cdf.append(cs[-1])
        else:
            j = next(k for k in range(len(xs) - 1) if xs[k] <= x <= xs[k + 1])
            frac = (x - xs[j]) / (xs[j + 1] - xs[j])
            cdf.append(cs[j] + frac * (cs[j + 1] - cs[j]))
    return NumericCdf(x_axis=x_axis, cdf=cdf)


class TestBinary:
    @pytest.mark.parametrize(
        "p_yes, resolved_yes, expected",
        [
            (0.73, True, True),
            (0.51, True, True),
            (0.50, True, False),  # exactly chance is not "beat chance"
            (0.40, True, False),
            (0.10, False, True),  # confident NO on a NO resolution
            (0.60, False, False),
        ],
    )
    def test_beat_chance_and_directional_agree(self, p_yes, resolved_yes, expected):
        assert beat_chance_binary(p_yes, resolved_yes) is expected
        # For binary the Tier-2 directional test is identical to Tier-1.
        assert directional_binary(p_yes, resolved_yes) is expected


class TestMultipleChoice:
    def test_beat_chance_uses_uniform_threshold(self):
        # 4 options, uniform is 0.25. Resolved option got 0.30 -> beats chance.
        assert beat_chance_multiple_choice([0.30, 0.30, 0.20, 0.20], resolved_index=0) is True
        # Resolved option got exactly uniform -> not beating chance.
        assert beat_chance_multiple_choice([0.25, 0.25, 0.25, 0.25], resolved_index=2) is False
        # Resolved option below uniform.
        assert beat_chance_multiple_choice([0.40, 0.40, 0.10, 0.10], resolved_index=2) is False

    def test_argmax_differs_from_beat_chance(self):
        probs = [0.30, 0.45, 0.25]  # argmax is index 1
        # Resolved index 0 beats chance (0.30 > 1/3? no -> 0.333). Actually 0.30 < 0.333.
        assert beat_chance_multiple_choice(probs, 0) is False
        assert argmax_multiple_choice(probs, 0) is False
        # Resolved index 1: argmax hit AND beats chance.
        assert argmax_multiple_choice(probs, 1) is True
        assert beat_chance_multiple_choice(probs, 1) is True

    def test_argmax_tie_counts_if_resolved_is_a_max(self):
        probs = [0.4, 0.4, 0.2]
        assert argmax_multiple_choice(probs, 0) is True
        assert argmax_multiple_choice(probs, 1) is True
        assert argmax_multiple_choice(probs, 2) is False

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError):
            beat_chance_multiple_choice([0.5, 0.5], resolved_index=5)


class TestNumeric:
    def test_percentile_inversion_linear(self):
        cdf = _linear_cdf(0.0, 100.0)
        assert cdf.value_at_percentile(0.25) == pytest.approx(25.0, abs=0.5)
        assert cdf.value_at_percentile(0.50) == pytest.approx(50.0, abs=0.5)
        assert cdf.value_at_percentile(0.75) == pytest.approx(75.0, abs=0.5)

    def test_within_iqr(self):
        # Tight distribution centered at 50: P25~45, P75~55.
        cdf = _cdf_from_percentiles([(0.0, 0.0), (45.0, 0.25), (50.0, 0.5), (55.0, 0.75), (100.0, 1.0)], 0.0, 100.0)
        assert within_iqr_numeric(cdf, 50.0) is True
        assert within_iqr_numeric(cdf, 46.0) is True
        assert within_iqr_numeric(cdf, 20.0) is False
        assert within_iqr_numeric(cdf, 90.0) is False

    def test_beat_chance_numeric_concentrated_vs_uniform(self):
        # Uniform forecast never beats the uniform baseline.
        uniform = _linear_cdf(0.0, 100.0)
        assert beat_chance_numeric(uniform, 50.0) is False
        # A forecast concentrated near 50 beats chance there, loses in the tail.
        peaked = _cdf_from_percentiles([(0.0, 0.0), (45.0, 0.25), (50.0, 0.5), (55.0, 0.75), (100.0, 1.0)], 0.0, 100.0)
        assert beat_chance_numeric(peaked, 50.0) is True
        assert beat_chance_numeric(peaked, 95.0) is False

    def test_out_of_bounds_returns_none(self):
        cdf = _linear_cdf(0.0, 100.0)
        assert beat_chance_numeric(cdf, 150.0) is None
        assert cdf.mass_in_bin_containing(-5.0) is None

    def test_log_scaled_axis_is_monotonic_increasing(self):
        locs = generate_cdf_locations(1.0, 1000.0, zero_point=0.0, cdf_size=201)
        assert locs[0] == pytest.approx(1.0)
        assert locs[-1] == pytest.approx(1000.0)
        assert all(locs[i] < locs[i + 1] for i in range(len(locs) - 1))
        # Log spacing puts the midpoint well below the linear midpoint (500.5).
        assert locs[100] < 200.0


class TestAggregation:
    def _verdicts(self):
        return [
            QuestionVerdict(
                1, "binary", answered=True, scorable=True, beat_chance=True, tier2_correct=True, peer_score=12.0
            ),
            QuestionVerdict(
                2, "binary", answered=True, scorable=True, beat_chance=False, tier2_correct=False, peer_score=-4.0
            ),
            QuestionVerdict(3, "numeric", answered=True, scorable=True, beat_chance=True, tier2_correct=False),
            QuestionVerdict(4, "numeric", answered=True, scorable=False),  # annulled -> excluded
            QuestionVerdict(5, "multiple_choice", answered=False, scorable=False),  # not answered
        ]

    def test_counts_and_percentages(self):
        s = summarize_bot("my-bot", self._verdicts(), rank=3)
        assert s.rank == 3
        binary = s.by_type["binary"]
        assert (binary.answered, binary.scorable) == (2, 2)
        assert binary.beat_chance_hits == 1
        assert binary.beat_chance_pct == 50.0
        assert binary.tier2_hits == 1
        assert binary.tier2_pct == 50.0
        assert binary.avg_peer_score == pytest.approx(4.0)  # (12 + -4)/2

        numeric = s.by_type["numeric"]
        # Q4 annulled -> excluded from scorable; Q3 only.
        assert numeric.answered == 2
        assert numeric.scorable == 1
        assert numeric.beat_chance_pct == 100.0
        assert numeric.tier2_pct == 0.0

        # Overall spans binary+numeric scorable (3), beat_chance hits 2.
        assert s.overall.scorable == 3
        assert s.overall.beat_chance_hits == 2
        assert s.overall.beat_chance_pct == pytest.approx(66.7)

    def test_empty_denominator_is_none_not_zero_div(self):
        s = summarize_bot("empty", [QuestionVerdict(9, "binary", answered=False, scorable=False)])
        assert s.by_type["binary"].beat_chance_pct is None
        assert not math.isnan(s.overall.beat_chance_pct or 0.0)

    def test_count_answered_by_type(self):
        assert count_answered_by_type(self._verdicts()) == {"binary": 2, "numeric": 2}
