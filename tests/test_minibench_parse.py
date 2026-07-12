"""Tests for parsing raw Metaculus JSON into verdicts."""

from metaculus_bot.minibench_analysis.parse import (
    iter_question_jsons,
    parse_resolution,
    type_bucket,
    verdict_from_question,
)


def _binary_q(resolution="yes", p_yes=0.8, forecasted=True):
    q = {"id": 1, "type": "binary", "resolution": resolution}
    if forecasted:
        q["my_forecasts"] = {"latest": {"forecast_values": [1 - p_yes, p_yes]}}
    return q


def _mc_q(resolution="B", probs=(0.2, 0.5, 0.3), forecasted=True):
    q = {"id": 2, "type": "multiple_choice", "resolution": resolution, "options": ["A", "B", "C"]}
    if forecasted:
        q["my_forecasts"] = {"latest": {"forecast_values": list(probs)}}
    return q


def _numeric_q(resolution="50", forecasted=True):
    cdf = [i / 200 for i in range(201)]  # uniform 0..100
    q = {
        "id": 3,
        "type": "numeric",
        "resolution": resolution,
        "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
    }
    if forecasted:
        q["my_forecasts"] = {"latest": {"forecast_values": cdf}}
    return q


class TestExtraction:
    def test_single_question(self):
        post = {"question": {"id": 1, "type": "binary"}}
        assert len(iter_question_jsons(post)) == 1

    def test_group_questions(self):
        post = {"group_of_questions": {"questions": [{"id": 1}, {"id": 2}]}}
        assert len(iter_question_jsons(post)) == 2

    def test_notebook_or_empty(self):
        assert iter_question_jsons({"notebook": {}}) == []

    def test_type_bucket(self):
        assert type_bucket({"type": "binary"}) == "binary"
        assert type_bucket({"type": "multiple_choice"}) == "multiple_choice"
        assert type_bucket({"type": "numeric"}) == "numeric"
        assert type_bucket({"type": "discrete"}) == "numeric"
        assert type_bucket({"type": "weird"}) is None


class TestResolution:
    def test_binary(self):
        assert parse_resolution({"resolution": "yes"}, "binary") == (True, True)
        assert parse_resolution({"resolution": "no"}, "binary") == (True, False)

    def test_non_scorable(self):
        assert parse_resolution({"resolution": "annulled"}, "binary") == (False, None)
        assert parse_resolution({"resolution": "ambiguous"}, "numeric") == (False, None)
        assert parse_resolution({"resolution": "above_upper_bound"}, "numeric") == (False, None)
        assert parse_resolution({"resolution": None}, "binary") == (False, None)

    def test_mc_index(self):
        q = {"resolution": "C", "options": ["A", "B", "C"]}
        assert parse_resolution(q, "multiple_choice") == (True, 2)
        q2 = {"resolution": "Z", "options": ["A", "B", "C"]}
        assert parse_resolution(q2, "multiple_choice") == (False, None)


class TestVerdicts:
    def test_binary_my_bot_hit(self):
        v = verdict_from_question(_binary_q("yes", 0.8), is_my_bot=True)
        assert v.answered and v.scorable
        assert v.beat_chance is True
        assert v.tier2_correct is True  # directional

    def test_binary_miss(self):
        v = verdict_from_question(_binary_q("no", 0.8), is_my_bot=True)
        assert v.beat_chance is False
        assert v.tier2_correct is False

    def test_mc_argmax_vs_beat_chance(self):
        # Resolved "B" (index 1) got 0.5 -> argmax hit and beats 1/3.
        v = verdict_from_question(_mc_q("B", (0.2, 0.5, 0.3)), is_my_bot=True)
        assert v.beat_chance is True
        assert v.tier2_correct is True
        # Resolved "A" (0.2): below 1/3 and not argmax.
        v2 = verdict_from_question(_mc_q("A", (0.2, 0.5, 0.3)), is_my_bot=True)
        assert v2.beat_chance is False
        assert v2.tier2_correct is False

    def test_numeric_iqr(self):
        # Uniform forecast: 50 is within P25-P75 (25..75) but does NOT beat chance.
        v = verdict_from_question(_numeric_q("50"), is_my_bot=True)
        assert v.tier2_correct is True
        assert v.beat_chance is False

    def test_not_answered(self):
        v = verdict_from_question(_binary_q("yes", forecasted=False), is_my_bot=True)
        assert v.answered is False
        assert v.beat_chance is None
        assert v.tier2_correct is None

    def test_non_scorable_resolution(self):
        v = verdict_from_question(_binary_q("annulled", 0.8), is_my_bot=True)
        assert v.answered is True
        assert v.scorable is False
        assert v.beat_chance is None

    def test_other_bot_no_tier2(self):
        # Supplying another bot's forecast values; tier2 must stay None.
        v = verdict_from_question(_binary_q("yes"), is_my_bot=False, forecast_values=[0.1, 0.9])
        assert v.beat_chance is True
        assert v.tier2_correct is None
        assert v.peer_score is None

    def test_numeric_out_of_bounds_value_excluded(self):
        # Resolution numerically outside the axis -> not scorable.
        v = verdict_from_question(_numeric_q("150"), is_my_bot=True)
        assert v.scorable is False
