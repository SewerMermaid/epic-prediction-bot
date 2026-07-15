"""End-to-end MiniBench CLI test with a fully mocked Metaculus client (no network)."""

import pandas as pd

from metaculus_bot.minibench_analysis import cli
from metaculus_bot.minibench_analysis.aggregate import summarize_bot
from metaculus_bot.minibench_analysis.parse import verdict_from_question
from metaculus_bot.minibench_analysis.report import (
    my_bot_accuracy_records,
    my_bot_answered_records,
    top10_records,
)


class _FakeClient:
    """Stand-in for MetaculusClient returning canned tournaments/posts/leaderboard."""

    def __init__(self):
        self.tournaments = [
            {"id": 1, "slug": "minibench-a", "name": "MiniBench A", "start_date": "2026-05-04"},
            {"id": 2, "slug": "minibench-b", "name": "MiniBench B", "start_date": "2026-05-18"},
            {"id": 3, "slug": "minibench", "name": "MiniBench C", "start_date": "2026-06-01"},
        ]

    def get_me(self):
        return {"id": 99, "username": "my-bot"}

    def list_minibench_tournaments(self):
        return self.tournaments

    def get_leaderboard(self, project_id):
        return [
            {"rank": 1, "username": "alpha", "user_id": 10, "score": 55.0, "take": 500, "peer_score": 12.0},
            {"rank": 2, "username": "beta", "user_id": 11, "score": 40.0, "take": 300, "peer_score": 8.0},
        ]

    def get_resolved_posts(self, tournament):
        # Two binary (one hit, one miss), one numeric uniform (IQR hit, not beat-chance).
        return [
            {
                "id": 101,
                "slug": "q-one",
                "title": "Question one?",
                "question": {
                    "id": 1,
                    "type": "binary",
                    "resolution": "yes",
                    "my_forecasts": {"latest": {"forecast_values": [0.2, 0.8]}},
                },
            },
            {
                "id": 102,
                "slug": "q-two",
                "title": "Question two?",
                "question": {
                    "id": 2,
                    "type": "binary",
                    "resolution": "no",
                    "my_forecasts": {"latest": {"forecast_values": [0.3, 0.7]}},
                },
            },
            {
                "id": 103,
                "slug": "q-three",
                "title": "Question three?",
                "question": {
                    "id": 3,
                    "type": "numeric",
                    "resolution": "50",
                    "scaling": {"range_min": 0.0, "range_max": 100.0, "zero_point": None},
                    "my_forecasts": {"latest": {"forecast_values": [i / 200 for i in range(201)]}},
                },
            },
        ]


def test_two_sessions_ago_writes_files_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: _FakeClient())
    rc = cli.main(["--mode", "two-sessions-ago", "--output-dir", str(tmp_path), "--session-offset", "2"])
    assert rc == 0

    answered = pd.read_csv(tmp_path / "my_bot_answered.csv").iloc[0]
    assert answered["total_answered"] == 3
    assert answered["binary_answered"] == 2
    assert answered["numeric_answered"] == 1

    acc = pd.read_csv(tmp_path / "my_bot_accuracy.csv").iloc[0]
    # Binary: one hit of two scorable -> 50%.
    assert acc["binary_beatchance_pct"] == 50.0
    # Numeric uniform: IQR hit (tier2 100%) but beat-chance 0%.
    assert acc["numeric_tier2_pct"] == 100.0
    assert acc["numeric_beatchance_pct"] == 0.0

    top = pd.read_csv(tmp_path / "top_bots_accuracy.csv")
    assert list(top["bot"]) == ["alpha", "beta"]
    assert not top["per_question_available"].any()  # other bots' forecasts not exposed

    questions = pd.read_csv(tmp_path / "my_bot_questions.csv")
    assert len(questions) == 3
    assert set(questions["question_url"]) == {
        "https://www.metaculus.com/questions/101/q-one/",
        "https://www.metaculus.com/questions/102/q-two/",
        "https://www.metaculus.com/questions/103/q-three/",
    }
    assert (questions["question_url"] == questions["my_answer_url"]).all()
    q1 = questions[questions["question_id"] == 1].iloc[0]
    assert q1["my_prediction"] == "80% yes"
    assert q1["title"] == "Question one?"
    assert q1["accurate"] == "yes"  # 80% yes, resolved yes -> directional hit
    # Q2 forecast 70% yes but resolved no -> inaccurate.
    q2 = questions[questions["question_id"] == 2].iloc[0]
    assert q2["accurate"] == "no"


def test_two_sessions_ago_targets_correct_tournament(tmp_path, monkeypatch):
    """offset=2 with 3 tournaments -> the oldest (index -3) is analyzed."""
    captured = {}
    fake = _FakeClient()
    original = fake.get_resolved_posts

    def _capture(tid):
        captured["tid"] = tid
        return original(tid)

    fake.get_resolved_posts = _capture  # type: ignore
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)
    cli.main(["--mode", "two-sessions-ago", "--output-dir", str(tmp_path), "--session-offset", "2"])
    assert captured["tid"] == 1  # MiniBench A, two sessions before current (id=3)


def test_all_except_current_excludes_latest(tmp_path, monkeypatch):
    fake = _FakeClient()
    seen = []
    orig = fake.get_resolved_posts

    def _track(tid):
        seen.append(tid)
        return orig(tid)

    fake.get_resolved_posts = _track  # type: ignore
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)
    rc = cli.main(["--mode", "all-except-current", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert seen == [1, 2]  # current (id=3) excluded

    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert len(hist) == 2
    assert list(hist["total_answered"]) == [3, 3]

    q_hist = pd.read_csv(tmp_path / "my_bot_history_questions.csv")
    assert len(q_hist) == 6  # 3 questions x 2 past minibenches
    assert set(q_hist["minibench"]) == {"MiniBench A", "MiniBench B"}


def test_all_except_current_single_tournament_degrades(tmp_path, monkeypatch):
    """With only one visible tournament, analyze it (with a note) instead of writing nothing."""
    fake = _FakeClient()
    fake.tournaments = [{"id": 1, "slug": "minibench", "name": "MiniBench", "start_date": "2026-06-01"}]
    monkeypatch.setattr(cli, "MetaculusClient", lambda *a, **k: fake)

    summary = cli.run_all_except_current(fake, str(tmp_path))
    assert "only one MiniBench tournament" in summary
    assert "current included" in summary

    hist = pd.read_csv(tmp_path / "my_bot_history_answered.csv")
    assert len(hist) == 1  # a file is produced, not an empty dir
    assert hist.iloc[0]["total_answered"] == 3


def test_all_except_current_no_tournaments_writes_message(monkeypatch, tmp_path):
    fake = _FakeClient()
    fake.tournaments = []
    summary = cli.run_all_except_current(fake, str(tmp_path))
    assert "No MiniBench tournaments found" in summary


def test_report_records_shapes():
    posts_q = {
        "id": 1,
        "type": "binary",
        "resolution": "yes",
        "my_forecasts": {"latest": {"forecast_values": [0.1, 0.9]}},
    }
    v = verdict_from_question(posts_q, is_my_bot=True)
    s = summarize_bot("my-bot", [v], rank=1)
    assert my_bot_answered_records(s, label="X")["binary_answered"] == 1
    assert my_bot_accuracy_records(s, label="X")["binary_beatchance_pct"] == 100.0
    assert top10_records([s], {"my-bot": {"leaderboard_score": 10}})[0]["leaderboard_score"] == 10
