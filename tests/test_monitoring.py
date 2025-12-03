from datetime import datetime, timedelta
import json

import pandas as pd
import pytest

from monitoring import DailyMonitor, main as monitoring_main


# -------------------------------------------------------------------
# Tests for helpers / data loading
# -------------------------------------------------------------------


def test_build_post_record_submission_and_comment():
    monitor = DailyMonitor()

    submission_item = {
        "id": "s1",
        "subreddit": "testsub",
        "title": "Hello",
        "selftext": "Body",
        "created_utc": 123,
        "permalink": "/r/testsub/s1",
    }
    comment_item = {
        "id": "c1",
        "subreddit": "testsub",
        "body": "Comment body",
        "created_utc": 456,
        "permalink": "/r/testsub/comments/xyz/c1",
    }

    sub_record = monitor._build_post_record(
        item=submission_item, username="user1", is_comment=False
    )
    com_record = monitor._build_post_record(
        item=comment_item, username="user1", is_comment=True
    )

    # submission preserves title/selftext
    assert sub_record["post_id"] == "s1"
    assert sub_record["title"] == "Hello"
    assert sub_record["selftext"] == "Body"
    assert sub_record["author"] == "user1"
    assert sub_record["permalink"].endswith("/r/testsub/s1")

    # comment has empty title and body from "body" field
    assert com_record["post_id"] == "c1"
    assert com_record["title"] == ""
    assert com_record["selftext"] == "Comment body"
    assert com_record["author"] == "user1"
    assert com_record["permalink"].endswith("/r/testsub/comments/xyz/c1")


def test_load_flagged_users_reads_csv(tmp_path):
    data_dir = tmp_path
    users_file = data_dir / "users_risk.csv"
    df = pd.DataFrame(
        [
            {"username": "u_high", "user_risk_score": 0.9},
            {"username": "u_low", "user_risk_score": 0.3},
            {"username": "AutoModerator", "user_risk_score": 0.99},
            {"username": "[deleted]", "user_risk_score": 0.8},
        ]
    )
    df.to_csv(users_file, index=False)

    monitor = DailyMonitor(data_dir=str(data_dir))
    users = monitor.load_flagged_users(min_score=0.5)

    # High-risk only, filtered from AutoModerator and [deleted]
    assert "u_high" in users
    assert "u_low" not in users
    assert "AutoModerator" not in users
    assert "[deleted]" not in users


def test_load_flagged_users_missing_file_returns_empty(tmp_path, capsys):
    # No users_risk.csv in this dir
    monitor = DailyMonitor(data_dir=str(tmp_path))
    users = monitor.load_flagged_users(min_score=0.5)

    assert users == []
    captured = capsys.readouterr()
    assert "users_risk.csv not found" in captured.out


# -------------------------------------------------------------------
# Tests for fetch_user_posts
# -------------------------------------------------------------------


def test_fetch_user_posts_uses_cutoff(monkeypatch):
    monitor = DailyMonitor()
    now_ts = datetime.now().timestamp()
    old_ts = (datetime.now() - timedelta(hours=monitor.check_hours + 1)).timestamp()

    def fake_get_json(path, params=None):
        if "submitted" in path:
            return {
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "new_post",
                                "subreddit": "test",
                                "created_utc": now_ts,
                                "title": "t",
                                "selftext": "b",
                                "permalink": "/p",
                            }
                        },
                        {
                            "data": {
                                "id": "old_post",
                                "subreddit": "test",
                                "created_utc": old_ts,
                                "title": "t",
                                "selftext": "b",
                                "permalink": "/p2",
                            }
                        },
                    ]
                }
            }
        if "comments" in path:
            return {
                "data": {
                    "children": [
                        {
                            "data": {
                                "id": "cnew",
                                "subreddit": "test",
                                "created_utc": now_ts,
                                "body": "hi",
                                "permalink": "/c",
                            }
                        }
                    ]
                }
            }
        return None

    monkeypatch.setattr(monitor.scraper, "_get_json", fake_get_json)

    posts = monitor.fetch_user_posts("u1")
    # old_post filtered by cutoff; we keep one submission + one comment
    ids = {p["post_id"] for p in posts}
    assert ids == {"new_post", "cnew"}


def test_fetch_user_posts_handles_error(monkeypatch, capsys):
    monitor = DailyMonitor()

    def fake_get_json(path, params=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(monitor.scraper, "_get_json", fake_get_json)

    posts = monitor.fetch_user_posts("u1")
    assert posts == []  # should swallow the error and return empty

    captured = capsys.readouterr()
    assert "Error fetching u/u1" in captured.out


# -------------------------------------------------------------------
# Tests for run() main loop
# -------------------------------------------------------------------


def test_run_creates_alerts(monkeypatch):
    monitor = DailyMonitor()
    # Use a simple fake scorer with high risk
    class FakeScorer:
        def score_text(self, text):
            return {
                "risk_score": 0.9,
                "violence_type": "call_to_violence",
                "violent_hits": 0,
                "hate_hits_strong": 0,
                "hate_hits_generic": 0,
                "all_caps_words": 0,
                "exclamations": 0,
                "explanation": "fake",
            }

    monitor.scorer = FakeScorer()

    # 1 high-risk user
    monkeypatch.setattr(monitor, "load_flagged_users", lambda min_score=0.5: ["user1"])

    # user has one post
    monkeypatch.setattr(
        monitor,
        "fetch_user_posts",
        lambda username: [
            {
                "post_id": "p1",
                "subreddit": "test",
                "title": "kill them",
                "selftext": "",
                "permalink": "https://reddit.com/p1",
            }
        ],
    )

    # moderation client mock
    import monitoring

    def fake_check_moderation_flag(text):
        return True, {"violence": 0.99}

    monkeypatch.setattr(monitoring, "check_moderation_flag", fake_check_moderation_flag)

    # avoid real sleep & file writes
    monkeypatch.setattr("monitoring.time.sleep", lambda s: None)
    monkeypatch.setattr(monitor, "save_alerts", lambda: None)

    alerts = monitor.run()
    assert len(alerts) == 1
    alert = alerts[0]
    assert alert["username"] == "user1"
    assert alert["risk_score"] == 0.9
    assert alert["moderation_flagged"] is True
    assert alert["moderation_categories"]["violence"] == 0.99


def test_run_no_users_if_loader_returns_empty(monkeypatch):
    monitor = DailyMonitor()

    # load_flagged_users returns []
    monkeypatch.setattr(monitor, "load_flagged_users", lambda min_score=0.5: [])

    # if run() tried to fetch posts, blow up
    monkeypatch.setattr(
        monitor,
        "fetch_user_posts",
        lambda username: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )

    alerts = monitor.run(users=None)
    assert alerts == []  # early return when no users


def test_run_below_threshold_no_alerts(monkeypatch):
    monitor = DailyMonitor(alert_threshold=0.8)

    class FakeLowScorer:
        def score_text(self, text):
            return {
                "risk_score": 0.3,  # below threshold
                "violence_type": "none",
                "violent_hits": 0,
                "hate_hits_strong": 0,
                "hate_hits_generic": 0,
                "all_caps_words": 0,
                "exclamations": 0,
                "explanation": "fake-low",
            }

    monitor.scorer = FakeLowScorer()

    monkeypatch.setattr(monitor, "load_flagged_users", lambda min_score=0.5: ["user1"])

    # user has one post, but low risk
    monkeypatch.setattr(
        monitor,
        "fetch_user_posts",
        lambda username: [
            {
                "post_id": "p1",
                "subreddit": "test",
                "title": "hello there",
                "selftext": "nice text",
                "permalink": "https://reddit.com/p1",
            }
        ],
    )

    # if save_alerts is called, fail the test
    monkeypatch.setattr(
        monitor,
        "save_alerts",
        lambda: (_ for _ in ()).throw(RuntimeError("should not save alerts")),
    )

    monkeypatch.setattr("monitoring.time.sleep", lambda s: None)

    alerts = monitor.run()
    assert alerts == []  # no alerts when below threshold


# -------------------------------------------------------------------
# Tests for save_alerts and main()
# -------------------------------------------------------------------


def test_save_alerts_writes_json(tmp_path, monkeypatch):
    monitor = DailyMonitor(data_dir=str(tmp_path))
    monitor.alerts = [
        {
            "username": "u1",
            "post_id": "p1",
            "subreddit": "test",
            "text_preview": "preview",
            "risk_score": 0.9,
            "violence_type": "call_to_violence",
            "moderation_flagged": True,
            "moderation_categories": {"violence": 0.99},
            "permalink": "https://reddit.com/p1",
            "alert_time": "2025-01-01T00:00:00",
        }
    ]

    # Avoid dealing with exact timestamp in filename by not mocking datetime;
    # just check we get exactly one JSON file with expected content.
    monitor.save_alerts()

    alerts_dir = tmp_path / "alerts"
    json_files = list(alerts_dir.glob("alerts_*.json"))
    assert len(json_files) == 1

    data = json.loads(json_files[0].read_text())
    assert isinstance(data, list)
    assert data[0]["username"] == "u1"
    assert data[0]["risk_score"] == 0.9


def test_main_calls_run(monkeypatch):
    # Patch DailyMonitor to a fake that records calls
    calls = {"run_called": False}

    class FakeMonitor:
        def __init__(self):
            pass

        def run(self, users=None):
            calls["run_called"] = True
            return []

    import monitoring

    monkeypatch.setattr(monitoring, "DailyMonitor", FakeMonitor)

    monitoring_main()
    assert calls["run_called"] is True
