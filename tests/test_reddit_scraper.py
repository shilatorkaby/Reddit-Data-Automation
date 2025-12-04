# mocking requests & internal methods

import pytest

from reddit_scraper import RedditScraper, REQUEST_DELAY


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text=""):
        self.status_code = status_code
        self._json = json_data or {}
        self.text = text

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# -------------------------------------------------------------------
# Existing tests
# -------------------------------------------------------------------


def test_parse_post_valid():
    data = {
        "id": "abc",
        "author": "user1",
        "created_utc": 1234567890,
        "title": "hi",
        "selftext": "body",
        "subreddit": "test",
        "permalink": "/r/test/abc",
        "url": "https://example.com",
    }
    scraper = RedditScraper()

    post = scraper._parse_post(data, query="q", subreddit=None)
    assert post["post_id"] == "abc"
    assert post["author"] == "user1"
    assert post["query"] == "q"


def test_search_posts_uses_get_json(monkeypatch):
    scraper = RedditScraper()

    # one page with two children, then stop
    payload = {
        "data": {
            "children": [
                {"data": {"id": "1", "author": "u1", "created_utc": 111, "permalink": "/p1"}},
                {"data": {"id": "2", "author": "u2", "created_utc": 222, "permalink": "/p2"}},
            ],
            "after": None,
        }
    }

    def fake_get_json(path, params=None):
        return payload

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)

    posts = scraper.search_posts(query="q", subreddit="testsub", limit=10, enrich_html=False)
    assert len(posts) == 2
    assert {p["post_id"] for p in posts} == {"1", "2"}


def test_get_user_history_stops_at_cutoff(monkeypatch):
    scraper = RedditScraper()

    # first page has a recent and an old submission; we should break on old
    recent_ts = 10_000_000_000
    old_ts = 1_000_000_000

    def fake_get_json(path, params=None):
        if "submitted" in path:
            return {
                "data": {
                    "children": [
                        {"data": {"id": "new", "author": "u1", "created_utc": recent_ts, "permalink": "/s1"}},
                        {"data": {"id": "old", "author": "u1", "created_utc": old_ts, "permalink": "/s2"}},
                    ],
                    "after": None,
                }
            }
        elif "comments" in path:
            return {
                "data": {
                    "children": [
                        {"data": {"id": "c1", "author": "u1", "created_utc": recent_ts, "permalink": "/c1", "body": "hi"}},
                        {"data": {"id": "c_old", "author": "u1", "created_utc": old_ts, "permalink": "/c2", "body": "bye"}},
                    ],
                    "after": None,
                }
            }
        return None

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)

    history = scraper.get_user_history("u1", months=120)  # very large window to include 'recent_ts'
    assert not history["error"]
    assert len(history["submissions"]) == 1
    assert history["submissions"][0]["post_id"] == "new"
    assert len(history["comments"]) == 1
    assert history["comments"][0]["post_id"] == "c1"


# -------------------------------------------------------------------
# New tests for more coverage
# -------------------------------------------------------------------


def test_rate_limit_sleeps_when_called(monkeypatch):
    scraper = RedditScraper()

    # force elapsed < REQUEST_DELAY so we hit the sleep branch
    times = [0.0]  # first call
    sleep_calls = []

    def fake_time():
        # always return same time so elapsed == current - 0 == 0
        return times[0]

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    # patch time and random.uniform
    monkeypatch.setattr("reddit_scraper.time.time", fake_time)
    monkeypatch.setattr("reddit_scraper.time.sleep", fake_sleep)
    monkeypatch.setattr("reddit_scraper.random.uniform", lambda a, b: 0.2)

    scraper._rate_limit()
    # we should have slept at least once
    assert len(sleep_calls) == 1
    # exact value isn't super important, but we know it's > REQUEST_DELAY
    assert sleep_calls[0] >= REQUEST_DELAY


def test_get_json_success(monkeypatch):
    scraper = RedditScraper()

    def fake_get(url, headers=None, params=None, timeout=None):
        return DummyResponse(
            status_code=200,
            json_data={"data": {"ok": True}},
        )

    # avoid real sleeping in _rate_limit
    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)
    monkeypatch.setattr("reddit_scraper.random.uniform", lambda a, b: 0.0)

    result = scraper._get_json("/test", params={"q": "x"})
    assert result == {"data": {"ok": True}}


def test_get_json_handles_429_then_recovers(monkeypatch):
    scraper = RedditScraper()

    calls = {"count": 0}

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(status_code=429)  # first call rate limited
        return DummyResponse(status_code=200, json_data={"data": {"ok": True}})

    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)
    monkeypatch.setattr("reddit_scraper.random.uniform", lambda a, b: 0.0)

    result = scraper._get_json("/test", params={"q": "x"})
    assert result == {"data": {"ok": True}}
    assert calls["count"] >= 2  # retried at least once


def test_get_json_fails_after_retries(monkeypatch):
    scraper = RedditScraper()

    def fake_get(url, headers=None, params=None, timeout=None):
        # always return server error
        return DummyResponse(status_code=500)

    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)
    monkeypatch.setattr("reddit_scraper.random.uniform", lambda a, b: 0.0)

    result = scraper._get_json("/test", params={"q": "x"})
    assert result is None  # after retries, we give up


def test_get_html_success(monkeypatch):
    scraper = RedditScraper()

    html_text = "<html><head><title>Test title</title></head><body></body></html>"

    def fake_get(url, headers=None, timeout=None):
        return DummyResponse(status_code=200, text=html_text)

    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)
    monkeypatch.setattr("reddit_scraper.random.uniform", lambda a, b: 0.0)

    result = scraper._get_html("https://example.com")
    assert "Test title" in result


def test_get_html_handles_429_then_ok(monkeypatch):
    scraper = RedditScraper()

    calls = {"count": 0}

    def fake_get(url, headers=None, timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return DummyResponse(status_code=429)
        return DummyResponse(status_code=200, text="<html></html>")

    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)

    result = scraper._get_html("https://example.com")
    assert result == "<html></html>"
    assert calls["count"] >= 2


def test_enrich_post_with_html_extracts_title(monkeypatch):
    scraper = RedditScraper()

    def fake_get_html(url):
        return "<html><head><title>  My Cool Title  </title></head><body></body></html>"

    monkeypatch.setattr(scraper, "_get_html", fake_get_html)

    post = {"permalink": "https://reddit.com/somepost"}
    enriched = scraper.enrich_post_with_html(post)

    assert enriched["html_title"] == "My Cool Title"


def test_enrich_post_with_html_no_permalink_no_change(monkeypatch):
    scraper = RedditScraper()
    post = {"title": "hi"}  # no permalink, function should early-return

    enriched = scraper.enrich_post_with_html(post)
    assert enriched == post


def test_search_posts_without_subreddit(monkeypatch):
    scraper = RedditScraper()

    payload = {
        "data": {
            "children": [
                {"data": {"id": "1", "author": "u1", "created_utc": 111, "permalink": "/p1"}},
            ],
            "after": None,
        }
    }

    def fake_get_json(path, params=None):
        # path should be /search.json when subreddit is None
        assert path == "/search.json"
        return payload

    monkeypatch.setattr(scraper, "_get_json", fake_get_json)

    posts = scraper.search_posts(query="q", subreddit=None, limit=5, enrich_html=False)
    assert len(posts) == 1
    assert posts[0]["post_id"] == "1"


def test_collect_targeted_posts_deduplicates(monkeypatch):
    scraper = RedditScraper()

    # Two posts with the same post_id should be deduped
    def fake_search_posts(query, subreddit, sort, time_filter, limit, enrich_html):
        return [
            {"post_id": "same", "subreddit": subreddit, "query": query},
            {"post_id": "same", "subreddit": subreddit, "query": query},
        ]

    monkeypatch.setattr(scraper, "search_posts", fake_search_posts)

    df = scraper.collect_targeted_posts(
        subreddits=["sub1"],
        search_terms=["term1"],
        limit_per_combo=10,
        enrich_html=False,
    )
    assert len(df) == 1
    assert df.iloc[0]["post_id"] == "same"


def test_collect_targeted_posts_no_posts_returns_empty(monkeypatch):
    scraper = RedditScraper()

    monkeypatch.setattr(scraper, "search_posts", lambda *a, **k: [])

    df = scraper.collect_targeted_posts(
        subreddits=["sub1"],
        search_terms=["term1"],
        limit_per_combo=5,
        enrich_html=False,
    )
    assert df.empty


def test_get_user_history_invalid_username():
    scraper = RedditScraper()
    history = scraper.get_user_history("[deleted]", months=2)
    assert history["submissions"] == []
    assert history["comments"] == []
    assert history["error"] == "invalid_username"


def test_enrich_users_with_history_handles_error(monkeypatch):
    scraper = RedditScraper()

    def fake_get_user_history(username, months=2, max_posts=500):
        return {"submissions": [], "comments": [], "error": "invalid_username"}

    monkeypatch.setattr(scraper, "get_user_history", fake_get_user_history)

    df = scraper.enrich_users_with_history(["u1", "u2"], months=2)
    assert df.empty  # no posts collected because all users errored


def test_enrich_users_with_history_collects_posts(monkeypatch):
    scraper = RedditScraper()

    def fake_get_user_history(username, months=2, max_posts=500):
        return {
            "submissions": [
                {
                    "post_id": f"s_{username}",
                    "author": username,
                    "subreddit": "test",
                    "title": "t",
                    "selftext": "b",
                    "created_utc": 111,
                    "score": 1,
                    "num_comments": 0,
                    "permalink": "/s",
                    "url": "",
                    "query": "",
                    "html_title": None,
                }
            ],
            "comments": [],
            "error": None,
        }

    monkeypatch.setattr(scraper, "get_user_history", fake_get_user_history)

    df = scraper.enrich_users_with_history(["u1", "u2"], months=2)
    # one submission per user, no comments
    assert len(df) == 2
    assert set(df["author"]) == {"u1", "u2"}
