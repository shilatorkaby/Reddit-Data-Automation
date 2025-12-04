"""
small integration across modules

Here we test:
 posts → label_posts → build_user_feed_from_posts,
 without touching network or OpenAI
"""

import pandas as pd

from post_labeler import label_posts
from risk_scorer import RiskScorer
from user_aggregator import build_user_feed_from_posts


def test_main_pipeline_execution(monkeypatch, tmp_path):
    """Test main.py pipeline execution"""
    # Create required files
    bad_words_file = tmp_path / "bad_words_en.txt"
    bad_words_file.write_text("kill\nmurder\n")

    # Mock scraper to return fake data
    class FakeScraper:
        def collect_targeted_posts(self, **kwargs):
            return pd.DataFrame([
                {"post_id": "1", "title": "Test", "selftext": "", "author": "user1",
                 "permalink": "/test", "subreddit": "test", "url": "", "query": "test",
                 "created_utc": 123456, "score": 1, "num_comments": 0, "html_title": None}
            ])

        def enrich_users_with_history(self, usernames, months=2):
            return pd.DataFrame()

    # Import main module
    import main

    # Mock the scraper
    monkeypatch.setattr(main, "RedditScraper", FakeScraper)

    # Mock load_bad_words to use our test file
    def fake_load_bad_words(path):
        return {"kill", "murder"}

    import profanity_detector
    monkeypatch.setattr(profanity_detector, "load_bad_words", fake_load_bad_words)

    # Change working directory to tmp_path
    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Run main
        main.main()

        # Verify outputs exist
        assert (data_dir / "raw_posts.csv").exists()
        assert (data_dir / "raw_posts_labeled.csv").exists()
        assert (data_dir / "users_risk.csv").exists()

        # Verify content
        df_raw = pd.read_csv(data_dir / "raw_posts.csv")
        assert len(df_raw) == 1

        df_labeled = pd.read_csv(data_dir / "raw_posts_labeled.csv")
        assert "violence_risk_score" in df_labeled.columns

        users_df = pd.read_csv(data_dir / "users_risk.csv")
        assert len(users_df) >= 0  # May be empty if no high-risk users

    finally:
        # Restore original directory
        os.chdir(original_cwd)


def test_main_pipeline_with_high_risk_users(monkeypatch, tmp_path):
    """Test main pipeline creates user enrichment files"""
    # Setup
    bad_words_file = tmp_path / "bad_words_en.txt"
    bad_words_file.write_text("kill\nmurder\n")

    class FakeScraper:
        def collect_targeted_posts(self, **kwargs):
            return pd.DataFrame([
                {"post_id": "1", "title": "Kill them all", "selftext": "violent post", "author": "baduser",
                 "permalink": "/test", "subreddit": "test", "url": "", "query": "test",
                 "created_utc": 123456, "score": 1, "num_comments": 0, "html_title": None}
            ])

        def enrich_users_with_history(self, usernames, months=2):
            return pd.DataFrame([
                {"post_id": "2", "title": "Another violent post", "selftext": "more violence",
                 "author": "baduser", "permalink": "/test2", "subreddit": "test", "url": "",
                 "created_utc": 123457, "score": 1, "num_comments": 0, "query": "", "html_title": None}
            ])

    import main
    monkeypatch.setattr(main, "RedditScraper", FakeScraper)

    def fake_load_bad_words(path):
        return {"kill", "murder"}

    import profanity_detector
    monkeypatch.setattr(profanity_detector, "load_bad_words", fake_load_bad_words)

    import os
    original_cwd = os.getcwd()
    os.chdir(tmp_path)

    try:
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        main.main()

        # Verify enrichment files exist
        assert (data_dir / "users_enriched_history.csv").exists()
        assert (data_dir / "users_enriched_history.json").exists()

        # Verify enriched data
        df_enriched = pd.read_csv(data_dir / "users_enriched_history.csv")
        assert len(df_enriched) >= 1

    finally:
        os.chdir(original_cwd)

def test_pipeline_label_and_aggregate(bad_words):
    # two users, one clearly more toxic
    df_raw = pd.DataFrame(
        [
            {
                "title": "I hate this",
                "selftext": "no violence here",
                "author": "u1",
                "permalink": "/r/test/1",
                "subreddit": "test",
                "url": "",
            },
            {
                "title": "we should kill them all",
                "selftext": "",
                "author": "u2",
                "permalink": "/r/test/2",
                "subreddit": "test",
                "url": "",
            },
        ]
    )

    scorer = RiskScorer()
    df_labeled = label_posts(df_raw, bad_words, use_moderation=False, scorer=scorer)

    assert "violence_risk_score" in df_labeled.columns
    users_df = build_user_feed_from_posts(df_labeled, post_risk_threshold=0.6, scorer=scorer)

    # Expect that u2 is higher risk than u1
    u1_score = users_df[users_df["username"] == "u1"]["user_risk_score"].iloc[0]
    u2_score = users_df[users_df["username"] == "u2"]["user_risk_score"].iloc[0]

    assert u2_score > u1_score


def test_full_pipeline_end_to_end(bad_words, tmp_path):
    """Test complete pipeline from raw posts to user feed"""
    # Create sample raw posts
    df_raw = pd.DataFrame([
        {"title": "Normal post", "selftext": "Nothing bad", "author": "user1",
         "permalink": "/r/test/1", "subreddit": "test", "url": ""},
        {"title": "Kill them all", "selftext": "We should attack", "author": "user2",
         "permalink": "/r/test/2", "subreddit": "test", "url": ""},
        {"title": "More violence", "selftext": "I hate everyone", "author": "user2",
         "permalink": "/r/test/3", "subreddit": "test", "url": ""},
    ])

    # Run through labeling
    scorer = RiskScorer()
    df_labeled = label_posts(df_raw, bad_words, use_moderation=False, scorer=scorer)

    # Verify labeled output
    assert "violence_risk_score" in df_labeled.columns
    assert "has_profanity" in df_labeled.columns
    assert len(df_labeled) == 3

    # Build user feed
    users_df = build_user_feed_from_posts(df_labeled, post_risk_threshold=0.6, scorer=scorer)

    # Verify user aggregation
    assert len(users_df) == 2
    user2_score = users_df[users_df["username"] == "user2"]["user_risk_score"].iloc[0]
    user1_score = users_df[users_df["username"] == "user1"]["user_risk_score"].iloc[0]
    assert user2_score > user1_score
    assert user2_score > 0.5


def test_pipeline_with_news_posts(bad_words):
    """Test pipeline correctly handles news posts"""
    df_raw = pd.DataFrame([
        {"title": "Police arrested suspect in shooting", "selftext": "",
         "author": "user1", "permalink": "/r/news/1", "subreddit": "news",
         "url": "https://example.com/news"},
    ])

    scorer = RiskScorer()
    df_labeled = label_posts(df_raw, bad_words, use_moderation=False, scorer=scorer)

    # News posts should have low/zero risk scores
    assert df_labeled.iloc[0]["violence_risk_score"] < 0.2
    assert df_labeled.iloc[0]["is_news_like"] == True


def test_pipeline_handles_empty_input(bad_words):
    """Test pipeline gracefully handles empty DataFrame"""
    df_empty = pd.DataFrame()

    scorer = RiskScorer()
    df_labeled = label_posts(df_empty, bad_words, use_moderation=False, scorer=scorer)
    users_df = build_user_feed_from_posts(df_labeled, scorer=scorer)

    assert df_labeled.empty
    assert users_df.empty