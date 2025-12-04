# unit + mocking LLM

import pandas as pd

from post_labeler import label_posts, is_news_like_post
from config import NEWS_SUBREDDITS, NEWS_KEYWORDS
import time


def test_label_posts_performance_large_dataset(bad_words, fake_low_risk_scorer):
    """Test performance with large dataset"""
    # Create 1000 posts
    df_large = pd.DataFrame([
        {"title": f"Post {i}", "selftext": "Some text here", "author": f"user{i % 100}",
         "permalink": f"/r/test/{i}", "subreddit": "test", "url": ""}
        for i in range(1000)
    ])

    start_time = time.time()
    result = label_posts(df_large, bad_words, use_moderation=False, scorer=fake_low_risk_scorer)
    elapsed = time.time() - start_time

    assert len(result) == 1000
    assert elapsed < 30  # Should complete in reasonable time
def test_label_posts_handles_missing_columns():
    """Test graceful handling of missing required columns"""
    df = pd.DataFrame([{"title": "test"}])  # missing 'author', 'selftext', etc.
    # Should either raise clear error or handle gracefully


def test_label_posts_empty_dataframe(bad_words, fake_low_risk_scorer):
    """Test handling of empty DataFrame"""
    df = pd.DataFrame()
    result = label_posts(df, bad_words, use_moderation=False, scorer=fake_low_risk_scorer)
    assert result.empty


def test_label_posts_news_filtering_edge_cases(bad_words, fake_high_risk_scorer):
    """Test news filtering with various edge cases"""
    test_cases = [
        # News subreddit + keyword but with selftext
        {"subreddit": "news", "title": "Police arrested suspect", "selftext": "I think we should kill them all",
         "url": ""},
        # News subreddit without news keywords
        {"subreddit": "news", "title": "Random discussion", "selftext": "", "url": "http://example.com"},
    ]

def test_label_posts_with_null_values(bad_words, fake_low_risk_scorer):
    """Test handling of None/NaN values in text fields"""
    df = pd.DataFrame([
        {"title": None, "selftext": None, "author": "user1", "permalink": "/test", "subreddit": "test", "url": ""},
        {"title": "test", "selftext": None, "author": "user2", "permalink": "/test2", "subreddit": "test", "url": ""},
    ])
    result = label_posts(df, bad_words, use_moderation=False, scorer=fake_low_risk_scorer)
    assert len(result) == 2
    assert all(pd.notna(result["violence_risk_score"]))

def test_is_news_like_post_true_for_news_headline():
    row = {
        "subreddit": next(iter(NEWS_SUBREDDITS)),
        "title": f"Suspect arrested after {next(iter(NEWS_KEYWORDS))}",
        "selftext": "",
        "url": "https://example.com/news",
    }
    assert is_news_like_post(row) == True


def test_is_news_like_post_false_for_non_news():
    row = {
        "subreddit": "pics",
        "title": "look at this",
        "selftext": "",
        "url": "https://example.com",
    }
    assert is_news_like_post(row) is False


def test_label_posts_basic_scoring_no_moderation(bad_words, fake_high_risk_scorer):
    df = pd.DataFrame(
        [
            {
                "title": "you are badword",
                "selftext": "",
                "author": "user1",
                "permalink": "/r/test/1",
                "subreddit": "test",
                "url": "",
            }
        ]
    )

    labeled = label_posts(df, bad_words, use_moderation=False, scorer=fake_high_risk_scorer)

    assert labeled.loc[0, "has_profanity"] == True
    assert "badword" in labeled.loc[0, "bad_words"]
    assert labeled.loc[0, "violence_risk_score"] == fake_high_risk_scorer.base_score
    assert labeled.loc[0, "moderation_flagged"] == False


def test_label_posts_with_moderation_called(monkeypatch, bad_words, fake_high_risk_scorer):
    df = pd.DataFrame(
        [
            {
                "title": "we should kill them",
                "selftext": "",
                "author": "user2",
                "permalink": "/r/test/2",
                "subreddit": "test",
                "url": "",
            }
        ]
    )

    called = {}

    def fake_check(text):
        called["text"] = text
        return True, {"violence": 0.99}

    import post_labeler

    monkeypatch.setattr(post_labeler, "check_moderation_flag", fake_check)

    labeled = label_posts(df, bad_words, use_moderation=True, scorer=fake_high_risk_scorer)

    assert called["text"].startswith("we should kill")
    assert labeled.loc[0, "moderation_flagged"] == True
    assert "violence:0.99" in labeled.loc[0, "moderation_categories"]
