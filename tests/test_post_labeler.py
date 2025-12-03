# unit + mocking LLM

import pandas as pd

from post_labeler import label_posts, is_news_like_post
from config import NEWS_SUBREDDITS, NEWS_KEYWORDS


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
