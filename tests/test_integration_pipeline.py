"""
small integration across modules

Here we test:
 posts → label_posts → build_user_feed_from_posts,
 without touching network or OpenAI
"""
import pandas as pd

from post_labeler import label_posts
from user_aggregator import build_user_feed_from_posts
from risk_scorer import RiskScorer


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
