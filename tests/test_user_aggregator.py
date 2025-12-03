#unit + small integration

import pandas as pd

from user_aggregator import build_user_feed_from_posts


def test_build_user_feed_empty_df_returns_empty():
    df = pd.DataFrame(columns=["author", "violence_risk_score"])
    users_df = build_user_feed_from_posts(df)
    assert users_df.empty
    assert list(users_df.columns) == [
        "username",
        "user_risk_score",
        "high_risk_posts",
        "total_posts",
        "explanation",
    ]


def test_build_user_feed_basic_aggregation(risk_scorer):
    data = [
        {"author": "u1", "violence_risk_score": 0.9},
        {"author": "u1", "violence_risk_score": 0.4},
        {"author": "u2", "violence_risk_score": 0.1},
    ]
    df = pd.DataFrame(data)

    users_df = build_user_feed_from_posts(df, post_risk_threshold=0.5, scorer=risk_scorer)

    row_u1 = users_df[users_df["username"] == "u1"].iloc[0]
    row_u2 = users_df[users_df["username"] == "u2"].iloc[0]

    assert row_u1["high_risk_posts"] == 1
    assert row_u1["total_posts"] == 2
    assert "2 posts analyzed" in row_u1["explanation"]

    assert row_u2["high_risk_posts"] == 0
    assert row_u2["total_posts"] == 1
