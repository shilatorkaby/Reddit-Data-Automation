# unit + small integration

import pandas as pd

from user_aggregator import build_user_feed_from_posts

import pytest


def test_user_aggregation_many_users(risk_scorer):
    """Test aggregation with many users"""
    # Create 500 users with 5 posts each
    posts = []
    for user_id in range(500):
        for post_id in range(5):
            posts.append({
                "author": f"user{user_id}",
                "violence_risk_score": (user_id % 10) / 10  # Varying scores
            })

    df = pd.DataFrame(posts)
    result = build_user_feed_from_posts(df, scorer=risk_scorer)

    assert len(result) == 500
    assert all(result["total_posts"] == 5)


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


def test_build_user_feed_missing_violence_risk_score_column():
    """Test error handling when required column is missing"""
    df = pd.DataFrame([{"author": "user1", "title": "test"}])
    with pytest.raises(ValueError, match="violence_risk_score"):
        build_user_feed_from_posts(df)


def test_build_user_feed_with_nan_scores(risk_scorer):
    """Test handling of NaN values in risk scores"""
    df = pd.DataFrame([
        {"author": "u1", "violence_risk_score": 0.8},
        {"author": "u1", "violence_risk_score": float('nan')},
        {"author": "u1", "violence_risk_score": 0.3},
    ])
    result = build_user_feed_from_posts(df, scorer=risk_scorer)
    assert len(result) == 1
    # Should handle NaN gracefully


def test_build_user_feed_single_high_risk_post(risk_scorer):
    """Test user with only one very high risk post"""
    df = pd.DataFrame([{"author": "u1", "violence_risk_score": 0.95}])
    result = build_user_feed_from_posts(df, post_risk_threshold=0.6, scorer=risk_scorer)
    assert result.iloc[0]["user_risk_score"] > 0.7
    assert result.iloc[0]["high_risk_posts"] == 1


def test_build_user_feed_filters_invalid_usernames():
    """Test filtering of various invalid username formats"""
    df = pd.DataFrame([
        {"author": "[deleted]", "violence_risk_score": 0.9},
        {"author": "AutoModerator", "violence_risk_score": 0.9},
        {"author": "", "violence_risk_score": 0.9},
        {"author": None, "violence_risk_score": 0.9},
        {"author": "valid_user", "violence_risk_score": 0.9},
    ])
    result = build_user_feed_from_posts(df)
    assert len(result) == 1
    assert result.iloc[0]["username"] == "valid_user"
