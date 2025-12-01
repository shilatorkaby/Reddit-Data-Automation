"""
user_aggregator.py

User-level risk aggregation based on labeled posts.

Responsibilities:
  - Group posts by author.
  - Aggregate post-level violence risk into a user-level score.
  - Provide explanations per user.

This can be extended later to include full 2-month history per user.
"""

from typing import Optional

import pandas as pd

from src.risk_scorer import RiskScorer


def build_user_feed_from_posts(
    df_posts: pd.DataFrame,
    post_risk_threshold: float = 0.6,
    scorer: Optional[RiskScorer] = None,
) -> pd.DataFrame:
    """
    Aggregate user-level risk based on the labeled posts we already collected.

    For each user:
      - take all violence_risk_score values
      - compute user_risk_score (weighted max + mean)
      - count how many posts exceed the post_risk_threshold
      - produce an explanation string

    Returns:
      users_df with columns:
        - username
        - user_risk_score
        - high_risk_posts
        - total_posts
        - explanation
    """
    if scorer is None:
        scorer = RiskScorer()

    if df_posts.empty:
        return pd.DataFrame(
            columns=["username", "user_risk_score", "high_risk_posts", "total_posts", "explanation"]
        )

    if "violence_risk_score" not in df_posts.columns:
        raise ValueError("df_posts must contain 'violence_risk_score' column.")

    rows = []

    for username, group in df_posts.groupby("author"):
        if not isinstance(username, str) or username in ("[deleted]", "AutoModerator"):
            continue

        scores = group["violence_risk_score"].fillna(0.0).tolist()
        user_score = scorer.aggregate_user_score(scores)
        high_risk = int((group["violence_risk_score"] >= post_risk_threshold).sum())
        total = int(len(scores))

        explanation = (
            f"{total} posts analyzed; "
            f"{high_risk} post(s) with violence_risk_score >= {post_risk_threshold:.2f}. "
            "User-level score emphasizes worst and average behavior."
        )

        rows.append(
            {
                "username": username,
                "user_risk_score": user_score,
                "high_risk_posts": high_risk,
                "total_posts": total,
                "explanation": explanation,
            }
        )

    users_df = pd.DataFrame(rows)
    return users_df
