"""
post_labeler.py

Label posts using:
  - dictionary-based profanity detection
  - context-aware violence / hate risk scoring (RiskScorer)
  - optional OpenAI Moderation API (LLM moderation classifier)
"""

from typing import Set

import pandas as pd

from src.profanity_detector import analyze_post
from src.moderation_client import check_moderation_flag
from src.risk_scorer import RiskScorer

from config import NEWS_SUBREDDITS, NEWS_KEYWORDS


def is_news_like_post(row) -> bool:
    """
    Heuristic to detect posts that are likely neutral news headlines
    about violent events, not user advocacy of violence.

    We treat these as non-relevant for violence *risk* scoring.
    """
    subreddit = (row.get("subreddit") or "").lower()
    if subreddit not in NEWS_SUBREDDITS:
        return False

    title = (row.get("title") or "").lower()
    selftext = (row.get("selftext") or "").strip()
    url = (row.get("url") or "").strip()

    # Link post with almost no selftext -> classic news share
    if url and len(selftext) < 30:
        # Common "hard news" vocabulary
        if any(k in title for k in NEWS_KEYWORDS):
            return True

    return False


def label_posts(
        df: pd.DataFrame,
        bad_words: Set[str],
        use_moderation: bool = False,
        scorer: RiskScorer = None,
) -> pd.DataFrame:
    """
    For each post in df, compute:

      - Dictionary-based profanity detection:
          has_profanity (bool)
          bad_words (str)

      - Violence / hate risk scoring (RiskScorer):
          violence_risk_score (float)
          violence_type (str)
          violence_explanation (str)

      - Optional OpenAI Moderation API classification:
          moderation_flagged (bool)
          moderation_categories (str)

    Expected input columns:
      - 'title'
      - 'selftext'
      - 'author'
      - 'permalink'
    """
    if scorer is None:
        scorer = RiskScorer()

    df = df.copy()

    # Initialize columns
    df["has_profanity"] = False
    df["bad_words"] = ""

    df["violence_risk_score"] = 0.0
    df["violence_type"] = "none"
    df["violence_explanation"] = ""

    df["moderation_flagged"] = False
    df["moderation_categories"] = ""

    for i, row in df.iterrows():
        title = row.get("title")
        body = row.get("selftext")
        text = (title or "") + " " + (body or "")

        # 1) Dictionary-based profanity detection
        has_prof, matched_words = analyze_post(title, body, bad_words)
        df.at[i, "has_profanity"] = has_prof
        df.at[i, "bad_words"] = ", ".join(matched_words)

        # 2) Violence / hate risk scoring (context-aware)
        risk_info = scorer.score_text(text)
        news_like = is_news_like_post(row)
        df.at[i, "is_news_like"] = news_like  # optional extra column

        if news_like and risk_info["violence_type"] in {"none", "descriptive"}:
            # Treat as neutral news report: no risk
            df.at[i, "violence_risk_score"] = 0.0
            df.at[i, "violence_type"] = "none"
            df.at[i, "violence_explanation"] = (
                    risk_info["explanation"]
                    + "; classified as news report (describing an external event, not advocating violence)."
            )
        else:
            df.at[i, "violence_risk_score"] = risk_info["risk_score"]
            df.at[i, "violence_type"] = risk_info["violence_type"]
            df.at[i, "violence_explanation"] = risk_info["explanation"]

        # Only ask OpenAI for posts that our model thinks are somewhat risky
        if use_moderation and risk_info["risk_score"] >= 0.8 and not news_like:
            flagged, categories = check_moderation_flag(text)
            df.at[i, "moderation_flagged"] = flagged
            if flagged and categories:
                cat_str = "; ".join(f"{k}:{v:.2f}" for k, v in categories.items())
                df.at[i, "moderation_categories"] = cat_str

            # If we think it's high risk but OpenAI doesn't flag anything relevant,
            # we can be downscale the risk.
            if risk_info["risk_score"] >= 0.8 and not flagged:
                df.at[i, "violence_risk_score"] = risk_info["risk_score"] * 0.6

    return df
