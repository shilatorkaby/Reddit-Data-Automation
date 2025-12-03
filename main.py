"""
End-to-end pipeline:
- Collect harmful / controversial posts from Reddit (public JSON + HTML).
- Detect profanity using a dictionary.
- Compute violence / hate risk with a context-aware model (RiskScorer).
- Optionally classify content using OpenAI Moderation API.
- Export:
    - full labeled post dataset
    - offensive subset
    - user-level risk feed
"""

from pathlib import Path

from reddit_scraper import RedditScraper
from profanity_detector import load_bad_words
from post_labeler import label_posts
from risk_scorer import RiskScorer
from user_aggregator import build_user_feed_from_posts
from config import SEARCH_TERMS_BY_CATEGORY, TARGET_SUBREDDITS


def main() -> None:
    # Ensure data directory exists
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load dictionary-based profanity lexicon
    bad_words_path = data_dir / "profanity" / "bad_words_en.txt"
    if not bad_words_path.exists():
        # fallback to local file in project root if needed
        bad_words_path = Path("bad_words_en.txt")

    print(f"Loading bad words from: {bad_words_path}")
    bad_words = load_bad_words(str(bad_words_path))

    # 2. Collect targeted Reddit posts
    grouped_queries = {
        cat: " OR ".join(terms)
        for cat, terms in SEARCH_TERMS_BY_CATEGORY.items()
    }

    # Use centralized config for target subreddits
    search_terms = list(grouped_queries.values())

    scraper = RedditScraper()
    print("Collecting posts from Reddit...")
    df_raw = scraper.collect_targeted_posts(
        subreddits=TARGET_SUBREDDITS,
        search_terms=search_terms,
        limit_per_combo=5,
        enrich_html=False,
    )
    raw_posts_path = data_dir / "raw_posts.csv"
    df_raw.to_csv(raw_posts_path, index=False)
    print(f"Collected {len(df_raw)} raw posts. Saved to {raw_posts_path}")

    # 3. Label posts (dictionary + violence model + optional LLM moderation)
    print("Labeling posts with profanity + violence risk model...")
    scorer = RiskScorer()
    df_labeled = label_posts(df_raw, bad_words, use_moderation=True, scorer=scorer)
    labeled_path = data_dir / "raw_posts_labeled.csv"
    df_labeled.to_csv(labeled_path, index=False)
    print(f"Labeled posts saved to {labeled_path}")

    # 4. Offensive subset: high-violence OR profanity OR LLM flagged
    print("Filtering offensive / high-risk posts...")
    high_risk_mask = (
            (df_labeled["has_profanity"])
            | (df_labeled["violence_risk_score"] >= 0.6)
            | (df_labeled.get("moderation_flagged", False))
    )
    df_offensive = df_labeled[high_risk_mask]
    offensive_path = data_dir / "posts_offensive_subset.csv"
    df_offensive.to_csv(offensive_path, index=False)
    print(f"Offensive / high-risk posts found: {len(df_offensive)}. Saved to {offensive_path}")

    # 5. User-level risk feed (from posts we collected)
    print("Building user-level risk feed...")
    users_df = build_user_feed_from_posts(
        df_labeled,
        post_risk_threshold=0.6,
        scorer=scorer
    )
    users_path = data_dir / "users_risk.csv"
    users_df.to_csv(users_path, index=False)
    print(f"User-level risk entries: {len(users_df)}. Saved to {users_path}")

    # Optional: print top high-risk users for quick inspection
    top_users = users_df.sort_values("user_risk_score", ascending=False).head(10)
    print("Top high-risk users (sample):")
    print(top_users[["username", "user_risk_score", "high_risk_posts", "total_posts"]])

    # 6. Enrich high-risk users with 2 months of history
    if not users_df.empty:
        print("\n[STEP 6] Enriching high-risk users with 2 months of history...")
        # high_risk_users = users_df[users_df["user_risk_score"] >= 0.5]["username"].tolist()[:20]  # Top 20
        high_risk_users = users_df.head(20)["username"].tolist()

        if high_risk_users:
            print(f"[INFO] Enriching {len(high_risk_users)} users...")
            df_enriched = scraper.enrich_users_with_history(high_risk_users, months=2)

            if not df_enriched.empty:
                print("[INFO] Labeling enriched posts...")
                df_enriched_labeled = label_posts(df_enriched, bad_words, use_moderation=False, scorer=scorer)

                # Save
                enriched_csv = data_dir / "users_enriched_history.csv"
                enriched_json = data_dir / "users_enriched_history.json"
                df_enriched_labeled.to_csv(enriched_csv, index=False)
                df_enriched_labeled.to_json(enriched_json, orient="records", indent=2)
                print(f"[INFO] Saved {len(df_enriched_labeled)} enriched posts")


if __name__ == "__main__":
    main()
