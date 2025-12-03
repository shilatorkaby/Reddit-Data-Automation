"""
Daily Monitoring System for Flagged Users.

Monitors high-risk users for new posts and generates alerts.

Usage:
    python -m monitoring
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd

from moderation_client import check_moderation_flag
from reddit_scraper import RedditScraper
from risk_scorer import RiskScorer


class DailyMonitor:
    """Monitor flagged users for new high-risk content."""

    def __init__(
            self,
            data_dir: str = "data",
            alert_threshold: float = 0.6,
            check_hours: int = 48,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.alert_threshold = alert_threshold
        self.check_hours = check_hours
        self.scraper = RedditScraper()
        self.scorer = RiskScorer()
        self.alerts: List[Dict] = []

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _build_post_record(
            self,
            *,
            item: Dict,
            username: str,
            is_comment: bool,
    ) -> Dict:
        """
        Normalize a Reddit submission or comment to a common post record.
        """
        if is_comment:
            text = item.get("body", "")
        else:
            text = item.get("selftext", "")
        return {
            "post_id": item.get("id"),
            "author": username,
            "subreddit": item.get("subreddit"),
            "title": "" if is_comment else item.get("title", ""),
            "selftext": text,
            "created_utc": item.get("created_utc"),
            "permalink": f"https://www.reddit.com{item.get('permalink', '')}",
        }

    # ------------------------------------------------------------------ #
    # Data loading / fetching
    # ------------------------------------------------------------------ #

    def load_flagged_users(self, min_score: float = 0.5) -> List[str]:
        """Load high-risk users from users_risk.csv."""
        users_file = self.data_dir / "users_risk.csv"

        if not users_file.exists():
            print("[WARN] users_risk.csv not found. Run main.py first.")
            return []

        df = pd.read_csv(users_file)
        high_risk = df[df["user_risk_score"] >= min_score]
        high_risk = high_risk.sort_values("user_risk_score", ascending=False)

        users = [
            u for u in high_risk["username"].tolist()
            if u and u not in ("[deleted]", "AutoModerator")
        ]

        print(f"[INFO] Loaded {len(users)} high-risk users to monitor")
        return users[:100]  # Limit to 100 users

    def fetch_user_posts(self, username: str) -> List[Dict]:
        """Fetch user's recent posts."""
        cutoff = (datetime.now() - timedelta(hours=self.check_hours)).timestamp()
        posts = []

        try:
            # Fetch submissions
            data = self.scraper._get_json(
                f"/user/{username}/submitted.json",
                {"limit": "25"}
            )
            if data and "data" in data:
                for child in data["data"].get("children", []):
                    p = child.get("data", {})
                    if p.get("created_utc", 0) >= cutoff:
                        posts.append(
                            self._build_post_record(
                                item=p,
                                username=username,
                                is_comment=False,
                            )
                        )
            # Fetch comments
            data = self.scraper._get_json(
                f"/user/{username}/comments.json",
                {"limit": "25"}
            )
            if data and "data" in data:
                for child in data["data"].get("children", []):
                    c = child.get("data", {})
                    if c.get("created_utc", 0) >= cutoff:
                        posts.append(
                            self._build_post_record(
                                item=c,
                                username=username,
                                is_comment=True,
                            )
                        )
        except Exception as e:
            print(f"[WARN] Error fetching u/{username}: {e}")

        return posts

    # ------------------------------------------------------------------ #
    # Main monitoring loop
    # ------------------------------------------------------------------ #
    def run(self, users: List[str] = None) -> List[Dict]:
        """Run monitoring and return alerts."""
        print("=" * 50)
        print("DAILY MONITORING")
        print(f"Time: {datetime.now().isoformat()}")
        print("=" * 50)

        if users is None:
            users = self.load_flagged_users()

        if not users:
            return []

        self.alerts = []

        for i, username in enumerate(users):
            if (i + 1) % 10 == 0:
                print(f"[INFO] Progress: {i + 1}/{len(users)}")

            posts = self.fetch_user_posts(username)

            for post in posts:
                text = (post.get("title") or "") + " " + (post.get("selftext") or "")
                risk = self.scorer.score_text(text)

                if risk["risk_score"] >= self.alert_threshold:
                    # Check with moderation API for high-risk posts
                    flagged, categories = check_moderation_flag(text)

                    alert = {
                        "username": username,
                        "post_id": post["post_id"],
                        "subreddit": post["subreddit"],
                        "text_preview": text[:200],
                        "risk_score": risk["risk_score"],
                        "violence_type": risk["violence_type"],
                        "moderation_flagged": flagged,
                        "moderation_categories": categories,
                        "permalink": post["permalink"],
                        "alert_time": datetime.now().isoformat(),
                    }
                    self.alerts.append(alert)

                    print(f"\n[ALERT] u/{username} - Risk: {risk['risk_score']:.2f}")
                    print(f"  Type: {risk['violence_type']}")
                    print(f"  Preview: {text[:80]}...")

            time.sleep(0.5)  # Rate limiting

        # Save alerts
        if self.alerts:
            self.save_alerts()

        print(f"\n[DONE] Checked {len(users)} users, {len(self.alerts)} alerts")
        return self.alerts

    def save_alerts(self) -> None:
        """Save alerts to JSON file."""
        alerts_dir = self.data_dir / "alerts"
        alerts_dir.mkdir(exist_ok=True)

        filename = f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = alerts_dir / filename

        with open(filepath, "w") as f:
            json.dump(self.alerts, f, indent=2)

        print(f"[INFO] Alerts saved to {filepath}")


def main():
    monitor = DailyMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
