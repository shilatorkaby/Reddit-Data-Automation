"""
monitoring.py

Daily Monitoring System for Flagged Users.

Monitors high-risk users for new posts and generates alerts.

Usage:
    python -m src.monitoring
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd

from src.reddit_scraper import RedditScraper
from src.risk_scorer import RiskScorer
from src.moderation_client import check_moderation_flag


class DailyMonitor:
    """Monitor flagged users for new high-risk content."""

    def __init__(
        self,
        data_dir: str = "data",
        alert_threshold: float = 0.6,
        check_hours: int = 48,
    ):
        self.data_dir = Path(data_dir)
        self.alert_threshold = alert_threshold
        self.check_hours = check_hours
        self.scraper = RedditScraper()
        self.scorer = RiskScorer()
        self.alerts: List[Dict] = []

    def load_flagged_users(self, min_score: float = 0.5) -> List[str]:
        """Load high-risk users from users_risk.csv."""
        users_file = self.data_dir / "users_risk.csv"

        if not users_file.exists():
            print("[WARN] users_risk.csv not found. Run main.py first.")
            return []

        df = pd.read_csv(users_file)
        high_risk = df[df["user_risk_score"] >= min_score]
        high_risk = high_risk.sort_values("user_risk_score", ascending=False)

        users = [u for u in high_risk["username"].tolist()
                 if u and u not in ("[deleted]", "AutoModerator")]

        print(f"[INFO] Loaded {len(users)} high-risk users to monitor")
        return users[:100]  # Limit to 100 users

    def fetch_user_posts(self, username: str) -> List[Dict]:
        """Fetch user's recent posts."""
        cutoff = (datetime.now() - timedelta(hours=self.check_hours)).timestamp()
        posts = []

        try:
            # Fetch submissions
            data = self.scraper._get_json(f"/user/{username}/submitted.json", {"limit": "25"})
            if data and "data" in data:
                for child in data["data"].get("children", []):
                    p = child.get("data", {})
                    if p.get("created_utc", 0) >= cutoff:
                        posts.append({
                            "post_id": p.get("id"),
                            "author": username,
                            "subreddit": p.get("subreddit"),
                            "title": p.get("title", ""),
                            "selftext": p.get("selftext", ""),
                            "created_utc": p.get("created_utc"),
                            "permalink": f"https://www.reddit.com{p.get('permalink', '')}",
                        })

            # Fetch comments
            data = self.scraper._get_json(f"/user/{username}/comments.json", {"limit": "25"})
            if data and "data" in data:
                for child in data["data"].get("children", []):
                    c = child.get("data", {})
                    if c.get("created_utc", 0) >= cutoff:
                        posts.append({
                            "post_id": c.get("id"),
                            "author": username,
                            "subreddit": c.get("subreddit"),
                            "title": "",
                            "selftext": c.get("body", ""),
                            "created_utc": c.get("created_utc"),
                            "permalink": f"https://www.reddit.com{c.get('permalink', '')}",
                        })
        except Exception as e:
            print(f"[WARN] Error fetching u/{username}: {e}")

        return posts

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

    def save_alerts(self):
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
