"""
Reddit scraper using public JSON endpoints (no API keys or OAuth).
- Demonstrates HTML parsing with BeautifulSoup for extra metadata.
- Includes user history enrichment (2+ months of data per user).
"""

from typing import List, Dict, Optional

import time
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup

from config import (
    BASE_URL,
    HEADERS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    TARGET_SUBREDDITS,
    ALL_SEARCH_TERMS,
)

# Rate limiting - Reddit allows ~60 requests per minute
REQUEST_DELAY = 0.5  # seconds between requests


class RedditScraper:
    """
    Simple wrapper around Reddit's public JSON and HTML endpoints.
    Handles edge cases: deleted/suspended users, private profiles, rate limiting.
    """

    def __init__(
            self,
            base_url: str = BASE_URL,
            headers: Dict[str, str] = None,
            timeout: int = REQUEST_TIMEOUT,
            max_retries: int = MAX_RETRIES,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or HEADERS.copy()
        self.timeout = timeout
        self.max_retries = max_retries
        self._last_request = 0.0

    def _rate_limit(self):
        # Wait between requests to avoid 429 errors.
        elapsed = time.time() - self._last_request
        if elapsed < REQUEST_DELAY:
            sleep_time = REQUEST_DELAY - elapsed + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
        self._last_request = time.time()

    def _get_json(self, path: str, params: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        # Make a GET request to Reddit and return parsed JSON.
        url = f"{self.base_url}{path}"

        self._rate_limit()  # Always wait before request

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=self.timeout,
                )

                if resp.status_code == 429:
                    # Rate limited - wait longer
                    sleep_time = 10 * attempt  # 10s, 20s, 30s
                    print(f"[WARN] Rate limited (429), waiting {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue

                if resp.status_code == 404:
                    # Not found (deleted user, private profile, etc.)
                    return None

                if resp.status_code in (500, 502, 503, 504):
                    sleep_time = 3 * attempt
                    print(f"[WARN] Server error {resp.status_code}, retry in {sleep_time}s")
                    time.sleep(sleep_time)
                    continue

                resp.raise_for_status()
                return resp.json()

            except requests.RequestException as e:
                print(f"[ERROR] Request error: {e}")
                time.sleep(2)

        print(f"[ERROR] Failed after {self.max_retries} attempts: {path}")
        return None

    def _get_html(self, url: str) -> Optional[str]:
        """Fetch raw HTML from a URL."""
        self._rate_limit()

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.get(url, headers=self.headers, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(10 * attempt)
                    continue
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as e:
                print(f"[ERROR] HTML error: {e}")
                time.sleep(2)
        return None

    def enrich_post_with_html(self, post: Dict) -> Dict:
        """Fetch HTML and extract title using BeautifulSoup."""
        permalink = post.get("permalink")
        if not permalink:
            return post

        html = self._get_html(permalink)
        if not html:
            return post

        try:
            soup = BeautifulSoup(html, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                post["html_title"] = title_tag.get_text(strip=True)
        except Exception as e:
            print(f"[WARN] HTML parse error: {e}")

        return post

    def search_posts(
            self,
            query: str,
            subreddit: Optional[str] = None,
            sort: str = "top",
            time_filter: str = "year",
            limit: int = 50,
            enrich_html: bool = False,
    ) -> List[Dict]:
        """Search Reddit for posts matching a query."""
        posts: List[Dict] = []
        after: Optional[str] = None

        while len(posts) < limit:
            if subreddit:
                path = f"/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "1",
                    "sort": sort,
                    "t": time_filter,
                    "limit": min(25, limit - len(posts)),  # Smaller batches
                }
            else:
                path = "/search.json"
                params = {
                    "q": query,
                    "sort": sort,
                    "t": time_filter,
                    "limit": min(25, limit - len(posts)),
                }

            if after:
                params["after"] = after

            data = self._get_json(path, params=params)
            if not data or "data" not in data:
                break

            children = data["data"].get("children", [])
            if not children:
                break

            for child in children:
                post_data = child.get("data", {})
                post = self._parse_post(post_data, query=query, subreddit=subreddit)
                if post:
                    if enrich_html:
                        post = self.enrich_post_with_html(post)
                    posts.append(post)

                if len(posts) >= limit:
                    break

            after = data["data"].get("after")
            if not after:
                break

        print(f"[INFO] Found {len(posts)} posts for '{query}' in r/{subreddit or 'all'}")
        print(f"[INFO] Found {len(posts)} posts for '{query}' in r/{subreddit or 'all'}")
        return posts

    def collect_targeted_posts(
            self,
            subreddits: Optional[List[str]] = None,
            search_terms: Optional[List[str]] = None,
            limit_per_combo: int = 10,
            enrich_html: bool = False,
    ) -> pd.DataFrame:
        """
        Collect posts across subreddits and search terms.

        Default limit reduced to 10 per combo to avoid rate limits.
        """
        if subreddits is None:
            subreddits = TARGET_SUBREDDITS
        if search_terms is None:
            search_terms = ALL_SEARCH_TERMS

        all_posts: List[Dict] = []
        seen_ids = set()

        total_combos = len(subreddits) * len(search_terms)
        current = 0

        for subreddit in subreddits:
            for term in search_terms:
                current += 1
                print(f"[INFO] Searching {current}/{total_combos}: r/{subreddit} - '{term}'")

                posts = self.search_posts(
                    query=term,
                    subreddit=subreddit,
                    sort="new",
                    time_filter="month",
                    limit=limit_per_combo,
                    enrich_html=enrich_html,
                )

                for p in posts:
                    key = p["post_id"]
                    if key not in seen_ids:
                        seen_ids.add(key)
                        all_posts.append(p)

        if not all_posts:
            print("[WARN] No posts collected.")
            return pd.DataFrame()

        df = pd.DataFrame(all_posts)
        print(f"[INFO] Total unique posts: {len(df)}")
        return df

    def get_user_history(self, username: str, months: int = 2, max_posts: int = 500) -> Dict[str, List[Dict]]:
        """Fetch user's 2+ months of posts/comments."""
        if not username or username in ("[deleted]", "[removed]", "AutoModerator"):
            return {"submissions": [], "comments": [], "error": "invalid_username"}

        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=months * 30)).timestamp()
        submissions, comments = [], []

        # Fetch submissions
        try:
            after = None
            while len(submissions) < max_posts:
                data = self._get_json(f"/user/{username}/submitted.json",
                                      {"limit": "100", "after": after} if after else {"limit": "100"})
                if not data or "data" not in data:
                    break
                children = data["data"].get("children", [])
                if not children:
                    break
                for child in children:
                    p = child.get("data", {})
                    if p.get("created_utc", 0) < cutoff:
                        break
                    post = self._parse_post(p, query="", subreddit=None)
                    if post:
                        submissions.append(post)
                if children and children[-1].get("data", {}).get("created_utc", 0) < cutoff:
                    break
                after = data["data"].get("after")
                if not after:
                    break
        except Exception as e:
            print(f"[WARN] Error fetching u/{username} submissions: {e}")

        # Fetch comments (same logic)
        try:
            after = None
            while len(comments) < max_posts:
                data = self._get_json(f"/user/{username}/comments.json",
                                      {"limit": "100", "after": after} if after else {"limit": "100"})
                if not data or "data" not in data:
                    break
                children = data["data"].get("children", [])
                if not children:
                    break
                for child in children:
                    c = child.get("data", {})
                    if c.get("created_utc", 0) < cutoff:
                        break
                    comment = {
                        "post_id": c.get("id"), "author": username, "subreddit": c.get("subreddit", ""),
                        "title": "", "selftext": c.get("body", ""), "created_utc": c.get("created_utc"),
                        "score": c.get("score", 0), "num_comments": 0,
                        "permalink": f"https://www.reddit.com{c.get('permalink', '')}", "url": "", "query": ""
                    }
                    comments.append(comment)
                if children and children[-1].get("data", {}).get("created_utc", 0) < cutoff:
                    break
                after = data["data"].get("after")
                if not after:
                    break
        except Exception as e:
            print(f"[WARN] Error fetching u/{username} comments: {e}")

        print(f"[INFO] u/{username}: {len(submissions)} submissions, {len(comments)} comments")
        return {"submissions": submissions, "comments": comments, "error": None}

    def enrich_users_with_history(self, usernames: List[str], months: int = 2) -> pd.DataFrame:
        """Fetch 2+ months history for multiple users."""
        all_posts = []
        for i, username in enumerate(usernames, 1):
            print(f"[INFO] Fetching history for u/{username} ({i}/{len(usernames)})")
            history = self.get_user_history(username, months=months)
            if history.get("error"):
                print(f"[WARN] Skipping u/{username}: {history['error']}")
                continue
            all_posts.extend(history["submissions"])
            all_posts.extend(history["comments"])

        if not all_posts:
            print("[WARN] No user history collected.")
            return pd.DataFrame()

        df = pd.DataFrame(all_posts)
        print(f"[INFO] Collected {len(df)} historical posts from {len(usernames)} users")
        return df

    def _parse_post(self, data: Dict, query: str, subreddit: Optional[str]) -> Optional[Dict]:
        """Extract post dictionary from Reddit JSON."""
        try:
            post_id = data.get("id")
            author = data.get("author")
            created_utc = data.get("created_utc")

            if not post_id or not author or created_utc is None:
                return None

            return {
                "post_id": post_id,
                "subreddit": data.get("subreddit") or subreddit,
                "query": query,
                "author": author,
                "title": data.get("title"),
                "selftext": data.get("selftext"),
                "created_utc": created_utc,
                "score": data.get("score"),
                "num_comments": data.get("num_comments"),
                "permalink": f"https://www.reddit.com{data.get('permalink', '')}",
                "url": data.get("url"),
                "html_title": None,
            }
        except Exception as e:
            print(f"[ERROR] Failed to parse post: {e}")
            return None


if __name__ == "__main__":
    scraper = RedditScraper()
    df = scraper.collect_targeted_posts(limit_per_combo=5, enrich_html=False)
    print(f"Collected {len(df)} posts")
    if not df.empty:
        print(df[["subreddit", "title", "author"]].head())
