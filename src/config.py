"""
config.py

Central configuration for the Reddit harmful-content collection pipeline.
Uses Reddit's public JSON endpoints (no API key required).
"""

from typing import List, Dict

# ---------------------------------------------------------------------------
# HTTP / Reddit settings
# ---------------------------------------------------------------------------

BASE_URL: str = "https://www.reddit.com"

# A real-looking User-Agent is important to avoid being blocked
HEADERS: Dict[str, str] = {
    "User-Agent": "harmful-content-detector/1.0 (research; contact: example@example.com)"
}

REQUEST_TIMEOUT: int = 10  # seconds
MAX_RETRIES: int = 3  # retry on transient errors like 429, 5xx

# ---------------------------------------------------------------------------
# Target subreddits
# ---------------------------------------------------------------------------

TARGET_SUBREDDITS: List[str] = [
    # Politics / news – often heated, controversial discussions
    "politics",
    "worldnews",
    "news",

    # Opinion / venting – personal stories, potential hate/harassment
    "unpopularopinion",
    "TrueOffMyChest",
    "AmItheAsshole",

    # Conflict / drama / controversial topics
    "PublicFreakout",
    "justice",
    "changemyview",
]

# ---------------------------------------------------------------------------
# Search terms for harmful / violent / hateful content
#
# These are *search queries*, not the internal bad-word lexicon.
# They are used to *find* relevant posts in the first place.
# ---------------------------------------------------------------------------

SEARCH_TERMS_BY_CATEGORY: Dict[str, List[str]] = {
    "violence": [
        "kill",
        "murder",
        "shooting",
        "stabbed",
        "bomb attack",
        "violent attack",
    ],
    "threats": [
        "deserve to die",
        "should be killed",
        "i will kill you",
        "we will attack",
        "you will pay for this",
        "death threat",
        "i will hurt",

    ],
    "hate_speech": [
        "hate speech",
        "racist slur",
        "racial hatred",
        "go back to your country",
        "they don't belong here",
    ],
    "dehumanization": [
        "they are animals",
        "vermin",
        "subhuman",
        "cockroaches",
        "parasites",
    ],
}

# ---------------------------------------------------------------------------
# Flattened list of search terms for simple iteration
# ---------------------------------------------------------------------------
ALL_SEARCH_TERMS: List[str] = []
for _category, _terms in SEARCH_TERMS_BY_CATEGORY.items():
    ALL_SEARCH_TERMS.extend(_terms)


# ---------------------------------------------------------------------------
# News subreddits- search queries and keywords for news posts.
# News report about violence - that should be ignored (or scored as 0).
# ---------------------------------------------------------------------------

NEWS_SUBREDDITS = {"news", "worldnews"}
NEWS_KEYWORDS = {
    "arrest", "arrested", "charged", "indicted",
    "investigation", "probe", "police", "authorities",
    "suspect", "suspected", "raid",
    "attack", "shooting", "explosion", "blast", "killing",
}

