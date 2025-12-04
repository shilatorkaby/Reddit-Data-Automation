"""
Central configuration for the Reddit harmful-content collection pipeline.
Uses Reddit's public JSON endpoints (no API key required).
"""
import os
from typing import List, Dict


# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()  # This loads .env file into os.environ
except ImportError:
    print("[WARN] python-dotenv not installed. Using default config or environment variables.")
    # pip install python-dotenv

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

# Flattened list of search terms for simple iteration
ALL_SEARCH_TERMS: List[str] = [
    term
    for terms in SEARCH_TERMS_BY_CATEGORY.values()
    for term in terms
]

# ---------------------------------------------------------------------------
# News-like content detection helpers
# ---------------------------------------------------------------------------

NEWS_SUBREDDITS = {"news", "worldnews"}
NEWS_KEYWORDS = {
    "arrest", "arrested", "charged", "indicted",
    "investigation", "probe", "police", "authorities",
    "suspect", "suspected", "raid",
    "attack", "shooting", "explosion", "blast", "killing",
}

ENRICH_TOP_N_USERS = 100

