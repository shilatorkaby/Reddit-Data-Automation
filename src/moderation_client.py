"""
moderation_client.py

LLM Moderation Classifier using the OpenAI Moderation API.

This module wraps the moderation API in a small helper function
Features:
`check_moderation_flag`, which can be used to classify text for
hate, harassment, violence, self-harm, explicit content, etc.
"""

import time
from typing import Tuple, Dict, Optional
from openai import OpenAI, RateLimitError

client = OpenAI()

# Rate limiting settings
_last_request_time = 0.0
_MIN_INTERVAL = 0.5  # seconds between requests
_cache: Dict[str, Tuple[bool, Optional[Dict[str, float]]]] = {}


def check_moderation_flag(expression: str) -> Tuple[bool, Optional[Dict[str, float]]]:
    """
    Checks if the given expression is flagged for moderation.

    Args:
        expression (str): The text expression to check.

    Returns:
        tuple:
            - flagged (bool): True if the text is flagged,
            - flagged_categories (dict or None): flagged categories -> score,
              or None if not flagged.
    """
    global _last_request_time

    if not expression:
        return False, None

    # Check cache first
    cache_key = expression[:500]  # Use first 500 chars as key
    if cache_key in _cache:
        return _cache[cache_key]

    # Rate limiting - wait if needed
    elapsed = time.time() - _last_request_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)

    # Retry logic for rate limits
    for attempt in range(3):
        try:
            _last_request_time = time.time()

            moderation_response = client.moderations.create(
                model="omni-moderation-latest",
                input=expression[:10000],  # API limit
            )

            result = moderation_response.results[0]
            categories = result.categories
            category_scores = result.category_scores
            flagged = result.flagged

            if flagged:
                flagged_categories: Dict[str, float] = {}
                for category, is_flagged in categories.model_dump().items():
                    if is_flagged:
                        flagged_categories[category] = category_scores.model_dump()[category]
                response = (True, flagged_categories)
            else:
                response = (False, None)

            # Cache result
            _cache[cache_key] = response
            return response

        except RateLimitError:
            wait_time = 2 ** attempt  # 1s, 2s, 4s
            print(f"[WARN] Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            print(f"[ERROR] Moderation API error: {e}")
            return False, None

    return False, None
