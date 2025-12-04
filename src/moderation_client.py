"""
LLM Moderation Classifier using the OpenAI Moderation API.

Provides:
    - check_moderation_flag: classify text for hate, harassment, violence,
      self-harm, explicit content, etc., using OpenAI's moderation endpoint.
"""
import os
import time
from typing import Tuple, Dict, Optional

# Module level - this runs when file is imported
try:
    from openai import OpenAI, RateLimitError, APIError

    OPENAI_AVAILABLE = True
    client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    OPENAI_AVAILABLE = False
    client = None

_last_request_time = 0.0
_MIN_INTERVAL = 0.5
_cache: Dict[str, Tuple[bool, Optional[Dict[str, float]]]] = {}

# Cache size limit to prevent memory issues
MAX_CACHE_SIZE = 1000


def check_moderation_flag(expression: str) -> Tuple[bool, Optional[Dict[str, float]]]:
    """
Check if text is flagged.
    Args:
        expression (str): The text expression to check.

    Returns:
        (flagged, flagged_categories) where:
          - flagged: True if the text is flagged by the model.
          - flagged_categories: mapping {category: score}, or None if not flagged.
    """
    global _last_request_time

    # Limit cache size
    if len(_cache) > MAX_CACHE_SIZE:
        # Remove oldest 20% of entries
        remove_count = MAX_CACHE_SIZE // 5
        keys_to_remove = list(_cache.keys())[:remove_count]
        for key in keys_to_remove:
            del _cache[key]

    if not OPENAI_AVAILABLE:
        print("[WARN] OpenAI not available. Install with: pip install openai")
        return False, None

    if not expression:
        return False, None

    # Check cache first (use first 500 chars as key)
    cache_key = expression[:500]
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
                categories_dict = categories.model_dump()
                scores_dict = category_scores.model_dump()
                for category, is_flagged in categories_dict.items():
                    if is_flagged:
                        flagged_categories[category] = scores_dict[category]
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
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] Moderation API error: {e}")
            return False, None

    return False, None
