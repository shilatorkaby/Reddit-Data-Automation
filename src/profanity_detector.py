"""
Simple dictionary-based profanity / abusive language detector.
Uses a dynamic bad-word list loaded from a text file.
"""

import string
from typing import List, Set, Tuple, Optional


def load_bad_words(path: str) -> Set[str]:
    """
    Load bad words from a text file (one term per line).

    Lines starting with '#' are treated as comments and ignored.
    """
    bad_words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            bad_words.add(line.lower())
    return bad_words


def detect_bad_words(text: str, bad_words: Set[str]) -> List[str]:
    """
    Return a list of distinct bad words found in the given text.
    """
    if not text:
        return []

    text = text.lower()
    # sting.punctuation = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    for p in string.punctuation:
        text = text.replace(p, " ")

    tokens = text.split()
    matched = {t for t in tokens if t in bad_words}
    return sorted(matched)


def analyze_post(
        title: Optional[str],
        body: Optional[str],
        bad_words: Set[str],
) -> Tuple[bool, List[str]]:
    """
    Analyze a post (title + body) for bad words.

    Returns:
        has_profanity (bool), matched_words (List[str])
    """
    text = (title or "") + " " + (body or "")
    matched = detect_bad_words(text, bad_words)
    return bool(matched), matched  # return false if matched list is empty
