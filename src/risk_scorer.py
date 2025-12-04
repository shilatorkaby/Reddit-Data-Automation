"""
Context-aware violence / hate content classifier.

This module provides:
  - RiskScorer.score_text(text): returns a violence risk score (0–1),
    violence type, and a detailed explanation.
  - RiskScorer.detect_language(text): simple language detection.
  - RiskScorer.aggregate_user_score(scores): aggregate scores per user.

Violence types:
  - none
  - self_directed       (e.g. "kill me")
  - descriptive         (e.g. "they killed him in the movie")
  - hate_speech         (dehumanizing language towards people/groups)
  - call_to_violence    (e.g. "we should kill them")
"""

from typing import List, Dict, Any, Optional
import re

from langdetect import detect, LangDetectException


class RiskScorer:
    def __init__(self) -> None:
        # Violent verbs (non-exhaustive, but good enough for the assignment)
        self.violent_verbs = {
            "kill", "shoot", "attack", "bomb", "murder",
            "stab", "lynch", "slaughter", "execute",
        }

        # Strong hate / dehumanization terms
        self.hate_terms = {
            "exterminate", "genocide",
            "subhuman", "vermin", "cockroach", "cockroaches",
            "parasite", "parasites",
        }

        # Generic hate words (we treat these as much weaker signals)
        self.generic_hate = {"hate", "destroy"}

        # Pronouns for context
        self.self_pronouns = {"i", "me", "myself"}
        self.other_pronouns = {
            "you", "him", "her", "them", "they",
            "those", "these", "people",
        }

    # ---------------- Language detection ----------------

    def detect_language(self, text: str) -> Optional[str]:
        try:
            return detect(text) if text.strip() else None
        except LangDetectException:
            return None

    # ---------------- Tokenization helper ----------------

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z']+", text.lower())

    # ---------------- Violence type classification ----------------

    def classify_violence_type(self, text: str) -> Dict[str, Any]:
        tokens = self._tokenize(text)
        if not tokens:
            return {"violence_type": "none", "details": "no text", "has_violence": False}

        has_violent_verb = any(t in self.violent_verbs for t in tokens)
        has_strong_hate = any(t in self.hate_terms for t in tokens)
        has_generic_hate = any(t in self.generic_hate for t in tokens)

        # No violent or hate-related terms at all
        if not has_violent_verb and not has_strong_hate and not has_generic_hate:
            return {
                "violence_type": "none",
                "details": "no violent or hate-related terms detected",
                "has_violence": False,
            }

        violence_mentions: List[tuple[str, str]] = []

        for i, tok in enumerate(tokens):
            if tok not in self.violent_verbs:
                continue

            window = tokens[max(0, i - 3): i + 4]
            window_str = " ".join(window)

            # Self-directed: "kill me", "shoot myself"
            if any(p in window for p in self.self_pronouns):
                violence_mentions.append(("self_directed", window_str))

            # Other-directed / call to violence: "kill them", "we should kill", "go kill"
            elif any(p in window for p in self.other_pronouns):
                violence_mentions.append(("call_to_violence", window_str))
            elif re.search(r"\bkill all\b|\bshoot all\b|\bwe should kill\b|\bgo kill\b", window_str):
                violence_mentions.append(("call_to_violence", window_str))
            else:
                # Not clearly self- or other-directed: treat as descriptive
                violence_mentions.append(("descriptive", window_str))

        # Strong hate / dehumanization without explicit violent verb
        if not violence_mentions and has_strong_hate:
            return {
                "violence_type": "hate_speech",
                "details": "dehumanizing expressions without explicit violent verb",
                "has_violence": True,
            }

        # Only generic hate (e.g. "I hate this") – weak, not hate_speech
        if not violence_mentions and has_generic_hate:
            return {
                "violence_type": "none",
                "details": "only generic hate words in non-violent context",
                "has_violence": False,
            }

        if not violence_mentions:
            return {
                "violence_type": "none",
                "details": "violent context unclear",
                "has_violence": False,
            }

        severity_order = ["none", "descriptive", "self_directed", "hate_speech", "call_to_violence"]
        chosen_type = "none"
        chosen_windows: List[str] = []

        for v_type, w in violence_mentions:
            if severity_order.index(v_type) > severity_order.index(chosen_type):
                chosen_type = v_type
                chosen_windows = [w]
            elif v_type == chosen_type:
                chosen_windows.append(w)

        details = f"violence type: {chosen_type}; example windows: " + " | ".join(chosen_windows)
        return {
            "violence_type": chosen_type,
            "details": details,
            "has_violence": chosen_type != "none",
        }

    # ---------------- Risk score computation ----------------

    def score_text(self, text: str) -> Dict[str, Any]:
        """
        Compute a risk score and explanation for a text.
        """
        tokens = self._tokenize(text)
        lower = text.lower()

        classification = self.classify_violence_type(text)
        v_type = classification["violence_type"]

        # Token-based counting (no substring tricks)
        violent_hits = sum(1 for t in tokens if t in self.violent_verbs)
        strong_hate_hits = sum(1 for t in tokens if t in self.hate_terms)
        generic_hate_hits = sum(1 for t in tokens if t in self.generic_hate)

        words = text.split()
        all_caps_words = sum(
            1 for w in words if w.isalpha() and w.isupper() and len(w) > 3
        )
        exclamations = lower.count("!")

        base_by_type = {
            "none": 0.0,
            "descriptive": 0.05,  # very mild
            "self_directed": 0.15,  # concerning but not towards others
            "hate_speech": 0.5,  # serious, but we avoid starting at 0.6
            "call_to_violence": 0.75,  # strong, but leaves room to go up with extra signals
        }
        base = base_by_type.get(v_type, 0.0)

        bonus = 0.0

        if tokens:
            # Violent words still matter, but less
            bonus += min(violent_hits * 0.03, 0.15)
            # Strong dehumanizing hate terms
            bonus += min(strong_hate_hits * 0.07, 0.25)
            # Generic "hate/destroy" -> very small
            bonus += min(generic_hate_hits * 0.02, 0.10)

            if all_caps_words >= 2:
                bonus += 0.05
            if exclamations >= 3:
                bonus += 0.05

        raw = base + bonus
        risk_score = max(0.0, min(1.0, raw))

        # After computing risk_score
        if risk_score >= 0.9:
            # Only allow 0.9–1.0 if it's either call_to_violence or strong hate_speech
            if not (v_type in {"call_to_violence", "hate_speech"} and strong_hate_hits + violent_hits >= 2):
                risk_score = 0.8

        explanation_parts = [
            f"violence_type={v_type}",
            classification["details"],
            f"violent_hits={violent_hits}",
            f"hate_hits_strong={strong_hate_hits}",
            f"hate_hits_generic={generic_hate_hits}",
        ]
        if all_caps_words >= 2:
            explanation_parts.append("intense tone (many ALL CAPS words)")
        if exclamations >= 3:
            explanation_parts.append("emotional tone (many exclamation marks)")

        if v_type == "self_directed":
            explanation_parts.append(
                "self-directed wording (e.g. 'kill me') – not a call to harm others, "
                "but still potentially concerning."
            )
        elif v_type == "descriptive":
            explanation_parts.append(
                "violent words appear in descriptive context, not as a call to action."
            )

        return {
            "risk_score": risk_score,
            "violence_type": v_type,
            "violent_hits": violent_hits,
            "hate_hits_strong": strong_hate_hits,
            "hate_hits_generic": generic_hate_hits,
            "all_caps_words": all_caps_words,
            "exclamations": exclamations,
            "explanation": "; ".join(explanation_parts),
        }

    # ---------------- User-level aggregation ----------------

    def aggregate_user_score(self, post_scores: List[float]) -> float:
        """
        Aggregate a list of post-level scores into a single user-level score.
        Weight is biased towards the user's worst post.

        Score = 0.7 * max_score + 0.3 * mean_score
        """
        if not post_scores:
            return 0.0
        max_s = max(post_scores)
        mean_s = sum(post_scores) / len(post_scores)
        return 0.7 * max_s + 0.3 * mean_s
