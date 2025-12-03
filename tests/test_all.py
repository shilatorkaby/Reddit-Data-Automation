"""
test_all.py

Comprehensive test suite for the Reddit harmful content detection system.

Tests cover:
- Profanity detection
- Risk scoring
- Post labeling
- User aggregation
- Edge cases (deleted users, empty posts, etc.)

Run with: python -m pytest test_all.py -v
or: python test_all.py
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from profanity_detector import load_bad_words, detect_bad_words, analyze_post
from risk_scorer import RiskScorer
from post_labeler import label_posts, is_news_like_post
from user_aggregator import build_user_feed_from_posts


# ============================================================================== 
# FIXTURES
# ==============================================================================

@pytest.fixture
def bad_words_file():
    """Create a temporary bad words file for testing."""
    content = """# Test bad words
shit
fuck
kill
murder
hate
bastard
"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(content)
        path = f.name
    yield path
    Path(path).unlink()


@pytest.fixture
def bad_words():
    """Return a basic set of bad words for testing."""
    return {"shit", "fuck", "kill", "murder", "hate", "bastard"}


@pytest.fixture
def scorer():
    """Return a RiskScorer instance."""
    return RiskScorer()


@pytest.fixture
def sample_posts():
    """Return sample post data for testing."""
    return pd.DataFrame([
        {
            "post_id": "1",
            "author": "user1",
            "title": "I hate this policy",
            "selftext": "This is terrible",
            "subreddit": "politics",
            "created_utc": 1609459200,
            "score": 10,
            "num_comments": 5,
            "permalink": "/r/politics/test1",
            "url": "",
            "query": "hate",
        },
        {
            "post_id": "2",
            "author": "user2",
            "title": "We should kill all mosquitoes",
            "selftext": "They are parasites",
            "subreddit": "science",
            "created_utc": 1609459300,
            "score": 50,
            "num_comments": 20,
            "permalink": "/r/science/test2",
            "url": "",
            "query": "kill",
        },
        {
            "post_id": "3",
            "author": "user3",
            "title": "Breaking: Shooting in downtown",
            "selftext": "",
            "subreddit": "news",
            "created_utc": 1609459400,
            "score": 100,
            "num_comments": 50,
            "permalink": "/r/news/test3",
            "url": "https://example.com/news",
            "query": "shooting",
        },
    ])


# ==============================================================================
# PROFANITY DETECTOR TESTS
# ==============================================================================

class TestProfanityDetector:
    """Tests for profanity detection module."""

    def test_load_bad_words(self, bad_words_file):
        """Test loading bad words from file."""
        words = load_bad_words(bad_words_file)
        assert "shit" in words
        assert "fuck" in words
        assert "kill" in words
        assert len(words) >= 5

    def test_load_bad_words_missing_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_bad_words("/nonexistent/file.txt")

    def test_detect_bad_words_basic(self, bad_words):
        """Test basic bad word detection."""
        text = "This is shit and I hate it"
        matches = detect_bad_words(text, bad_words)
        assert "shit" in matches
        assert "hate" in matches
        assert len(matches) == 2

    def test_detect_bad_words_case_insensitive(self, bad_words):
        """Test case-insensitive matching."""
        text = "FUCK this SHIT"
        matches = detect_bad_words(text, bad_words)
        assert "fuck" in matches
        assert "shit" in matches

    def test_detect_bad_words_with_punctuation(self, bad_words):
        """Test detection with punctuation."""
        text = "What the fuck! This is shit."
        matches = detect_bad_words(text, bad_words)
        assert "fuck" in matches
        assert "shit" in matches

    def test_detect_bad_words_empty_text(self, bad_words):
        """Test with empty text."""
        matches = detect_bad_words("", bad_words)
        assert matches == []

    def test_analyze_post_with_profanity(self, bad_words):
        """Test analyzing a post with profanity."""
        has_prof, matches = analyze_post(
            "This is a fuck test",
            "I hate this shit bastard",
            bad_words
        )
        assert has_prof is True
        assert "fuck" in matches
        assert "hate" in matches
        assert "shit" in matches

    def test_analyze_post_without_profanity(self, bad_words):
        """Test analyzing a clean post."""
        has_prof, matches = analyze_post(
            "This is a nice day",
            "I love sunshine",
            bad_words
        )
        assert has_prof is False
        assert matches == []

    def test_analyze_post_none_values(self, bad_words):
        """Test with None values."""
        has_prof, matches = analyze_post(None, None, bad_words)
        assert has_prof is False
        assert matches == []


# ==============================================================================
# RISK SCORER TESTS
# ==============================================================================

class TestRiskScorer:
    """Tests for risk scoring module."""

    def test_score_text_no_violence(self, scorer):
        """Test scoring text with no violence."""
        result = scorer.score_text("This is a normal post about cats")
        assert result["risk_score"] == 0.0
        assert result["violence_type"] == "none"

    def test_score_text_descriptive_violence(self, scorer):
        """Test scoring descriptive violence."""
        result = scorer.score_text("The movie had a killing scene")
        assert result["violence_type"] in ["none", "descriptive"]
        assert result["risk_score"] < 0.3

    def test_score_text_self_directed(self, scorer):
        """Test scoring self-directed violence."""
        result = scorer.score_text("I want to kill myself")
        assert result["violence_type"] == "self_directed"
        assert 0.1 < result["risk_score"] < 0.5

    def test_score_text_call_to_violence(self, scorer):
        """Test scoring call to violence."""
        result = scorer.score_text("We should kill them all")
        assert result["violence_type"] == "call_to_violence"
        assert result["risk_score"] >= 0.7

    def test_score_text_hate_speech(self, scorer):
        """Test scoring hate speech."""
        result = scorer.score_text("They are subhuman parasites")
        assert result["violence_type"] == "hate_speech"
        assert result["risk_score"] >= 0.5

    def test_score_text_generic_hate(self, scorer):
        """Test scoring generic hate (weak signal)."""
        result = scorer.score_text("I hate Mondays")
        assert result["violence_type"] == "none"
        assert result["risk_score"] < 0.2

    def test_score_text_intensity_indicators(self, scorer):
        """Test that intensity indicators increase score."""
        normal = scorer.score_text("kill them")
        intense = scorer.score_text("KILL THEM ALL!!!")
        assert intense["risk_score"] > normal["risk_score"]

    def test_score_text_empty(self, scorer):
        """Test scoring empty text."""
        result = scorer.score_text("")
        assert result["risk_score"] == 0.0
        assert result["violence_type"] == "none"

    def test_aggregate_user_score(self, scorer):
        """Test user score aggregation."""
        scores = [0.2, 0.5, 0.9, 0.3]
        user_score = scorer.aggregate_user_score(scores)
        # Should be 0.7 * max + 0.3 * mean = 0.7 * 0.9 + 0.3 * 0.475
        expected = 0.7 * 0.9 + 0.3 * 0.475
        assert abs(user_score - expected) < 0.01

    def test_aggregate_user_score_empty(self, scorer):
        """Test aggregation with empty list."""
        assert scorer.aggregate_user_score([]) == 0.0


# ==============================================================================
# POST LABELER TESTS
# ==============================================================================

class TestPostLabeler:
    """Tests for post labeling module."""

    def test_label_posts_basic(self, sample_posts, bad_words, scorer):
        """Test basic post labeling."""
        df = label_posts(sample_posts, bad_words, use_moderation=False, scorer=scorer)

        assert "has_profanity" in df.columns
        assert "violence_risk_score" in df.columns
        assert len(df) == len(sample_posts)

    def test_label_posts_empty(self, bad_words, scorer):
        """Test labeling empty DataFrame."""
        df = pd.DataFrame()
        result = label_posts(df, bad_words, scorer=scorer)
        assert result.empty

    def test_label_posts_detects_profanity(self, sample_posts, bad_words, scorer):
        """Test that profanity is detected."""
        df = label_posts(sample_posts, bad_words, scorer=scorer)
        # Post with "hate" should be flagged
        assert df[df["title"].str.contains("hate", case=False, na=False)]["has_profanity"].any()

    def test_label_posts_risk_scores(self, sample_posts, bad_words, scorer):
        """Test that risk scores are assigned."""
        df = label_posts(sample_posts, bad_words, scorer=scorer)
        assert (df["violence_risk_score"] >= 0).all()
        assert (df["violence_risk_score"] <= 1).all()

    def test_is_news_like_post(self):
        """Test news post detection."""
        # News subreddit with news keywords
        row = {
            "subreddit": "news",
            "title": "Police arrest suspect in shooting",
            "selftext": "",
            "url": "https://example.com/news",
        }
        assert is_news_like_post(row) is True

        # Non-news subreddit
        row["subreddit"] = "politics"
        assert is_news_like_post(row) is False


# ==============================================================================
# USER AGGREGATOR TESTS
# ==============================================================================

class TestUserAggregator:
    """Tests for user aggregation module."""

    def test_build_user_feed_basic(self, scorer):
        """Test basic user feed building."""
        posts = pd.DataFrame([
            {"author": "user1", "violence_risk_score": 0.5, "title": "", "selftext": ""},
            {"author": "user1", "violence_risk_score": 0.8, "title": "", "selftext": ""},
            {"author": "user2", "violence_risk_score": 0.3, "title": "", "selftext": ""},
        ])

        users = build_user_feed_from_posts(posts, scorer=scorer)

        assert len(users) == 2
        assert "username" in users.columns
        assert "user_risk_score" in users.columns
        assert "high_risk_posts" in users.columns

    def test_build_user_feed_empty(self, scorer):
        """Test with empty DataFrame."""
        users = build_user_feed_from_posts(pd.DataFrame(), scorer=scorer)
        assert users.empty

    def test_build_user_feed_filters_deleted(self, scorer):
        """Test that deleted users are filtered out."""
        posts = pd.DataFrame([
            {"author": "user1", "violence_risk_score": 0.5, "title": "", "selftext": ""},
            {"author": "[deleted]", "violence_risk_score": 0.8, "title": "", "selftext": ""},
            {"author": "AutoModerator", "violence_risk_score": 0.9, "title": "", "selftext": ""},
        ])

        users = build_user_feed_from_posts(posts, scorer=scorer)

        assert len(users) == 1
        assert users.iloc[0]["username"] == "user1"

    def test_build_user_feed_aggregation(self, scorer):
        """Test score aggregation logic."""
        posts = pd.DataFrame([
            {"author": "user1", "violence_risk_score": 0.2, "title": "post1", "selftext": ""},
            {"author": "user1", "violence_risk_score": 0.5, "title": "post2", "selftext": ""},
            {"author": "user1", "violence_risk_score": 0.9, "title": "post3", "selftext": ""},
        ])

        users = build_user_feed_from_posts(posts, post_risk_threshold=0.6, scorer=scorer)

        user = users.iloc[0]
        assert user["total_posts"] == 3
        assert user["high_risk_posts"] == 1  # Only 0.9 is above 0.6


# ==============================================================================
# EDGE CASES TESTS
# ==============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_title_and_body(self, bad_words):
        """Test handling of empty title and body."""
        has_prof, matches = analyze_post("", "", bad_words)
        assert has_prof is False
        assert matches == []

    def test_none_title_and_body(self, bad_words):
        """Test handling of None title and body."""
        has_prof, matches = analyze_post(None, None, bad_words)
        assert has_prof is False
        assert matches == []

    def test_very_long_text(self, scorer):
        """Test handling of very long text."""
        long_text = "word " * 10000
        result = scorer.score_text(long_text)
        assert isinstance(result["risk_score"], float)
        assert 0 <= result["risk_score"] <= 1

    def test_special_characters(self, bad_words):
        """Test handling of special characters."""
        text = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
        matches = detect_bad_words(text, bad_words)
        assert matches == []

    def test_unicode_text(self, scorer):
        """Test handling of unicode text."""
        text = "こんにちは世界"  # Japanese
        result = scorer.score_text(text)
        assert result["risk_score"] == 0.0


# ==============================================================================
# RUN TESTS
# ==============================================================================

if __name__ == "__main__":
    # Run with pytest if available, otherwise just run basic tests
    try:
        pytest.main([__file__, "-v", "--tb=short"])
    except:
        print("Running basic tests (pytest not available)...")
        print("\n" + "="*70)
        print("RUNNING TESTS")
        print("="*70 + "\n")
        
        # Run a few basic tests
        scorer = RiskScorer()
        bad_words = {"kill", "hate", "shit"}
        
        # Test 1: Risk scoring
        result = scorer.score_text("I hate this")
        print(f"✓ Test 1 passed: Basic risk scoring works (score={result['risk_score']})")
        
        # Test 2: Profanity detection
        has_prof, matches = analyze_post("This is shit", "", bad_words)
        assert has_prof == True
        print(f"✓ Test 2 passed: Profanity detection works (matched: {matches})")
        
        # Test 3: Edge case
        result = scorer.score_text("")
        assert result["risk_score"] == 0.0
        print(f"✓ Test 3 passed: Empty text handled correctly")
        
        print("\n" + "="*70)
        print("ALL BASIC TESTS PASSED!")
        print("="*70)
        print("\nFor comprehensive testing, install pytest: pip install pytest")
        print("Then run: python -m pytest test_all.py -v")
