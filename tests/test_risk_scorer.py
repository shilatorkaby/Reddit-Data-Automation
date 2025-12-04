"""
Test suite for the risk scoring and classification system.
Validates accuracy, edge cases, and reliability of the pipeline.

Run with: python -m pytest test_risk_scorer.py -v
or: python test_risk_scorer.py
"""

import unittest

from reddit_scraper import RedditScraper
from risk_scorer import RiskScorer
from profanity_detector import detect_bad_words, load_bad_words
from user_aggregator import build_user_feed_from_posts
import pandas as pd
import pytest

@pytest.mark.parametrize("text,expected_type", [
    ("I want to kill myself", "self_directed"),
    ("We should kill them all", "call_to_violence"),
    ("They are subhuman vermin", "hate_speech"),
    ("He will kill the character in the movie", "descriptive"),  # Changed - uses base form "kill"
    ("Nice weather today", "none"),
    ("", "none"),
])
def test_classify_violence_type_parametrized(text, expected_type):
    """Parameterized test for violence classification"""
    scorer = RiskScorer()
    result = scorer.classify_violence_type(text)
    assert result["violence_type"] == expected_type

@pytest.mark.parametrize("score_list,expected_min,expected_max", [
    ([0.1, 0.2, 0.9], 0.7, 0.8),  # High max should dominate
    ([0.5, 0.5, 0.5], 0.5, 0.5),  # Uniform scores
    ([0.0], 0.0, 0.0),  # Single zero
    ([1.0], 1.0, 1.0),  # Single max
    ([], 0.0, 0.0),  # Empty list
])
def test_aggregate_user_score_parametrized(score_list, expected_min, expected_max):
    """Parameterized test for user score aggregation"""
    scorer = RiskScorer()
    result = scorer.aggregate_user_score(score_list)
    assert expected_min <= result <= expected_max

class TestRiskScorer(unittest.TestCase):
    """Test suite for RiskScorer class."""

    def setUp(self):
        """Initialize scorer before each test."""
        self.scorer = RiskScorer()

    def test_classify_none_when_no_violence(self):
        result = self.scorer.classify_violence_type("I like ice cream")
        assert result["violence_type"] == "none"
        assert result["has_violence"] is False

    def test_classify_self_directed(self):
        result = self.scorer.classify_violence_type("I want to kill myself")
        assert result["violence_type"] == "self_directed"
        assert result["has_violence"] is True

    def test_classify_call_to_violence(self):
        result = self.scorer.classify_violence_type("we should kill them all")
        assert result["violence_type"] == "call_to_violence"
        assert result["has_violence"] is True

    def test_classify_hate_speech_without_violent_verb(self):
        result = self.scorer.classify_violence_type("they are subhuman vermin")
        assert result["violence_type"] == "hate_speech"
        assert result["has_violence"] is True

    def test_score_text_more_risky_for_violent_text(self):
        neutral = self.scorer.score_text("nice weather today")
        violent = self.scorer.score_text("we should kill them all!!!")

        assert violent["risk_score"] > neutral["risk_score"]
        assert violent["risk_score"] > 0.5
        assert violent["violence_type"] in {"call_to_violence", "hate_speech"}

    def test_aggregate_user_score_biases_max(self):
        score = self.scorer.aggregate_user_score([0.1, 0.2, 0.9])
        # 0.7*0.9 + 0.3*mean(0.1,0.2,0.9)=0.63 + 0.3*0.4=0.75
        assert 0.74 < score < 0.76

    def test_no_violence(self):
        """Test posts with no violent content."""
        text = "I really enjoy programming and building cool projects."
        result = self.scorer.score_text(text)
        
        self.assertEqual(result["violence_type"], "none")
        self.assertLess(result["risk_score"], 0.2)
        self.assertEqual(result["violent_hits"], 0)

    def test_call_to_violence(self):
        """Test clear calls to violence."""
        text = "We should kill them all! They deserve to die!"
        result = self.scorer.score_text(text)
        
        self.assertEqual(result["violence_type"], "call_to_violence")
        self.assertGreaterEqual(result["risk_score"], 0.7)
        self.assertGreater(result["violent_hits"], 0)

    def test_self_directed_violence(self):
        """Test self-directed violent language."""
        text = "I just want to kill myself, I can't take it anymore."
        result = self.scorer.score_text(text)
        
        self.assertEqual(result["violence_type"], "self_directed")
        self.assertGreaterEqual(result["risk_score"], 0.15)
        self.assertLess(result["risk_score"], 0.5)

    def test_descriptive_violence(self):
        """Test descriptive/narrative violence (e.g., movie descriptions)."""
        text = "The character was violently killed during the attack scene in the movie."
        result = self.scorer.score_text(text)

        # Should be descriptive (not clearly directed at someone) or have low score
        self.assertIn(result["violence_type"], ["descriptive", "none"])
        self.assertLess(result["risk_score"], 0.4)

    def test_hate_speech(self):
        """Test dehumanizing hate speech."""
        text = "They are all vermin and cockroaches, subhuman parasites."
        result = self.scorer.score_text(text)

        self.assertEqual(result["violence_type"], "hate_speech")
        self.assertGreaterEqual(result["risk_score"], 0.5)
        self.assertGreater(result["hate_hits_strong"], 0)

    def test_news_report(self):
        """Test that news-like content is not over-scored."""
        text = "Police arrested suspect in shooting that killed 3 people."
        result = self.scorer.score_text(text)

        # Should be descriptive or low-risk
        self.assertIn(result["violence_type"], ["descriptive", "none"])
        self.assertLess(result["risk_score"], 0.4)

    def test_empty_text(self):
        """Test handling of empty or whitespace-only text."""
        result = self.scorer.score_text("")
        self.assertEqual(result["violence_type"], "none")
        self.assertEqual(result["risk_score"], 0.0)

    def test_special_characters(self):
        """Test text with special characters and emojis."""
        text = "💀💀💀 This is crazy!!! 😱😱"
        result = self.scorer.score_text(text)

        # Should handle gracefully without errors
        self.assertIsInstance(result["risk_score"], float)
        self.assertGreaterEqual(result["risk_score"], 0.0)
        self.assertLessEqual(result["risk_score"], 1.0)

    def test_intensity_signals(self):
        """Test that intensity signals (CAPS, exclamations) affect score."""
        text_normal = "i hate this situation"
        text_intense = "I HATE THIS SITUATION!!!"

        result_normal = self.scorer.score_text(text_normal)
        result_intense = self.scorer.score_text(text_intense)

        # Intense version should score higher
        self.assertGreaterEqual(result_intense["risk_score"], result_normal["risk_score"])

    def test_user_aggregation(self):
        """Test user-level score aggregation."""
        # User with mixed post scores
        post_scores = [0.1, 0.3, 0.8, 0.2]
        user_score = self.scorer.aggregate_user_score(post_scores)

        # Should be weighted toward max
        max_score = max(post_scores)
        mean_score = sum(post_scores) / len(post_scores)
        expected = 0.7 * max_score + 0.3 * mean_score

        self.assertAlmostEqual(user_score, expected, places=5)

    def test_user_aggregation_empty(self):
        """Test user aggregation with no posts."""
        user_score = self.scorer.aggregate_user_score([])
        self.assertEqual(user_score, 0.0)


class TestProfanityDetector(unittest.TestCase):
    """Test suite for profanity detection."""

    def setUp(self):
        """Load bad words before each test."""
        try:
            self.bad_words = load_bad_words("../data/profanity/bad_words_en.txt")
        except FileNotFoundError:
            self.bad_words = load_bad_words("bad_words_en.txt")

    def test_detect_profanity(self):
        """Test detection of profane words."""
        text = "This is some shit and fuck content."
        detected = detect_bad_words(text, self.bad_words)

        self.assertGreater(len(detected), 0)
        self.assertIn("shit", detected)
        self.assertIn("fuck", detected)

    def test_no_profanity(self):
        """Test clean text."""
        text = "This is perfectly clean content."
        detected = detect_bad_words(text, self.bad_words)

        self.assertEqual(len(detected), 0)

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        text = "SHIT Shit ShIt"
        detected = detect_bad_words(text, self.bad_words)

        # Should detect as single word (deduplicated)
        self.assertIn("shit", detected)

    def test_punctuation_handling(self):
        """Test that punctuation doesn't break detection."""
        text = "This is shit! Really, it's shit."
        detected = detect_bad_words(text, self.bad_words)

        self.assertIn("shit", detected)


class TestUserAggregator(unittest.TestCase):
    """Test suite for user-level aggregation."""

    def setUp(self):
        """Create sample post data."""
        self.df_posts = pd.DataFrame([
            {"author": "user1", "violence_risk_score": 0.8},
            {"author": "user1", "violence_risk_score": 0.3},
            {"author": "user1", "violence_risk_score": 0.9},
            {"author": "user2", "violence_risk_score": 0.2},
            {"author": "user2", "violence_risk_score": 0.1},
            {"author": "[deleted]", "violence_risk_score": 0.9},
        ])

    def test_user_feed_generation(self):
        """Test generation of user feed."""
        users_df = build_user_feed_from_posts(self.df_posts, post_risk_threshold=0.6)

        # Should have 2 users (excluding [deleted])
        self.assertEqual(len(users_df), 2)

        # Check user1 has correct high-risk post count
        user1 = users_df[users_df["username"] == "user1"].iloc[0]
        self.assertEqual(user1["high_risk_posts"], 2)  # 0.8 and 0.9
        self.assertEqual(user1["total_posts"], 3)

    def test_deleted_users_filtered(self):
        """Test that deleted users are filtered out."""
        users_df = build_user_feed_from_posts(self.df_posts)

        self.assertNotIn("[deleted]", users_df["username"].values)

    def test_empty_posts(self):
        """Test handling of empty DataFrame."""
        df_empty = pd.DataFrame()
        users_df = build_user_feed_from_posts(df_empty)

        self.assertTrue(users_df.empty)


class TestEdgeCases(unittest.TestCase):
    """Test suite for edge cases and error handling."""

    def setUp(self):
        """Initialize scorer."""
        self.scorer = RiskScorer()

    def test_very_long_text(self):
        """Test handling of very long text."""
        text = "kill " * 1000  # 1000 repetitions
        result = self.scorer.score_text(text)

        # Should not crash and should cap at 1.0
        self.assertLessEqual(result["risk_score"], 1.0)
        self.assertIsInstance(result["risk_score"], float)

    def test_unicode_characters(self):
        """Test handling of Unicode/non-ASCII characters."""
        text = "这是中文 kill العربية murder עברית"
        result = self.scorer.score_text(text)

        # Should detect English violent words
        self.assertGreater(result["violent_hits"], 0)
        self.assertIsInstance(result["risk_score"], float)

    def test_mixed_case_violence(self):
        """Test case variations of violent terms."""
        texts = ["KILL THEM ALL", "Kill Them All", "kill them all"]

        for text in texts:
            result = self.scorer.score_text(text)
            self.assertEqual(result["violence_type"], "call_to_violence")
            self.assertGreater(result["risk_score"], 0.6)

    def test_language_detection(self):
        """Test language detection for non-English text."""
        text_en = "This is English text about killing"
        text_es = "Este es un texto en español"

        lang_en = self.scorer.detect_language(text_en)
        lang_es = self.scorer.detect_language(text_es)

        self.assertEqual(lang_en, "en")
        self.assertEqual(lang_es, "es")


class TestScoreConsistency(unittest.TestCase):
    """Test consistency and reliability of scoring."""

    def setUp(self):
        """Initialize scorer."""
        self.scorer = RiskScorer()

    def test_score_bounds(self):
        """Test that scores are always between 0 and 1."""
        test_texts = [
            "",
            "normal text",
            "kill kill kill murder murder attack",
            "KILL THEM ALL NOW!!!",
            "They are subhuman vermin parasites cockroaches",
        ]

        for text in test_texts:
            result = self.scorer.score_text(text)
            self.assertGreaterEqual(result["risk_score"], 0.0)
            self.assertLessEqual(result["risk_score"], 1.0)

    def test_deterministic_scoring(self):
        """Test that same text produces same score."""
        text = "We should kill them all, they are vermin!"

        result1 = self.scorer.score_text(text)
        result2 = self.scorer.score_text(text)

        self.assertEqual(result1["risk_score"], result2["risk_score"])
        self.assertEqual(result1["violence_type"], result2["violence_type"])

    def test_incremental_severity(self):
        """Test that more severe content scores higher."""
        texts_ordered = [
            "I disagree with this policy",
            "I hate this policy",
            "Someone should stop this policy",
            "We should attack those who support this",
            "We should kill everyone who supports this",
        ]

        scores = [self.scorer.score_text(t)["risk_score"] for t in texts_ordered]

        # Generally, each should be >= previous (with some tolerance)
        for i in range(1, len(scores)):
            # Allow some flexibility due to context analysis
            self.assertGreaterEqual(scores[i] + 0.15, scores[i-1])


def run_test_suite():
    """Run all tests and print results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRiskScorer))
    suite.addTests(loader.loadTestsFromTestCase(TestProfanityDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestUserAggregator))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestScoreConsistency))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*70)

    return result


def test_risk_scorer_handles_only_punctuation():
    """Test scorer with only punctuation"""
    scorer = RiskScorer()
    result = scorer.score_text("!!! ??? ...")
    assert result["risk_score"] == 0.0


def test_reddit_scraper_handles_malformed_json(monkeypatch):
    """Test scraper handles malformed JSON response"""
    scraper = RedditScraper()

    def fake_get(url, headers=None, params=None, timeout=None):
        class FakeResponse:
            status_code = 200

            def json(self):
                raise ValueError("Invalid JSON")

            def raise_for_status(self):
                pass

        return FakeResponse()

    monkeypatch.setattr("reddit_scraper.requests.get", fake_get)
    monkeypatch.setattr("reddit_scraper.time.sleep", lambda s: None)

    result = scraper._get_json("/test")
    assert result is None


def test_moderation_client_handles_network_timeout(monkeypatch):
    """Test moderation client handles network timeout"""
    import moderation_client

    class TimeoutError(Exception):
        pass

    class FailingModerations:
        def create(self, model, input):
            raise TimeoutError("Network timeout")

    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)
    fake_client = type('obj', (object,), {'moderations': FailingModerations()})()
    monkeypatch.setattr(moderation_client, "client", fake_client)
    monkeypatch.setattr(moderation_client.time, "sleep", lambda s: None)

    flagged, categories = moderation_client.check_moderation_flag("test")
    assert flagged is False
    assert categories is None

if __name__ == "__main__":
    run_test_suite()