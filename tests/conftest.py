# config test
import pytest
import pandas as pd

from risk_scorer import RiskScorer


@pytest.fixture
def bad_words():
    # simple lexicon for tests
    return {"badword", "nasty", "ugly"}


@pytest.fixture
def risk_scorer():
    return RiskScorer()


class FakeRiskScorer:
    """Deterministic scorer for tests of label_posts / monitoring."""

    def __init__(self, base_score=0.5, v_type="none"):
        self.base_score = base_score
        self.v_type = v_type

    def score_text(self, text: str):
        return {
            "risk_score": self.base_score,
            "violence_type": self.v_type,
            "violent_hits": 0,
            "hate_hits_strong": 0,
            "hate_hits_generic": 0,
            "all_caps_words": 0,
            "exclamations": 0,
            "explanation": f"fake scorer: {self.v_type} with {self.base_score}",
        }

    def aggregate_user_score(self, scores):
        # simple mean for tests
        return sum(scores) / len(scores) if scores else 0.0


@pytest.fixture
def fake_high_risk_scorer():
    # Always returns high risk
    return FakeRiskScorer(base_score=0.9, v_type="call_to_violence")


@pytest.fixture
def fake_low_risk_scorer():
    return FakeRiskScorer(base_score=0.1, v_type="none")


@pytest.fixture
def sample_posts_df():
    """Fixture providing sample posts DataFrame"""
    return pd.DataFrame([
        {"title": "Normal post", "selftext": "Nothing bad", "author": "user1",
         "permalink": "/r/test/1", "subreddit": "test", "url": ""},
        {"title": "Violent post", "selftext": "kill them all", "author": "user2",
         "permalink": "/r/test/2", "subreddit": "test", "url": ""},
    ])

@pytest.fixture
def sample_labeled_posts_df():
    """Fixture providing labeled posts DataFrame"""
    return pd.DataFrame([
        {"author": "user1", "violence_risk_score": 0.2, "has_profanity": False},
        {"author": "user2", "violence_risk_score": 0.8, "has_profanity": True},
        {"author": "user2", "violence_risk_score": 0.7, "has_profanity": False},
    ])

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory with structure"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "alerts").mkdir()
    return data_dir