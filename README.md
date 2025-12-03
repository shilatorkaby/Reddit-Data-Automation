# Reddit Harmful Content Detection System

An automated Python system for identifying and monitoring violent hate speech on Reddit. This project delivers scored
data feeds of posts and prioritized lists of high-risk user accounts.

## Table of Contents
- [Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Risk Scoring Methodology](#risk-scoring-methodology)
- [Output Files](#output-files)
- [Edge Cases](#edge-cases--reliability)
- [Testing & Validation](#testing--validation)
- [Configuration](#configuration)
- [Limitations & Future Work
](#limitations--future-work)

## Objective

Automatically collect and score Reddit posts discussing controversial, harmful, and violent content, providing:

- **Post-level data feed** with language, date, text, and risk score
- **User-level risk feed** with calculated scores and explanations

## Key Features

- 🔍 **Targeted scraping** using Reddit's public JSON API (no authentication required)
- 📊 **Multi-layered risk scoring**: dictionary-based + context-aware + LLM moderation
- 🎯 **Context-aware classification**: distinguishes news reports from advocacy
- 🚨 **Daily monitoring** of flagged users with automated alerts
- 📝 **Detailed explanations** for every risk assessment

## Architecture

```
┌─────────────────┐
│ Reddit API      │
│ (Public JSON)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ Data Collection │─────▶│ Profanity Filter │
│ (reddit_scraper)│      │ (dictionary)     │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│ Risk Scorer     │◀─────│ OpenAI Moderation│
│ (context-aware) │      │ (validation)     │
└────────┬────────┘      └──────────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ User Aggregator │─────▶│ Daily Monitor    │
└────────┬────────┘      └────────┬─────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐      ┌──────────────────┐
│  Data Outputs   │      │ Alert System     │
└─────────────────┘      └──────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/reddit-harmful-content-detection.git
cd reddit-harmful-content-detection

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas requests beautifulsoup4 openai langdetect pytest pytest-mock

# Set OpenAI API key (optional)
export OPENAI_API_KEY='your-key-here'
```

### Run Pipeline

```bash
# Collect and analyze posts
python main.py

# Monitor high-risk users
python -m src.monitoring

# Run tests
python test_risk_scorer.py

```

## Methodology

### Data Collection

- **Search Strategy**: Combines search terms (e.g., "kill", "death threat") with targeted subreddits (politics, news, PublicFreakout)
- **Source**: Reddit's public JSON API (no authentication needed)
- **Rate Limiting**: 0.5s between requests to avoid 429 errors
- **User History**: Collects recent post history for identified high-risk users

### Risk Scoring Methodology

**1. Profanity Detection** (`profanity_detector.py`)

- Match against curated lexicon of profanity and violent terms
- Case-insensitive normalization
- Fast initial filtering for explicit content
- Produces has_profanity and list of matched terms

**2. Context-Aware Classification** (`risk_scorer.py`)

- **Violence Type Detection**:
   - `none` - No violent content
   - `descriptive` - Narrative/news context
   - `self_directed` - Self-harm language
   - `hate_speech` - Dehumanizing language
   - `call_to_violence` - Direct threats
- **Contextual Analysis**: Examines pronoun usage and sentence structure
- **Scoring Formula**: Base score by type + bonuses for intensity signals
- **News Filtering**: Distinguishes reporting from advocacy

**3. Optional LLM Moderation** (`moderation_client.py`)

- OpenAI Moderation API validation `omni-moderation-latest` for high-risk posts (score ≥ 0.8)
- Caching and retry logic for reliability
- Score calibration based on API feedback

**4. User-Level Aggregation** (`user_aggregator.py`)

- Formula: `user_score = 0.7 × max_post_score + 0.3 × mean_post_score`
- Emphasizes worst behavior while considering patterns
- Counts high-risk posts per user

**5. Daily Monitoring** (`monitoring.py`)

- Tracks users with risk score ≥ 0.5
- Checks posts from last 48 hours
- Generates timestamped alerts for new high-risk content

## Output Files

| File                         | Description            | Key Fields                                                              |
|------------------------------|------------------------|-------------------------------------------------------------------------|
| `raw_posts.csv`              | All collected posts    | post_id, author, title, selftext, subreddit, created_utc                |
| `raw_posts_labeled.csv`      | Posts with risk scores | + violence_risk_score, violence_type, has_profanity, moderation_flagged |
| `posts_offensive_subset.csv` | High-risk posts only   | Posts with risk ≥ 0.6 or profanity or API flagged                       |
| `users_risk.csv`             | User-level assessments | username, user_risk_score, high_risk_posts, explanation                 |
| `alerts/*.json`              | Monitoring alerts      | Timestamped alerts for new high-risk content from monitored users       |

### Risk Score Ranges

| Score   | Severity     | Action           |
|---------|--------------|------------------|
| 0.0-0.4 | Low-Moderate | Monitor          |
| 0.4-0.6 | Medium       | Review           |
| 0.6-0.8 | High         | Priority review  |
| 0.8-1.0 | Critical     | Immediate action |

## Edge Cases & Reliability
- Deleted users → Labeled as “deleted_user”
- Private profiles → Labeled “private_profile”, assigned score 0
- No post history → Assigned score 0 with explanation
- News-like posts → Automatically filtered and scored 0
- Rate limiting → Handled with exponential backoff and retries
- Duplicate posts → Removed using post ID deduplication

## Testing & Validation
**Test types covered:**
- **Unit tests**
  - `test_profanity_detector.py` - lexicon loading & profanity detection
  - `test_risk_scorer.py` – violence type classification & scoring formula
  - `test_post_labeler.py` – end-to-end labeling of posts
  - `test_user_aggregator.py` – user-level risk aggregation

- **Mocked / API-facing tests**
  - `test_moderation_client.py` – OpenAI moderation client, caching, retries, rate limiting
  - `test_reddit_scraper.py` – Reddit scraping logic using mocked HTTP responses
  - `test_monitoring.py` – daily monitor loop, alert generation, and file outputs

- **Integration**
  - Pipeline-style tests that go from labeled posts → user risk feed


### Run Tests

This project uses `pytest` with a mix of unit, integration, and mocked tests.

```bash
# Run full test suite
pytest -q

# Or run a specific file
pytest tests/test_reddit_scraper.py -q
pytest tests/test_monitoring.py -q

## Configuration

Edit `src/config.py` to customize:

```python
# Target subreddits
TARGET_SUBREDDITS = ["politics", "worldnews", "PublicFreakout"]

# Search terms by category
SEARCH_TERMS_BY_CATEGORY = {
    "violence": ["kill", "murder", "shooting"],
    "threats": ["deserve to die", "should be killed"],
    "hate_speech": ["hate speech", "racist slur"],
}
```

## Limitations & Future Work

**Current Limitations:**

- Heuristic-based (not deep learning)
- English-only support
- Challenges with sarcasm/satire
- Misspellings may evade detection

