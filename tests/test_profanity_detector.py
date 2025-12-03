# unit + filesystem
from pathlib import Path

from profanity_detector import load_bad_words, detect_bad_words, analyze_post


def test_load_bad_words_ignores_comments_and_blank_lines(tmp_path):
    content = """
    # this is a comment
    BadWord
    nasty

    ugly
    """
    path = tmp_path / "bad_words.txt"
    path.write_text(content, encoding="utf-8")

    bad_words = load_bad_words(str(path))
    assert bad_words == {"badword", "nasty", "ugly"}


def test_detect_bad_words_simple(bad_words):
    text = "This is a BadWord and another ugly thing!"
    hits = detect_bad_words(text, bad_words)
    assert hits == ["badword", "ugly"]


def test_analyze_post_aggregates_title_and_body(bad_words):
    title = "Nasty comment"
    body = "You are an ugly person"
    has_prof, words = analyze_post(title, body, bad_words)
    assert has_prof is True
    assert set(words) == {"nasty", "ugly"}


def test_analyze_post_no_profanity(bad_words):
    title = "Hello world"
    body = "Nice day"
    has_prof, words = analyze_post(title, body, bad_words)
    assert has_prof is False
    assert words == []
