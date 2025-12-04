# mocking OpenAI

import types
import pytest

import moderation_client


# ---------- helpers ----------

class FakeCategories:
    def __init__(self, mapping):
        self._mapping = mapping

    def model_dump(self):
        return self._mapping


class FakeResult:
    def __init__(self, flagged=True):
        cats = {"violence": True, "hate": False}
        scores = {"violence": 0.98, "hate": 0.01}
        self.categories = FakeCategories(cats)
        self.category_scores = FakeCategories(scores)
        self.flagged = flagged


class FakeModerations:
    def __init__(self, flagged=True, results=None):
        self._flagged = flagged
        self._results = results  # optional custom result list
        self.calls = 0

    def create(self, model, input):
        self.calls += 1
        if self._results is not None:
            # use pre-supplied FakeResult list
            return types.SimpleNamespace(results=self._results)
        return types.SimpleNamespace(results=[FakeResult(flagged=self._flagged)])


@pytest.fixture(autouse=True)
def reset_globals():
    """
    Reset module-level cache and last_request_time between tests.
    """
    moderation_client._cache.clear()
    moderation_client._last_request_time = 0.0
    yield
    moderation_client._cache.clear()
    moderation_client._last_request_time = 0.0


# ---------- basic behavior ----------

def test_check_moderation_flag_returns_false_when_openai_unavailable(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", False)
    flagged, categories = moderation_client.check_moderation_flag("anything")
    assert flagged is False
    assert categories is None


def test_check_moderation_flag_empty_text_returns_false(monkeypatch):
    # Even if available, empty string should be no-op
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)

    # Make sure client would explode if called, so we know it isn't
    class BombModerations:
        def create(self, *a, **k):
            raise AssertionError("moderations.create should not be called for empty text")

    fake_client = types.SimpleNamespace(moderations=BombModerations())
    monkeypatch.setattr(moderation_client, "client", fake_client)

    flagged, categories = moderation_client.check_moderation_flag("")
    assert flagged is False
    assert categories is None


def test_check_moderation_flag_flagged_true(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)
    fake_mod = FakeModerations(flagged=True)
    fake_client = types.SimpleNamespace(moderations=fake_mod)
    monkeypatch.setattr(moderation_client, "client", fake_client)

    flagged, categories = moderation_client.check_moderation_flag("test text")
    assert flagged is True
    assert "violence" in categories
    assert categories["violence"] == pytest.approx(0.98)


def test_check_moderation_flag_not_flagged(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)
    fake_mod = FakeModerations(flagged=False)
    fake_client = types.SimpleNamespace(moderations=fake_mod)
    monkeypatch.setattr(moderation_client, "client", fake_client)

    flagged, categories = moderation_client.check_moderation_flag("harmless text")
    assert flagged is False
    assert categories is None


# ---------- cache & rate limiting ----------

def test_check_moderation_flag_uses_cache(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)

    fake_mod = FakeModerations(flagged=True)
    fake_client = types.SimpleNamespace(moderations=fake_mod)
    monkeypatch.setattr(moderation_client, "client", fake_client)

    text = "some long expression that will be cached"

    # First call -> hits API
    flagged1, cats1 = moderation_client.check_moderation_flag(text)
    # Second call -> should come from cache, not call API again
    flagged2, cats2 = moderation_client.check_moderation_flag(text)

    assert flagged1 is True
    assert flagged2 is True
    assert cats1 == cats2
    assert fake_mod.calls == 1  # only one API call


def test_check_moderation_flag_rate_limit_sleep(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)

    fake_mod = FakeModerations(flagged=False)
    fake_client = types.SimpleNamespace(moderations=fake_mod)
    monkeypatch.setattr(moderation_client, "client", fake_client)

    sleep_calls = []

    # Simulate "now"
    def fake_time():
        return 100.5

    def fake_sleep(seconds):
        sleep_calls.append(seconds)

    # Pretend we made a request very recently: elapsed = 100.5 - 100.3 = 0.2 < _MIN_INTERVAL
    moderation_client._last_request_time = 100.3

    monkeypatch.setattr(moderation_client.time, "time", fake_time)
    monkeypatch.setattr(moderation_client.time, "sleep", fake_sleep)

    flagged, categories = moderation_client.check_moderation_flag("text")
    assert flagged is False  # our FakeModerations returns not flagged
    assert categories is None

    # There should be exactly one sleep due to rate limiting
    assert len(sleep_calls) == 1
    # And it should be close to _MIN_INTERVAL - elapsed (about 0.3 here)
    assert sleep_calls[0] == pytest.approx(moderation_client._MIN_INTERVAL - 0.2, rel=1e-3)


# ---------- retry & error handling ----------

def test_check_moderation_flag_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)

    # Custom RateLimitError to be sure it's catchable
    class MyRateLimitError(Exception):
        pass

    monkeypatch.setattr(moderation_client, "RateLimitError", MyRateLimitError)

    calls = {"count": 0}

    class FlakyModerations:
        def create(self, model, input):
            calls["count"] += 1
            # First two attempts -> rate limit, third -> success
            if calls["count"] < 3:
                raise MyRateLimitError("Rate limited")
            return types.SimpleNamespace(results=[FakeResult(flagged=True)])

    fake_client = types.SimpleNamespace(moderations=FlakyModerations())
    monkeypatch.setattr(moderation_client, "client", fake_client)

    # Avoid real sleeping during retries
    monkeypatch.setattr(moderation_client.time, "sleep", lambda s: None)
    monkeypatch.setattr(moderation_client.time, "time", lambda: 0.0)

    flagged, categories = moderation_client.check_moderation_flag("violent text")
    assert flagged is True
    assert "violence" in categories
    assert calls["count"] == 3  # 2 failures + 1 success


def test_check_moderation_flag_generic_exception(monkeypatch, capsys):
    monkeypatch.setattr(moderation_client, "OPENAI_AVAILABLE", True)

    class ExplodingModerations:
        def create(self, model, input):
            raise RuntimeError("boom")

    fake_client = types.SimpleNamespace(moderations=ExplodingModerations())
    monkeypatch.setattr(moderation_client, "client", fake_client)

    flagged, categories = moderation_client.check_moderation_flag("anything")
    assert flagged is False
    assert categories is None

    captured = capsys.readouterr()
    assert "Moderation API error" in captured.out
