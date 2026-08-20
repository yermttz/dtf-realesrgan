import pytest

from http_app import get_config, set_processor
from tests.helpers import AUTH, CALLBACK_SECRET, FakeProcessor


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKER_AUTH_TOKEN", AUTH)
    monkeypatch.setenv("CALLBACK_URL", "http://kanonico.test/dtf/api/images/upscale/result")
    monkeypatch.setenv("CALLBACK_SECRET", CALLBACK_SECRET)
    monkeypatch.setenv("TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("MAX_INPUT_MB", "20")
    monkeypatch.setenv("MAX_INPUT_PIXELS", "8000000")
    get_config.cache_clear()
    yield tmp_path
    get_config.cache_clear()
    set_processor(None)


@pytest.fixture
def processor():
    fake = FakeProcessor(seen_paths=[])
    set_processor(fake)
    return fake
