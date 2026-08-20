import pytest

from handler import set_download_fn, set_processor
from tests.helpers import CALLBACK_SECRET, FakeProcessor, jpeg_bytes, public_resolver


@pytest.fixture
def worker_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CALLBACK_URL", "https://kanonico.test/dtf/api/images/upscale/result")
    monkeypatch.setenv("CALLBACK_SECRET", CALLBACK_SECRET)
    monkeypatch.setenv("TMP_DIR", str(tmp_path / "tmp"))
    monkeypatch.setenv("WEIGHTS_DIR", str(tmp_path / "weights"))
    monkeypatch.setenv("MAX_INPUT_MB", "20")
    monkeypatch.setenv("MAX_INPUT_PIXELS", "8000000")
    monkeypatch.setattr("downloader.socket.getaddrinfo", public_resolver)
    yield tmp_path
    set_processor(None)
    set_download_fn(None)


@pytest.fixture
def processor():
    fake = FakeProcessor(seen_paths=[])
    set_processor(fake)
    return fake


@pytest.fixture
def jpeg_download(processor):
    body = jpeg_bytes(16, 12)

    def download_fn(url, dest_path, **kwargs):
        with open(dest_path, "wb") as handle:
            handle.write(body)
        return len(body)

    set_download_fn(download_fn)
    return body
