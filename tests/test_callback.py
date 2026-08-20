import json

import requests

from callback import send_png_result
from tests.helpers import CALLBACK_SECRET, JOB_ID, png_bytes


class DummyResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {
            "success": True,
            "image_url": "/uploads/71872588/processed/x.png",
            "image_id": "1787248816746-hgiat1",
        }

    def json(self):
        return self._payload


def test_callback_posts_png_multipart(monkeypatch):
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen["url"] = url
        seen["headers"] = headers
        seen["files"] = files
        seen["data"] = data
        seen["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(requests, "post", fake_post)
    png = png_bytes(64, 48)
    result = send_png_result(
        callback_url="http://kanonico.test/dtf/api/images/upscale/result",
        callback_secret=CALLBACK_SECRET,
        timeout_sec=30,
        job_id=JOB_ID,
        image_id="1787248816746-hgiat1",
        user_id="71872588",
        png_bytes=png,
    )
    assert result.image_url.endswith("x.png")
    assert seen["data"]["job_id"] == JOB_ID
    assert seen["data"]["image_id"] == "1787248816746-hgiat1"
    assert seen["data"]["user_id"] == "71872588"
    filename, body, mime = seen["files"]["file"]
    assert filename.endswith(".png")
    assert mime == "image/png"
    assert body[:8] == b"\x89PNG\r\n\x1a\n"
    assert seen["headers"]["Authorization"] == f"Bearer {CALLBACK_SECRET}"
    assert seen["headers"]["x-kanonico-result-secret"] == CALLBACK_SECRET
    assert "image_url" not in seen["data"]
    dumped = json.dumps({k: str(v) for k, v in seen["data"].items()})
    assert "image_base64" not in dumped
