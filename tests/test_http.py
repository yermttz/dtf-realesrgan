import json

import requests
from fastapi.testclient import TestClient

from http_app import app, set_processor
from public_errors import PUBLIC_FAIL_MESSAGE
from tests.helpers import (
    AUTH,
    CALLBACK_SECRET,
    FILE_NAME,
    FakeProcessor,
    IMAGE_ID,
    JOB_ID,
    USER_ID,
    jpeg_bytes,
    kanonico_fields,
)


INFRA = ("runpod", "real-esrgan", "realesrgan", "gpu", "worker", "endpoint", "docker", "cuda")


def _assert_public(body: dict):
    dumped = json.dumps(body).lower()
    for word in INFRA:
        assert word not in dumped
    if body.get("output"):
        assert body["output"]["error"] == PUBLIC_FAIL_MESSAGE or body["output"].get("success") is True


def _client():
    return TestClient(app)


def _auth():
    return {"Authorization": f"Bearer {AUTH}"}


def _mock_callback(monkeypatch, status=200, payload=None):
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen.update(url=url, headers=headers, files=files, data=data, timeout=timeout)
        class Resp:
            status_code = status

            def json(self):
                return payload or {
                    "success": True,
                    "image_url": "/uploads/71872588/processed/x.png",
                    "image_id": IMAGE_ID,
                }

        return Resp()

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not download")))
    return seen


def test_health_unauthenticated():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_missing_auth(worker_env, processor):
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post("/runsync", data=kanonico_fields(), files=files)
    assert response.status_code == 401
    _assert_public(response.json())


def test_multipart_success_kanonico_contract(worker_env, processor, monkeypatch):
    seen = _mock_callback(monkeypatch)
    image = jpeg_bytes(524, 698)
    files = {"file": (FILE_NAME, image, "image/jpeg")}
    response = _client().post(
        "/abcd1234/runsync",
        headers=_auth(),
        data=kanonico_fields(),
        files=files,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "COMPLETED"
    assert body["output"]["success"] is True
    assert body["output"]["image_url"] == "/uploads/71872588/processed/x.png"
    assert body["output"]["image_id"] == IMAGE_ID
    assert body["output"]["input_width"] == 524
    assert body["output"]["input_height"] == 698
    assert body["output"]["output_width"] == 2096
    assert body["output"]["output_height"] == 2792
    assert body["output"]["model"] == "anime"
    assert seen["data"]["job_id"] == JOB_ID
    assert seen["data"]["user_id"] == USER_ID
    assert seen["files"]["file"][2] == "image/png"
    _assert_public(body)


def test_runsync_root_path(worker_env, processor, monkeypatch):
    _mock_callback(monkeypatch)
    files = {"file": (FILE_NAME, jpeg_bytes(16, 12), "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(model="normal"), files=files)
    assert response.status_code == 200
    assert response.json()["output"]["model"] == "normal"


def test_missing_file(worker_env, processor):
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields())
    assert response.status_code == 400
    body = response.json()
    assert body["status"] == "FAILED"
    assert body["output"]["success"] is False
    _assert_public(body)


def test_empty_file(worker_env, processor):
    files = {"file": (FILE_NAME, b"", "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 400
    _assert_public(response.json())


def test_invalid_mime_claimed(worker_env, processor):
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "text/plain")}
    response = _client().post(
        "/runsync",
        headers=_auth(),
        data=kanonico_fields(image_mime="text/plain"),
        files=files,
    )
    assert response.status_code == 400
    _assert_public(response.json())


def test_non_image_bytes_with_jpeg_mime(worker_env, processor):
    files = {"file": (FILE_NAME, b"not-an-image", "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 400
    _assert_public(response.json())


def test_invalid_model(worker_env, processor):
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post(
        "/runsync",
        headers=_auth(),
        data=kanonico_fields(model="ultra"),
        files=files,
    )
    assert response.status_code == 400
    _assert_public(response.json())


def test_invalid_scale(worker_env, processor):
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post(
        "/runsync",
        headers=_auth(),
        data=kanonico_fields(scale="2"),
        files=files,
    )
    assert response.status_code == 400
    _assert_public(response.json())


def test_processing_error_is_generic(worker_env, monkeypatch):
    set_processor(FakeProcessor(fail=True, seen_paths=[]))
    _mock_callback(monkeypatch)
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 502
    body = response.json()
    assert body["output"]["error"] == PUBLIC_FAIL_MESSAGE
    _assert_public(body)


def test_rejects_json_image_url_without_downloading(worker_env, processor, monkeypatch):
    seen_get = []

    def fake_get(*args, **kwargs):
        seen_get.append(args)
        raise AssertionError("must not download image_url")

    monkeypatch.setattr(requests, "get", fake_get)
    response = _client().post(
        "/runsync",
        headers={**_auth(), "Content-Type": "application/json"},
        json={
            "image_url": "https://example.test/secret.jpg?token=abc",
            "job_id": JOB_ID,
            "model": "anime",
            "scale": 4,
        },
    )
    assert response.status_code == 400
    assert seen_get == []
    dumped = json.dumps(response.json()).lower()
    assert "image_url" not in dumped or response.json()["status"] == "FAILED"
    _assert_public(response.json())


def test_temp_dir_empty_after_http_success(worker_env, processor, monkeypatch):
    _mock_callback(monkeypatch)
    tmp = worker_env / "tmp"
    files = {"file": (FILE_NAME, jpeg_bytes(10, 10), "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 200
    leftover = list(tmp.iterdir()) if tmp.exists() else []
    assert leftover == []


def test_callback_failure_is_generic(worker_env, processor, monkeypatch):
    _mock_callback(monkeypatch, status=500, payload={"success": False})
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 502
    _assert_public(response.json())
    leftover = list((worker_env / "tmp").iterdir()) if (worker_env / "tmp").exists() else []
    assert leftover == []


def test_logs_do_not_include_secrets(worker_env, processor, monkeypatch, capsys):
    _mock_callback(monkeypatch)
    files = {"file": (FILE_NAME, jpeg_bytes(8, 8), "image/jpeg")}
    response = _client().post("/runsync", headers=_auth(), data=kanonico_fields(), files=files)
    assert response.status_code == 200
    logs = capsys.readouterr().out
    assert AUTH not in logs
    assert CALLBACK_SECRET not in logs
    assert "Bearer" not in logs
    assert "[AI] job=" in logs
