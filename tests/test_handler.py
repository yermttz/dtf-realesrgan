import json

from downloader import DownloadError
from handler import handler, set_download_fn, set_processor
from public_errors import PUBLIC_FAIL_MESSAGE
from tests.helpers import CALLBACK_SECRET, FakeProcessor, IMAGE_ID, JOB_ID, jpeg_bytes, job_input


INFRA = ("runpod", "real-esrgan", "realesrgan", "gpu", "pytorch", "cuda", "endpoint", "docker")


def _assert_public(payload: dict):
    dumped = json.dumps(payload).lower()
    for word in INFRA:
        assert word not in dumped
    if payload.get("success") is False:
        assert payload["error"] == PUBLIC_FAIL_MESSAGE


def test_handler_success(worker_env, processor, jpeg_download, monkeypatch):
    seen = {}

    def fake_post(url, headers=None, files=None, data=None, timeout=None):
        seen.update(url=url, headers=headers, files=files, data=data)

        class Resp:
            status_code = 200

            def json(self):
                return {"success": True, "image_url": "/uploads/x.png", "image_id": IMAGE_ID}

        return Resp()

    monkeypatch.setattr("callback.requests.post", fake_post)
    result = handler({"input": job_input(scale="4")})
    assert result["success"] is True
    assert result["image_id"] == IMAGE_ID
    assert result["input_width"] == 16
    assert result["output_width"] == 64
    assert "error" not in result
    assert seen["data"]["job_id"] == JOB_ID
    assert seen["files"]["file"][2] == "image/png"
    assert seen["headers"]["x-kanonico-result-secret"] == CALLBACK_SECRET
    _assert_public(result)
    leftover = list((worker_env / "tmp").iterdir()) if (worker_env / "tmp").exists() else []
    assert leftover == []


def test_handler_invalid_model(worker_env, processor, jpeg_download):
    result = handler({"input": job_input(model="ultra")})
    assert result["success"] is False
    _assert_public(result)


def test_handler_invalid_scale(worker_env, processor, jpeg_download):
    result = handler({"input": job_input(scale=2)})
    assert result["success"] is False
    _assert_public(result)


def test_handler_http_url_rejected(worker_env, processor, jpeg_download):
    result = handler({"input": job_input(image_url="http://kanonico.example/a.jpg")})
    assert result["success"] is False
    _assert_public(result)


def test_handler_localhost_rejected(worker_env, processor, jpeg_download):
    result = handler({"input": job_input(image_url="https://127.0.0.1/secret.jpg")})
    assert result["success"] is False
    _assert_public(result)


def test_handler_private_ip_rejected(worker_env, processor, jpeg_download):
    result = handler({"input": job_input(image_url="https://192.168.0.10/a.jpg")})
    assert result["success"] is False
    _assert_public(result)


def test_handler_download_404(worker_env, processor):
    def download_fn(*_args, **_kwargs):
        raise DownloadError("http_404")

    set_download_fn(download_fn)
    result = handler({"input": job_input()})
    assert result["success"] is False
    _assert_public(result)


def test_handler_download_timeout(worker_env, processor):
    def download_fn(*_args, **_kwargs):
        raise DownloadError("timeout")

    set_download_fn(download_fn)
    result = handler({"input": job_input()})
    assert result["success"] is False
    _assert_public(result)


def test_handler_not_an_image(worker_env, processor):
    def download_fn(url, dest_path, **kwargs):
        with open(dest_path, "wb") as handle:
            handle.write(b"not-an-image")
        return 12

    set_download_fn(download_fn)
    result = handler({"input": job_input()})
    assert result["success"] is False
    _assert_public(result)


def test_handler_processing_error_hides_infra(worker_env):
    set_processor(FakeProcessor(fail=True, seen_paths=[]))

    def download_fn(url, dest_path, **kwargs):
        with open(dest_path, "wb") as handle:
            handle.write(jpeg_bytes(8, 8))
        return 100

    set_download_fn(download_fn)
    result = handler({"input": job_input()})
    assert result["success"] is False
    _assert_public(result)


def test_logs_do_not_include_secrets_or_full_url(worker_env, processor, jpeg_download, monkeypatch, capsys):
    monkeypatch.setattr(
        "callback.requests.post",
        lambda *a, **k: type("R", (), {"status_code": 200, "json": lambda self: {"success": True, "image_url": "/x.png", "image_id": IMAGE_ID}})(),
    )
    secret_url = "https://kanonico.example/uploads/a.jpg?token=super-secret-token"
    handler({"input": job_input(image_url=secret_url)})
    logs = capsys.readouterr().out
    assert CALLBACK_SECRET not in logs
    assert "super-secret-token" not in logs
    assert "Bearer" not in logs
    assert "[AI] job=" in logs
