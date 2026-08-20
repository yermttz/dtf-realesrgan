import os

from callback import CallbackResult
from pipeline import ProcessingError, process_job
from public_errors import PUBLIC_FAIL_MESSAGE
from tests.helpers import FakeProcessor, JOB_ID, jpeg_bytes, job_input
from validation import parse_job_input


def _run(tmp_path, processor, download_fn, send=None):
    calls = []

    def _send(job_id, image_id, user_id, png_bytes):
        calls.append((job_id, image_id, user_id, png_bytes[:8]))
        if send:
            return send(job_id, image_id, user_id, png_bytes)
        return CallbackResult(image_url="/uploads/u/processed/x.png", image_id=image_id)

    result = process_job(
        job=parse_job_input(job_input()),
        processor=processor,
        send_callback=_send,
        tmp_dir=str(tmp_path),
        max_input_bytes=20 * 1024 * 1024,
        max_input_pixels=8_000_000,
        download_connect_timeout_sec=5,
        download_read_timeout_sec=5,
        download_max_redirects=3,
        allowed_hosts=frozenset(),
        download_fn=download_fn,
    )
    return result, calls


def test_successful_processing_and_temp_cleanup(tmp_path):
    seen = []
    body = jpeg_bytes(16, 12)

    def download_fn(url, dest_path, **kwargs):
        with open(dest_path, "wb") as handle:
            handle.write(body)
        return len(body)

    result, calls = _run(tmp_path, FakeProcessor(seen_paths=seen), download_fn)
    assert result.input_width == 16
    assert result.output_width == 64
    assert result.model == "anime"
    assert calls[0][0] == JOB_ID
    assert calls[0][3] == b"\x89PNG\r\n\x1a\n"
    for input_path, output_path in seen:
        assert not os.path.exists(input_path)
        assert not os.path.exists(output_path)
        assert "original.jpg" not in input_path
    assert list(tmp_path.iterdir()) == []


def test_processing_error_is_generic_and_cleans_temp(tmp_path):
    seen = []
    body = jpeg_bytes(8, 8)

    def download_fn(url, dest_path, **kwargs):
        with open(dest_path, "wb") as handle:
            handle.write(body)
        return len(body)

    try:
        _run(tmp_path, FakeProcessor(fail=True, seen_paths=seen), download_fn)
        assert False, "expected ProcessingError"
    except ProcessingError as exc:
        message = str(exc)
        assert message == PUBLIC_FAIL_MESSAGE
        lowered = message.lower()
        for word in ("runpod", "real-esrgan", "gpu", "cuda", "pytorch"):
            assert word not in lowered
    for input_path, output_path in seen:
        assert not os.path.exists(input_path)
        assert not os.path.exists(output_path)
