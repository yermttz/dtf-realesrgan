import os

from callback import CallbackResult
from pipeline import ProcessingError, process_job
from public_errors import PUBLIC_FAIL_MESSAGE
from tests.helpers import FakeProcessor, JOB_ID, jpeg_bytes
from validation import parse_job_fields


def _job():
    return parse_job_fields(
        JOB_ID,
        "1787248816746-hgiat1",
        "71872588",
        "anime",
        4,
        "photo.jpg",
        "image/jpeg",
    )


def test_successful_processing_and_temp_cleanup(tmp_path):
    seen = []
    processor = FakeProcessor(seen_paths=seen)
    calls = []

    def send(job_id, image_id, user_id, png_bytes):
        calls.append((job_id, image_id, user_id, png_bytes[:8]))
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
        return CallbackResult(image_url="/uploads/u/processed/x.png", image_id=image_id)

    result = process_job(
        job=_job(),
        file_bytes=jpeg_bytes(16, 12),
        processor=processor,
        send_callback=send,
        tmp_dir=str(tmp_path),
        max_input_bytes=20 * 1024 * 1024,
        max_input_pixels=8_000_000,
    )
    assert result.image_url == "/uploads/u/processed/x.png"
    assert result.input_width == 16
    assert result.output_width == 64
    assert result.model == "anime"
    assert calls[0][0] == JOB_ID
    assert seen
    for input_path, output_path in seen:
        assert not os.path.exists(input_path)
        assert not os.path.exists(output_path)
        assert "photo.jpg" not in input_path
    leftover = list(tmp_path.iterdir())
    assert leftover == []


def test_processing_error_is_generic_and_cleans_temp(tmp_path):
    seen = []
    processor = FakeProcessor(fail=True, seen_paths=seen)
    try:
        process_job(
            job=_job(),
            file_bytes=jpeg_bytes(8, 8),
            processor=processor,
            send_callback=lambda *a: None,
            tmp_dir=str(tmp_path),
            max_input_bytes=20 * 1024 * 1024,
            max_input_pixels=8_000_000,
        )
        assert False, "expected ProcessingError"
    except ProcessingError as exc:
        message = str(exc)
        assert message == PUBLIC_FAIL_MESSAGE
        lowered = message.lower()
        for word in ("runpod", "real-esrgan", "gpu", "cuda", "worker"):
            assert word not in lowered
    for input_path, output_path in seen:
        assert not os.path.exists(input_path)
        assert not os.path.exists(output_path)
