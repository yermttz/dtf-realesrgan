import os

import pytest

from public_errors import PUBLIC_FAIL_MESSAGE
from tests.helpers import JOB_ID, PUBLIC_IMAGE_URL, jpeg_bytes, job_input
from validation import JobValidationError, parse_job_input, validate_image_file


def test_parse_valid_job_input():
    job = parse_job_input(job_input())
    assert job.model == "anime"
    assert job.scale == 4
    assert job.image_url == PUBLIC_IMAGE_URL
    assert job.job_id == JOB_ID


@pytest.mark.parametrize("model", ["foo", "ANIME-X", ""])
def test_invalid_model(model):
    with pytest.raises(JobValidationError, match="model"):
        parse_job_input(job_input(model=model))


@pytest.mark.parametrize("scale", ["2", "8", "nope", None])
def test_invalid_scale(scale):
    with pytest.raises(JobValidationError, match="scale"):
        parse_job_input(job_input(scale=scale))


def test_invalid_job_id():
    with pytest.raises(JobValidationError, match="job_id"):
        parse_job_input(job_input(job_id="not-a-uuid"))


def test_http_url_rejected_at_parse():
    with pytest.raises(JobValidationError, match="blocked image_url"):
        parse_job_input(job_input(image_url="http://kanonico.example/original.jpg"))


def test_missing_image_url():
    payload = job_input()
    del payload["image_url"]
    with pytest.raises(JobValidationError, match="image_url"):
        parse_job_input(payload)


def test_file_name_and_mime_are_optional_metadata():
    payload = job_input()
    payload["image_mime"] = "text/plain"
    del payload["file_name"]
    job = parse_job_input(payload)
    assert job.image_mime == "text/plain"
    assert job.file_name == ""


def test_empty_file_rejected(tmp_path):
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    with pytest.raises(JobValidationError, match="empty"):
        validate_image_file(str(path), 20 * 1024 * 1024, 8_000_000)


def test_non_image_rejected(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"this is not an image")
    with pytest.raises(JobValidationError, match="not an image"):
        validate_image_file(str(path), 20 * 1024 * 1024, 8_000_000)


def test_real_jpeg_accepted(tmp_path):
    path = tmp_path / "in.jpg"
    path.write_bytes(jpeg_bytes(524, 698))
    width, height, fmt = validate_image_file(str(path), 20 * 1024 * 1024, 8_000_000)
    assert (width, height) == (524, 698)
    assert fmt == "JPEG"


def test_pixel_limit(tmp_path):
    path = tmp_path / "in.jpg"
    path.write_bytes(jpeg_bytes(10, 10))
    with pytest.raises(JobValidationError, match="too large"):
        validate_image_file(str(path), 20 * 1024 * 1024, 50)


def test_public_error_has_no_infra_words():
    lowered = PUBLIC_FAIL_MESSAGE.lower()
    for word in ("runpod", "real-esrgan", "gpu", "pytorch"):
        assert word not in lowered


def test_validate_does_not_keep_temp_elsewhere(tmp_path):
    path = tmp_path / "in.jpg"
    path.write_bytes(jpeg_bytes(8, 8))
    validate_image_file(str(path), 20 * 1024 * 1024, 8_000_000)
    assert os.path.isfile(path)
