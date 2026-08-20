import pytest

from public_errors import PUBLIC_FAIL_MESSAGE
from validation import JobValidationError, parse_job_fields, validate_image_bytes
from tests.helpers import JOB_ID, jpeg_bytes


def test_parse_valid_kanonico_fields():
    job = parse_job_fields(
        JOB_ID,
        "1787248816746-hgiat1",
        "71872588",
        "anime",
        "4",
        "photo.jpg",
        "image/jpeg",
    )
    assert job.model == "anime"
    assert job.scale == 4


@pytest.mark.parametrize("model", ["foo", "ANIME-X", ""])
def test_invalid_model(model):
    with pytest.raises(JobValidationError, match="model"):
        parse_job_fields(JOB_ID, "i", "u", model, "4", "a.jpg", "image/jpeg")


@pytest.mark.parametrize("scale", ["2", "8", "nope", None])
def test_invalid_scale(scale):
    with pytest.raises(JobValidationError, match="scale"):
        parse_job_fields(JOB_ID, "i", "u", "normal", scale, "a.jpg", "image/jpeg")


def test_invalid_job_id():
    with pytest.raises(JobValidationError, match="job_id"):
        parse_job_fields("not-a-uuid", "i", "u", "normal", "4", "a.jpg", "image/jpeg")


def test_invalid_claimed_mime():
    with pytest.raises(JobValidationError, match="image_mime"):
        parse_job_fields(JOB_ID, "i", "u", "normal", "4", "a.jpg", "text/plain")


def test_empty_file_rejected():
    with pytest.raises(JobValidationError, match="empty"):
        validate_image_bytes(b"", 20 * 1024 * 1024, 8_000_000)


def test_non_image_rejected_even_if_mime_says_jpeg():
    with pytest.raises(JobValidationError, match="not an image"):
        validate_image_bytes(b"this is not an image", 20 * 1024 * 1024, 8_000_000)


def test_real_jpeg_accepted():
    data = jpeg_bytes(524, 698)
    width, height, fmt = validate_image_bytes(data, 20 * 1024 * 1024, 8_000_000)
    assert (width, height) == (524, 698)
    assert fmt == "JPEG"


def test_pixel_limit():
    data = jpeg_bytes(10, 10)
    with pytest.raises(JobValidationError, match="too large"):
        validate_image_bytes(data, 20 * 1024 * 1024, 50)


def test_public_error_has_no_infra_words():
    lowered = PUBLIC_FAIL_MESSAGE.lower()
    for word in ("runpod", "real-esrgan", "gpu", "worker", "endpoint"):
        assert word not in lowered
