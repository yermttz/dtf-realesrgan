from __future__ import annotations

import io
import uuid
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

ALLOWED_MODELS = frozenset({"anime", "normal"})
ALLOWED_SCALES = frozenset({4})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF"})
ALLOWED_CLAIMED_MIMES = frozenset({"image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"})


class JobValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class JobRequest:
    job_id: str
    image_id: str
    user_id: str
    model: str
    scale: int
    file_name: str
    image_mime: str


def _required_text(value: str | None, field: str, *, max_len: int = 256) -> str:
    text = (value or "").strip()
    if not text:
        raise JobValidationError(f"missing {field}")
    if len(text) > max_len:
        raise JobValidationError(f"invalid {field}")
    return text


def parse_job_fields(
    job_id: str | None,
    image_id: str | None,
    user_id: str | None,
    model: str | None,
    scale: str | int | None,
    file_name: str | None,
    image_mime: str | None,
) -> JobRequest:
    parsed_job_id = _required_text(job_id, "job_id", max_len=64)
    try:
        uuid.UUID(parsed_job_id)
    except ValueError as exc:
        raise JobValidationError("invalid job_id") from exc

    parsed_model = _required_text(model, "model", max_len=32).lower()
    if parsed_model not in ALLOWED_MODELS:
        raise JobValidationError("invalid model")

    if scale is None or str(scale).strip() == "":
        raise JobValidationError("missing scale")
    try:
        parsed_scale = int(str(scale).strip())
    except ValueError as exc:
        raise JobValidationError("invalid scale") from exc
    if parsed_scale not in ALLOWED_SCALES:
        raise JobValidationError("invalid scale")

    claimed_mime = _required_text(image_mime, "image_mime", max_len=64).lower()
    if claimed_mime not in ALLOWED_CLAIMED_MIMES:
        raise JobValidationError("invalid image_mime")

    return JobRequest(
        job_id=parsed_job_id,
        image_id=_required_text(image_id, "image_id", max_len=128),
        user_id=_required_text(user_id, "user_id", max_len=64),
        model=parsed_model,
        scale=parsed_scale,
        file_name=_required_text(file_name, "file_name", max_len=255),
        image_mime=claimed_mime,
    )


def validate_image_bytes(data: bytes, max_bytes: int, max_pixels: int) -> tuple[int, int, str]:
    if not data:
        raise JobValidationError("empty file")
    if len(data) > max_bytes:
        raise JobValidationError("file too large")

    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            fmt = (img.format or "").upper()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise JobValidationError("not an image") from exc

    if width <= 0 or height <= 0:
        raise JobValidationError("invalid image size")
    if width * height > max_pixels:
        raise JobValidationError("image too large")
    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise JobValidationError("unsupported image format")
    return width, height, fmt
