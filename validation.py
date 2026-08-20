from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from PIL import Image, UnidentifiedImageError

ALLOWED_MODELS = frozenset({"anime", "normal"})
ALLOWED_SCALES = frozenset({4})
ALLOWED_IMAGE_FORMATS = frozenset({"JPEG", "PNG", "WEBP", "GIF"})


class JobValidationError(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class JobRequest:
    image_url: str
    job_id: str
    image_id: str
    user_id: str
    model: str
    scale: int
    file_name: str
    image_mime: str


def _required_text(value: object, field: str, *, max_len: int = 256) -> str:
    text = str(value or "").strip()
    if not text:
        raise JobValidationError(f"missing {field}")
    if len(text) > max_len:
        raise JobValidationError(f"invalid {field}")
    return text


def parse_job_input(data: dict) -> JobRequest:
    if not isinstance(data, dict):
        raise JobValidationError("invalid input")

    image_url = _required_text(data.get("image_url"), "image_url", max_len=2048)
    if not image_url.lower().startswith("https://"):
        raise JobValidationError("blocked image_url")

    parsed_job_id = _required_text(data.get("job_id"), "job_id", max_len=64)
    try:
        uuid.UUID(parsed_job_id)
    except ValueError as exc:
        raise JobValidationError("invalid job_id") from exc

    parsed_model = _required_text(data.get("model"), "model", max_len=32).lower()
    if parsed_model not in ALLOWED_MODELS:
        raise JobValidationError("invalid model")

    scale_raw = data.get("scale")
    if scale_raw is None or str(scale_raw).strip() == "":
        raise JobValidationError("missing scale")
    try:
        parsed_scale = int(str(scale_raw).strip())
    except ValueError as exc:
        raise JobValidationError("invalid scale") from exc
    if parsed_scale not in ALLOWED_SCALES:
        raise JobValidationError("invalid scale")

    return JobRequest(
        image_url=image_url,
        job_id=parsed_job_id,
        image_id=_required_text(data.get("image_id"), "image_id", max_len=128),
        user_id=_required_text(data.get("user_id"), "user_id", max_len=64),
        model=parsed_model,
        scale=parsed_scale,
        file_name=str(data.get("file_name") or "").strip()[:255],
        image_mime=str(data.get("image_mime") or "").strip()[:64],
    )


def validate_image_file(path: str, max_bytes: int, max_pixels: int) -> tuple[int, int, str]:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise JobValidationError("empty file") from exc
    if size <= 0:
        raise JobValidationError("empty file")
    if size > max_bytes:
        raise JobValidationError("file too large")

    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
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
