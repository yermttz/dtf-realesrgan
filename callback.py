from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import requests

from logging_utils import safe_log
from public_errors import PUBLIC_FAIL_MESSAGE


class CallbackError(RuntimeError):
    pass


@dataclass(frozen=True)
class CallbackResult:
    image_url: str
    image_id: str


CallbackFn = Callable[[str, str, str, bytes], CallbackResult]


def send_png_result(
    *,
    callback_url: str,
    callback_secret: str,
    timeout_sec: int,
    job_id: str,
    image_id: str,
    user_id: str,
    png_bytes: bytes,
) -> CallbackResult:
    if not callback_url:
        safe_log(job_id, "callback not configured")
        raise CallbackError(PUBLIC_FAIL_MESSAGE)
    if not callback_url.lower().startswith("https://"):
        safe_log(job_id, "callback not https")
        raise CallbackError(PUBLIC_FAIL_MESSAGE)
    if not png_bytes:
        safe_log(job_id, "empty result")
        raise CallbackError(PUBLIC_FAIL_MESSAGE)

    files = {
        "file": ("result.png", png_bytes, "image/png"),
    }
    data = {
        "job_id": job_id,
        "image_id": image_id,
        "user_id": user_id,
    }
    headers = {
        "Authorization": f"Bearer {callback_secret}",
        "x-kanonico-result-secret": callback_secret,
    }

    safe_log(job_id, "callback sending")
    try:
        response = requests.post(
            callback_url,
            headers=headers,
            files=files,
            data=data,
            timeout=timeout_sec,
        )
    except requests.RequestException as exc:
        safe_log(job_id, f"callback failed transport={type(exc).__name__}")
        raise CallbackError(PUBLIC_FAIL_MESSAGE) from exc

    if response.status_code >= 400:
        safe_log(job_id, f"callback failed status={response.status_code}")
        raise CallbackError(PUBLIC_FAIL_MESSAGE)

    try:
        payload = response.json()
    except ValueError as exc:
        safe_log(job_id, "callback failed invalid-json")
        raise CallbackError(PUBLIC_FAIL_MESSAGE) from exc

    image_url = str(payload.get("image_url") or "")
    returned_image_id = str(payload.get("image_id") or image_id)
    if not payload.get("success") or not image_url:
        safe_log(job_id, "callback failed missing-url")
        raise CallbackError(PUBLIC_FAIL_MESSAGE)

    safe_log(job_id, "callback ok")
    return CallbackResult(image_url=image_url, image_id=returned_image_id)
