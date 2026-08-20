from __future__ import annotations

import os
import uuid
from dataclasses import dataclass

from typing import Protocol

from callback import CallbackFn, CallbackResult
from logging_utils import safe_log
from public_errors import PUBLIC_FAIL_MESSAGE
from validation import JobRequest, validate_image_bytes


class ImageProcessor(Protocol):
    def enhance(self, input_path: str, output_path: str, model: str, scale: int) -> tuple[int, int, int, int]:
        ...


class ProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineResult:
    job_id: str
    image_id: str
    image_url: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    model: str
    scale: int


def _remove(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError:
        safe_log(None, "temp cleanup failed")


def process_job(
    *,
    job: JobRequest,
    file_bytes: bytes,
    processor: ImageProcessor,
    send_callback: CallbackFn,
    tmp_dir: str,
    max_input_bytes: int,
    max_input_pixels: int,
) -> PipelineResult:
    input_path = None
    output_path = None
    try:
        input_width, input_height, _fmt = validate_image_bytes(
            file_bytes, max_input_bytes, max_input_pixels
        )
        safe_log(job.job_id, "received file")
        safe_log(job.job_id, f"size={len(file_bytes)}")
        safe_log(job.job_id, f"input={input_width}x{input_height}")
        safe_log(job.job_id, f"model={job.model}")
        safe_log(job.job_id, f"scale={job.scale}")

        os.makedirs(tmp_dir, exist_ok=True)
        token = uuid.uuid4().hex
        input_path = os.path.join(tmp_dir, f"{token}-input")
        output_path = os.path.join(tmp_dir, f"{token}-output.png")

        with open(input_path, "wb") as handle:
            handle.write(file_bytes)

        safe_log(job.job_id, "processing")
        try:
            in_w, in_h, out_w, out_h = processor.enhance(
                input_path, output_path, job.model, job.scale
            )
        except Exception as exc:
            safe_log(job.job_id, f"processing failed type={type(exc).__name__}")
            raise ProcessingError(PUBLIC_FAIL_MESSAGE) from exc

        if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
            safe_log(job.job_id, "processing failed missing-output")
            raise ProcessingError(PUBLIC_FAIL_MESSAGE)

        with open(output_path, "rb") as handle:
            png_bytes = handle.read()

        try:
            callback_result: CallbackResult = send_callback(
                job.job_id, job.image_id, job.user_id, png_bytes
            )
        except Exception as exc:
            safe_log(job.job_id, f"callback failed type={type(exc).__name__}")
            raise ProcessingError(PUBLIC_FAIL_MESSAGE) from exc
        safe_log(job.job_id, "completed")
        return PipelineResult(
            job_id=job.job_id,
            image_id=callback_result.image_id,
            image_url=callback_result.image_url,
            input_width=in_w,
            input_height=in_h,
            output_width=out_w,
            output_height=out_h,
            model=job.model,
            scale=job.scale,
        )
    finally:
        _remove(input_path)
        _remove(output_path)
