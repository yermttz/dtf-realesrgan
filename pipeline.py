from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Callable, Protocol

from callback import CallbackFn, CallbackResult
from downloader import DownloadError, download_to_file, url_host
from logging_utils import safe_log
from public_errors import PUBLIC_FAIL_MESSAGE
from validation import JobRequest, JobValidationError, validate_image_file

DownloadFn = Callable[..., int]


class ImageProcessor(Protocol):
    def enhance(self, input_path: str, output_path: str, model: str, scale: int) -> tuple[int, int, int, int]:
        ...


class ProcessingError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineResult:
    job_id: str
    image_id: str
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
    processor: ImageProcessor,
    send_callback: CallbackFn,
    tmp_dir: str,
    max_input_bytes: int,
    max_input_pixels: int,
    download_connect_timeout_sec: int,
    download_read_timeout_sec: int,
    download_max_redirects: int,
    allowed_hosts: frozenset[str],
    download_fn: DownloadFn = download_to_file,
) -> PipelineResult:
    input_path = None
    output_path = None
    try:
        os.makedirs(tmp_dir, exist_ok=True)
        token = uuid.uuid4().hex
        input_path = os.path.join(tmp_dir, f"{token}-input")
        output_path = os.path.join(tmp_dir, f"{token}-output.png")

        safe_log(job.job_id, f"fetch host={url_host(job.image_url)}")
        try:
            size = download_fn(
                job.image_url,
                input_path,
                max_bytes=max_input_bytes,
                connect_timeout_sec=download_connect_timeout_sec,
                read_timeout_sec=download_read_timeout_sec,
                max_redirects=download_max_redirects,
                allowed_hosts=allowed_hosts,
                job_id=job.job_id,
            )
        except JobValidationError:
            raise
        except DownloadError as exc:
            safe_log(job.job_id, f"fetch failed reason={exc.reason}")
            raise ProcessingError(PUBLIC_FAIL_MESSAGE) from exc
        except Exception as exc:
            safe_log(job.job_id, f"fetch failed type={type(exc).__name__}")
            raise ProcessingError(PUBLIC_FAIL_MESSAGE) from exc

        input_width, input_height, _fmt = validate_image_file(
            input_path, max_input_bytes, max_input_pixels
        )
        safe_log(job.job_id, "received file")
        safe_log(job.job_id, f"size={size}")
        safe_log(job.job_id, f"input={input_width}x{input_height}")
        safe_log(job.job_id, f"model={job.model}")
        safe_log(job.job_id, f"scale={job.scale}")

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
