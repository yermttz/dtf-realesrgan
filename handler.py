"""RunPod Serverless worker entrypoint."""

from __future__ import annotations

from callback import send_png_result
from config import WorkerConfig, load_config
from downloader import assert_safe_https_url
from logging_utils import safe_log
from pipeline import ProcessingError, process_job
from public_errors import public_error_payload
from validation import JobValidationError, parse_job_input

_processor = None
_download_fn = None


def set_processor(processor) -> None:
    global _processor
    _processor = processor


def set_download_fn(fn) -> None:
    global _download_fn
    _download_fn = fn


def get_processor(cfg: WorkerConfig):
    global _processor
    if _processor is None:
        from processor import RealEsrganProcessor

        _processor = RealEsrganProcessor(cfg.weights_dir, tile_size=cfg.tile_size)
    return _processor


def handler(job: dict) -> dict:
    cfg = load_config()
    job_id = None
    try:
        data = job.get("input") if isinstance(job, dict) else None
        parsed = parse_job_input(data or {})
        job_id = parsed.job_id
        assert_safe_https_url(parsed.image_url, allowed_hosts=cfg.allowed_download_hosts)
        safe_log(job_id, "accepted")

        def send_callback(cb_job_id: str, cb_image_id: str, cb_user_id: str, png_bytes: bytes):
            return send_png_result(
                callback_url=cfg.callback_url,
                callback_secret=cfg.callback_secret,
                timeout_sec=cfg.callback_timeout_sec,
                job_id=cb_job_id,
                image_id=cb_image_id,
                user_id=cb_user_id,
                png_bytes=png_bytes,
            )

        kwargs = dict(
            job=parsed,
            processor=get_processor(cfg),
            send_callback=send_callback,
            tmp_dir=cfg.tmp_dir,
            max_input_bytes=cfg.max_input_bytes,
            max_input_pixels=cfg.max_input_pixels,
            download_connect_timeout_sec=cfg.download_connect_timeout_sec,
            download_read_timeout_sec=cfg.download_read_timeout_sec,
            download_max_redirects=cfg.download_max_redirects,
            allowed_hosts=cfg.allowed_download_hosts,
        )
        if _download_fn is not None:
            kwargs["download_fn"] = _download_fn
        result = process_job(**kwargs)
        return {
            "success": True,
            "image_id": result.image_id,
            "input_width": result.input_width,
            "input_height": result.input_height,
            "output_width": result.output_width,
            "output_height": result.output_height,
        }
    except JobValidationError as exc:
        safe_log(job_id, f"rejected reason={exc.reason}")
        return public_error_payload()
    except ProcessingError:
        return public_error_payload()
    except Exception as exc:
        safe_log(job_id, f"failed type={type(exc).__name__}")
        return public_error_payload()


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
