from __future__ import annotations

import asyncio
import hmac
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from callback import send_png_result
from config import WorkerConfig, load_config
from logging_utils import safe_log
from pipeline import ProcessingError, process_job
from public_errors import public_error_payload
from validation import JobValidationError, parse_job_fields

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

_processor = None


def _apply_upload_limit(max_bytes: int) -> None:
    extra = max(max_bytes + 1024 * 1024, 21 * 1024 * 1024)
    if hasattr(MultiPartParser, "max_part_size"):
        MultiPartParser.max_part_size = extra


def set_processor(processor) -> None:
    global _processor
    _processor = processor


def get_processor():
    global _processor
    if _processor is None:
        from processor import RealEsrganProcessor

        cfg = load_config()
        _processor = RealEsrganProcessor(cfg.weights_dir, tile_size=cfg.tile_size)
    return _processor


@lru_cache(maxsize=1)
def get_config() -> WorkerConfig:
    cfg = load_config()
    _apply_upload_limit(cfg.max_input_bytes)
    return cfg


def _failed_response(status_code: int = 400) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "FAILED",
            "output": public_error_payload(),
        },
    )


def _completed_response(result) -> dict:
    return {
        "status": "COMPLETED",
        "id": result.job_id,
        "job_id": result.job_id,
        "output": {
            "success": True,
            "image_url": result.image_url,
            "image_id": result.image_id,
            "input_width": result.input_width,
            "input_height": result.input_height,
            "output_width": result.output_width,
            "output_height": result.output_height,
            "model": result.model,
            "scale": result.scale,
        },
    }


def _extract_bearer(authorization: str | None) -> str:
    if not authorization:
        return ""
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix) :].strip()
    return ""


def require_auth(
    authorization: Annotated[str | None, Header()] = None,
    config: WorkerConfig = Depends(get_config),
) -> None:
    provided = _extract_bearer(authorization)
    expected = config.auth_token
    if not expected:
        # Gateway-authenticated deployments may not forward the bearer token.
        return
    a = expected.encode("utf-8")
    b = provided.encode("utf-8")
    if len(a) != len(b) or not hmac.compare_digest(a, b):
        raise HTTPException(status_code=401, detail=public_error_payload())


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request: Request, _exc: RequestValidationError):
    return _failed_response(400)


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    payload = exc.detail if isinstance(exc.detail, dict) else public_error_payload()
    if exc.status_code == 401:
        return JSONResponse(status_code=401, content=payload)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "FAILED", "output": payload if "success" in payload else public_error_payload()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception):
    safe_log(None, f"unhandled type={type(exc).__name__}")
    return _failed_response(500)


@app.get("/health")
@app.get("/ping")
async def health():
    return {"ok": True}


async def _read_upload(upload: UploadFile | None, max_bytes: int) -> bytes:
    if upload is None:
        raise JobValidationError("missing file")
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise JobValidationError("file too large")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise JobValidationError("empty file")
    return data


async def handle_enhance(
    *,
    file: UploadFile | None,
    job_id: str | None,
    image_id: str | None,
    user_id: str | None,
    model: str | None,
    scale: str | None,
    file_name: str | None,
    image_mime: str | None,
    config: WorkerConfig,
    processor,
):
    job = None
    try:
        job = parse_job_fields(job_id, image_id, user_id, model, scale, file_name, image_mime)
        safe_log(job.job_id, "accepted")
        file_bytes = await _read_upload(file, config.max_input_bytes)

        def send_callback(cb_job_id: str, cb_image_id: str, cb_user_id: str, png_bytes: bytes):
            return send_png_result(
                callback_url=config.callback_url,
                callback_secret=config.callback_secret,
                timeout_sec=config.callback_timeout_sec,
                job_id=cb_job_id,
                image_id=cb_image_id,
                user_id=cb_user_id,
                png_bytes=png_bytes,
            )

        result = await asyncio.to_thread(
            process_job,
            job=job,
            file_bytes=file_bytes,
            processor=processor,
            send_callback=send_callback,
            tmp_dir=config.tmp_dir,
            max_input_bytes=config.max_input_bytes,
            max_input_pixels=config.max_input_pixels,
        )
        return _completed_response(result)
    except JobValidationError as exc:
        safe_log(job.job_id if job else None, f"rejected reason={exc.reason}")
        return _failed_response(400)
    except ProcessingError:
        return _failed_response(502)
    except Exception as exc:
        safe_log(job.job_id if job else None, f"failed type={type(exc).__name__}")
        return _failed_response(502)


@app.post("/runsync")
@app.post("/{endpoint_id}/runsync")
async def runsync(
    file: Annotated[UploadFile | None, File(default=None)] = None,
    job_id: Annotated[str | None, Form()] = None,
    image_id: Annotated[str | None, Form()] = None,
    user_id: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    scale: Annotated[str | None, Form()] = None,
    file_name: Annotated[str | None, Form()] = None,
    image_mime: Annotated[str | None, Form()] = None,
    _: None = Depends(require_auth),
    config: WorkerConfig = Depends(get_config),
):
    return await handle_enhance(
        file=file,
        job_id=job_id,
        image_id=image_id,
        user_id=user_id,
        model=model,
        scale=scale,
        file_name=file_name,
        image_mime=image_mime,
        config=config,
        processor=get_processor(),
    )
