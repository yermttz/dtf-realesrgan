"""AI worker entrypoint.

HTTP multipart is the production contract (Kanonico DTF).
Classic queue-based serverless remains available when the job-queue env is set.
"""

from __future__ import annotations

import os

from public_errors import PUBLIC_FAIL_MESSAGE, public_error_payload


def runpod_handler(job: dict) -> dict:
    """Queue-based jobs no longer accept image_url. Multipart HTTP is required."""
    data = job.get("input") or {}
    if "image_url" in data:
        print("[AI] rejected image_url input", flush=True)
    return {
        "success": False,
        "error": PUBLIC_FAIL_MESSAGE,
        **public_error_payload(),
    }


def main() -> None:
    if os.environ.get("RUNPOD_WEBHOOK_GET_JOB"):
        import runpod

        runpod.serverless.start({"handler": runpod_handler})
        return

    import uvicorn

    from config import load_config

    cfg = load_config()
    uvicorn.run("http_app:app", host="0.0.0.0", port=cfg.port, log_level="info")


if __name__ == "__main__":
    main()
