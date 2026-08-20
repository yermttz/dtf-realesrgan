import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class WorkerConfig:
    auth_token: str
    callback_url: str
    callback_secret: str
    max_input_bytes: int
    max_input_pixels: int
    max_output_bytes: int
    tmp_dir: str
    weights_dir: str
    tile_size: int
    port: int
    callback_timeout_sec: int


def load_config() -> WorkerConfig:
    auth = (
        os.environ.get("WORKER_AUTH_TOKEN")
        or os.environ.get("AI_API_KEY")
        or os.environ.get("RUNPOD_API_KEY")
        or ""
    ).strip()
    callback_url = (
        os.environ.get("CALLBACK_URL")
        or os.environ.get("KANONICO_CALLBACK_URL")
        or ""
    ).strip()
    callback_secret = (
        os.environ.get("CALLBACK_SECRET")
        or os.environ.get("AI_RESULT_SECRET")
        or os.environ.get("RUNPOD_RESULT_SECRET")
        or ""
    ).strip()
    max_input_mb = _int_env("MAX_INPUT_MB", _int_env("AI_MAX_INPUT_MB", 20))
    max_output_mb = _int_env("MAX_OUTPUT_MB", _int_env("AI_MAX_OUTPUT_MB", 80))
    return WorkerConfig(
        auth_token=auth,
        callback_url=callback_url,
        callback_secret=callback_secret,
        max_input_bytes=max(1, max_input_mb) * 1024 * 1024,
        max_input_pixels=_int_env("MAX_INPUT_PIXELS", _int_env("AI_MAX_INPUT_PIXELS", 8_000_000)),
        max_output_bytes=max(1, max_output_mb) * 1024 * 1024,
        tmp_dir=os.environ.get("TMP_DIR") or os.environ.get("TEMP") or "/tmp",
        weights_dir=os.environ.get("WEIGHTS_DIR") or "/app/weights",
        tile_size=max(0, _int_env("TILE_SIZE", 128)),
        port=_int_env("PORT", 8000),
        callback_timeout_sec=max(5, _int_env("CALLBACK_TIMEOUT_SEC", 120)),
    )
