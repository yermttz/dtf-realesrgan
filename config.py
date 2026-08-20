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


def _host_set(raw: str) -> frozenset[str]:
    hosts = []
    for part in raw.split(","):
        host = part.strip().lower().strip(".")
        if host:
            hosts.append(host)
    return frozenset(hosts)


@dataclass(frozen=True)
class WorkerConfig:
    callback_url: str
    callback_secret: str
    max_input_bytes: int
    max_input_pixels: int
    tmp_dir: str
    weights_dir: str
    tile_size: int
    callback_timeout_sec: int
    download_connect_timeout_sec: int
    download_read_timeout_sec: int
    download_max_redirects: int
    allowed_download_hosts: frozenset[str]


def load_config() -> WorkerConfig:
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
    return WorkerConfig(
        callback_url=callback_url,
        callback_secret=callback_secret,
        max_input_bytes=max(1, max_input_mb) * 1024 * 1024,
        max_input_pixels=_int_env("MAX_INPUT_PIXELS", _int_env("AI_MAX_INPUT_PIXELS", 8_000_000)),
        tmp_dir=os.environ.get("TMP_DIR") or os.environ.get("TEMP") or "/tmp",
        weights_dir=os.environ.get("WEIGHTS_DIR") or "/app/weights",
        tile_size=max(0, _int_env("TILE_SIZE", 128)),
        callback_timeout_sec=max(5, _int_env("CALLBACK_TIMEOUT_SEC", 120)),
        download_connect_timeout_sec=max(1, _int_env("DOWNLOAD_CONNECT_TIMEOUT_SEC", 10)),
        download_read_timeout_sec=max(1, _int_env("DOWNLOAD_READ_TIMEOUT_SEC", 120)),
        download_max_redirects=max(0, _int_env("DOWNLOAD_MAX_REDIRECTS", 3)),
        allowed_download_hosts=_host_set(os.environ.get("DOWNLOAD_ALLOWED_HOSTS") or ""),
    )
