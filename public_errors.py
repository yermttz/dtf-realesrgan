"""Public-facing error text. Never include infrastructure details."""

PUBLIC_FAIL_MESSAGE = "No fue posible procesar la imagen."

# Substrings that must never appear in HTTP bodies returned to callers.
_INFRA_MARKERS = (
    "runpod",
    "real-esrgan",
    "realesrgan",
    "gpu",
    "cuda",
    "worker",
    "endpoint",
    "provider",
    "docker",
    "hostname",
    "traceback",
    "pytorch",
    "basicsr",
        "uvicorn",
        "fastapi",
)


def is_safe_public_text(text: str) -> bool:
    lowered = text.lower()
    return not any(marker in lowered for marker in _INFRA_MARKERS)


def public_error_payload(message: str | None = None) -> dict:
    text = message if message and is_safe_public_text(message) else PUBLIC_FAIL_MESSAGE
    return {"success": False, "error": text}
