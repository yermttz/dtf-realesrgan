from __future__ import annotations

import re

_REDACT_KEYS = re.compile(
    r"(authorization|api[_-]?key|secret|token|bearer|password)",
    re.IGNORECASE,
)


def safe_log(job_id: str | None, message: str) -> None:
    prefix = f"[AI] job={job_id} " if job_id else "[AI] "
    print(f"{prefix}{message}", flush=True)


def redact(value: str) -> str:
    if not value:
        return ""
    if _REDACT_KEYS.search(value):
        return "[redacted]"
    return value
