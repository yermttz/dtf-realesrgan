from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urljoin, urlparse

import requests

from logging_utils import safe_log
from public_errors import PUBLIC_FAIL_MESSAGE
from validation import JobValidationError

BLOCKED_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata.google.internal",
        "metadata",
        "instance-data",
        "kubernetes",
        "kubernetes.default",
        "kubernetes.default.svc",
    }
)
BLOCKED_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".lan", ".home")
BLOCKED_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
)


class DownloadError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(PUBLIC_FAIL_MESSAGE)
        self.reason = reason


def url_host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""


def _is_blocked_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return True
    if ip.is_unspecified or ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
        return True
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return _is_blocked_ip(str(ip.ipv4_mapped))
    return any(ip in network for network in BLOCKED_NETWORKS)


def _host_allowed(hostname: str, allowed_hosts: frozenset[str]) -> bool:
    if not allowed_hosts:
        return True
    host = hostname.lower().strip(".")
    if host in allowed_hosts:
        return True
    return any(host.endswith("." + allowed) for allowed in allowed_hosts)


def assert_safe_https_url(
    url: str,
    *,
    allowed_hosts: frozenset[str] = frozenset(),
    resolver=None,
) -> str:
    resolve = resolver or socket.getaddrinfo
    if not url or not isinstance(url, str):
        raise JobValidationError("missing image_url")
    try:
        parsed = urlparse(url.strip())
    except Exception as exc:
        raise JobValidationError("invalid image_url") from exc

    if parsed.scheme.lower() != "https":
        raise JobValidationError("blocked image_url")
    if parsed.username or parsed.password:
        raise JobValidationError("blocked image_url")
    hostname = (parsed.hostname or "").lower().strip(".")
    if not hostname:
        raise JobValidationError("invalid image_url")
    if hostname in BLOCKED_HOSTS or any(hostname.endswith(sfx) for sfx in BLOCKED_HOST_SUFFIXES):
        raise JobValidationError("blocked image_url")
    if not _host_allowed(hostname, allowed_hosts):
        raise JobValidationError("blocked image_url")

    try:
        ipaddress.ip_address(hostname)
        host_is_ip = True
    except ValueError:
        host_is_ip = False

    if host_is_ip and _is_blocked_ip(hostname):
        raise JobValidationError("blocked image_url")

    try:
        infos = resolve(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise JobValidationError("blocked image_url") from exc
    ips = {item[4][0] for item in infos}
    if not ips or any(_is_blocked_ip(ip) for ip in ips):
        raise JobValidationError("blocked image_url")
    return url.strip()


def download_to_file(
    url: str,
    dest_path: str,
    *,
    max_bytes: int,
    connect_timeout_sec: int,
    read_timeout_sec: int,
    max_redirects: int,
    allowed_hosts: frozenset[str],
    job_id: str | None = None,
    http_get=None,
    resolver=None,
) -> int:
    get = http_get or requests.get
    resolve = resolver or socket.getaddrinfo
    current = assert_safe_https_url(url, allowed_hosts=allowed_hosts, resolver=resolve)
    redirects = 0
    timeout = (connect_timeout_sec, read_timeout_sec)

    while True:
        host = url_host(current)
        safe_log(job_id, f"downloading host={host}")
        try:
            response = get(
                current,
                stream=True,
                timeout=timeout,
                allow_redirects=False,
            )
        except requests.Timeout as exc:
            safe_log(job_id, "download timeout")
            raise DownloadError("timeout") from exc
        except requests.RequestException as exc:
            safe_log(job_id, f"download failed type={type(exc).__name__}")
            raise DownloadError("transport") from exc

        try:
            status = int(response.status_code)
            if status in (301, 302, 303, 307, 308):
                location = response.headers.get("Location") or ""
                if redirects >= max_redirects or not location:
                    safe_log(job_id, f"download failed status={status}")
                    raise DownloadError(f"http_{status}")
                nxt = urljoin(current, location)
                current = assert_safe_https_url(nxt, allowed_hosts=allowed_hosts, resolver=resolve)
                redirects += 1
                continue
            if status == 404:
                safe_log(job_id, "download failed status=404")
                raise DownloadError("http_404")
            if status >= 500:
                safe_log(job_id, f"download failed status={status}")
                raise DownloadError("http_500")
            if status != 200:
                safe_log(job_id, f"download failed status={status}")
                raise DownloadError(f"http_{status}")

            length_header = response.headers.get("Content-Length")
            if length_header:
                try:
                    declared = int(length_header)
                except ValueError:
                    declared = 0
                if declared > max_bytes:
                    safe_log(job_id, "download failed too-large")
                    raise DownloadError("too_large")

            total = 0
            with open(dest_path, "wb") as handle:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        handle.close()
                        safe_log(job_id, "download failed too-large")
                        raise DownloadError("too_large")
                    handle.write(chunk)
            if total <= 0:
                safe_log(job_id, "download failed empty")
                raise DownloadError("empty")
            safe_log(job_id, f"downloaded size={total}")
            return total
        finally:
            response.close()
