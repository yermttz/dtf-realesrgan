import pytest
import requests

from downloader import DownloadError, assert_safe_https_url, download_to_file
from tests.helpers import FakeResponse, jpeg_bytes, private_resolver, public_resolver
from validation import JobValidationError


def _download(url, dest, http_get, max_bytes=20 * 1024 * 1024):
    return download_to_file(
        url,
        dest,
        max_bytes=max_bytes,
        connect_timeout_sec=5,
        read_timeout_sec=5,
        max_redirects=3,
        allowed_hosts=frozenset(),
        http_get=http_get,
        resolver=public_resolver,
    )


def test_valid_https_url_with_public_dns():
    assert_safe_https_url("https://kanonico.example/uploads/a.jpg", resolver=public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "http://kanonico.example/a.jpg",
        "file:///etc/passwd",
        "ftp://kanonico.example/a.jpg",
    ],
)
def test_non_https_rejected(url):
    with pytest.raises(JobValidationError, match="blocked image_url"):
        assert_safe_https_url(url, resolver=public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost/a.jpg",
        "https://127.0.0.1/a.jpg",
        "https://0.0.0.0/a.jpg",
        "https://[::1]/a.jpg",
    ],
)
def test_localhost_rejected(url):
    with pytest.raises(JobValidationError, match="blocked image_url"):
        assert_safe_https_url(url, resolver=public_resolver)


@pytest.mark.parametrize(
    "url",
    [
        "https://10.0.0.4/a.jpg",
        "https://192.168.1.20/a.jpg",
        "https://172.16.5.5/a.jpg",
        "https://169.254.169.254/latest/meta-data",
    ],
)
def test_private_ip_rejected(url):
    with pytest.raises(JobValidationError, match="blocked image_url"):
        assert_safe_https_url(url, resolver=public_resolver)


def test_dns_to_private_ip_rejected():
    with pytest.raises(JobValidationError, match="blocked image_url"):
        assert_safe_https_url("https://evil.example/a.jpg", resolver=private_resolver)


def test_allowed_hosts_enforced():
    with pytest.raises(JobValidationError, match="blocked image_url"):
        assert_safe_https_url(
            "https://other.example/a.jpg",
            allowed_hosts=frozenset({"kanonico.example"}),
            resolver=public_resolver,
        )


def test_download_success_streams_to_file(tmp_path):
    body = jpeg_bytes(32, 24)
    dest = str(tmp_path / "in.jpg")

    def http_get(url, **kwargs):
        assert kwargs.get("stream") is True
        assert kwargs.get("allow_redirects") is False
        return FakeResponse(200, body)

    size = _download("https://kanonico.example/a.jpg", dest, http_get)
    assert size == len(body)
    assert open(dest, "rb").read() == body


def test_download_http_404(tmp_path):
    def http_get(url, **kwargs):
        return FakeResponse(404, b"missing")

    with pytest.raises(DownloadError) as exc:
        _download("https://kanonico.example/a.jpg", str(tmp_path / "in.jpg"), http_get)
    assert exc.value.reason == "http_404"


def test_download_http_500(tmp_path):
    def http_get(url, **kwargs):
        return FakeResponse(500, b"err")

    with pytest.raises(DownloadError) as exc:
        _download("https://kanonico.example/a.jpg", str(tmp_path / "in.jpg"), http_get)
    assert exc.value.reason == "http_500"


def test_download_timeout(tmp_path):
    def http_get(url, **kwargs):
        raise requests.Timeout("read timed out")

    with pytest.raises(DownloadError) as exc:
        _download("https://kanonico.example/a.jpg", str(tmp_path / "in.jpg"), http_get)
    assert exc.value.reason == "timeout"


def test_download_too_large_content_length(tmp_path):
    def http_get(url, **kwargs):
        return FakeResponse(200, b"abc", headers={"Content-Length": "99999999"})

    with pytest.raises(DownloadError) as exc:
        _download("https://kanonico.example/a.jpg", str(tmp_path / "in.jpg"), http_get, max_bytes=10)
    assert exc.value.reason == "too_large"


def test_download_too_large_stream(tmp_path):
    def http_get(url, **kwargs):
        return FakeResponse(200, b"x" * 50)

    with pytest.raises(DownloadError) as exc:
        _download("https://kanonico.example/a.jpg", str(tmp_path / "in.jpg"), http_get, max_bytes=10)
    assert exc.value.reason == "too_large"
