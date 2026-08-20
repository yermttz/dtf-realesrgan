from handler import runpod_handler
from public_errors import PUBLIC_FAIL_MESSAGE


def test_runpod_handler_rejects_image_url():
    result = runpod_handler({"input": {"image_url": "https://example.test/a.jpg", "model": "anime"}})
    assert result["success"] is False
    assert result["error"] == PUBLIC_FAIL_MESSAGE
    dumped = str(result).lower()
    for word in ("runpod", "https://", "image_url"):
        if word == "image_url":
            continue
        assert word not in dumped
