from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_no_cpu_fallback_in_processor():
    text = _read("processor.py")
    assert "half=True" in text
    assert "half=torch.cuda.is_available()" not in text
    assert "assert_cuda_ready()" in text
    assert "device='cpu'" not in text
    assert 'device="cpu"' not in text
    assert text.index("apply_torchvision_shim()") < text.index("from realesrgan import")
    assert text.index("apply_torchvision_shim()") < text.index("basicsr.archs")


def test_cuda_required_has_no_cpu_path():
    text = _read("cuda_compat.py")
    assert 'raise RuntimeError("CUDA is required")' in text
    assert "torch.device('cpu')" not in text
    assert 'torch.device("cpu")' not in text


def test_entrypoint_is_runpod_serverless():
    handler = _read("handler.py")
    assert 'runpod.serverless.start({"handler": handler})' in handler
    dockerfile = _read("Dockerfile")
    assert 'CMD ["python3", "-u", "/app/handler.py"]' in dockerfile


def test_image_url_contract_unchanged():
    validation = _read("validation.py")
    handler = _read("handler.py")
    pipeline = _read("pipeline.py")
    assert 'data.get("image_url")' in validation
    assert "parsed.image_url" in handler
    assert "job.image_url" in pipeline


def test_callback_contract_unchanged():
    callback = _read("callback.py")
    config = _read("config.py")
    assert "CALLBACK_URL" in config
    assert "CALLBACK_SECRET" in config
    assert "x-kanonico-result-secret" in callback
    assert "Authorization" in callback


def test_no_fastapi_multipart_s3_or_base64_protocol():
    names = [
        "handler.py",
        "pipeline.py",
        "downloader.py",
        "validation.py",
        "callback.py",
        "processor.py",
        "config.py",
        "Dockerfile",
    ]
    blob = "\n".join(_read(name) for name in names).lower()
    assert "fastapi" not in blob
    assert "uvicorn" not in blob
    assert "python-multipart" not in blob
    assert "boto3" not in blob
    assert "s3://" not in blob
    assert "base64" not in blob
    assert "from fastapi" not in blob


def test_dockerfile_uses_cu128_and_no_pypi_torch():
    docker = _read("Dockerfile")
    assert "FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime" in docker
    assert "https://download.pytorch.org/whl/cu128" in docker
    assert "torch==2.9.1+cu128" in docker
    assert "torchvision==0.24.1+cu128" in docker
    assert "opencv-python-headless==4.10.0.84" in docker
    assert "basicsr==1.4.2" in docker
    assert "realesrgan==0.3.0" in docker
    assert "--no-deps" in docker
    assert "sm_120" in docker
    assert "numpy==1.26.4" in docker
    lines = [line for line in docker.splitlines() if "opencv-python==" in line]
    assert lines == []
