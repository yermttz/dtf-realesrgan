from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def test_cuda_kernel_probe_fp16():
    from cuda_compat import assert_cuda_ready, has_sm120, is_blackwell_capability

    assert_cuda_ready()
    device = torch.device("cuda:0")
    cap = torch.cuda.get_device_capability(0)
    arch_list = list(torch.cuda.get_arch_list())
    x = torch.randn(32, 32, device=device, dtype=torch.float16)
    y = x @ x
    torch.cuda.synchronize()
    assert y.shape == (32, 32)
    if is_blackwell_capability(cap):
        assert cap == (12, 0)
        assert has_sm120(arch_list)


def test_realesrgan_normal_runs_on_gpu_and_scales_4x(tmp_path):
    pytest.importorskip("realesrgan")
    pytest.importorskip("basicsr")
    from PIL import Image

    from processor import RealEsrganProcessor

    weights_dir = Path(os.environ.get("WEIGHTS_DIR") or "/app/weights")
    if not (weights_dir / "RealESRGAN_x4plus.pth").is_file():
        env_dir = Path("/app/weights")
        if (env_dir / "RealESRGAN_x4plus.pth").is_file():
            weights_dir = env_dir
        else:
            pytest.skip("RealESRGAN_x4plus.pth is not present")

    src = tmp_path / "in.png"
    dst = tmp_path / "out.png"
    Image.new("RGB", (16, 12), (12, 80, 160)).save(src)

    processor = RealEsrganProcessor(str(weights_dir), tile_size=128)
    in_w, in_h, out_w, out_h = processor.enhance(str(src), str(dst), "normal", 4)
    upsampler = processor._upsamplers["normal"]
    param = next(upsampler.model.parameters())

    assert param.device.type == "cuda"
    assert (in_w, in_h) == (16, 12)
    assert (out_w, out_h) == (64, 48)
    with Image.open(dst) as result:
        assert result.size == (64, 48)
