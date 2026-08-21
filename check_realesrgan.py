"""Minimal Real-ESRGAN GPU check. Uses existing WEIGHTS_DIR checkpoints."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import torch
from PIL import Image

from processor import RealEsrganProcessor


def main() -> int:
    weights_dir = os.environ.get("WEIGHTS_DIR") or "/app/weights"
    local = Path(weights_dir) / "RealESRGAN_x4plus.pth"
    if not local.is_file():
        raise SystemExit(f"missing checkpoint: {local}")

    processor = RealEsrganProcessor(weights_dir, tile_size=128)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.png"
        dst = Path(tmp) / "out.png"
        Image.new("RGB", (16, 12), (12, 80, 160)).save(src)
        in_w, in_h, out_w, out_h = processor.enhance(str(src), str(dst), "normal", 4)

    param = next(processor._upsamplers["normal"].model.parameters())
    if param.device.type != "cuda":
        raise SystemExit("Real-ESRGAN is not on CUDA")
    if (in_w, in_h, out_w, out_h) != (16, 12, 64, 48):
        raise SystemExit(f"unexpected scale: {(in_w, in_h, out_w, out_h)}")
    print("realesrgan_gpu_ok", torch.cuda.get_device_name(0), "4x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
