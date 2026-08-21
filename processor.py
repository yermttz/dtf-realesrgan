"""Real-ESRGAN upscaler. Processing settings match the previous worker."""

from __future__ import annotations

import os
import threading
from typing import Protocol

from torchvision_compat import apply_torchvision_shim

apply_torchvision_shim()

import numpy as np
import torch
from PIL import Image

from cuda_compat import assert_cuda_ready
from model_weights import resolve_model_path

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:  # pragma: no cover - package path varies by version
    from basicsr.archs.rrdbnet_archs import RRDBNet
from realesrgan import RealESRGANer


class ImageProcessor(Protocol):
    def enhance(self, input_path: str, output_path: str, model: str, scale: int) -> tuple[int, int, int, int]:
        """Process input_path into a PNG at output_path. Returns input/output sizes."""


class RealEsrganProcessor:
    def __init__(self, weights_dir: str, tile_size: int = 128):
        assert_cuda_ready()
        self.weights_dir = weights_dir
        self.tile_size = tile_size
        self._lock = threading.Lock()
        self._upsamplers: dict[str, RealESRGANer] = {}

    def enhance(self, input_path: str, output_path: str, model: str, scale: int) -> tuple[int, int, int, int]:
        with Image.open(input_path) as img:
            rgb = img.convert("RGB")
            width, height = rgb.size
            img_np = np.array(rgb)

        with self._lock:
            upsampler = self._get_upsampler(model)
            output, _ = upsampler.enhance(img_np, outscale=scale)

        result_img = Image.fromarray(output)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        result_img.save(output_path, "PNG")
        out_w, out_h = result_img.size
        return width, height, out_w, out_h

    def _get_upsampler(self, model: str) -> RealESRGANer:
        cached = self._upsamplers.get(model)
        if cached is not None:
            return cached

        if model == "anime":
            model_arch = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
            )
        else:
            model_arch = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4
            )

        model_path = resolve_model_path(self.weights_dir, model)
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model_arch,
            tile=self.tile_size,
            tile_pad=10,
            pre_pad=0,
            half=True,
        )
        self._upsamplers[model] = upsampler
        return upsampler
