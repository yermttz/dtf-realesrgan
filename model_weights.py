"""Local Real-ESRGAN checkpoint paths. Production never uses GitHub URLs."""

from __future__ import annotations

import os

ANIME_WEIGHTS_FILE = "RealESRGAN_x4plus_anime_6B.pth"
NORMAL_WEIGHTS_FILE = "RealESRGAN_x4plus.pth"


def weights_filename(model: str) -> str:
    if model == "anime":
        return ANIME_WEIGHTS_FILE
    return NORMAL_WEIGHTS_FILE


def resolve_model_path(weights_dir: str, model: str) -> str:
    path = os.path.join(weights_dir, weights_filename(model))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"missing weights file: {path}")
    return path
