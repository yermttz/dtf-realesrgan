from pathlib import Path

import pytest

from model_weights import resolve_model_path, weights_filename


def test_anime_resolves_local_weights_path(tmp_path):
    path = tmp_path / "RealESRGAN_x4plus_anime_6B.pth"
    path.write_bytes(b"ckpt")
    assert resolve_model_path(str(tmp_path), "anime") == str(path)


def test_normal_resolves_local_weights_path(tmp_path):
    path = tmp_path / "RealESRGAN_x4plus.pth"
    path.write_bytes(b"ckpt")
    assert resolve_model_path(str(tmp_path), "normal") == str(path)


def test_missing_weights_fail_fast(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing weights file"):
        resolve_model_path(str(tmp_path), "anime")
    with pytest.raises(FileNotFoundError, match="missing weights file"):
        resolve_model_path(str(tmp_path), "normal")


def test_production_weight_filenames_under_app_weights():
    assert weights_filename("anime") == "RealESRGAN_x4plus_anime_6B.pth"
    assert weights_filename("normal") == "RealESRGAN_x4plus.pth"
    assert str(Path("/app/weights") / weights_filename("anime")).replace("\\", "/") == (
        "/app/weights/RealESRGAN_x4plus_anime_6B.pth"
    )
    assert str(Path("/app/weights") / weights_filename("normal")).replace("\\", "/") == (
        "/app/weights/RealESRGAN_x4plus.pth"
    )
