from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")


def test_assert_cuda_ready_rejects_cpu():
    from cuda_compat import assert_cuda_ready

    with patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA is required"):
            assert_cuda_ready()


def test_assert_cuda_ready_rejects_blackwell_without_sm120():
    from cuda_compat import assert_cuda_ready

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA RTX PRO 6000"),
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(torch.cuda, "get_arch_list", return_value=["sm_80", "sm_86", "sm_90"]),
    ):
        with pytest.raises(RuntimeError, match="CUDA is required"):
            assert_cuda_ready()


def test_assert_cuda_ready_maps_missing_kernel_image():
    from cuda_compat import assert_cuda_ready

    def _boom(*_args, **_kwargs):
        raise RuntimeError("CUDA error: no kernel image is available for execution on the device")

    with (
        patch.object(torch.cuda, "is_available", return_value=True),
        patch.object(torch.cuda, "get_device_name", return_value="NVIDIA RTX PRO 6000"),
        patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        patch.object(torch.cuda, "get_arch_list", return_value=["sm_100", "sm_120"]),
        patch.object(torch, "randn", side_effect=_boom),
    ):
        with pytest.raises(RuntimeError, match="CUDA is required"):
            assert_cuda_ready()
