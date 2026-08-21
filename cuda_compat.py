"""CUDA capability checks. Logs stay internal; do not send these strings to clients."""

from __future__ import annotations

from logging_utils import safe_log


def has_sm120(arch_list: list[str]) -> bool:
    for arch in arch_list:
        name = arch.replace("compute_", "sm_")
        if name == "sm_120" or name.startswith("sm_120"):
            return True
    return False


def is_blackwell_capability(capability: tuple[int, int]) -> bool:
    return capability[0] >= 12


def missing_blackwell_kernels(capability: tuple[int, int], arch_list: list[str]) -> bool:
    return is_blackwell_capability(capability) and not has_sm120(arch_list)


def assert_cuda_ready() -> None:
    import torch

    safe_log(None, f"torch={torch.__version__}")
    safe_log(None, f"torch.version.cuda={torch.version.cuda}")
    if not torch.cuda.is_available():
        safe_log(None, "cuda_available=false")
        raise RuntimeError("CUDA is required")

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    arch_list = list(torch.cuda.get_arch_list())
    safe_log(None, f"cuda_available=true")
    safe_log(None, f"device={name}")
    safe_log(None, f"capability={cap[0]}.{cap[1]}")
    safe_log(None, f"arch_list={arch_list}")

    if missing_blackwell_kernels(cap, arch_list):
        safe_log(None, "missing sm_120 kernels for this GPU")
        raise RuntimeError("CUDA is required")

    try:
        x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
        y = x @ x
        torch.cuda.synchronize()
        _ = float(y.mean().detach().cpu())
    except Exception as exc:
        safe_log(None, f"cuda kernel probe failed type={type(exc).__name__}")
        raise RuntimeError("CUDA is required") from exc

    safe_log(None, "cuda kernel probe ok")
