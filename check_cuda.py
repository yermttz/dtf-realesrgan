"""Manual CUDA probe for Blackwell (sm_120). Run inside the GPU container."""

from __future__ import annotations

import torch

from cuda_compat import assert_cuda_ready, has_sm120, is_blackwell_capability


def main() -> int:
    assert_cuda_ready()
    cap = torch.cuda.get_device_capability(0)
    arch_list = list(torch.cuda.get_arch_list())
    print("cuda_available", torch.cuda.is_available())
    print("torch", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("device", torch.cuda.get_device_name(0))
    print("capability", cap)
    print("arch_list", arch_list)
    if is_blackwell_capability(cap):
        assert cap[0] == 12 and cap[1] == 0, cap
        assert has_sm120(arch_list), arch_list
        print("sm_120_ok")
    else:
        print("sm_120_not_required_for_this_gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
