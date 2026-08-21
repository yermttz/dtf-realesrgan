"""Shim for BasicSR 1.4.2 on torchvision >= 0.17 (functional_tensor was removed)."""

from __future__ import annotations

import sys
import types


def apply_torchvision_shim() -> None:
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return
    from torchvision.transforms.functional import rgb_to_grayscale

    mod = types.ModuleType("torchvision.transforms.functional_tensor")
    mod.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = mod
