from cuda_compat import has_sm120, is_blackwell_capability, missing_blackwell_kernels


def test_has_sm120_accepts_sm_and_compute_tags():
    assert has_sm120(["sm_90", "sm_120"])
    assert has_sm120(["compute_120"])
    assert has_sm120(["sm_120a"])
    assert not has_sm120(["sm_50", "sm_60", "sm_90"])
    assert not has_sm120([])


def test_blackwell_is_compute_12():
    assert is_blackwell_capability((12, 0))
    assert is_blackwell_capability((12, 1))
    assert not is_blackwell_capability((8, 6))
    assert not is_blackwell_capability((8, 9))
    assert not is_blackwell_capability((9, 0))


def test_missing_kernels_only_when_blackwell_without_sm120():
    old_arches = ["sm_50", "sm_60", "sm_61", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
    new_arches = old_arches + ["sm_100", "sm_120"]
    assert missing_blackwell_kernels((12, 0), old_arches)
    assert not missing_blackwell_kernels((12, 0), new_arches)
    assert not missing_blackwell_kernels((8, 6), old_arches)
    assert not missing_blackwell_kernels((8, 9), new_arches)
