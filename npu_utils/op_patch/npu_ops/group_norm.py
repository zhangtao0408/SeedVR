import torch
from torch.nn import functional as F
from npu_utils.op_patch.utils import ORIGINALS

def patch_group_norm() -> None:
    """
    ISSUE: https://github.com/zhangtao0408/SeedVR/issues/9
    作用：优先走 NPU 特化实现，必要时回退原始实现。
    """

    if "group_norm" not in ORIGINALS:
        ORIGINALS["group_norm"] = F.group_norm

    # autocast disable
    @torch.amp.autocast('npu', enabled=False)
    def _group_norm(x, num_groups, weight=None, bias=None, eps: float = 1e-5):
        return ORIGINALS["group_norm"](x, num_groups, weight, bias, eps)

    F.group_norm = _group_norm