from npu_utils.op_patch.npu_ops.rmsnorm import patch_rmsnorm
from npu_utils.op_patch.npu_ops.argsort import patch_argsort
from npu_utils.op_patch.npu_ops.group_norm import patch_group_norm

__all__ = [
    "patch_rmsnorm",
    "patch_argsort",
    "patch_group_norm",
]