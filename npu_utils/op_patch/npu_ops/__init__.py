from npu_utils.op_patch.npu_ops.rmsnorm import patch_rmsnorm
from npu_utils.op_patch.npu_ops.argsort import patch_argsort

__all__ = [
    "patch_rmsnorm",
    "patch_argsort",
]