from npu_utils.op_patch.utils import register, PATCHES, NpuPatch
from npu_utils.op_patch.npu_ops import patch_rmsnorm

register(
    NpuPatch(
        name="rmsnorm",
        issue="https://github.com/zhangtao0408/SeedVR/issues/11",
        summary="RMSNorm 调用 NPU 融合算子（torch_npu.npu_rms_norm）",
        apply_fn=patch_rmsnorm,
    )
)