from npu_utils.op_patch.utils import register, PATCHES, NpuPatch
from npu_utils.op_patch.npu_ops import patch_rmsnorm, patch_argsort, patch_group_norm

register(
    NpuPatch(
        name="rmsnorm",
        issue="https://github.com/zhangtao0408/SeedVR/issues/11",
        summary="RMSNorm 调用 NPU 融合算子（torch_npu.npu_rms_norm）",
        apply_fn=patch_rmsnorm,
    )
)

register(
    NpuPatch(
        name="argsort",
        issue="https://github.com/zhangtao0408/SeedVR/issues/8",
        summary="argsort 对不支持的 dtype 先降精度，再转回原 dtype（torch.argsort）",
        apply_fn=patch_argsort,
    )
)

register(
    NpuPatch(
        name="group_norm",
        issue="https://github.com/zhangtao0408/SeedVR/issues/9",
        summary="group_norm 调用 NPU 融合算子，缺参时回退原始实现（torch.nn.functional.group_norm）",
        apply_fn=patch_group_norm,
    )
)
