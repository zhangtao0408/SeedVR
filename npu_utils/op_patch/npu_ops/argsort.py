import torch
import torch_npu
from npu_utils.op_patch.utils import ORIGINALS

def patch_argsort() -> None:
    """
    ISSUE: https://github.com/zhangtao0408/SeedVR/issues/8
    作用：在不支持 fp32 的场景下降级到 float，再转回原 dtype。
    """
    if "argsort" not in ORIGINALS:
        ORIGINALS["argsort"] = torch.argsort

    def _argsort(input, dim: int = -1, descending: bool = False, stable: bool = False):
        original_dtype = input.dtype
        if original_dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
        ):
            input = input.float()

        # 兼容不同版本的 torch.argsort 签名
        try:
            out = ORIGINALS["argsort"](input, dim=dim, descending=descending, stable=stable)  # type: ignore[arg-type]
        except TypeError:
            out = ORIGINALS["argsort"](input, dim=dim, descending=descending)

        return out.to(original_dtype)

    torch.argsort = _argsort