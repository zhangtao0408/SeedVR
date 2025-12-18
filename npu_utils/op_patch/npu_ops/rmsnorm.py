import torch
import torch_npu
from npu_utils.op_patch.utils import ORIGINALS

def patch_rmsnorm() -> None:
    """
    ISSUE: https://github.com/zhangtao0408/SeedVR/issues/11
    作用：RMSNorm 前向调用 NPU 实现；必要时保留 bias。
    """
    from torch import nn
    try:
        from diffusers.models.normalization import RMSNorm
    except Exception as exc:  # pragma: no cover - diffusers 缺失时显式抛出
        raise RuntimeError("需要 diffusers 才能应用 RMSNorm patch") from exc

    if "RMSNorm.forward" not in ORIGINALS:
        ORIGINALS["RMSNorm.__init__"] = RMSNorm.__init__
        ORIGINALS["RMSNorm.forward"] = RMSNorm.forward

    def _rmsnorm_init(self, *args, **kwargs):
        ORIGINALS["RMSNorm.__init__"](self, *args, **kwargs)
        if self.weight is None:
            self.weight = torch.ones(kwargs['dim'], dtype=torch.bfloat16, device="npu")

    def _rmsnorm_npu_forward(self, hidden_states):
        if self.weight is not None and self.weight.dtype in (torch.float16, torch.bfloat16):
            hidden_states = hidden_states.to(self.weight.dtype)

        hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.eps)[0]
        
        # 兼容旧版 diffusion 中 rmsnorm 无 bias 参数的情况
        if hasattr(self, 'bias') and self.bias is not None:
            hidden_states = hidden_states + self.bias
        return hidden_states

    RMSNorm.__init__ = _rmsnorm_init
    RMSNorm.forward = _rmsnorm_npu_forward