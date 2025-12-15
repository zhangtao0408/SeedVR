# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

import torch

try:
    from flash_attn import flash_attn_varlen_func
except:
    flash_attn_varlen_func = None
    print('Note!!!!!! flash_attn is not avaliable!')

try:
    import torch_npu
    npu_available = True
except ImportError:
    npu_available = False

try:
    import mindiesd
    from mindiesd import attention_forward_varlen
    mindiesd_available = True
except:
    mindiesd_available = False

import torch.nn.functional as F
from torch import nn

class TorchAttention(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        assert len(args) == 0 or len(args) > 2, "query, key should both provided by args / kwargs"
        q = kwargs.get("query") or args[0]
        k = kwargs.get("key") or args[1]
        b, h, sq, d = q.shape
        b, h, sk, d = k.shape
        return b * h * (4 * d * (sq / 1e6) * (sk / 1e6))

    def forward(self, *args, **kwargs):
        return F.scaled_dot_product_attention(*args, **kwargs)


class FlashAttentionVarlen(nn.Module):
    def tflops(self, args, kwargs, output) -> float:
        cu_seqlens_q = kwargs["cu_seqlens_q"]
        cu_seqlens_k = kwargs["cu_seqlens_k"]
        _, h, d = output.shape
        seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]) / 1e6
        seqlens_k = (cu_seqlens_k[1:] - cu_seqlens_k[:-1]) / 1e6
        return h * (4 * d * (seqlens_q * seqlens_k).sum())

    def forward(self, *args, **kwargs):
        if npu_available:
            return self._forward_npu(*args, **kwargs)
        else:
            return self._forward_gpu(*args, **kwargs)

    def _forward_gpu(self, *args, **kwargs):
        kwargs["deterministic"] = torch.are_deterministic_algorithms_enabled()
        return flash_attn_varlen_func(*args, **kwargs)

    def _forward_npu(self, *args, **kwargs):
        del kwargs['max_seqlen_q']
        del kwargs['max_seqlen_k']

        if isinstance(kwargs['cu_seqlens_q'], torch.Tensor):
            kwargs['cu_seqlens_q'] = kwargs['cu_seqlens_q'].tolist()
        if isinstance(kwargs['cu_seqlens_k'], torch.Tensor):
            kwargs['cu_seqlens_k'] = kwargs['cu_seqlens_k'].tolist()

        if mindiesd_available:
            output = attention_forward_varlen(*args, **kwargs)
        else:
            _, num_heads, head_dim = kwargs['q'].shape
            atten_mask = None
            sparse_mode = 0

            output = torch_npu.npu_fusion_attention(
                query=kwargs['q'],
                key=kwargs['k'],
                value=kwargs['v'],
                head_num=num_heads,
                pse=None,
                padding_mask=None,
                atten_mask=atten_mask,
                scale=1.0 / (head_dim ** 0.5),
                keep_prob=1,
                input_layout="TND",
                actual_seq_qlen=kwargs['cu_seqlens_q'][1:],
                actual_seq_kvlen=kwargs['cu_seqlens_k'][1:],
                sparse_mode=sparse_mode,
            )[0]
        return output