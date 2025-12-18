"""
面向 NPU 的常用性能 patch 工具。

使用示例：
    from codes.SeedVR.npu_utils.op_patch import apply_patches, list_patches
    print(list_patches())                       # 查看可用 patch
    apply_patches()                             # 应用全部 patch
    apply_patches(selected=["argsort"])         # 只应用指定 patch
    apply_patches(skip=["rmsnorm"])             # 跳过指定 patch
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

import torch
import torch_npu

from npu_utils.op_patch.utils import PATCHES, APPLIED, ORIGINALS
from npu_utils.op_patch.utils import op_register

logger = logging.getLogger(__name__)

def list_patches() -> List[Dict[str, str]]:
    """列出可用 patch 及其说明。"""
    return [
        {"name": p.name, "issue": p.issue, "summary": p.summary}
        for p in PATCHES.values()
    ]


def apply_patches(
    selected: Optional[Sequence[str]] = None,
    skip: Optional[Iterable[str]] = None,
    strict: bool = True,
    verbose: bool = False,
) -> None:
    """
    应用 NPU patch。

    - selected: 仅应用指定名称；None 表示全部。
    - skip: 跳过的 patch 名称。
    - strict: True 时任一 patch 失败直接抛错；False 时记录警告继续。
    - verbose: True 时输出简单的日志到 logger。
    """
    desired = set(selected) if selected else set(PATCHES.keys())
    if skip:
        desired -= set(skip)

    for name in desired:
        if name not in PATCHES:
            msg = f"未知 patch: {name}"
            if strict:
                raise KeyError(msg)
            logger.warning(msg)
            continue

        if name in APPLIED:
            if verbose:
                logger.info("patch %s 已应用，跳过", name)
            continue

        patch = PATCHES[name]
        try:
            patch.apply_fn()
            APPLIED.add(name)
            if verbose:
                logger.info("patch %s 应用完成，issue=%s", name, patch.issue)
        except Exception as exc:
            msg = f"patch {name} 应用失败: {exc}"
            if strict:
                raise
            logger.warning(msg)