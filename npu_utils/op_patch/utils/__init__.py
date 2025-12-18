from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

@dataclass(frozen=True)
class NpuPatch:
    name: str
    issue: str
    summary: str
    apply_fn: Callable[[], None]

def register(patch: NpuPatch) -> None:
    PATCHES[patch.name] = patch

PATCHES: Dict[str, NpuPatch] = {}
APPLIED: Set[str] = set()
ORIGINALS: Dict[str, Callable[..., object]] = {}