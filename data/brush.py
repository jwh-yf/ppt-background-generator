"""模型笔触。用 SD 生成的表面纹理给学科线条加一点墨色不匀。"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BRUSH_DIR = ROOT / "data" / "brush"

DISCIPLINE_BRUSH: Dict[str, str] = {
    "math": "chalk",
    "mech": "blueprint",
    "philo": "paper",
}

_CACHE: Dict[Tuple[str, int, int, int], np.ndarray] = {}


def _load_tile(name: str) -> Optional[Image.Image]:
    p = BRUSH_DIR / f"{name}.png"
    if not p.is_file():
        return None
    return Image.open(p).convert("L")


def _tile_to(w: int, h: int, tile: Image.Image, seed: int) -> np.ndarray:
    # 512 的块按 1:1 平铺，奇偶块镜像，接缝不会露。
    # 试过放大 3.75 倍铺满：颗粒变成 4px，2-3px 的制图线被压成虚线，读起来就是错的。
    rng = np.random.default_rng(int(hashlib.sha256(str(seed).encode()).hexdigest()[:16], 16))
    arr = np.asarray(tile, dtype=np.float32) / 255.0
    th, tw = arr.shape[:2]
    flip0 = bool(rng.random() < 0.5)
    ty, tx = np.indices((h, w))
    by = (ty // th) % 2 == 1
    bx = (tx // tw) % 2 == 1
    if flip0:
        bx = ~bx
    yy = ty % th
    xx = tx % tw
    yy = np.where(by, th - 1 - yy, yy)
    xx = np.where(bx, tw - 1 - xx, xx)
    return arr[yy, xx]


def grain(w: int, h: int, discipline: str, seed: int, contrast: float = 0.35) -> Optional[np.ndarray]:
    """(h,w,1) 的乘性颗粒场，均值 1。"""
    # TODO 只在 1-4 px 的线宽上验过。剖面块、大色块这类面状填充没单独看过，
    #      当前学科元素都落在这个线宽区间里，所以暂时不影响出图。
    name = DISCIPLINE_BRUSH.get(discipline, "paper")
    key = (name, w, h, int(seed) & 0xFFFF, round(float(contrast), 3))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    tile = _load_tile(name)
    if tile is None:
        return None
    t = _tile_to(w, h, tile, seed)
    # 先减均值再归一化，纹理只改不匀、不改整体亮度，否则整张会偏灰
    t = t - float(t.mean())
    sd = float(t.std()) or 1.0
    t = 1.0 + (t / sd) * (0.18 * contrast * 3.0)
    # 下界卡在 0.82：线只有 2-3px，压到 0.5 就会出现断口，制图上断口=虚线，意思变了
    t = np.clip(t, 0.82, 1.30)[..., None].astype(np.float32)
    _CACHE[key] = t
    return t


def apply_ink(before: np.ndarray, after: np.ndarray, discipline: str, seed: int,
              contrast: float = 0.35) -> np.ndarray:
    """把这一笔画出来的那一层（after-before）乘上颗粒场，几何不动。"""
    g = grain(after.shape[1], after.shape[0], discipline, seed, contrast)
    if g is None:
        return after
    delta = after - before
    return np.clip(before + delta * g, 0.0, 255.0).astype(np.float32)
