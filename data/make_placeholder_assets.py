"""生成程序化母图素材 —— 它们是本作品训练数据的唯一来源，不是"占位符"。"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

KINDS = (
    "gradient_blobs",
    "geometric",
    "paper",
    "stripes",
    "bokeh",
    "wave",
    "fbm_cloud",
    "fbm_silk",
)

TEXTURE_KINDS = ("paper", "stripes", "wave", "fbm_silk")

def _fbm_field(
    rng: random.Random, size: int, octaves: int = 5, persistence: float = 0.55
) -> np.ndarray:
    acc = np.zeros((size, size), dtype=np.float32)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        # o=0 是最粗的一层（振幅最大），o 越大越细、振幅越小 —— 顺序不能反，
        res = max(2, size // (2 ** (octaves + 2 - o)))
        gen = np.random.default_rng(rng.randrange(1 << 30))
        base = (gen.random((res, res)) * 255.0).astype(np.uint8)
        layer = np.asarray(
            Image.fromarray(base).resize((size, size), Image.BICUBIC), dtype=np.float32
        ) / 255.0
        acc += layer * amp
        norm += amp
        amp *= persistence
    acc /= norm
    lo, hi = np.percentile(acc, 2.0), np.percentile(acc, 98.0)
    return np.clip((acc - lo) / max(1e-6, float(hi - lo)), 0.0, 1.0)

def _muted_duo(rng: random.Random, lo_gray: float, hi_gray: float) -> tuple:
    tint = np.array([rng.uniform(-16, 16) for _ in range(3)], dtype=np.float32)
    lo = np.clip(lo_gray + tint, 0, 255)
    hi = np.clip(hi_gray + tint * 0.6, 0, 255)
    return lo, hi

def _duotone(field: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> np.ndarray:
    t = field[..., None]
    return c1[None, None, :] * (1.0 - t) + c2[None, None, :] * t

def draw_fbm_cloud(rng: random.Random, size: int) -> Image.Image:
    field = _fbm_field(rng, size, octaves=rng.randint(4, 6))
    field = 0.5 + (field - 0.5) * rng.uniform(0.45, 0.7)
    g = rng.uniform(105, 165)
    lo, hi = _muted_duo(rng, g - rng.uniform(28, 55), g + rng.uniform(30, 70))
    img = _duotone(field, lo, hi)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(size * 0.004)
    )

def draw_fbm_silk(rng: random.Random, size: int) -> Image.Image:
    field = _fbm_field(rng, size, octaves=rng.randint(3, 5))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / float(size)
    ang = rng.uniform(0, math.pi)
    axis = xx * math.cos(ang) + yy * math.sin(ang)
    bands = 0.5 + 0.5 * np.sin((field * rng.uniform(4.0, 9.0) + axis * rng.uniform(1.0, 2.6)) * math.pi)
    mixed = np.clip(0.62 * bands + 0.38 * field, 0.0, 1.0)
    mixed = 0.5 + (mixed - 0.5) * rng.uniform(0.5, 0.75)
    g = rng.uniform(100, 165)
    lo, hi = _muted_duo(rng, g - rng.uniform(25, 50), g + rng.uniform(25, 60))
    img = _duotone(mixed, lo, hi)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(size * 0.003)
    )

def draw_gradient_blobs(rng: random.Random, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = np.zeros((size, size, 3), dtype=np.float32)
    for _ in range(rng.randint(3, 6)):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(size * 0.15, size * 0.55)
        col = np.array([rng.uniform(40, 235) for _ in range(3)], dtype=np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        w = np.clip(1.0 - d / r, 0, 1) ** 2
        img += w[..., None] * col
    img = img / max(1.0, img.max() / 235.0)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(size * 0.02)
    )

def draw_geometric(rng: random.Random, size: int) -> Image.Image:
    img = Image.new("RGB", (size, size), tuple(rng.randint(200, 250) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(8, 18)):
        x0, y0 = rng.uniform(-0.2, 1.0) * size, rng.uniform(-0.2, 1.0) * size
        x1, y1 = x0 + rng.uniform(0.08, 0.45) * size, y0 + rng.uniform(0.08, 0.45) * size
        col = tuple(rng.randint(60, 220) for _ in range(3))
        if rng.random() < 0.5:
            d.rectangle((x0, y0, x1, y1), fill=col)
        else:
            d.ellipse((x0, y0, x1, y1), fill=col)
    return img.filter(ImageFilter.GaussianBlur(size * 0.004))

def draw_paper(rng: random.Random, size: int) -> Image.Image:
    base = np.array([rng.uniform(215, 245) for _ in range(3)], dtype=np.float32)
    gen = np.random.default_rng(rng.randrange(1 << 30))
    small = gen.normal(0, 1, size=(size // 6, size // 6)).astype(np.float32)
    tex = np.asarray(
        Image.fromarray(np.clip(small * 30 + 128, 0, 255).astype(np.uint8)).resize(
            (size, size), Image.BICUBIC
        ),
        dtype=np.float32,
    )
    img = base[None, None, :] + (tex - 128.0)[..., None] * 0.55
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

def draw_stripes(rng: random.Random, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    angle = rng.uniform(0, math.pi)
    freq = rng.uniform(0.01, 0.06)
    s = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * freq)
    c1 = np.array([rng.uniform(30, 120) for _ in range(3)], dtype=np.float32)
    c2 = c1 + rng.uniform(40, 120)
    t = (s * 0.5 + 0.5)[..., None]
    img = c1[None, None, :] * (1 - t) + c2[None, None, :] * t
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

def draw_bokeh(rng: random.Random, size: int) -> Image.Image:
    img = np.zeros((size, size, 3), dtype=np.float32)
    col_base = np.array([rng.uniform(30, 120) for _ in range(3)], dtype=np.float32)
    img += col_base[None, None, :]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    for _ in range(rng.randint(12, 26)):
        cx, cy = rng.uniform(0, size), rng.uniform(0, size)
        r = rng.uniform(size * 0.03, size * 0.14)
        col = np.array([rng.uniform(120, 255) for _ in range(3)], dtype=np.float32)
        d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        w = np.clip(1.0 - (d / r) ** 2, 0, 1)
        img += w[..., None] * col * 0.35
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB").filter(
        ImageFilter.GaussianBlur(size * 0.01)
    )

def draw_wave(rng: random.Random, size: int) -> Image.Image:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    k1, k2, ph = rng.uniform(2, 6), rng.uniform(2, 6), rng.uniform(0, 6.28)
    v = np.sin(xx / size * k1 * math.pi + ph) + np.sin(yy / size * k2 * math.pi)
    t = ((v + 2) / 4)[..., None]
    c1 = np.array([rng.uniform(30, 130) for _ in range(3)], dtype=np.float32)
    c2 = np.array([rng.uniform(150, 250) for _ in range(3)], dtype=np.float32)
    img = c1[None, None, :] * (1 - t) + c2[None, None, :] * t
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), "RGB")

DRAWERS = {
    "gradient_blobs": draw_gradient_blobs,
    "geometric": draw_geometric,
    "paper": draw_paper,
    "stripes": draw_stripes,
    "bokeh": draw_bokeh,
    "wave": draw_wave,
    "fbm_cloud": draw_fbm_cloud,
    "fbm_silk": draw_fbm_silk,
}

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="生成程序化母图素材（训练数据的唯一来源）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-kind", type=int, default=36, help=f"每类生成多少张（共 {len(KINDS)} 类）")
    ap.add_argument("--count", type=int, default=None, help="[兼容旧用法] 总数，必须是 6 的倍数")
    ap.add_argument("--size", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    if args.count is not None:
        if args.count % len(KINDS):
            ap.error(f"--count 必须是 {len(KINDS)} 的倍数（当前 {args.count}）")
        per_kind = args.count // len(KINDS)
    else:
        per_kind = args.per_kind
    total = per_kind * len(KINDS)

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    i = 0
    for kind in KINDS:
        for _ in range(per_kind):
            img = DRAWERS[kind](rng, args.size)
            name = f"placeholder_{i:03d}_{kind}.jpg"
            img.save(args.out / name, format="JPEG", quality=90)
            sidecar = {
                "asset_id": f"placeholder_{i:03d}_{kind}",
                "origin": "team-generated: data/make_placeholder_assets.py",
                "license": "team-owned",
                "author": "参赛团队（程序化生成）",
                "retrieved": "",
                "role": "texture" if kind in TEXTURE_KINDS else "photo",
                "notes": "程序化生成的母图素材，团队自有；训练数据的唯一来源，非第三方素材。",
                "allowed_for_training": True,
                "allowed_for_redistribution": True,
            }
            (args.out / (name + ".json")).write_text(
                json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            i += 1
    print(
        f"生成母图素材 {total} 张（{per_kind} 张/类 × {len(KINDS)} 类，含侧车 JSON）-> {args.out}"
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
