"""PPT 背景数据集合成器。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from presets import ASPECTS, LAYOUTS, PRESETS, TONES, Preset, Tone  # noqa: E402

GENERATOR_VERSION = "1.5.0"

args_seed_base = 20261016

def stable_hash(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")

def build_curve_lut(curve: Sequence[Tuple[float, float]]) -> np.ndarray:
    """把 (in, out) 控制点插值成 256 项 LUT，用分段线性保证单调、无溢出。"""
    xs = np.array([p[0] for p in curve], dtype=np.float64)
    ys = np.array([p[1] for p in curve], dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 256)
    lut = np.interp(grid, xs, ys)
    return np.clip(lut * 255.0, 0, 255)

def apply_tone(img: np.ndarray, tone: Tone) -> np.ndarray:
    """对 float32 RGB(0-255) 应用亮度曲线、饱和度与色调叠加。
    
    入参必须是 (H, W, 3)。灰度 (H, W) 会让色调向量按最后一维广播，报出"""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"apply_tone 需要 (H, W, 3) 的 RGB 图，收到 {img.shape}；"
            "灰度图请先 convert('RGB') 再 np.asarray"
        )
    lut = build_curve_lut(tone.curve)
    idx = np.clip(img, 0, 255).astype(np.uint8)
    out = lut[idx].astype(np.float32)

    if abs(tone.saturation - 1.0) > 1e-3:
        luma = (
            out[..., 0] * 0.299 + out[..., 1] * 0.587 + out[..., 2] * 0.114
        )[..., None]
        out = luma + (out - luma) * tone.saturation
        out = np.clip(out, 0, 255)

    if tone.tint_strength > 0:
        tint = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
        a = tone.tint_strength
        out = out * (1.0 - a) + (out * 0.35 + tint * 0.65) * a
        out = np.clip(out, 0, 255)

    return out

def cover_resize(img: Image.Image, w: int, h: int, zoom: float = 1.0) -> Image.Image:
    tw, th = int(w), int(h)
    src = img
    if zoom > 1.0:
        zw, zh = int(math.ceil(w * zoom)), int(math.ceil(h * zoom))
        sw, sh = img.size
        scale = max(zw / sw, zh / sh)
        nw, nh = max(zw, int(round(sw * scale))), max(zh, int(round(sh * scale)))
        src = img.resize((nw, nh), Image.LANCZOS)
    sw, sh = src.size
    scale = max(tw / sw, th / sh)
    nw, nh = max(tw, int(round(sw * scale))), max(th, int(round(sh * scale)))
    src = src.resize((nw, nh), Image.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return src.crop((left, top, left + tw, top + th))

def apply_photo_geometry(img: Image.Image, recipe: Dict[str, Any]) -> Image.Image:
    """对母图做透视形变 / 旋转 / 镜像 / 百分比裁切。"""
    if recipe.get("photo_mirror"):
        img = ImageOps.mirror(img)

    warp = float(recipe.get("photo_warp", 0.0))
    if warp > 0.001:
        w, h = img.size
        jx, jy = w * warp, h * warp
        rnd = random.Random(recipe.get("photo_warp_seed", 0))
        dx = [rnd.uniform(-jx, jx) for _ in range(4)]
        dy = [rnd.uniform(-jy, jy) for _ in range(4)]
        quad = (
            0.0 + dx[0], 0.0 + dy[0],
            0.0 + dx[1], float(h) + dy[1],
            float(w) + dx[2], float(h) + dy[2],
            float(w) + dx[3], 0.0 + dy[3],
        )
        img = img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)

    rot = float(recipe.get("photo_rotate", 0.0))
    if rot:
        img = img.rotate(rot, resample=Image.BICUBIC, expand=True)

    swirl = float(recipe.get("photo_swirl", 0.0))
    if abs(swirl) > 0.01:
        rotated = _swirl(img, swirl, float(recipe.get("photo_swirl_at", 0.5)))
        if rotated is not None:
            img = rotated

    zoom = float(recipe.get("photo_zoom", 1.0))
    if zoom > 1.001:
        w, h = img.size
        tw, th = max(8, int(w / zoom)), max(8, int(h / zoom))
        ox = recipe.get("photo_crop_x", 0.5)
        oy = recipe.get("photo_crop_y", 0.5)
        left = int(round((w - tw) * ox))
        top = int(round((h - th) * oy))
        left = min(max(0, left), w - tw)
        top = min(max(0, top), h - th)
        img = img.crop((left, top, left + tw, top + th))
    return img

def _swirl(img: Image.Image, strength: float, cx_rel: float, quality: int = 3) -> Optional[Image.Image]:
    try:
        w, h = img.size
        cx, cy = w * cx_rel, h * 0.45
        radius = max(w, h) * 0.62
        step = max(16, max(w, h) // 28)
        xs = list(range(0, w, step)) + [w - 1]
        ys = list(range(0, h, step)) + [h - 1]
        mesh: List[float] = []
        for y in ys:
            for x in xs:
                dx, dy = x - cx, y - cy
                r = math.hypot(dx, dy)
                falloff = max(0.0, 1.0 - r / radius) ** 2
                theta = strength * falloff
                cos_t, sin_t = math.cos(theta), math.sin(theta)
                sx = cx + dx * cos_t + dy * sin_t
                sy = cy - dx * sin_t + dy * cos_t
                mesh.extend((min(max(sx, 0.0), w - 1.0), min(max(sy, 0.0), h - 1.0)))
        return img.transform((w, h), Image.MESH, mesh, resample=Image.BICUBIC)
    except Exception:  # noqa: BLE001 —— 形变只是加分项，失败就用原图
        return None

def prepare_base(asset_path: Path, w: int, h: int, recipe: Dict[str, Any], tone: Tone) -> np.ndarray:
    """读取母图 → 构图变换 → 缩放 → 影调映射 → 光线与边缘，返回 float32 RGB。"""
    img = Image.open(asset_path).convert("RGB")
    img = apply_photo_geometry(img, recipe)
    img = cover_resize(img, w, h, zoom=recipe["photo_scale"])
    if recipe["photo_blur_px"] > 0:
        img = img.filter(ImageFilter.GaussianBlur(recipe["photo_blur_px"]))
    arr = np.asarray(img, dtype=np.float32)
    arr = apply_photo_color(arr, recipe, tone)
    arr = apply_radial_light(arr, recipe)
    arr = apply_posterize(
        arr,
        float(recipe.get("photo_posterize", 0.0)),
        float(recipe.get("photo_posterize_strength", 1.0)),
    )
    arr = apply_tone(arr, tone)
    return arr

def apply_radial_light(arr: np.ndarray, recipe: Dict[str, Any]) -> np.ndarray:
    amount = float(recipe.get("photo_radial", 0.0))
    if abs(amount) < 0.01:
        return arr
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx = w * float(recipe.get("photo_radial_x", 0.5))
    cy = h * float(recipe.get("photo_radial_y", 0.4))
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (max(w, h) * 0.75)
    falloff = np.clip(1.0 - d, 0.0, 1.0) ** 2
    gain = 1.0 + amount * falloff
    return np.clip(arr * gain[..., None], 0.0, 255.0)

def apply_posterize(arr: np.ndarray, levels: float, strength: float = 1.0) -> np.ndarray:
    """色调分离：把亮度压成 N 档，做出像是丝网印或者类似剪纸质感的色块边缘。"""
    if levels <= 1.01 or strength <= 0.01:
        return arr
    luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    step = 255.0 / (levels - 1.0)
    quant = np.round(luma / step) * step
    scale = np.where(luma > 1.0, quant / np.maximum(luma, 1.0), 1.0)
    posterized = np.clip(arr * scale[..., None], 0.0, 255.0)
    return arr * (1.0 - strength) + posterized * strength

def apply_photo_color(arr: np.ndarray, recipe: Dict[str, Any], tone: Tone) -> np.ndarray:
    amount = float(recipe.get("photo_duotone", 0.0))
    if amount <= 0.01:
        return arr
    luma = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]) / 255.0
    t = np.clip(luma, 0.0, 1.0)[..., None]
    base = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
    dark = base * 0.45
    light = np.clip(base * (1.15 + tone.tint_strength) + 40.0 * (1.0 - tone.tint_strength), 0, 255)
    duo = dark + (light - dark) * t
    return arr * (1.0 - amount) + duo * amount

def make_texture(w: int, h: int, strength: float, rng: random.Random) -> Optional[np.ndarray]:
    if strength <= 0:
        return None
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = rng.random()
    if base < 0.5:
        # 注意 h//4*4 != h（例如 1247），放大后尺寸会对不上画布，必须按目标尺寸
        noise = rng.random() * 0.0
        sh, sw = max(2, round(h / 4)), max(2, round(w / 4))
        small = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0.0, 1.0, size=(sh, sw)
        ).astype(np.float32)
        n = np.asarray(
            Image.fromarray(
                np.clip(small * 40 + 128, 0, 255).astype(np.uint8)
            ).resize((w, h), Image.BILINEAR),
            dtype=np.float32,
        )
        tex = (n - 128.0) * strength
    else:
        freq = 0.02 + rng.random() * 0.05
        phase = rng.random() * math.tau
        tex = np.sin((xx + yy) * freq + phase) * strength * 8.0
    return tex[..., None]

def apply_panel_shadow(
    canvas: np.ndarray,
    geom: Dict[str, Any],
    w: int,
    h: int,
    strength: float = 0.18,
) -> np.ndarray:
    if strength <= 0:
        return canvas
    x, y = int(geom["x"]), int(geom["y"])
    pw, ph = int(geom["w"]), int(geom["h"])
    pad = max(4, int(min(w, h) * 0.05))
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [x - pad, y - pad, x + pw + pad, y + ph + pad],
        radius=max(4, int(min(pw, ph) * 0.05)),
        fill=255,
    )
    mask = mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.035))
    alpha = (np.asarray(mask, dtype=np.float32) / 255.0)[..., None] * strength
    return canvas * (1.0 - alpha)

def apply_corner_glow(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    strength: float = 0.12,
) -> np.ndarray:
    if strength <= 0:
        return canvas
    corners = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    cx_list, cy_list = [], []
    for box in rects.values():
        x, y, rw, rh = [float(v) for v in box]
        cx_list.append((x + rw / 2) / w)
        cy_list.append((y + rh / 2) / h)
    mx = sum(cx_list) / max(1, len(cx_list)) if cx_list else 0.5
    my = sum(cy_list) / max(1, len(cy_list)) if cy_list else 0.5
    gx, gy = min(corners, key=lambda c: ((c[0] - mx) ** 2 + (c[1] - my) ** 2)) if False else max(
        corners, key=lambda c: (c[0] - mx) ** 2 + (c[1] - my) ** 2
    )
    cx = gx * w
    cy = gy * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.85 * max(w, h))
    falloff = np.clip(1.0 - rad, 0.0, 1.0) ** 1.8
    a = (falloff * strength)[..., None]
    return canvas * (1.0 - a) + np.array([255.0, 255.0, 255.0])[None, None, :] * a

def draw_panel_card(
    canvas: np.ndarray,
    geom: Dict[str, Any],
    w: int,
    h: int,
    tone: Tone,
    rng: random.Random,
    strength: float = 1.0,
) -> np.ndarray:
    """画一张看得见的磨砂卡片（带投影和高光边），只给模型出图那用。 """
    x, y = int(geom["x"]), int(geom["y"])
    pw, ph = int(geom["w"]), int(geom["h"])
    if pw < 8 or ph < 8:
        return canvas

    off = max(2, int(min(w, h) * 0.012))
    pad = max(4, int(min(w, h) * 0.045))
    sh = Image.new("L", (w, h), 0)
    ImageDraw.Draw(sh).rounded_rectangle(
        [x - pad + off, y - pad + off, x + pw + pad + off, y + ph + pad + off],
        radius=max(6, int(min(pw, ph) * 0.06)),
        fill=255,
    )
    sh = sh.filter(ImageFilter.GaussianBlur(min(w, h) * 0.032))
    sha = (np.asarray(sh, dtype=np.float32) / 255.0)[..., None] * (0.22 * strength)
    canvas = canvas * (1.0 - sha)

    region = canvas[y: y + ph, x: x + pw]
    blurred = np.asarray(
        Image.fromarray(np.clip(region, 0, 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(max(4.0, min(pw, ph) * 0.04))
        ),
        dtype=np.float32,
    )
    bg_lum = float(blurred.mean())
    ink = np.array(hex_to_rgb(tone.ink_hex), dtype=np.float32)
    if bg_lum >= 128.0:
        card = blurred * 0.52 + ink * 0.34 + 255.0 * 0.14
    else:
        card = blurred * 0.46 + 255.0 * 0.42 + ink * 0.12

    side = geom.get("side")
    mask_img = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask_img).rounded_rectangle(
        [0, 0, pw - 1, ph - 1],
        radius=max(4, int(min(pw, ph) * 0.055)) if side not in ("top", "bottom") else 0,
        fill=255,
    )
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(max(1.0, min(pw, ph) * 0.012)))
    if side in ("top", "bottom"):
        m = np.asarray(mask_img, dtype=np.float32) / 255.0
        t = np.linspace(0.0, 1.0, ph, dtype=np.float32)
        ramp = np.clip((1.0 - t) / 0.14, 0.0, 1.0)
        if side == "bottom":
            ramp = ramp[::-1]
        m = m * ramp[:, None]
        mask_img = Image.fromarray((m * 255.0).astype(np.uint8))
    a = (np.asarray(mask_img, dtype=np.float32) / 255.0)[..., None] * 0.72
    canvas[y: y + ph, x: x + pw] = region * (1.0 - a) + card * a

    edge = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(edge)
    lw = max(1, int(min(w, h) * 0.0016))
    if side in ("top", "bottom"):
        ea = None
    elif side == "right":
        d.rectangle([x, y, x + lw, y + ph], fill=255)
    else:
        d.rectangle([x + pw - lw, y, x + pw, y + ph], fill=255)
    if side not in ("top", "bottom"):
        ea = (np.asarray(edge, dtype=np.float32) / 255.0)[..., None] * (0.16 * strength)
        canvas = canvas * (1.0 - ea) + np.array([255.0, 255.0, 255.0])[None, None, :] * ea

    acc = Image.new("L", (w, h), 0)
    ad = ImageDraw.Draw(acc)
    lw2 = max(2, int(h * 0.0032))
    lh2 = max(12, int(ph * 0.30))
    ly = y + int(ph * 0.32)
    if geom.get("side") in ("top", "bottom"):
        ad.rectangle(
            [x + int(pw * 0.08), ly, x + int(pw * 0.08) + lh2, ly + lw2 * 2], fill=255
        )
    elif geom.get("side") == "right":
        lx = x + pw - int(pw * 0.10) - lw2
        ad.rectangle([lx, ly, lx + lw2, ly + lh2], fill=255)
    else:
        lx = x + int(pw * 0.10)
        ad.rectangle([lx, ly, lx + lw2, ly + lh2], fill=255)
    aa = (np.asarray(acc, dtype=np.float32) / 255.0)[..., None] * 0.85
    accent = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32)[None, None, :]
    canvas = canvas * (1.0 - aa) + accent * aa
    return canvas

def _zone_mask(
    w: int, h: int, rects: Dict[str, Tuple[int, int, int, int]], margin: float = 0.06
) -> np.ndarray:
    m = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(m)
    pad = int(min(w, h) * margin)
    for box in rects.values():
        x, y, rw, rh = [int(v) for v in box]
        d.rectangle([x - pad, y - pad, x + rw + pad, y + rh + pad], fill=0)
    arr = np.asarray(m, dtype=np.float32) / 255.0
    soft = np.asarray(
        Image.fromarray((arr * 255).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(min(w, h) * 0.03)
        ),
        dtype=np.float32,
    ) / 255.0
    return soft[..., None]

def apply_design_elements(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    """画设计元素：细线、大圆环、斜切块面、细网格。
    
    元素只长在压字区之外。背景变好看的代价绝不能是字压上去更模糊。
    """
    if amount <= 0:
        return canvas
    keep = _zone_mask(w, h, rects)
    yy, xx = np.mgrid[0: h, 0: w].astype(np.float32)
    ink = np.array(hex_to_rgb(tone.ink_hex), dtype=np.float32)
    accent = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32)
    out = canvas.copy()

    if rng.random() < 0.55:
        step = rng.choice([32, 40, 48, 56])
        gx = (np.abs((xx % step) - step / 2) < 0.6).astype(np.float32)
        gy = (np.abs((yy % step) - step / 2) < 0.6).astype(np.float32)
        grid = np.clip(gx + gy, 0.0, 1.0)[..., None] * (0.030 * amount)
        out = out * (1.0 - grid * keep) + ink[None, None, :] * (grid * keep)

    if rng.random() < 0.75:
        cx = rng.uniform(0.15, 0.85) * w
        cy = rng.uniform(0.15, 0.85) * h
        rad = rng.uniform(0.28, 0.52) * min(w, h)
        lw = rng.uniform(1.5, 3.0)
        d = np.abs(np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - rad)
        ring = np.clip(1.0 - d / lw, 0.0, 1.0)[..., None]
        ring = ring * (0.10 * amount) * keep
        col = accent if rng.random() < 0.5 else ink
        out = out * (1.0 - ring) + col[None, None, :] * ring

    if rng.random() < 0.7:
        pts = []
        n = rng.randint(3, 4)
        for i in range(n):
            ang = 2 * math.pi * i / n + rng.uniform(-0.25, 0.25)
            r = rng.uniform(0.35, 0.75) * max(w, h)
            pts.append((w / 2 + r * math.cos(ang), h / 2 + r * math.sin(ang)))
        lay = Image.new("L", (w, h), 0)
        ImageDraw.Draw(lay).polygon(pts, fill=255)
        lay = lay.filter(ImageFilter.GaussianBlur(min(w, h) * 0.06))
        a = (np.asarray(lay, dtype=np.float32) / 255.0)[..., None]
        a = a * (0.09 * amount) * keep
        col = ink if rng.random() < 0.5 else np.array([255.0, 255.0, 255.0])
        out = out * (1.0 - a) + col[None, None, :] * a

    if rng.random() < 0.85:
        vert = rng.random() < 0.5
        pos = (0.382 if rng.random() < 0.5 else 0.618) * (w if vert else h)
        ln = rng.uniform(0.40, 0.70) * (h if vert else w)
        start = rng.uniform(0.05, 0.95 - ln / (h if vert else w)) * (h if vert else w)
        lw = max(1.0, min(w, h) * 0.0018)
        lay = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(lay)
        if vert:
            d.rectangle([pos - lw, start, pos + lw, start + ln], fill=255)
        else:
            d.rectangle([start, pos - lw, start + ln, pos + lw], fill=255)
        a = (np.asarray(lay, dtype=np.float32) / 255.0)[..., None] * (0.55 * amount) * keep
        out = out * (1.0 - a) + accent[None, None, :] * a

    return out

def apply_structure(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float,
) -> np.ndarray:
    """在安全区之外叠一层大尺度明暗（软形状 + 方向光）。"""
    if amount <= 0:
        return canvas

    protect = np.zeros((h, w), np.float32)
    margin = int(min(w, h) * 0.08)
    for (x, y, zw, zh) in rects.values():
        x0 = max(0, int(x) - margin)
        y0 = max(0, int(y) - margin)
        x1 = min(w, int(x + zw) + margin)
        y1 = min(h, int(y + zh) + margin)
        protect[y0:y1, x0:x1] = 1.0

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    shape = np.zeros((h, w), np.float32)
    for _ in range(rng.randint(2, 4)):
        cx = rng.uniform(-0.15, 1.15) * w
        cy = rng.uniform(-0.15, 1.15) * h
        rad = rng.uniform(0.30, 0.85) * min(w, h)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad
        shape = np.maximum(shape, np.clip(1.0 - dist, 0.0, 1.0) ** 1.25)

    angle = rng.uniform(0.0, math.tau)
    lin = (np.cos(angle) * (xx / w - 0.5) + np.sin(angle) * (yy / h - 0.5)) * 2.0
    field = np.clip(0.62 * shape + 0.38 * (0.5 + 0.5 * np.clip(lin, -1.0, 1.0)), 0.0, 1.0)

    field = field * (1.0 - protect)
    feathered = Image.fromarray(
        np.clip(field * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).filter(ImageFilter.GaussianBlur(min(w, h) * 0.030))
    field = np.asarray(feathered, dtype=np.float32) / 255.0

    tint = np.array(tone.tint, dtype=np.float32)
    hi = tint + (255.0 - tint) * 0.55
    lo = tint * 0.55

    a = field[..., None] * amount
    out = canvas + (hi.reshape(1, 1, 3) - canvas) * a
    b = ((1.0 - field) * amount * 0.65)[..., None]
    out = out + (lo.reshape(1, 1, 3) - out) * b
    return out

def panel_geometry(layout: str, w: int, h: int, rng: random.Random) -> Dict[str, Any]:
    spec = LAYOUTS[layout]
    side = spec["panel_side"]
    if side == "none":
        return {"x": 0, "y": 0, "w": 0, "h": 0, "side": "none"}

    ratio = spec["panel_width_ratio"] * (0.92 + rng.random() * 0.16)
    pw = int(w * ratio)
    if side in ("top", "bottom"):
        ph = int(h * spec.get("panel_height_ratio", 0.30) * (0.90 + rng.random() * 0.20))
        return {"x": 0, "y": 0 if side == "top" else h - ph, "w": w, "h": ph, "side": side}
    if spec.get("panel_anchor") == "corner":
        ph = int(h * spec.get("panel_height_ratio", 0.46) * (0.90 + rng.random() * 0.20))
        py = 0 if rng.random() < 0.5 else h - ph
    elif layout == "corner_soft":
        ph = int(h * (0.55 + rng.random() * 0.25))
        py = 0 if rng.random() < 0.5 else h - ph
    else:
        ph = h
        py = 0
    px = 0 if side == "left" else w - pw
    return {"x": px, "y": py, "w": pw, "h": ph, "side": side}

def rounded_mask(w: int, h: int, radius: int, feather: int, edge_soften: float = 0.0) -> np.ndarray:

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=radius, fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    arr = np.asarray(m, dtype=np.float32) / 255.0
    if edge_soften > 0:
        t = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
        ramp = np.clip((1.0 - t) / max(1e-6, edge_soften), 0.0, 1.0)
        arr = arr * ramp
    return arr[..., None]

def draw_panel(
    canvas: np.ndarray,
    geom: Dict[str, Any],
    panel_style: str,
    ink_hex: str,
    accent_hex: str,
    tone: Tone,
    rng: random.Random,
) -> np.ndarray:
    """在画布上绘制面板磨砂玻璃 / 实色 / 渐变。"""
    if geom["w"] <= 0 or geom["h"] <= 0:
        return canvas

    x, y, pw, ph = geom["x"], geom["y"], geom["w"], geom["h"]
    region = canvas[y : y + ph, x : x + pw, :]
    rh, rw = region.shape[0], region.shape[1]

    ink = np.array(hex_to_rgb(ink_hex), dtype=np.float32)
    radius = int(min(rw, rh) * 0.06)
    feather = max(2, int(min(rw, rh) * 0.02))
    # 否则合成图会直接暴露出一块矩形色块。
    edge_soften = 0.70 if panel_style == "gradient" else 0.22
    if geom["side"] in ("top", "bottom"):
        edge_soften = 0.0
    mask = rounded_mask(rw, rh, radius, feather, edge_soften=edge_soften)
    if geom["side"] == "right":
        mask = mask[:, ::-1, :]

    if panel_style == "frosted":
        blurred = np.asarray(
            Image.fromarray(np.clip(region, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(max(6.0, min(rw, rh) * 0.05))
            ),
            dtype=np.float32,
        )
        base = blurred * 0.62 + 255.0 * 0.30 + ink * 0.08
        alpha = 0.86
    elif panel_style == "solid":
        # 实色块如果真的拍一层纯色：深墨色板会在浅底上糊出一块巨大黑矩形。
        base = ink * 0.92 + region * 0.08
        alpha = 0.78
    else:
        t = np.linspace(0.0, 1.0, rw, dtype=np.float32)[None, :, None]
        if geom["side"] == "right":
            t = t[:, ::-1, :]
        near = ink * 0.92 + 10.0
        far = np.array([250.0, 249.0, 246.0], dtype=np.float32)
        base = near * (1.0 - t) + far * t
        alpha = 0.94

    canvas[y : y + ph, x : x + pw, :] = region * (1.0 - alpha * mask) + base * (alpha * mask)

    # 附近，否则一条离谱的竖线会正好落在最扎眼的地方。
    line_w = max(3, int(min(rw, rh) * 0.012))
    accent = np.array(hex_to_rgb(accent_hex), dtype=np.float32)
    inset = int(pw * (0.30 if panel_style == "gradient" else 0.40))
    lx = x + inset if geom["side"] == "left" else x + pw - inset
    lx = max(x + 2, min(lx, x + pw - line_w - 2))
    ly0 = y + int(ph * (0.20 + rng.random() * 0.12))
    ly1 = y + int(ph * (0.52 + rng.random() * 0.18))
    canvas[ly0:ly1, lx : lx + line_w, :] = accent
    return canvas

def apply_skin_rules(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    if amount <= 0:
        return canvas
    keep = _zone_mask(w, h, rects, margin=0.05)
    acc = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32)[None, None, :]
    ink = np.array(hex_to_rgb(tone.ink_hex), dtype=np.float32)[None, None, :]

    step = max(28, int(min(w, h) / 22))
    g = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(g)
    for gx in range(step, w, step):
        gd.line([(gx, 0), (gx, h)], fill=255)
    for gy in range(step, h, step):
        gd.line([(0, gy), (w, gy)], fill=255)
    ga = (np.asarray(g, dtype=np.float32) / 255.0)[..., None] * (0.055 * amount) * keep
    canvas = canvas * (1.0 - ga) + acc * ga

    rl = Image.new("L", (w, h), 0)
    rd = ImageDraw.Draw(rl)
    lwid = max(2, int(min(w, h) * 0.0022))
    vx = int(w * (0.382 if rng.random() < 0.5 else 0.618))
    hy = int(h * (0.382 if rng.random() < 0.5 else 0.618))
    rd.rectangle([vx, 0, vx + lwid, h], fill=255)
    rd.rectangle([0, hy, w, hy + lwid], fill=255)
    ra = (np.asarray(rl, dtype=np.float32) / 255.0)[..., None] * (0.42 * amount) * keep
    canvas = canvas * (1.0 - ra) + ink * ra

    rg = Image.new("L", (w, h), 0)
    rgd = ImageDraw.Draw(rg)
    rad = int(min(w, h) * (0.42 + rng.random() * 0.10))
    cx = int(w * (0.86 if rng.random() < 0.5 else 0.14))
    cy = int(h * (0.18 if rng.random() < 0.5 else 0.82))
    ow = max(2, int(min(w, h) * 0.0035))
    rgd.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=255, width=ow)
    rga = (np.asarray(rg, dtype=np.float32) / 255.0)[..., None] * (0.13 * amount) * keep
    canvas = canvas * (1.0 - rga) + ink * rga
    return canvas

def apply_skin_blocks(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    if amount <= 0:
        return canvas
    keep = _zone_mask(w, h, rects, margin=0.05)
    acc = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32)[None, None, :]

    left = rng.random() < 0.5
    top = rng.random() < 0.5
    span = 0.30 + rng.random() * 0.10
    deep = 0.46 + rng.random() * 0.16
    if left:
        poly = [(0, 0), (int(w * span), 0), (int(w * (span - 0.10)), int(h * deep)), (0, int(h * deep))]
        if not top:
            poly = [(x, h - y) for x, y in poly]
    else:
        poly = [(w, 0), (int(w * (1 - span)), 0), (int(w * (1 - span + 0.10)), int(h * deep)), (w, int(h * deep))]
        if not top:
            poly = [(x, h - y) for x, y in poly]

    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon(poly, fill=255)
    ma = (np.asarray(m, dtype=np.float32) / 255.0)[..., None] * (0.17 * amount) * keep
    lit = np.clip(canvas * 0.55 + 255.0 * 0.45, 0, 255)
    canvas = canvas * (1.0 - ma) + lit * ma

    e = Image.new("L", (w, h), 0)
    ImageDraw.Draw(e).line([poly[1], poly[2]], fill=255, width=max(4, int(min(w, h) * 0.007)))
    ea = (np.asarray(e, dtype=np.float32) / 255.0)[..., None] * (0.88 * amount) * keep
    canvas = canvas * (1.0 - ea) + acc * ea
    return canvas

def apply_skin_glow(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    if amount <= 0:
        return canvas
    keep = _zone_mask(w, h, rects, margin=0.02)
    soft_keep = 1.0 - (1.0 - keep) * 0.55
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32)[None, None, :]
    for _ in range(2 + (1 if rng.random() < 0.5 else 0)):
        cx = w * (0.10 + rng.random() * 0.80)
        cy = h * (0.10 + rng.random() * 0.80)
        rad = max(w, h) * (0.42 + rng.random() * 0.38)
        fall = np.clip(1.0 - np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / rad, 0.0, 1.0) ** 1.7
        a = (fall * (0.10 + rng.random() * 0.09) * amount * soft_keep[..., 0])[..., None]
        canvas = canvas * (1.0 - a) + (canvas * 0.35 + base * 0.65) * a
    return canvas

def suppress_blotches(
    canvas: np.ndarray,
    fine_ratio: float = 0.020,
    coarse_ratio: float = 0.090,
    strength: float = 0.80,
) -> np.ndarray:
    """压掉模型画出来的圆形色斑。
    
    训练集里有 bokeh 母图（12–26 个圆盘叠加），caption 还写着虚焦短语，模型就学成
    在纯色场上点圆斑：512 下像光斑，放大到 1920 就是一块洗不掉的污点。
    """
    if strength <= 0:
        return canvas
    h, w = canvas.shape[:2]
    base = min(w, h)
    fine = max(2.0, base * fine_ratio)
    coarse = max(fine * 1.5, base * coarse_ratio)
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    lo = np.asarray(img.filter(ImageFilter.GaussianBlur(fine)), dtype=np.float32)
    hi = np.asarray(img.filter(ImageFilter.GaussianBlur(coarse)), dtype=np.float32)
    return np.clip(canvas - (lo - hi) * strength, 0, 255)

def _blur_arr(canvas: np.ndarray, sigma: float) -> np.ndarray:
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))
    return np.asarray(img.filter(ImageFilter.GaussianBlur(sigma)), dtype=np.float32)

def despeckle_discs(
    canvas: np.ndarray,
    sigma_fine: float = 6.0,
    sigma_mid: float = 30.0,
    k_sigma: float = 4.0,
    dilate: float = 12.0,
    strength: float = 1.0,
) -> tuple[np.ndarray, float]:
    band = _blur_arr(canvas, sigma_fine) - _blur_arr(canvas, sigma_mid)
    m = np.abs(band).mean(axis=2)
    thr = float(m.mean() + k_sigma * m.std())
    mask = (m > thr).astype(np.float32)
    if mask.sum() < 40:
        return canvas, 0.0
    soft = _blur_arr(mask[..., None].repeat(3, axis=2), dilate)[..., 0]
    soft = np.clip(soft * 1.6, 0.0, 1.0)
    cover = float((soft > 0.05).mean())
    return np.clip(canvas - band * (soft * strength)[..., None], 0, 255), cover

def blob_score(canvas: np.ndarray) -> float:
    """给模型点出来的圆斑打一个可比的分数，用于检测到就重抽。
    
    所以阈值必须在真实出图上标定，见 tools/calibrate_blob.py，报告 §4.8 里也记了一次。"""
    band = _blur_arr(canvas, 6.0) - _blur_arr(canvas, 30.0)
    m = np.abs(band).mean(axis=2)
    med = float(np.median(m))
    hi = float(np.percentile(m, 99.0))
    return float((float(m.max()) - med) / max(1e-6, hi - med))

def _free_corner(
    rects: Dict[str, Tuple[int, int, int, int]], w: int, h: int
) -> Tuple[int, int, int]:
    pts = []
    for x, y, rw, rh in rects.values():
        pts.append((x + rw / 2.0, y + rh / 2.0))
    if not pts:
        pts = [(w / 2.0, h / 2.0)]
    best = (-1, 1, -1.0)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx = w / 2.0 + sx * w * 0.33
            cy = h / 2.0 + sy * h * 0.33
            dist = min((cx - px) ** 2 + (cy - py) ** 2 for px, py in pts)
            if dist > best[2]:
                best = (sx, sy, dist)
    return best

def _bg_luma(canvas: np.ndarray) -> float:
    return float(canvas[..., :3].mean())

def _line_color(tone: Tone, dark_bg: bool) -> Tuple[int, int, int]:
    if dark_bg:
        return (238, 242, 248)
    return hex_to_rgb(tone.ink_hex)

def _zone_mask_solid(
    w: int, h: int, rects: Dict[str, Tuple[int, int, int, int]], margin: float = 0.06
) -> np.ndarray:
    m = Image.new("L", (w, h), 255)
    d = ImageDraw.Draw(m)
    pad = int(min(w, h) * margin)
    for box in rects.values():
        x, y, rw, rh = [int(v) for v in box]
        d.rectangle([x - pad, y - pad, x + rw + pad, y + rh + pad], fill=0)
    return (np.asarray(m, dtype=np.float32) / 255.0)[..., None]

def _zone_mask_pair(
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    margin: float = 0.06,
    floor: float = 0.30,
    small_ratio: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    hard = _zone_mask(w, h, rects, margin=margin)
    small = {k: v for k, v in rects.items() if float(v[2]) * float(v[3]) < w * h * small_ratio}
    soft = hard * (1.0 - floor) + floor
    if small:
        soft = soft * _zone_mask_solid(w, h, small, margin)
    return hard, soft

def _math_font(size: int):
    try:
        from PIL import ImageFont

        for name in ("simsun.ttc", "simhei.ttf", "msyh.ttc"):
            p = Path("C:/Windows/Fonts") / name
            if p.is_file():
                return ImageFont.truetype(str(p), size)
    except Exception:  # noqa: BLE001
        return None
    return None

def apply_motif_math(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    keep_hard, keep = _zone_mask_pair(w, h, rects, margin=0.07, floor=0.30)
    dark = _bg_luma(canvas) < 128.0
    base = _line_color(tone, dark)
    accent = hex_to_rgb(tone.accent_hex) if not dark else (232, 238, 246)
    col = np.array(base, dtype=np.float32)[None, None, :]
    acc = np.array(accent, dtype=np.float32)[None, None, :]

    grid = Image.new("L", (w, h), 0)
    gd = ImageDraw.Draw(grid)
    step = int(min(w, h) * rng.uniform(0.030, 0.055))
    lwg = 1
    for x in range(step, w, step):
        gd.line([(x, 0), (x, h)], fill=255, width=lwg)
    for y in range(step, h, step):
        gd.line([(0, y), (w, y)], fill=255, width=lwg)
    ga = (np.asarray(grid, dtype=np.float32) / 255.0)[..., None] * (0.080 * amount) * keep
    canvas = canvas * (1.0 - ga) + col * ga

    fc_x, fc_y, _ = _free_corner(rects, w, h)
    ax_w = int(w * rng.uniform(0.22, 0.34))
    ax_h = int(h * rng.uniform(0.26, 0.40))
    ox = int(w * (rng.uniform(0.16, 0.30) if fc_x >= 0 else rng.uniform(0.48, 0.66)))
    oy = int(h * (rng.uniform(0.52, 0.64) if fc_y <= 0 else rng.uniform(0.60, 0.74)))
    lw = max(2, int(min(w, h) * 0.0020))
    axes = Image.new("L", (w, h), 0)
    ad = ImageDraw.Draw(axes)
    ad.line([(ox, oy), (ox + ax_w, oy)], fill=255, width=lw)
    ad.line([(ox, oy), (ox, oy - ax_h)], fill=255, width=lw)
    ah = lw * 3
    ad.polygon([(ox + ax_w, oy), (ox + ax_w - ah * 2, oy - ah), (ox + ax_w - ah * 2, oy + ah)], fill=255)
    ad.polygon([(ox, oy - ax_h), (ox - ah, oy - ax_h + ah * 2), (ox + ah, oy - ax_h + ah * 2)], fill=255)
    for k in range(1, rng.randint(3, 6)):
        tx = ox + ax_w * k / 5.0
        ad.line([(tx, oy - lw * 2), (tx, oy + lw * 2)], fill=200, width=lw)
        ty = oy - ax_h * k / 5.0
        ad.line([(ox - lw * 2, ty), (ox + lw * 2, ty)], fill=200, width=lw)

    kind = rng.choice(["sin", "gauss", "exp", "cubic", "log"])
    xs = np.linspace(0.0, 1.0, 120)
    if kind == "sin":
        ys = 0.5 - 0.34 * np.sin(xs * math.pi * rng.uniform(1.2, 2.2) + rng.uniform(0, 1))
    elif kind == "gauss":
        mu, sg = rng.uniform(0.3, 0.7), rng.uniform(0.12, 0.22)
        ys = 0.95 - 0.75 * np.exp(-((xs - mu) ** 2) / (2 * sg * sg))
    elif kind == "exp":
        ys = 0.9 - 0.72 * xs ** rng.uniform(1.6, 3.0)
    elif kind == "cubic":
        ys = 0.5 + rng.uniform(0.18, 0.3) * np.sin(xs * math.pi * 1.5) * (0.4 + xs)
    else:
        ys = 0.9 - 0.55 * np.log1p(xs * rng.uniform(4, 9)) / math.log(10)
    pts = [(ox + ax_w * t, oy - ax_h * (0.5 - y) if False else oy - ax_h * (1.0 - y)) for t, y in zip(xs, ys)]
    curve = Image.new("L", (w, h), 0)
    cd = ImageDraw.Draw(curve)
    cd.line(pts, fill=255, width=lw + 1, joint="curve")
    if rng.random() < 0.55:
        kind2 = rng.choice(["sin", "exp", "cubic"])
        if kind2 == "sin":
            ys2 = 0.5 - 0.28 * np.sin(xs * math.pi * rng.uniform(0.8, 1.6) + rng.uniform(1, 3))
        elif kind2 == "exp":
            ys2 = 0.15 + 0.6 * xs ** rng.uniform(1.2, 2.4)
        else:
            ys2 = 0.5 + rng.uniform(0.2, 0.34) * np.cos(xs * math.pi * 1.2)
        pts2 = [(ox + ax_w * t, oy - ax_h * (1.0 - y)) for t, y in zip(xs, ys2)]
        cd.line(pts2, fill=170, width=lw, joint="curve")
    n_dot = rng.randint(0, 10)
    for _ in range(n_dot):
        t = rng.random()
        y = float(np.interp(t, xs, ys))
        px = ox + ax_w * t
        py = oy - ax_h * (1.0 - y)
        r = max(2.0, min(w, h) * rng.uniform(0.0026, 0.0042))
        cd.ellipse([px - r, py - r, px + r, py + r], fill=255)
    ca = (np.asarray(curve, dtype=np.float32) / 255.0)[..., None] * (0.42 * amount) * keep
    canvas = canvas * (1.0 - ca) + acc * ca

    aa = (np.asarray(axes, dtype=np.float32) / 255.0)[..., None] * (0.46 * amount) * keep
    canvas = canvas * (1.0 - aa) + col * aa

    font = _math_font(int(min(w, h) * rng.uniform(0.055, 0.085)))
    if font is not None:
        glyphs = rng.sample(["√", "∫", "∑", "≈", "π", "∞", "∂", "λ"], k=rng.randint(1, 3))
        st = Image.new("L", (w, h), 0)
        sd = ImageDraw.Draw(st)
        for i, gch in enumerate(glyphs):
            gx = int(w * rng.uniform(0.06, 0.88))
            gy = int(h * rng.uniform(0.08, 0.86))
            sd.text((gx, gy), gch, font=font, fill=255)
        sa = (np.asarray(st, dtype=np.float32) / 255.0)[..., None] * (0.26 * amount) * keep
        canvas = canvas * (1.0 - sa) + acc * sa
    return canvas

def apply_motif_mech(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    keep_hard, keep = _zone_mask_pair(w, h, rects, margin=0.045, floor=0.38)
    dark = _bg_luma(canvas) < 128.0
    base = _line_color(tone, dark)
    accent = hex_to_rgb(tone.accent_hex) if not dark else (240, 244, 250)
    col = np.array(base, dtype=np.float32)[None, None, :]
    acc = np.array(accent, dtype=np.float32)[None, None, :]

    m1 = int(min(w, h) * rng.uniform(0.030, 0.048))
    m2 = m1 * 2
    lw_outer = max(3, int(min(w, h) * 0.0026))
    lw_inner = max(2, int(min(w, h) * 0.0018))
    iw, ih = w - 2 * m2, h - 2 * m2

    frame = Image.new("L", (w, h), 0)
    fd = ImageDraw.Draw(frame)
    fd.rectangle((m1, m1, w - m1, h - m1), outline=255, width=lw_outer)
    fd.rectangle((m2, m2, w - m2, h - m2), outline=255, width=lw_inner)

    tl = int(min(w, h) * rng.uniform(0.018, 0.030))
    nt = rng.randint(2, 4)
    for k in range(1, nt + 1):
        xx = m2 + iw * k / (nt + 1.0)
        fd.line([(xx, m1), (xx, m1 + tl)], fill=255, width=lw_inner)
        fd.line([(xx, h - m1), (xx, h - m1 - tl)], fill=255, width=lw_inner)
        yy = m2 + ih * k / (nt + 1.0)
        fd.line([(m1, yy), (m1 + tl, yy)], fill=255, width=lw_inner)
        fd.line([(w - m1, yy), (w - m1 - tl, yy)], fill=255, width=lw_inner)
    fa = (np.asarray(frame, dtype=np.float32) / 255.0)[..., None] * (0.74 * amount)
    canvas = canvas * (1.0 - fa) + col * fa

    blk = Image.new("L", (w, h), 0)
    bd = ImageDraw.Draw(blk)
    kind = rng.choice(["title", "title", "bom"])
    if kind == "title":
        tw, th = int(iw * rng.uniform(0.22, 0.30)), int(ih * rng.uniform(0.13, 0.18))
        tx, ty = w - m2 - tw, h - m2 - th
        bd.rectangle((tx, ty, tx + tw, ty + th), outline=255, width=lw_inner)
        rows = rng.choice([3, 3, 4])
        for i in range(1, rows):
            bd.line([(tx, ty + th * i / rows), (tx + tw, ty + th * i / rows)], fill=255, width=lw_inner)
        bd.line([(tx + tw * rng.uniform(0.28, 0.40), ty), (tx + tw * 0.34, ty + th)], fill=255, width=lw_inner)
    else:
        tw, th = int(iw * rng.uniform(0.26, 0.34)), int(ih * rng.uniform(0.18, 0.24))
        tx, ty = w - m2 - tw, h - m2 - th
        bd.rectangle((tx, ty, tx + tw, ty + th), outline=255, width=lw_inner)
        rows = rng.randint(5, 7)
        for i in range(1, rows):
            bd.line([(tx, ty + th * i / rows), (tx + tw, ty + th * i / rows)], fill=255, width=lw_inner)
        for f in (0.12, 0.46, 0.72):
            bd.line([(tx + tw * f, ty), (tx + tw * f, ty + th)], fill=255, width=lw_inner)
    ba = (np.asarray(blk, dtype=np.float32) / 255.0)[..., None] * (0.82 * amount)
    canvas = canvas * (1.0 - ba) + col * ba

    dim = Image.new("L", (w, h), 0)
    dd = ImageDraw.Draw(dim)
    ah = lw_inner * 3
    n_chain = rng.choice([1, 2, 2])
    for ci in range(n_chain):
        if ci == 0:
            dy = h - m2 - int(ih * rng.uniform(0.035, 0.060))
            x_lo, x_hi = m2, tx - int(iw * 0.03)
        else:
            dy = m1 + int(ih * rng.uniform(0.020, 0.040))
            x_lo, x_hi = m2 + int(iw * rng.uniform(0.02, 0.10)), w - m2
        if x_hi - x_lo < iw * 0.3:
            continue
        dd.line([(x_lo, dy), (x_hi, dy)], fill=255, width=lw_inner)
        dd.polygon([(x_lo, dy), (x_lo + ah * 2, dy - ah), (x_lo + ah * 2, dy + ah)], fill=255)
        dd.polygon([(x_hi, dy), (x_hi - ah * 2, dy - ah), (x_hi - ah * 2, dy + ah)], fill=255)
        for ex in (x_lo, x_hi):
            ey = (h - m2) if ci == 0 else m2
            dd.line([(ex, min(dy, ey) - (ah * 2 if ci == 0 else 0)),
                     (ex, max(dy, ey) + (ah * 2 if ci else 0))], fill=235, width=lw_inner)
        seg = rng.randint(2, 3)
        for k in range(1, seg):
            sx = x_lo + (x_hi - x_lo) * k / seg
            dd.line([(sx, dy - ah * 1.6), (sx, dy + ah * 1.6)], fill=235, width=lw_inner)

    if rng.random() < 0.75:
        vx = m2 + int(iw * rng.uniform(0.01, 0.03))
        y_lo, y_hi = m2 + int(ih * 0.06), h - m2 - int(ih * rng.uniform(0.06, 0.16))
        dd.line([(vx, y_lo), (vx, y_hi)], fill=255, width=lw_inner)
        dd.polygon([(vx, y_lo), (vx - ah, y_lo + ah * 2), (vx + ah, y_lo + ah * 2)], fill=255)
        dd.polygon([(vx, y_hi), (vx - ah, y_hi - ah * 2), (vx + ah, y_hi - ah * 2)], fill=255)
    da = (np.asarray(dim, dtype=np.float32) / 255.0)[..., None] * (0.85 * amount)
    canvas = canvas * (1.0 - da) + acc * da

    fb = Image.new("L", (w, h), 0)
    fd2 = ImageDraw.Draw(fb)
    fc_x, fc_y, _ = _free_corner(rects, w, h)
    blk2 = Image.new("L", (w, h), 0)
    bd2 = ImageDraw.Draw(blk2)
    fill2 = Image.new("L", (w, h), 0)
    fd2fill = ImageDraw.Draw(fill2)
    for _ in range(rng.choice([1, 1, 2])):
        bw2 = int(iw * rng.uniform(0.12, 0.22))
        bh2 = int(ih * rng.uniform(0.16, 0.28))
        pad_x = int(iw * rng.uniform(0.04, 0.10))
        pad_y = int(ih * rng.uniform(0.06, 0.16))
        bx = m2 + pad_x if fc_x >= 0 else w - m2 - pad_x - bw2
        by = m2 + pad_y if fc_y >= 0 else h - m2 - pad_y - bh2
        step_h = max(6, int(min(w, h) * rng.uniform(0.009, 0.014)))
        sign = 1 if rng.random() < 0.5 else -1
        for off in range(-bh2, bw2 + bh2, step_h):
            x0, y0 = bx + off, by
            x1_, y1_ = bx + off - sign * bh2, by + bh2
            fd2.line([(x0, y0), (x1_, y1_)], fill=255, width=max(1, lw_inner - 1))
        fd2fill.rectangle((bx, by, bx + bw2, by + bh2), fill=255)
        bd2.rectangle((bx, by, bx + bw2, by + bh2), outline=255, width=lw_inner)
        cym = by + bh2 // 2
        xx = bx - int(bw2 * 0.16)
        while xx < bx + bw2 + int(bw2 * 0.16):
            bd2.line([(xx, cym), (min(xx + step_h * 3, bx + bw2 + int(bw2 * 0.16)), cym)],
                     fill=255, width=max(1, lw_inner - 1))
            xx += step_h * 5
    # 只保留块内的剖面线（用填充掩码，不能用边框掩码），再叠上边框与中心线
    fb = Image.composite(fb, Image.new("L", (w, h), 0), fill2)
    fb = Image.fromarray(
        np.maximum(np.asarray(fb, dtype=np.uint8), np.asarray(blk2, dtype=np.uint8))
    )
    hb = (np.asarray(fb, dtype=np.float32) / 255.0)[..., None] * (0.58 * amount) * keep_hard
    canvas = canvas * (1.0 - hb) + col * hb

    # 有一半是这个。制图里引线必须指到被注对象，否则就不该画。
    misc = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(misc)
    has_detail = rng.random() < 0.55
    cx2 = cy2 = rad = 0
    if has_detail:
        rad = int(min(w, h) * rng.uniform(0.055, 0.085))
        cx2 = m2 + int(iw * rng.uniform(0.08, 0.30)) if fc_x >= 0 else w - m2 - int(iw * rng.uniform(0.08, 0.30))
        cy2 = m2 + int(ih * rng.uniform(0.55, 0.80)) if fc_y >= 0 else h - m2 - int(ih * rng.uniform(0.20, 0.40))
        md.ellipse([cx2 - rad, cy2 - rad, cx2 + rad, cy2 + rad], outline=255, width=lw_inner)
        md.line([(cx2 - rad, cy2), (cx2 + rad, cy2)], fill=190, width=max(1, lw_inner - 1))
    if rng.random() < 0.60:
        bs = int(min(w, h) * rng.uniform(0.030, 0.042))
        bx2 = m2 + int(iw * rng.uniform(0.55, 0.80))
        by2 = m2 + int(ih * rng.uniform(0.06, 0.16)) if fc_y >= 0 else h - m2 - int(ih * rng.uniform(0.10, 0.20)) - bs
        by2 = max(m2 + 4, min(by2, h - m2 - bs - 4))
        md.rectangle((bx2, by2, bx2 + bs, by2 + bs), outline=255, width=lw_inner)
        md.line([(bx2 + bs * 0.5, by2 + bs), (bx2 + bs * 0.5, by2 + bs * 2.2)], fill=255, width=lw_inner)
        md.polygon([(bx2 + bs * 0.5, by2 + bs), (bx2 + bs * 0.5 - lw_inner * 3, by2 + bs + lw_inner * 4),
                    (bx2 + bs * 0.5 + lw_inner * 3, by2 + bs + lw_inner * 4)], fill=255)
    if has_detail:
        ang = rng.uniform(-0.6, 0.6) + (0.0 if fc_x >= 0 else math.pi)
        rx0 = cx2 + int(math.cos(ang) * rad)
        ry0 = cy2 + int(math.sin(ang) * rad)
        dx = int(iw * rng.uniform(0.05, 0.09)) * (1 if fc_x >= 0 else -1)
        dy = -int(ih * rng.uniform(0.03, 0.06))
        md.line([(rx0, ry0), (rx0 + dx, ry0 + dy)], fill=255, width=lw_inner)
        md.line([(rx0 + dx, ry0 + dy), (rx0 + dx * 2, ry0 + dy)], fill=255, width=lw_inner)
        r3 = max(2.0, min(w, h) * 0.0032)
        md.ellipse([rx0 - r3, ry0 - r3, rx0 + r3, ry0 + r3], fill=255)
    ma2 = (np.asarray(misc, dtype=np.float32) / 255.0)[..., None] * (0.62 * amount) * keep_hard
    canvas = canvas * (1.0 - ma2) + col * ma2
    return canvas

def apply_motif_philo(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 1.0,
) -> np.ndarray:
    keep_hard, keep = _zone_mask_pair(w, h, rects, margin=0.08, floor=0.30)
    dark = _bg_luma(canvas) < 128.0
    ink = np.array(_line_color(tone, dark), dtype=np.float32)[None, None, :]

    if rng.random() < 0.88:
        ruled = Image.new("L", (w, h), 0)
        rd = ImageDraw.Draw(ruled)
        lw3 = max(1, int(min(w, h) * 0.0012))
        gap = max(38, int(h * rng.uniform(0.050, 0.078)))
        y0 = int(h * rng.uniform(0.10, 0.18))
        while y0 < h * 0.92:
            rd.line([(int(w * 0.06), y0), (int(w * 0.94), y0)], fill=255, width=lw3)
            y0 += gap
        ra = (np.asarray(ruled, dtype=np.float32) / 255.0)[..., None] * (rng.uniform(0.07, 0.12) * amount) * keep
        canvas = canvas * (1.0 - ra) + ink * ra

    arc = Image.new("L", (w, h), 0)
    ad = ImageDraw.Draw(arc)
    sx, sy, _ = _free_corner(rects, w, h)
    n_stroke = rng.choice([1, 1, 2])
    for si in range(n_stroke):
        ax = w * rng.uniform(0.30, 0.68) + (0.06 * w * si * (1 if sx >= 0 else -1))
        ay = h * rng.uniform(0.34, 0.62) + (0.12 * h * si * (1 if sy >= 0 else -1))
        p0 = (ax - w * rng.uniform(0.10, 0.20), ay + h * rng.uniform(0.04, 0.14))
        p1 = (ax - w * rng.uniform(0.02, 0.09), ay - h * rng.uniform(0.14, 0.26))
        p2 = (ax + w * rng.uniform(0.08, 0.18), ay + h * rng.uniform(0.10, 0.26))
        p3 = (ax + w * rng.uniform(0.14, 0.24), ay - h * rng.uniform(0.04, 0.12))
        steps = 90
        for i in range(steps + 1):
            t = i / steps
            mt = 1.0 - t
            px = (mt ** 3) * p0[0] + 3 * (mt ** 2) * t * p1[0] + 3 * mt * (t ** 2) * p2[0] + (t ** 3) * p3[0]
            py = (mt ** 3) * p0[1] + 3 * (mt ** 2) * t * p1[1] + 3 * mt * (t ** 2) * p2[1] + (t ** 3) * p3[1]
            r = max(1.5, min(w, h) * rng.uniform(0.0055, 0.0085) * (1.0 - 0.6 * t))
            ad.ellipse([px - r, py - r, px + r, py + r], fill=255)
    aa = (np.asarray(arc, dtype=np.float32) / 255.0)[..., None] * (rng.uniform(0.5, 0.7) * amount) * keep
    canvas = canvas * (1.0 - aa) + ink * aa

    if rng.random() < 0.45:
        ens = Image.new("L", (w, h), 0)
        ed = ImageDraw.Draw(ens)
        rad = int(min(w, h) * rng.uniform(0.16, 0.26))
        cx2 = int(w * rng.uniform(0.2, 0.8))
        cy2 = int(h * rng.uniform(0.2, 0.8))
        start = rng.uniform(0, 360)
        ed.arc([cx2 - rad, cy2 - rad, cx2 + rad, cy2 + rad], start=start, end=start + rng.uniform(250, 330),
               fill=255, width=max(2, int(min(w, h) * 0.0035)))
        ea = (np.asarray(ens, dtype=np.float32) / 255.0)[..., None] * (0.10 * amount) * keep
        canvas = canvas * (1.0 - ea) + ink * ea

    chain = Image.new("L", (w, h), 0)
    cd = ImageDraw.Draw(chain)
    n_node = rng.randint(2, 4)
    cy = h * (rng.uniform(0.80, 0.88) if sy < 0 else rng.uniform(0.10, 0.18))
    seg = w * rng.uniform(0.09, 0.13)
    x_start = w * (rng.uniform(0.14, 0.24) if sx > 0 else rng.uniform(0.36, 0.48))
    lw = max(2, int(min(w, h) * 0.0018))
    for i in range(n_node):
        cx = x_start + i * seg
        r = max(3.0, min(w, h) * rng.uniform(0.0045, 0.0065))
        cd.ellipse([cx - r, cy - r, cx + r, cy + r], outline=255, width=lw)
        if i < n_node - 1:
            cd.line([(cx + r * 2, cy), (cx + seg - r * 2, cy)], fill=190, width=lw)
            ax2 = cx + seg - r * 2
            cd.polygon([(ax2, cy), (ax2 - lw * 4, cy - lw * 3), (ax2 - lw * 4, cy + lw * 3)], fill=190)
    ca = (np.asarray(chain, dtype=np.float32) / 255.0)[..., None] * (0.40 * amount) * keep_hard
    canvas = canvas * (1.0 - ca) + ink * ca

    if rng.random() < 0.5:
        seal = Image.new("L", (w, h), 0)
        sd2 = ImageDraw.Draw(seal)
        bs = int(min(w, h) * rng.uniform(0.048, 0.070))
        bx = int(w * rng.uniform(0.70, 0.86))
        by = int(h * rng.uniform(0.72, 0.86))
        sd2.rectangle((bx, by, bx + bs, by + bs), outline=255, width=max(2, int(min(w, h) * 0.0022)))
        sd2.line([(bx, by + bs * 0.5), (bx + bs, by + bs * 0.5)], fill=210, width=max(1, int(min(w, h) * 0.0016)))
        sa = (np.asarray(seal, dtype=np.float32) / 255.0)[..., None] * (0.42 * amount) * keep_hard
        canvas = canvas * (1.0 - sa) + ink * sa
    return canvas

def apply_soft_focus(
    canvas: np.ndarray,
    w: int,
    h: int,
    rng: random.Random,
    tone: Tone,
    amount: float = 0.0,
) -> np.ndarray:

    if amount <= 0:
        return canvas
    layers = []
    for res, amp in ((rng.choice((5, 6, 7)), 1.0), (rng.choice((11, 13, 14)), 0.45)):
        g = np.asarray(
            Image.fromarray(
                np.array(
                    [[rng.randrange(256) for _ in range(res)] for _ in range(res)],
                    dtype=np.uint8,
                ),
                "L",
            ).resize((w, h), Image.BICUBIC),
            dtype=np.float32,
        ) / 255.0
        layers.append((g - g.mean()) * amp)
    field = layers[0] + layers[1]
    peak = float(np.abs(field).max())
    if peak < 1e-6:
        return canvas
    field = field / peak
    delta = field[..., None] * (20.0 * amount)
    out = np.clip(canvas + delta, 0, 255)
    if amount > 0.15:
        warm = np.clip(field, 0, 1)[..., None] * (0.05 * amount)
        accent = np.array(hex_to_rgb(tone.accent_hex), dtype=np.float32).reshape(1, 1, 3)
        out = out * (1.0 - warm) + accent * warm
    return np.clip(out, 0, 255)

def _frost_composite(
    base: np.ndarray,
    canvas: np.ndarray,
    mask: Image.Image,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    amount: float,
) -> np.ndarray:
    keep = _zone_mask(w, h, rects, margin=0.05)[..., 0]
    atten = 0.18 + 0.82 * keep
    ma = (np.asarray(mask, dtype=np.float32) / 255.0)
    alpha = (ma * atten * (0.62 * amount))[..., None]
    out = canvas * (1.0 - alpha) + base * alpha
    rim = np.abs(ma - np.asarray(mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.004)), dtype=np.float32) / 255.0)
    out = out + rim[..., None] * (14.0 * amount)
    return np.clip(out, 0, 255)

def apply_frost_patches(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Tuple[int, int, int, int]],
    rng: random.Random,
    tone: Tone,
    amount: float = 0.0,
    shape: str = "square",
) -> np.ndarray:
    """磨砂块（模糊 B）：一大片磨砂玻璃压在画面上，像加学科元素之前那种大色块。
    
    早先是 1–2 片小圆斑，实测只有指甲盖大，根本看不出是"磨砂玻璃"；
    改成一大块之后才能感觉出"玻璃压在纸面上"。
    """
    if amount <= 0:
        return canvas
    src = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
    radius = max(10, int(min(w, h) * 0.030))
    base = np.asarray(src.filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32)
    gray = base.mean(axis=2, keepdims=True)
    base = base * 0.70 + gray * 0.30
    base = base * (1.0 - 0.14 * amount) + 255.0 * (0.14 * amount)

    _shape = shape if shape in ("square", "ellipse") else rng.choice(("square", "ellipse"))

    if _shape == "ellipse":
        big_r = int(min(w, h) * (0.52 + rng.random() * 0.16))
        squash = 1.25 + rng.random() * 0.45
        cx = int(w * (0.20 + rng.random() * 0.60))
        cy = int(h * (0.20 + rng.random() * 0.60))
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        md.ellipse(
            (cx - int(big_r * squash), cy - big_r, cx + int(big_r * squash), cy + big_r),
            fill=255,
        )
        if rng.random() < 0.6:
            r2 = int(big_r * (0.45 + rng.random() * 0.25))
            cx2 = int(w * (0.15 + rng.random() * 0.70))
            cy2 = int(h * (0.12 + rng.random() * 0.76))
            md.ellipse((cx2 - r2, cy2 - r2, cx2 + r2, cy2 + r2), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.075))
        return _frost_composite(base, canvas, mask, w, h, rects, amount)

    bw = int(w * (0.42 + rng.random() * 0.16))
    bh = int(h * (0.55 + rng.random() * 0.30))
    left_side = rng.random() < 0.5
    top_side = rng.random() < 0.5
    margin_x = int(w * (0.03 + rng.random() * 0.05))
    margin_y = int(h * (0.04 + rng.random() * 0.06))
    x0 = margin_x if left_side else w - bw - margin_x
    y0 = margin_y if top_side else h - bh - margin_y
    x0 = max(0, min(w - bw, x0))
    y0 = max(0, min(h - bh, y0))
    mask = Image.new("L", (w, h), 0)
    md = ImageDraw.Draw(mask)
    md.rounded_rectangle(
        (x0, y0, x0 + bw, y0 + bh),
        radius=int(min(bw, bh) * (0.06 + rng.random() * 0.05)),
        fill=255,
    )
    if rng.random() < 0.6:
        bw2 = int(bw * (0.45 + rng.random() * 0.22))
        bh2 = int(bh * (0.40 + rng.random() * 0.22))
        x2 = int(x0 + (bw - bw2) * rng.random())
        y2 = int(y0 + (bh - bh2) * rng.random())
        md.rounded_rectangle(
            (x2, y2, x2 + bw2, y2 + bh2),
            radius=int(min(bw2, bh2) * 0.08),
            fill=255,
        )
    mask = mask.filter(ImageFilter.GaussianBlur(min(w, h) * 0.022))

    return _frost_composite(base, canvas, mask, w, h, rects, amount)

def apply_vignette(canvas: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return canvas
    h, w = canvas.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / math.sqrt(2.0)
    falloff = np.clip(1.0 - strength * np.clip(r - 0.25, 0, 1) ** 1.6, 0, 1)
    return canvas * falloff[..., None]

def apply_grain(canvas: np.ndarray, amount: float, rng: random.Random) -> np.ndarray:
    if amount <= 0:
        return canvas
    gen = np.random.default_rng(rng.randrange(1 << 30))
    noise = gen.normal(0.0, 1.0, size=canvas.shape[:2]).astype(np.float32)
    return canvas + noise[..., None] * amount

def safe_zone_rects(layout: str, w: int, h: int, geom: Dict[str, Any]) -> Dict[str, Tuple[int, int, int, int]]:
    m = int(round(min(w, h) * 0.055))
    if layout == "left_panel":
        bx = geom["x"] + geom["w"] + m
        bw = w - bx - m
        return {
            "title": (bx, int(h * 0.18), bw, int(h * 0.20)),
            "body": (bx, int(h * 0.42), bw, int(h * 0.36)),
            "footer": (bx, h - m - int(h * 0.05), bw, int(h * 0.05)),
        }
    if layout == "right_panel":
        bw = w - geom["x"] - m
        return {
            "title": (m, int(h * 0.18), bw, int(h * 0.20)),
            "body": (m, int(h * 0.42), bw, int(h * 0.36)),
            "footer": (m, h - m - int(h * 0.05), bw, int(h * 0.05)),
        }
    if layout == "corner_card":
        right_side = geom["x"] < w / 2
        bx = geom["x"] + geom["w"] + m if right_side else m
        bw = w - geom["w"] - 2 * m
        top_card = geom["y"] <= 0
        ty = int(h * 0.20) if top_card else int(h * 0.14)
        return {
            "title": (bx, ty, bw, int(h * 0.18)),
            "body": (bx, ty + int(h * 0.24), bw, int(h * 0.34)),
            "footer": (bx, h - m - int(h * 0.05), bw, int(h * 0.05)),
        }
    if layout == "top_band":
        by = geom["h"] + m
        return {
            "title": (int(w * 0.08), by + int(h * 0.04), int(w * 0.84), int(h * 0.20)),
            "body": (int(w * 0.08), by + int(h * 0.30), int(w * 0.84), int(h * 0.30)),
            "footer": (int(w * 0.08), h - m - int(h * 0.04), int(w * 0.84), int(h * 0.04)),
        }
    if layout == "corner_soft":
        bx = geom["x"] + geom["w"] + m
        vx = geom["y"] > 0
        # 图注不能压在卡片边上：卡片贴左下角时，若图注仍放左下角，那条 6% 高的窄条
        # 正好横跨卡片右边缘，一半亮一半暗 —— 实测局部标准差 37（阈值 26），
        cap_x = (w - int(w * 0.4) - m) if vx else m
        return {
            "title": (bx, int(h * 0.14) if not vx else int(h * 0.12), w - bx - m, int(h * 0.18)),
            "body": (bx, int(h * 0.38) if not vx else int(h * 0.34), w - bx - m, int(h * 0.40)),
            "caption": (cap_x, h - m - int(h * 0.06), int(w * 0.4), int(h * 0.06)),
        }
    if layout == "center_band":
        return {
            "title": (int(w * 0.10), int(h * 0.10), int(w * 0.80), int(h * 0.16)),
            "body": (int(w * 0.10), int(h * 0.62), int(w * 0.80), int(h * 0.24)),
            "footer": (int(w * 0.10), h - m - int(h * 0.04), int(w * 0.80), int(h * 0.04)),
        }
    return {
        "title": (int(w * 0.12), int(h * 0.16), int(w * 0.76), int(h * 0.20)),
        "body": (int(w * 0.12), int(h * 0.46), int(w * 0.76), int(h * 0.34)),
        "footer": (int(w * 0.12), h - m - int(h * 0.05), int(w * 0.76), int(h * 0.05)),
    }

def analyze_zone(canvas: np.ndarray, rect: Tuple[int, int, int, int], region: str) -> Dict[str, Any]:
    x, y, w, h = rect
    h_img, w_img = canvas.shape[:2]
    x2, y2 = min(x + w, w_img), min(y + h, h_img)
    x, y = max(0, x), max(0, y)
    crop = canvas[y:y2, x:x2, :]
    if crop.size == 0:
        luma_std, luma_mean = 255.0, 128.0
    else:
        luma = crop[..., 0] * 0.299 + crop[..., 1] * 0.587 + crop[..., 2] * 0.114
        luma_std = float(luma.std())
        luma_mean = float(luma.mean())
    return {
        "region": region,
        "x": int(x),
        "y": int(y),
        "w": int(x2 - x),
        "h": int(y2 - y),
        "contrast_std": round(luma_std, 2),
        "mean_luma": round(luma_mean, 2),
    }

LAYOUT_PHRASE = {
    "left_panel": "left side occluded by a soft vertical panel, right side remains open for text",
    "right_panel": "right side occluded by a soft vertical panel, left side remains open for text",
    "corner_card": "a single rounded card rests in one corner, the opposite side stays wide and empty",
    "top_band": "a horizontal band runs along the top edge, the lower area stays open for text",
    "center_band": "a horizontal band crosses the middle, top and bottom areas stay clean",
    "corner_soft": "one corner holds a soft color block, the rest of the frame is calm and empty",
    "vignette_plain": "edges are darkened by a soft vignette, center stays legible",
}

PANEL_PHRASE = {
    "frosted": "frosted glass panel",
    "solid": "flat solid panel",
    "gradient": "smooth gradient panel",
    "none": "",
}

# TODO caption 只写到版式和色调，没写元素池抽了什么，模型其实学不到元素
def build_caption(preset: Preset, recipe: Dict[str, Any]) -> str:
    tone = TONES[preset.tone]
    parts = list(preset.keywords_en)
    parts.append(LAYOUT_PHRASE[recipe["layout"]])
    if recipe["panel_style"] != "none":
        parts.append(PANEL_PHRASE[recipe["panel_style"]])
    parts.append(f"{tone.name_en} palette")
    parts.append(f"accent color {recipe['accent_hex']}")
    if recipe.get("decor", 0) > 0:
        parts.append("thin decorative rules and a large geometric ring")
    if recipe.get("structure", 0) >= 0.30:
        parts.append("soft large-scale light gradient")
    if recipe["photo_blur_px"] >= 8:
        # 换成不含虚焦语义的说法，母图家族里的 bokeh 也一并从清单里去掉了。
        parts.append("even soft photographic tone")
    if recipe["grain"] > 0.6:
        parts.append("fine paper grain")
    return ", ".join(parts)

KIND_SUFFIXES = ("fbm_cloud", "fbm_silk", "gradient_blobs", "geometric", "paper", "stripes", "wave", "bokeh")

def _asset_kind(asset_id: str) -> str:
    for suf in KIND_SUFFIXES:
        if asset_id.endswith(suf):
            return suf
    return asset_id.rsplit("_", 1)[-1]

def load_assets(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
            for k in ("asset_id", "file", "origin", "license", "role"):
                if k not in row:
                    raise SystemExit(f"{path}:{lineno} 缺少字段 {k}")
            rows.append(row)
    if not rows:
        raise SystemExit(f"{path} 里没有可用素材")
    return rows

def pick_assets(pool: List[Dict[str, Any]], preset: Preset, rng: random.Random) -> List[Dict[str, Any]]:
    if "any" in preset.photo_roles:
        allowed = list(pool)
    else:
        allowed = [a for a in pool if a["role"] in preset.photo_roles]
    if not allowed:
        raise SystemExit(
            f"预设 {preset.key} 需要 role∈{preset.photo_roles} 的素材，但清单里一张都没有；"
            f"请补充素材或修改 preset.photo_roles"
        )
    main = rng.choice(allowed)
    chosen = [main]
    if preset.texture_weight > 0.15 and len(allowed) > 1 and rng.random() < 0.55:
        others = [a for a in allowed if a["asset_id"] != main["asset_id"]]
        if others:
            chosen.append(rng.choice(others))
    return chosen

def synthesize_background(
    preset_key: str,
    aspect: str,
    seed: int,
    assets_root: Path,
    pool: List[Dict[str, Any]],
    with_panel: bool = True,
) -> Tuple[Image.Image, Dict[str, Any]]:

    preset = PRESETS[preset_key]
    tone = TONES[preset.tone]
    w, h = ASPECTS[aspect]
    rng = random.Random(seed)

    layout = rng.choice(preset.layouts) if with_panel else "vignette_plain"
    geom = panel_geometry(layout, w, h, rng)
    panel_style = "none" if (not with_panel or layout == "vignette_plain") else rng.choice(
        ["frosted", "frosted", "gradient", "solid"]
    )

    chosen = pick_assets(pool, preset, rng)
    main_asset = chosen[0]
    recipe: Dict[str, Any] = {
        "layout": layout,
        "tone": tone.key,
        "panel_side": geom["side"],
        "panel_style": panel_style,
        "photo_scale": round(1.06 + rng.random() * 0.28, 3),
        "photo_blur_px": round(0.8 + rng.random() * 5.0, 2),
        "overlay_alpha": round(0.10 + rng.random() * 0.20, 3),
        "accent_hex": tone.accent_hex,
        "vignette": round(0.08 + rng.random() * 0.22, 3),
        "grain": round(rng.random() * 0.6, 3),
        # 不要全开会有网格/圆环会变成每张图的硬规则，
        "decor": 0.0 if rng.random() < 0.15 else round(0.45 + rng.random() * 0.55, 3),
        "photo_rotate": float(rng.choice([0, 0, 90, 180, 270, rng.uniform(-20.0, 20.0)])),
        "photo_mirror": rng.random() < 0.45,
        "photo_zoom": round(1.35 + (rng.random() ** 1.4) * 1.35, 3),
        "photo_crop_x": round(rng.random(), 3),
        "photo_crop_y": round(rng.random(), 3),
        "photo_duotone": round(max(0.0, rng.random() - 0.5) * 1.4, 3),
        "photo_warp": round(0.02 + rng.random() * 0.06, 4),
        "photo_warp_seed": rng.randrange(1 << 30),
        "photo_swirl": round((rng.random() - 0.5) * 2.0 * (1.2 if rng.random() < 0.75 else 2.6), 3),
        "photo_swirl_at": round(rng.uniform(0.25, 0.75), 3),
        "photo_posterize": float(rng.choice([0.0, 0.0, 0.0, rng.randint(12, 20), rng.randint(11, 16)])),
        "photo_posterize_strength": round(rng.uniform(0.25, 0.55), 3),
        "photo_radial": round((rng.random() - 0.42) * 0.5, 3),
        "photo_radial_x": round(rng.uniform(0.15, 0.85), 3),
        "photo_radial_y": round(rng.uniform(0.10, 0.70), 3),
    }
    _kind = _asset_kind(main_asset["asset_id"])
    if _kind in ("stripes", "wave", "fbm_cloud", "fbm_silk"):
        recipe["photo_blur_px"] = round(max(recipe["photo_blur_px"] * 2.2, 18.0), 2)

    canvas = prepare_base(assets_root / main_asset["file"], w, h, recipe, tone)
    if canvas.shape[:2] != (h, w):
        raise AssertionError(
            f"prepare_base 返回 {canvas.shape[:2]}，应为 {(h, w)}（aspect={aspect}）"
        )
    if recipe["overlay_alpha"] > 0:
        overlay = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
        canvas = canvas * (1.0 - recipe["overlay_alpha"]) + overlay * recipe["overlay_alpha"]

    tex = make_texture(w, h, preset.texture_weight, rng)
    if tex is not None:
        canvas = canvas + tex

    rects = safe_zone_rects(layout, w, h, geom) if with_panel else {"full": (0, 0, w, h)}
    if recipe.get("decor", 0) > 0:
        canvas = apply_design_elements(canvas, w, h, rects, rng, tone, recipe["decor"])

    if recipe.get("structure"):
        canvas = apply_structure(canvas, w, h, rects, rng, tone, recipe["structure"])

    if with_panel:
        canvas = draw_panel(canvas, geom, panel_style, tone.ink_hex, tone.accent_hex, tone, rng)
        canvas = apply_vignette(canvas, recipe["vignette"])
    else:
        canvas = apply_vignette(canvas, recipe["vignette"])
    canvas = apply_grain(canvas, recipe["grain"], rng)
    canvas = np.clip(canvas, 0, 255)

    img = Image.fromarray(canvas.astype(np.uint8), mode="RGB")
    meta = {
        "recipe": recipe,
        "layout": layout,
        "panel_geom": geom,
        "safe_zones": [analyze_zone(canvas, r, n) for n, r in rects.items()],
        "sources": [
            {
                "asset_id": a["asset_id"],
                "origin": a["origin"],
                "license": a["license"],
                "author": a.get("author", ""),
            }
            for a in chosen
        ],
    }
    return img, meta

def generate_one(
    idx: int,
    pool: List[Dict[str, Any]],
    preset_key: str,
    aspect: str,
    assets_root: Path,
    out_dir: Path,
    jpeg_quality: int,
    clean_threshold: float = 26.0,
    max_tries: int = 4,
) -> Dict[str, Any]:
    last: Optional[Dict[str, Any]] = None
    for attempt in range(max_tries):
        rec = _generate_one_once(
            idx + attempt * 100_000, pool, preset_key, aspect, assets_root, out_dir, jpeg_quality
        )
        want = out_dir / f"bg_{idx:06d}.jpg"
        got = out_dir / rec["file"]
        if got != want:
            got.replace(want)
            rec["file"] = want.name
        worst = max(
            (z["contrast_std"] for z in rec["safe_zones"] if z["region"] in ("title", "body")),
            default=0.0,
        )
        rec["zone_worst_std"] = round(float(worst), 2)
        last = rec
        if worst <= clean_threshold:
            return rec
    assert last is not None
    last["zone_warn"] = True
    return last

def _generate_one_once(
    idx: int,
    pool: List[Dict[str, Any]],
    preset_key: str,
    aspect: str,
    assets_root: Path,
    out_dir: Path,
    jpeg_quality: int,
) -> Dict[str, Any]:
    preset = PRESETS[preset_key]
    tone = TONES[preset.tone]
    w, h = ASPECTS[aspect]
    # 注意：必须用自带的稳定哈希，不能用内置 hash()——Python 的 str hash 每个进程都加了
    seed = args_seed_base + idx * 7919 + stable_hash(preset_key) % 100003
    rng = random.Random(seed)

    layout = rng.choice(preset.layouts)
    geom = panel_geometry(layout, w, h, rng)
    if layout == "vignette_plain":
        panel_style = "none"
    else:
        panel_style = rng.choice(["frosted", "frosted", "gradient", "solid"])

    recipe: Dict[str, Any] = {
        "layout": layout,
        "tone": tone.key,
        "panel_side": geom["side"],
        "panel_style": panel_style,
        "photo_scale": round(1.06 + rng.random() * 0.28, 3),
        "photo_blur_px": round(0.8 + rng.random() * 3.7, 2),
        "overlay_alpha": round(0.10 + rng.random() * 0.20, 3),
        "accent_hex": tone.accent_hex,
        "vignette": round(0.08 + rng.random() * 0.22, 3) if layout != "vignette_plain"
        else round(0.30 + rng.random() * 0.18, 3),
        "grain": round(rng.random() * 0.6, 3),
        # 不要全开否则有网格/圆环会变成每张图的硬规则，（重要的事情说两遍）
        "decor": 0.0 if rng.random() < 0.15 else round(0.45 + rng.random() * 0.55, 3),
        "photo_rotate": float(rng.choice([0, 0, 90, 180, 270, rng.uniform(-20.0, 20.0)])),
        "photo_mirror": rng.random() < 0.45,
        "photo_zoom": round(1.35 + (rng.random() ** 1.4) * 1.35, 3),
        "photo_crop_x": round(rng.random(), 3),
        "photo_crop_y": round(rng.random(), 3),
        "photo_duotone": round(max(0.0, rng.random() - 0.5) * 1.4, 3),
        "photo_blur_mult": round(1.0 + rng.random() * 2.5, 2),
        "photo_warp": round(0.02 + rng.random() * 0.06, 4),
        "photo_warp_seed": rng.randrange(1 << 30),
        "photo_swirl": round((rng.random() - 0.5) * 2.0 * (1.2 if rng.random() < 0.75 else 2.6), 3),
        "photo_swirl_at": round(rng.uniform(0.25, 0.75), 3),
        "photo_posterize": float(rng.choice([0.0, 0.0, 0.0, rng.randint(12, 20), rng.randint(11, 16)])),
        "photo_posterize_strength": round(rng.uniform(0.25, 0.55), 3),
        "photo_radial": round((rng.random() - 0.42) * 0.5, 3),
        "photo_radial_x": round(rng.uniform(0.15, 0.85), 3),
        "photo_radial_y": round(rng.uniform(0.10, 0.70), 3),
    }
    if preset_key == "clean_white":
        recipe["photo_blur_px"] = round(2.0 + rng.random() * 3.0, 2)
        recipe["overlay_alpha"] = round(0.20 + rng.random() * 0.20, 3)
        recipe["photo_duotone"] = round(rng.random() * 0.4, 3)
        recipe["photo_blur_mult"] = round(1.2 + rng.random() * 2.0, 2)
    if preset_key == "violet_tech":
        recipe["vignette"] = round(0.24 + rng.random() * 0.20, 3)
    recipe["photo_blur_px"] = round(recipe["photo_blur_px"] * recipe["photo_blur_mult"], 2)

    chosen = pick_assets(pool, preset, rng)
    main_asset = chosen[0]
    _kind = _asset_kind(main_asset["asset_id"])
    if _kind in ("stripes", "wave", "fbm_cloud", "fbm_silk"):
        recipe["photo_blur_px"] = round(max(recipe["photo_blur_px"] * 2.2, 18.0), 2)

    canvas = prepare_base(assets_root / main_asset["file"], w, h, recipe, tone)
    if canvas.shape[:2] != (h, w):
        raise AssertionError(
            f"prepare_base 返回 {canvas.shape[:2]}，应为 {(h, w)}（aspect={aspect}）"
        )

    if recipe["overlay_alpha"] > 0:
        overlay = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
        canvas = canvas * (1.0 - recipe["overlay_alpha"]) + overlay * recipe["overlay_alpha"]

    tex = make_texture(w, h, preset.texture_weight, rng)
    if tex is not None:
        canvas = canvas + tex

    if recipe.get("decor", 0) > 0:
        _rects = safe_zone_rects(layout, w, h, geom)
        canvas = apply_design_elements(canvas, w, h, _rects, rng, tone, recipe["decor"])
    canvas = draw_panel(canvas, geom, panel_style, tone.ink_hex, tone.accent_hex, tone, rng)
    canvas = apply_vignette(canvas, recipe["vignette"])
    canvas = apply_grain(canvas, recipe["grain"], rng)
    canvas = np.clip(canvas, 0, 255)

    img = Image.fromarray(canvas.astype(np.uint8), mode="RGB")
    fname = f"bg_{idx:06d}.jpg"
    out_dir.mkdir(parents=True, exist_ok=True)
    img.save(out_dir / fname, format="JPEG", quality=jpeg_quality, optimize=True, subsampling=1)

    zones = [analyze_zone(canvas, rect, name) for name, rect in safe_zone_rects(layout, w, h, geom).items()]

    sources = [
        {
            "asset_id": a["asset_id"],
            "origin": a["origin"],
            "license": a["license"],
            "author": a.get("author", ""),
            "retrieved": a.get("retrieved", ""),
            "role": "photo" if a is main_asset else "texture",
        }
        for a in chosen
    ]

    return {
        "id": f"bg_{idx:06d}",
        "file": fname,
        "image_sha256": hashlib.sha256((out_dir / fname).read_bytes()).hexdigest(),
        "width": w,
        "height": h,
        "aspect": aspect,
        "preset": preset.key,
        "seed": seed,
        "recipe": recipe,
        "safe_zones": zones,
        "caption": build_caption(preset, recipe),
        "sources": sources,
        "generator": {"tool": "make_backgrounds.py", "version": GENERATOR_VERSION},
    }

def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    v = value.lstrip("#")
    return int(v[0:2], 16), int(v[2:4], 16), int(v[4:6], 16)

def plan_distribution(count: int, aspects: Sequence[str], keys: Sequence[str], rng: random.Random) -> List[Tuple[str, str]]:
    plan: List[Tuple[str, str]] = []
    aspect_cycle = [a for a in aspects for _ in range(4 if a == "16:9" else 1)]
    i = 0
    while len(plan) < count:
        plan.append((keys[i % len(keys)], aspect_cycle[i % len(aspect_cycle)]))
        i += 1
    rng.shuffle(plan)
    return plan

def write_manifest_report(records: List[Dict[str, Any]], path: Path) -> None:
    counter: Counter = Counter()
    meta: Dict[str, Dict[str, Any]] = {}
    for r in records:
        for s in r["sources"]:
            counter[s["asset_id"]] += 1
            meta[s["asset_id"]] = s
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["资源名称及类型", "版本/来源", "许可证/授权类型", "使用方式", "关键义务/限制", "使用次数", "作者", "获取日期"]
        )
        for asset_id, n in counter.most_common():
            s = meta[asset_id]
            w.writerow(
                [
                    f"图像素材 {asset_id}",
                    s["origin"],
                    s["license"],
                    "作为合成背景的母图/纹理层，经裁剪、影调映射与模糊后使用；不单独再分发原图",
                    "按原许可保留署名与来源记录；不再分发原图本体",
                    n,
                    s.get("author", ""),
                    s.get("retrieved", ""),
                ]
            )

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="合成 PPT 背景数据集")
    ap.add_argument("--assets", type=Path, required=True, help="素材清单 manifest.jsonl")
    ap.add_argument("--assets-root", type=Path, default=None, help="素材根目录，默认取清单同级 raw/")
    ap.add_argument("--out", type=Path, required=True, help="输出目录")
    ap.add_argument("--count", type=int, default=600)
    ap.add_argument("--aspects", nargs="+", default=["16:9"], choices=list(ASPECTS))
    ap.add_argument("--presets", nargs="+", default=list(PRESETS))
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--jpeg-quality", type=int, default=88)
    args = ap.parse_args(argv)

    assets_root = args.assets_root or args.assets.parent / "raw"
    global args_seed_base
    args_seed_base = args.seed
    pool = load_assets(args.assets)
    rng = random.Random(args.seed)
    plan = plan_distribution(args.count, args.aspects, args.presets, rng)

    img_dir = args.out / "images"
    records: List[Dict[str, Any]] = []
    for i, (preset_key, aspect) in enumerate(plan):
        rec = generate_one(i, pool, preset_key, aspect, assets_root, img_dir, args.jpeg_quality)
        records.append(rec)
        if (i + 1) % 50 == 0:
            print(f"  已生成 {i + 1}/{len(plan)}", flush=True)

    rec_path = args.out / "records.jsonl"
    with rec_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    report_path = args.out / "manifest_report.csv"
    write_manifest_report(records, report_path)

    digest = hashlib.sha256(rec_path.read_bytes()).hexdigest()
    summary = {
        "generated": len(records),
        "aspects": dict(Counter(r["aspect"] for r in records)),
        "presets": dict(Counter(r["preset"] for r in records)),
        "layouts": dict(Counter(r["recipe"]["layout"] for r in records)),
        "panel_styles": dict(Counter(r["recipe"]["panel_style"] for r in records)),
        "seed": args.seed,
        "generator_version": GENERATOR_VERSION,
        "records_sha256": digest,
        "date": date.today().isoformat(),
    }
    (args.out / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n完成：{len(records)} 张 -> {img_dir}")
    print(f"记录：{rec_path}")
    print(f"资源清单：{report_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
