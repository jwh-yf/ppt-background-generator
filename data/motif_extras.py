# -*- coding: utf-8 -*-
"""给学科符号加元素池：每张图除了基础骨架，再随机抽 0–2 件附加元素。"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

Rect = Tuple[int, int, int, int]

__all__ = ["apply_motif_extras", "EXTRA_NAMES"]

def _free_box(rects: Dict[str, Rect], w: int, h: int, rng, size: float) -> Tuple[int, int, int, int]:
    # TODO 候选只有 24 个，丰富档（2 件）偶尔会擦到安全区边缘，实测最差 20.5（阈值 26）。
    #      要收的话：候选提到 40 个，或抽样时命中边缘就换一件。
    pad = 12
    s = int(min(w, h) * size)
    best = None
    for _ in range(24):
        x = int(rng.uniform(0.02, 0.98) * w)
        y = int(rng.uniform(0.02, 0.98) * h)
        x = max(0, min(w - s, x))
        y = max(0, min(h - s, y))
        hit = False
        for (rx, ry, rw, rh) in rects.values():
            if x < rx + rw + pad and x + s + pad > rx and y < ry + rh + pad and y + s + pad > ry:
                hit = True
                break
        if not hit:
            edge = min(x, y, w - x - s, h - y - s)
            if best is None or edge < best[0]:
                best = (edge, x, y, s)
    if best is None:
        x = max(0, min(w - s, (w - s) // 2))
        y = max(0, min(h - s, (h - s) // 2))
        return x, y, s, s
    return best[1], best[2], s, s

def _m_contours(d, box, rng, lw):
    x0, y0, s, _ = box
    n = rng.randint(3, 5)
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    for i in range(n):
        rx = s * (0.16 + i * 0.07)
        ry = rx * rng.uniform(0.55, 0.85)
        d.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=200, width=lw)

def _m_vectors(d, box, rng, lw):
    x0, y0, s, _ = box
    for _ in range(rng.randint(6, 11)):
        px = x0 + rng.uniform(0.08, 0.92) * s
        py = y0 + rng.uniform(0.08, 0.92) * s
        ang = rng.uniform(0, 2 * math.pi)
        ln = s * rng.uniform(0.08, 0.16)
        ex, ey = px + math.cos(ang) * ln, py + math.sin(ang) * ln
        d.line([(px, py), (ex, ey)], fill=210, width=lw)
        for sgn in (+1, -1):
            a2 = ang + sgn * 2.5
            d.line([(ex, ey), (ex + math.cos(a2) * ln * 0.32, ey + math.sin(a2) * ln * 0.32)],
                   fill=210, width=lw)

def _m_polar(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    rings = rng.randint(2, 3)
    for i in range(rings):
        r = s * (0.26 + i * 0.11)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=200, width=lw)
    n = rng.choice([12, 16, 24])
    r_out = s * (0.26 + (rings - 1) * 0.11)
    for k in range(n):
        a = 2 * math.pi * k / n
        d.line([(cx + math.cos(a) * r_out, cy + math.sin(a) * r_out),
                (cx + math.cos(a) * (r_out + s * 0.05), cy + math.sin(a) * (r_out + s * 0.05))],
               fill=190, width=lw)

def _m_matrix(d, box, rng, lw):
    x0, y0, s, _ = box
    bw, bh = s * 0.52, s * 0.60
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    L, R, T, B = cx - bw / 2, cx + bw / 2, cy - bh / 2, cy + bh / 2
    arm = bw * 0.18
    for xx, sgn in ((L, +1), (R, -1)):
        d.line([(xx, T), (xx + sgn * arm, T)], fill=205, width=lw)
        d.line([(xx, B), (xx + sgn * arm, B)], fill=205, width=lw)
        d.line([(xx, T), (xx, B)], fill=205, width=lw)

def _m_spiral(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    turns = rng.uniform(1.6, 2.6)
    pts = []
    steps = 90
    for i in range(steps):
        t = i / steps * turns * 2 * math.pi
        r = s * 0.42 * (i / steps)
        pts.append((cx + math.cos(t) * r, cy + math.sin(t) * r))
    d.line(pts, fill=200, width=lw)

def _m_table(d, box, rng, lw):
    x0, y0, s, _ = box
    rows = rng.randint(4, 6)
    x1 = x0 + s * 0.74
    y1 = y0 + s * 0.66
    d.rectangle((x0, y0, x1, y1), outline=200, width=lw)
    for i in range(1, rows):
        yy = y0 + (y1 - y0) * i / rows
        d.line([(x0, yy), (x1, yy)], fill=200, width=lw)
    d.line([(x0 + (x1 - x0) * 0.62, y0), (x0 + (x1 - x0) * 0.62, y1)], fill=200, width=lw)

# 要求撤掉了——制图里它们必须指向被注对象，池子里的元素偏偏没有对象可指。
def _k_section(d, box, rng, lw):
    x0, y0, s, _ = box
    y = y0 + s * 0.5
    d.line([(x0 + s * 0.10, y), (x0 + s * 0.90, y)], fill=200, width=lw)
    for xx, sgn in ((x0 + s * 0.10, +1), (x0 + s * 0.90, -1)):
        d.line([(xx, y), (xx + sgn * s * 0.10, y - s * 0.06)], fill=200, width=lw)
        d.line([(xx, y), (xx + sgn * s * 0.10, y + s * 0.06)], fill=200, width=lw)

def _k_pitch(d, box, rng, lw):
    x0, y0, s, _ = box
    y = y0 + s * 0.5
    xa, xb = x0 + s * 0.10, x0 + s * 0.90
    d.line([(xa, y), (xb, y)], fill=200, width=lw)
    n = rng.randint(4, 8)
    for i in range(n + 1):
        xx = xa + (xb - xa) * i / n
        d.line([(xx, y - s * 0.05), (xx, y + s * 0.05)], fill=200, width=lw)

def _k_thread(d, box, rng, lw):
    x0, y0, s, _ = box
    n = rng.randint(4, 7)
    for i in range(n):
        yy = y0 + s * (0.20 + 0.10 * i)
        d.line([(x0 + s * 0.14, yy), (x0 + s * 0.86, yy)], fill=200, width=lw)
    d.line([(x0 + s * 0.14, y0 + s * 0.20), (x0 + s * 0.14, y0 + s * 0.20 + s * 0.10 * (n - 1))],
           fill=200, width=lw)
    d.line([(x0 + s * 0.86, y0 + s * 0.20), (x0 + s * 0.86, y0 + s * 0.20 + s * 0.10 * (n - 1))],
           fill=200, width=lw)

def _k_datum(d, box, rng, lw):
    x0, y0, s, _ = box
    bx, by = x0 + s * 0.30, y0 + s * 0.30
    bs = s * 0.20
    d.rectangle((bx, by, bx + bs, by + bs), outline=205, width=lw)
    d.line([(bx + bs, by + bs / 2), (x0 + s * 0.80, by + bs / 2)], fill=205, width=lw)
    d.polygon([(x0 + s * 0.80, by + bs / 2),
               (x0 + s * 0.74, by + bs / 2 - s * 0.045),
               (x0 + s * 0.74, by + bs / 2 + s * 0.045)], fill=205)

def _k_keyway(d, box, rng, lw):
    x0, y0, s, _ = box
    ox, oy = x0 + s * 0.14, y0 + s * 0.30
    ow, oh = s * 0.62, s * 0.34
    d.rectangle((ox, oy, ox + ow, oy + oh), outline=200, width=lw)
    sh = oh * 0.42
    sy = oy + (oh - sh) * 0.5
    sx, sw = ox + ow * 0.16, ow * 0.68
    d.line([(sx + sh * 0.5, sy), (sx + sw - sh * 0.5, sy)], fill=200, width=lw)
    d.line([(sx + sh * 0.5, sy + sh), (sx + sw - sh * 0.5, sy + sh)], fill=200, width=lw)
    d.arc((sx, sy, sx + sh, sy + sh), 90, 270, fill=200, width=lw)
    d.arc((sx + sw - sh, sy, sx + sw, sy + sh), 270, 90, fill=200, width=lw)

def _k_roughness(d, box, rng, lw):
    x0, y0, s, _ = box
    a = (x0 + s * 0.26, y0 + s * 0.58)
    b = (x0 + s * 0.36, y0 + s * 0.38)
    c = (x0 + s * 0.46, y0 + s * 0.58)
    d.line([a, b], fill=200, width=lw)
    d.line([b, c], fill=200, width=lw)

def _k_detail(d, box, rng, lw):
    x0, y0, s, _ = box
    r = s * 0.26
    cx, cy = x0 + s * 0.58, y0 + s * 0.46
    d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=205, width=lw)
    d.line([(cx - r, cy + r), (x0 + s * 0.14, y0 + s * 0.86)], fill=205, width=lw)

def _p_enso(d, box, rng, lw):
    x0, y0, s, _ = box
    for i in range(2):
        r = s * (0.20 + 0.14 * i)
        cx, cy = x0 + s * (0.46 + 0.06 * i), y0 + s * 0.50
        start = rng.uniform(0, 360)
        d.arc((cx - r, cy - r, cx + r, cy + r), start=start, end=start + rng.uniform(230, 320),
              fill=150, width=lw)

def _p_vertical(d, box, rng, lw):
    x0, y0, s, _ = box
    for i in range(rng.randint(3, 5)):
        xx = x0 + s * (0.14 + 0.17 * i)
        d.line([(xx, y0 + s * 0.16), (xx, y0 + s * 0.84)], fill=170, width=lw)

def _p_quote(d, box, rng, lw):
    x0, y0, s, _ = box
    x1, y1 = x0 + s * 0.76, y0 + s * 0.62
    d.rectangle((x0, y0, x1, y1), outline=165, width=lw)
    d.line([(x0, y0), (x0, y1)], fill=140, width=lw * 3)

def _p_dots(d, box, rng, lw):
    x0, y0, s, _ = box
    n = rng.randint(5, 9)
    for i in range(n):
        px = x0 + s * (0.10 + 0.10 * i)
        py = y0 + s * (0.50 + 0.10 * math.sin(i * 1.1))
        r = max(1.5, s * 0.008)
        d.ellipse((px - r, py - r, px + r, py + r), fill=160)

def _p_seal(d, box, rng, lw):
    x0, y0, s, _ = box
    bs = s * 0.26
    bx, by = x0 + s * 0.36, y0 + s * 0.36
    d.rectangle((bx, by, bx + bs, by + bs), outline=150, width=lw * 2)
    d.line([(bx + bs / 2, by + bs * 0.22), (bx + bs / 2, by + bs * 0.78)], fill=150, width=lw)
    d.line([(bx + bs * 0.22, by + bs / 2), (bx + bs * 0.78, by + bs / 2)], fill=150, width=lw)

def _m_fourier(d, box, rng, lw):
    x0, y0, s, _ = box
    base = y0 + s * 0.55
    for k in range(rng.randint(2, 3)):
        f = rng.uniform(1.4, 3.2)
        ph = rng.uniform(0, 2 * math.pi)
        amp = s * 0.15 / (k + 1)
        pts = [(x0 + s * 0.10 + s * 0.80 * i / 60,
                base + amp * math.sin(2 * math.pi * f * i / 60 + ph)) for i in range(61)]
        d.line(pts, fill=200 - k * 12, width=lw)

def _m_series(d, box, rng, lw):
    x0, y0, s, _ = box
    n = rng.randint(4, 7)
    xa, xb = x0 + s * 0.14, x0 + s * 0.86
    step = (xb - xa) / n
    ys = []
    for i in range(n):
        ys.append(y0 + s * (0.80 - 0.34 * i / max(1, n - 1)))
    for i in range(n):
        d.line([(xa + i * step, ys[i]), (xa + (i + 1) * step, ys[i])], fill=190, width=lw)
        if i < n - 1:
            d.line([(xa + (i + 1) * step, ys[i]), (xa + (i + 1) * step, ys[i + 1])],
                   fill=190, width=lw)
    yl = ys[-1] - s * 0.035
    xx = xa
    while xx < xb:
        d.line([(xx, yl), (min(xx + s * 0.035, xb), yl)], fill=155, width=lw)
        xx += s * 0.07

def _m_delta(d, box, rng, lw):
    x0, y0, s, _ = box
    pts = [(x0 + s * 0.12 + s * 0.76 * i / 50,
            y0 + s * 0.74 - s * 0.34 * math.sin(math.pi * i / 50)) for i in range(51)]
    d.line(pts, fill=200, width=lw)
    i0, i1 = 14, rng.randint(36, 44)
    pa = pts[i0]
    pb = pts[i1]
    d.line([pa, pb], fill=170, width=lw)
    d.line([(pa[0], pa[1]), (pb[0], pa[1])], fill=165, width=lw)
    d.line([(pb[0], pa[1]), (pb[0], pb[1])], fill=165, width=lw)

def _m_shade(d, box, rng, lw):
    x0, y0, s, _ = box
    def curve(t):
        return y0 + s * 0.30 + s * 0.32 * max(t, 0.0) ** 1.6
    pts = []
    for i in range(61):
        pts.append((x0 + s * 0.12 + s * 0.76 * i / 60, curve(i / 60)))
    d.line(pts, fill=200, width=lw)
    t0 = rng.uniform(0.22, 0.40)
    t1 = min(0.92, t0 + rng.uniform(0.18, 0.30))
    base = y0 + s * 0.76
    xa = x0 + s * 0.12 + s * 0.76 * t0
    xb = x0 + s * 0.12 + s * 0.76 * t1
    xx = xa
    step = max(3.0, s * 0.012)
    while xx < xb:
        t = (xx - (x0 + s * 0.12)) / (s * 0.76)
        d.line([(xx, curve(t)), (xx, base)], fill=138, width=1)
        xx += step
    d.line([(xa, base), (xb, base)], fill=190, width=lw)

def _k_chamfer(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.34, y0 + s * 0.66
    ln = s * 0.14
    d.line([(cx, cy), (cx + ln, cy - ln)], fill=200, width=lw)
    d.line([(cx + ln * 0.55, cy + ln * 0.55), (cx + ln * 1.55, cy - ln * 0.45)],
           fill=200, width=lw)

def _k_fit(d, box, rng, lw):
    x0, y0, s, _ = box
    xa, xb = x0 + s * 0.14, x0 + s * 0.86
    yc = y0 + s * 0.50
    h1, h2 = s * 0.09, s * 0.13
    d.line([(xa, yc - h1), (xb, yc - h1)], fill=200, width=lw)
    d.line([(xa, yc + h2), (xb, yc + h2)], fill=200, width=lw)
    xx = xa
    step = max(4.0, s * 0.022)
    while xx < xb - (h1 + h2):
        d.line([(xx, yc + h2), (xx + (h1 + h2), yc - h1)], fill=148, width=1)
        xx += step

def _k_gear(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    r = s * 0.26
    teeth = rng.choice([12, 16, 20])
    pts = []
    for i in range(teeth * 2):
        a = 2 * math.pi * i / (teeth * 2)
        rr = r if i % 2 == 0 else r * 0.82
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr))
    d.line(pts + [pts[0]], fill=200, width=lw)
    d.ellipse((cx - r * 0.32, cy - r * 0.32, cx + r * 0.32, cy + r * 0.32),
              outline=200, width=lw)

def _k_centers(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    L = s * 0.36
    for dx, dy in ((1, 0), (0, 1)):
        t = 0.0
        x = cx - L * dx
        y = cy - L * dy
        while t < 2 * L:
            seg = s * 0.05 if int(t / (s * 0.10)) % 2 == 0 else s * 0.012
            seg = min(seg, 2 * L - t)
            d.line([(x + dx * t, y + dy * t), (x + dx * (t + seg), y + dy * (t + seg))],
                   fill=172, width=max(1, lw - 1))
            t += seg + s * 0.018
    d.ellipse((cx - s * 0.13, cy - s * 0.13, cx + s * 0.13, cy + s * 0.13),
              outline=200, width=lw)

def _p_book(d, box, rng, lw):
    x0, y0, s, _ = box
    for i, wd in enumerate((max(2, lw * 2), lw, max(3, lw * 3))):
        x = x0 + s * (0.24 + i * 0.14)
        d.line([(x, y0 + s * 0.16), (x, y0 + s * 0.84)], fill=136 + i * 12, width=wd)

def _p_enso2(d, box, rng, lw):
    x0, y0, s, _ = box
    cx, cy = x0 + s * 0.5, y0 + s * 0.5
    for k in range(2):
        r = s * (0.29 - k * 0.11)
        if r <= 4:
            continue
        start = rng.uniform(0, 360)
        d.arc((cx - r, cy - r, cx + r, cy + r), start, start + rng.uniform(225, 335),
              fill=138, width=max(1, lw + 1))

def _p_speck(d, box, rng, lw):
    x0, y0, s, _ = box
    for _ in range(rng.randint(10, 18)):
        px = x0 + s * rng.uniform(0.14, 0.86)
        py = y0 + s * rng.uniform(0.24, 0.76)
        r = max(0.8, s * rng.uniform(0.004, 0.012))
        d.ellipse((px - r, py - r, px + r, py + r), fill=128)

def _p_grid4(d, box, rng, lw):
    x0, y0, s, _ = box
    bs = s * 0.34
    bx, by = x0 + s * 0.33, y0 + s * 0.33
    d.rectangle((bx, by, bx + bs, by + bs), outline=148, width=lw)
    for i in range(10):
        t0 = i * bs / 10
        t1 = min(t0 + bs / 20, bs)
        d.line([(bx + t0, by + bs / 2), (bx + t1, by + bs / 2)], fill=138, width=1)
        d.line([(bx + bs / 2, by + t0), (bx + bs / 2, by + t1)], fill=138, width=1)

POOLS: Dict[str, List[Tuple[str, Callable, float]]] = {
    "math": [
        ("contours", _m_contours, 0.20),
        ("polar", _m_polar, 0.16),
        ("matrix", _m_matrix, 0.13),
        ("spiral", _m_spiral, 0.11),
        ("table", _m_table, 0.10),
        ("fourier", _m_fourier, 0.12),
        ("series", _m_series, 0.10),
        ("delta", _m_delta, 0.10),
        ("shade", _m_shade, 0.10),
    ],
    "mech": [
        ("thread", _k_thread, 0.16),
        ("roughness", _k_roughness, 0.15),
        ("pitch", _k_pitch, 0.14),
        ("keyway", _k_keyway, 0.15),
        ("chamfer", _k_chamfer, 0.13),
        ("fit", _k_fit, 0.13),
        ("gear", _k_gear, 0.14),
        ("centers", _k_centers, 0.14),
    ],
    "philo": [
        ("enso", _p_enso, 0.18),
        ("vertical", _p_vertical, 0.14),
        ("quote", _p_quote, 0.13),
        ("dots", _p_dots, 0.11),
        ("seal", _p_seal, 0.11),
        ("book", _p_book, 0.11),
        ("enso2", _p_enso2, 0.11),
        ("speck", _p_speck, 0.11),
        ("grid4", _p_grid4, 0.10),
    ],
}

EXTRA_NAMES = {k: [n for n, _, _ in v] for k, v in POOLS.items()}

_COUNT = {"sparse": 0, "normal": 1, "rich": 2}

def apply_motif_extras(
    canvas: np.ndarray,
    w: int,
    h: int,
    rects: Dict[str, Rect],
    rng,
    tone,
    discipline: str,
    richness: str = "normal",
    amount: float = 1.0,
) -> np.ndarray:
    """在画布上叠 0–2 件附加学科元素。discipline 取 math / mech / philo。"""
    pool = POOLS.get(discipline)
    if not pool or amount <= 0:
        return canvas
    k = _COUNT.get(richness, 1)
    if k <= 0:
        return canvas

    names = [n for n, _, _ in pool]
    weights = [wt for _, _, wt in pool]
    picks: List[str] = []
    for _ in range(k):
        avail = [i for i, n in enumerate(names) if n not in picks]
        if not avail:
            break
        wsum = sum(weights[i] for i in avail)
        r = rng.uniform(0, wsum)
        acc = 0.0
        for i in avail:
            acc += weights[i]
            if r <= acc:
                picks.append(names[i])
                break

    layer = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(layer)
    lw = max(1, int(min(w, h) * 0.0018))
    for name in picks:
        fn = next(f for n, f, _ in pool if n == name)
        size = rng.uniform(0.18, 0.30)
        box = _free_box(rects, w, h, rng, size)
        fn(d, box, rng, lw)

    from make_backgrounds import _zone_mask

    keep = _zone_mask(w, h, rects, margin=0.04)[..., 0]
    alpha = (np.asarray(layer, dtype=np.float32) / 255.0) * (0.28 + 0.72 * keep) * (0.95 * amount)
    a = alpha[..., None]
    from make_backgrounds import _line_color, _bg_luma  # noqa: E402

    dark = _bg_luma(canvas) < 128.0
    col = np.array(_line_color(tone, dark), dtype=np.float32)[None, None, :]
    return np.clip(canvas * (1.0 - a) + col * a, 0, 255)
