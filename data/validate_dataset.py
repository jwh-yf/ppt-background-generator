"""数据集体检：把生成完了变成可以拿去训练了。

C3 安全区  各语义区域的局部对比度是否低于阈值（越低越适合压字）"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from presets import ASPECTS, PRESETS, TONES  # noqa: E402

TONE_NAME_EN: Dict[str, str] = {k: t.name_en for k, t in TONES.items()}

SAFE_ZONE_STD_MAX = 40.0
SAFE_ZONE_STD_GOOD = 26.0
LUMA_DARK_MIN = 25.0
LUMA_LIGHT_MAX = 236.0
DHASH_DISTANCE = 4
COLOR_DIFF_MAX = 8.0
MAX_DUP_RATIO = 0.05

def load_records(path: Path) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return recs

def dhash(img: Image.Image, size: int = 8) -> int:
    """差值哈希。对缩放/压缩不敏感，够用来抓重复合成图。"""
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out

def color_signature(img: Image.Image, size: int = 8) -> np.ndarray:
    """缩到 8x8 的 RGB 均值向量，用来和 dHash 配合判断是不是同一张。
    
    否则近似重复报告会淹没真正的问题。"""
    small = img.convert("RGB").resize((size, size), Image.LANCZOS)
    return np.asarray(small, dtype=np.float32).reshape(-1)

def is_duplicate(h1: int, c1: np.ndarray, h2: int, c2: np.ndarray) -> Tuple[bool, int, float]:
    """返回 (是否重复, 汉明距离, 平均通道差)。结构接近 且 颜色接近才算重复。"""
    dist = hamming(h1, h2)
    cdiff = float(np.abs(c1 - c2).mean())
    return (dist <= DHASH_DISTANCE and cdiff <= COLOR_DIFF_MAX), dist, cdiff

def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")

def entropy(counts: Iterable[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c > 0)

def zone_stats(arr: np.ndarray, rect: Dict[str, Any]) -> Tuple[float, float, float]:
    x, y, w, h = int(rect["x"]), int(rect["y"]), int(rect["w"]), int(rect["h"])
    crop = arr[max(0, y) : y + h, max(0, x) : x + w, :]
    if crop.size == 0:
        return 255.0, 128.0, 255.0
    r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    dx = np.abs(np.diff(luma, axis=1)).mean() if luma.shape[1] > 1 else 0.0
    dy = np.abs(np.diff(luma, axis=0)).mean() if luma.shape[0] > 1 else 0.0
    return float(luma.std()), float(luma.mean()), float((dx + dy) / 2.0)

def audit(records: Sequence[Dict[str, Any]], images_root: Path) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []
    per_image: List[Dict[str, Any]] = []
    hashes: List[Tuple[str, int, np.ndarray]] = []

    for rec in records:
        img_id = rec.get("id") or rec.get("file", "?")
        path = images_root / rec.get("file", "")
        row: Dict[str, Any] = {"id": img_id, "file": rec.get("file", "")}

        if not path.is_file():
            issues.append({"code": "C1", "id": img_id, "msg": f"文件缺失：{path.name}"})
            row["ok"] = False
            per_image.append(row)
            continue
        try:
            with Image.open(path) as im:
                im.load()
                size = im.size
                arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            issues.append({"code": "C1", "id": img_id, "msg": f"无法解码：{exc}"})
            row["ok"] = False
            per_image.append(row)
            continue

        if size != (rec.get("width"), rec.get("height")):
            issues.append(
                {"code": "C1", "id": img_id, "msg": f"尺寸不符：文件 {size} vs 记录 {(rec.get('width'), rec.get('height'))}"}
            )

        # 但两者必须对得上，对不上说明记录不可信，所以顺手做一致性核对。
        zones = rec.get("safe_zones") or []
        if not zones:
            issues.append({"code": "C3", "id": img_id, "msg": "记录里没有安全区元数据"})
        zone_std, zone_luma, zone_hf = [], [], []
        drift = 0.0
        for z in zones:
            std, mean, hf = zone_stats(arr, z)
            zone_std.append(std)
            zone_luma.append(mean)
            zone_hf.append(hf)
            recorded = z.get("contrast_std")
            if recorded is not None:
                drift = max(drift, abs(float(recorded) - std))
            if std > SAFE_ZONE_STD_MAX:
                issues.append(
                    {"code": "C3", "id": img_id, "msg": f"安全区 {z.get('region')} 对比度 {std:.1f} > {SAFE_ZONE_STD_MAX}"}
                )
        if drift > 2.0:
            issues.append(
                {"code": "C3", "id": img_id, "msg": f"安全区对比度元数据与像素不符（最大偏差 {drift:.1f}）"}
            )
        row.update(
            {
                "n_zones": len(zones),
                "zone_std_max": round(max(zone_std), 2) if zone_std else None,
                "zone_std_mean": round(float(np.mean(zone_std)), 2) if zone_std else None,
                "zone_hf_mean": round(float(np.mean(zone_hf)), 3) if zone_hf else None,
                "zone_luma_mean": round(float(np.mean(zone_luma)), 2) if zone_luma else None,
                "zone_meta_drift": round(drift, 3),
            }
        )

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        row["global_luma"] = round(float(luma.mean()), 2)
        row["global_std"] = round(float(luma.std()), 2)
        row["dark_ratio"] = round(float((luma < 40).mean()), 4)
        row["bright_ratio"] = round(float((luma > 235).mean()), 4)
        if row["global_luma"] < LUMA_DARK_MIN:
            issues.append({"code": "C4", "id": img_id, "msg": f"整体过暗：{row['global_luma']}"})
        if row["global_luma"] > LUMA_LIGHT_MAX:
            issues.append({"code": "C4", "id": img_id, "msg": f"整体过亮：{row['global_luma']}"})

        with Image.open(path) as im_dup:
            h = dhash(im_dup)
            sig = color_signature(im_dup)
        row["dhash"] = f"{h:016x}"
        struct_only = 0
        for other_id, other_h, other_sig in hashes:
            dup, dist, cdiff = is_duplicate(h, sig, other_h, other_sig)
            if dup:
                issues.append(
                    {"code": "C5", "id": img_id,
                     "msg": f"与 {other_id} 近似重复（汉明 {dist}，色差 {cdiff:.1f}）"}
                )
            elif dist <= DHASH_DISTANCE:
                struct_only += 1
        hashes.append((img_id, h, sig))
        row["struct_similar_count"] = struct_only

        caption = rec.get("caption") or ""
        recipe = rec.get("recipe") or {}
        tone_key = recipe.get("tone")
        if not caption:
            issues.append({"code": "C6", "id": img_id, "msg": "caption 为空"})
        if "no text" not in caption.lower():
            issues.append({"code": "C6", "id": img_id, "msg": "caption 缺少no text负向约束"})
        if recipe.get("accent_hex") and recipe["accent_hex"].lower() not in caption.lower():
            issues.append({"code": "C6", "id": img_id, "msg": f"caption 未包含 accent 色 {recipe.get('accent_hex')}"})
        # 属于校验器比数据更严的假警报，必须按 name_en 判定。
        if tone_key:
            tone_name = TONE_NAME_EN.get(tone_key, tone_key.replace("_", " "))
            if tone_name.lower() not in caption.lower() and tone_key not in caption:
                issues.append(
                    {"code": "C6", "id": img_id, "msg": f"caption 未体现 tone={tone_key}（应为 '{tone_name}'）"}
                )
        sources = rec.get("sources") or []
        if not sources:
            issues.append({"code": "C6", "id": img_id, "msg": "缺少素材来源（sources）"})
        for s in sources:
            if not s.get("license"):
                issues.append({"code": "C6", "id": img_id, "msg": f"素材 {s.get('asset_id')} 未标注许可证"})

        row["preset"] = rec.get("preset")
        row["aspect"] = rec.get("aspect")
        row["layout"] = recipe.get("layout")
        row["panel_style"] = recipe.get("panel_style")
        row["tone"] = tone_key
        row["asset_main"] = sources[0].get("asset_id") if sources else None
        row["ok"] = True
        per_image.append(row)

    def dist(field: str) -> Dict[str, Any]:
        c = Counter(r.get(field) for r in per_image if r.get(field))
        total = sum(c.values())
        top_key, top_n = (c.most_common(1)[0] if c else (None, 0))
        return {
            "counts": dict(c.most_common()),
            "n_categories": len(c),
            "entropy_bits": round(entropy(c.values()), 3),
            "max_share": round(top_n / total, 4) if total else 0.0,
            "top": top_key,
        }

    presets_dist = dist("preset")
    layouts_dist = dist("layout")
    aspects_dist = dist("aspect")
    styles_dist = dist("panel_style")
    tones_dist = dist("tone")
    assets_dist = dist("asset_main")

    if presets_dist["max_share"] > 0.35:
        issues.append(
            {"code": "C2", "id": "-", "msg": f"预设分布失衡：{presets_dist['top']} 占 {presets_dist['max_share']:.1%}"}
        )
    if layouts_dist["max_share"] > 0.45:
        issues.append(
            {"code": "C2", "id": "-", "msg": f"版式分布失衡：{layouts_dist['top']} 占 {layouts_dist['max_share']:.1%}"}
        )
    if assets_dist["max_share"] > 0.10:
        issues.append(
            {
                "code": "C2",
                "id": "-",
                "msg": f"单一素材占比过高：{assets_dist['top']} 占 {assets_dist['max_share']:.1%}（模型会记住这张图）",
            }
        )

    dup_issues = [i for i in issues if i["code"] == "C5"]
    n_ok = sum(1 for r in per_image if r.get("ok"))
    dup_ratio = len(dup_issues) / max(1, n_ok)
    if dup_ratio > MAX_DUP_RATIO:
        issues.append(
            {"code": "C5", "id": "-", "msg": f"近似重复比例 {dup_ratio:.1%} 超过上限 {MAX_DUP_RATIO:.0%}"}
        )

    by_code: Dict[str, int] = defaultdict(int)
    for i in issues:
        by_code[i["code"]] += 1

    summary = {
        "records": len(records),
        "decodable": n_ok,
        "images_root": str(images_root),
        "criteria": {
            "safe_zone_std_max": SAFE_ZONE_STD_MAX,
            "safe_zone_std_good": SAFE_ZONE_STD_GOOD,
            "luma_range": [LUMA_DARK_MIN, LUMA_LIGHT_MAX],
            "dhash_distance": DHASH_DISTANCE,
            "color_diff_max": COLOR_DIFF_MAX,
            "max_dup_ratio": MAX_DUP_RATIO,
        },
        "distributions": {
            "preset": presets_dist,
            "layout": layouts_dist,
            "aspect": aspects_dist,
            "panel_style": styles_dist,
            "tone": tones_dist,
            "main_asset": {
                "n_categories": assets_dist["n_categories"],
                "max_share": assets_dist["max_share"],
                "top": assets_dist["top"],
            },
        },
        "aggregate": {
            "zone_std_max_mean": round(
                float(np.mean([r["zone_std_max"] for r in per_image if r.get("zone_std_max") is not None])), 2
            )
            if any(r.get("zone_std_max") is not None for r in per_image)
            else None,
            "zone_hf_mean": round(
                float(np.mean([r["zone_hf_mean"] for r in per_image if r.get("zone_hf_mean") is not None])), 4
            )
            if any(r.get("zone_hf_mean") is not None for r in per_image)
            else None,
            "global_luma_mean": round(float(np.mean([r["global_luma"] for r in per_image if r.get("global_luma")])), 2)
            if any(r.get("global_luma") for r in per_image)
            else None,
            "dup_ratio": round(dup_ratio, 4),
            "dup_count": len(dup_issues),
            "struct_similar_pairs": sum(int(r.get("struct_similar_count") or 0) for r in per_image),
            "zone_meta_drift_max": round(
                max((float(r.get("zone_meta_drift") or 0.0) for r in per_image), default=0.0), 3
            ),
        },
        "issues_by_code": dict(sorted(by_code.items())),
        "issues": issues[:400],
        "issues_truncated": max(0, len(issues) - 400),
        "verdict": verdict(issues, n_ok, len(records), dup_ratio),
    }
    return {"summary": summary, "per_image": per_image}

def verdict(
    issues: Sequence[Dict[str, Any]],
    n_ok: int,
    n_total: int,
    dup_ratio: float = 0.0,
) -> Dict[str, Any]:
    """判定等级。
    
    于是改到 0 重复才让过变成了不可能达成的目标，指标也就失去了指导作用。"""
    c5_hard = [i for i in issues if i["code"] == "C5" and i["id"] == "-" and dup_ratio > MAX_DUP_RATIO]
    hard = [i for i in issues if i["code"] == "C1" and i["id"] != "-"] + c5_hard
    soft = [i for i in issues if i["code"] in ("C3", "C4", "C6")]
    soft += [i for i in issues if i["code"] == "C5" and i["id"] != "-"]
    if n_ok < n_total * 0.98 or hard:
        level = "fail"
    elif len(soft) > n_total * 0.25 or dup_ratio > 0.0:
        level = "warn"
    else:
        level = "pass"
    return {
        "level": level,
        "hard_issues": len(hard),
        "soft_issues": len(soft),
        "dup_ratio": round(dup_ratio, 4),
        "note": {
            "pass": "可直接用于训练",
            "warn": "可训练；建议看一眼 issues 里的软性问题（安全区对比度偏高 / 少量近似重复）",
            "fail": "存在硬性问题（缺图/不可解码/重复率超上限），修好再训",
        }[level],
    }

def write_outputs(result: Dict[str, Any], out_prefix: Path) -> None:
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    (out_prefix.with_suffix(".json")).write_text(
        json.dumps(result["summary"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    csv_path = out_prefix.with_suffix(".csv")
    rows = result["per_image"]
    fields = [
        "id", "file", "ok", "preset", "aspect", "layout", "panel_style", "tone", "asset_main",
        "n_zones", "zone_std_max", "zone_std_mean", "zone_hf_mean", "zone_luma_mean", "zone_meta_drift",
        "global_luma", "global_std", "dark_ratio", "bright_ratio", "dhash", "struct_similar_count",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    issues_path = out_prefix.with_name(out_prefix.name + "_issues.jsonl")
    with issues_path.open("w", encoding="utf-8") as fh:
        for i in result["summary"]["issues"]:
            fh.write(json.dumps(i, ensure_ascii=False) + "\n")

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="合成数据集体检")
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--images-root", type=Path, default=None, help="默认取 records 同级的 images/")
    ap.add_argument("--out", type=Path, required=True, help="报告前缀，如 data/dataset_v1/report")
    args = ap.parse_args(argv)

    records = load_records(args.records)
    images_root = args.images_root or args.records.parent / "images"
    result = audit(records, images_root)
    write_outputs(result, args.out)

    s = result["summary"]
    print(json.dumps({k: s[k] for k in ("records", "decodable", "aggregate", "issues_by_code", "verdict")},
                     ensure_ascii=False, indent=2))
    print(f"\n报告：{args.out.with_suffix('.json')}")
    print(f"逐图指标：{args.out.with_suffix('.csv')}")
    print(f"问题明细：{args.out.with_name(args.out.name + '_issues.jsonl')}")
    return 0 if s["verdict"]["level"] != "fail" else 2

if __name__ == "__main__":
    raise SystemExit(main())
