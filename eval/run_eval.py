"""评测脚本：把有 LoRA / 无 LoRA / 纯程序化 / 随机噪声放到同一把尺子下量。

固定 40 条中文需求、同一 seed 基线，五个后端跑同一条 SDService 流水线。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
for _extra in (ROOT, ROOT / "data", ROOT / "app", ROOT / "train"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from presets import ASPECTS, PRESETS, TONES, compose_prompt  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_mkbg_eval", ROOT / "data" / "make_backgrounds.py")
_mkbg = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mkbg)  # type: ignore[union-attr]

_ASSETS_ROOT = ROOT / "data" / "assets" / "raw"
_MANIFEST_PATH = ROOT / "data" / "assets" / "manifest.jsonl"
_ASSET_POOL_CACHE: Optional[List[Dict[str, Any]]] = None

def _load_asset_pool() -> List[Dict[str, Any]]:
    global _ASSET_POOL_CACHE
    if _ASSET_POOL_CACHE is None:
        if not _MANIFEST_PATH.exists():
            raise SystemExit(
                f"找不到母图清单 {_MANIFEST_PATH}；请先运行 data\\make_placeholder_assets.py"
            )
        _ASSET_POOL_CACHE = _mkbg.load_assets(_MANIFEST_PATH)
    return _ASSET_POOL_CACHE

# procedural_raw 是消融后端（回答预设+版式+安全区约束值多少分），
BACKENDS = ("lora", "sd_base", "procedural", "procedural_raw", "noise")
# TODO 配色变体、柔焦、磨砂、笔触这些可调项都没进评测口径，第 6 节的数字对应的是默认档。
#      要补的话按 6 个可调项各跑一轮 40 条，五后端乘起来太贵，只跑 lora 一档就够看趋势。
METRIC_KEYS = (
    "safe_zone_std", "safe_zone_colored", "artifact_hf",
    "relevance", "contrast_headroom", "layout_ok", "seconds",
)

# 否则基线被不公平地压到 0 分，对比就没有意义了。
KEYWORD_PRESET: Dict[str, Tuple[str, ...]] = {
    "academic_blue": ("学术", "论文", "答辩", "课程", "科研", "实验", "报告", "建模", "算法", "数据"),
    "clean_white": ("极简", "产品", "商业", "计划", "简洁", "发布", "路演", "创业"),
    "warm_terra": ("人文", "社科", "文化", "公益", "历史", "文学", "志愿", "乡村", "教育"),
    "forest_deep": ("环保", "生态", "农业", "生命", "碳", "绿色", "能源", "气候", "林业"),
    "violet_tech": ("科技", "技术", "竞赛", "编程", "人工智能", "机器学习", "芯片", "网络", "安全"),
    "ink_mono": ("水墨", "艺术", "总结", "回顾", "正式", "周年", "纪念", "年度", "汇报"),
}

def zone_slice(arr: np.ndarray, z: Dict[str, Any]) -> np.ndarray:
    x, y, w, h = int(z["x"]), int(z["y"]), int(z["w"]), int(z["h"])
    return arr[max(0, y) : y + h, max(0, x) : x + w, :]

def laplacian_energy(luma: np.ndarray) -> float:
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    lap = (
        4.0 * luma[1:-1, 1:-1]
        - luma[:-2, 1:-1] - luma[2:, 1:-1]
        - luma[1:-1, :-2] - luma[1:-1, 2:]
    )
    return float(np.percentile(np.abs(lap), 95))

def measure(img: Image.Image, record: Dict[str, Any], prompt: str, preset_key: str,
            seconds: float) -> Dict[str, Any]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    zones = record.get("safe_zones") or []
    if not zones:
        h, w = arr.shape[:2]
        zones = [{"region": "full", "x": 0, "y": 0, "w": w, "h": h}]

    stds, coloreds, hfs, headrooms = [], [], [], []
    for z in zones:
        crop = zone_slice(arr, z)
        if crop.size == 0:
            continue
        r, g, b = crop[..., 0], crop[..., 1], crop[..., 2]
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        mx, mn = crop.max(axis=2), crop.min(axis=2)
        sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
        stds.append(float(luma.std()))
        coloreds.append(float((sat > 0.25).mean()))
        hfs.append(laplacian_energy(luma))
        headrooms.append(max(255.0 - float(luma.mean()), float(luma.mean())))

    relevance, layout_ok = score_relevance(preset_key, record, prompt)

    return {
        "safe_zone_std": round(float(np.mean(stds)), 2) if stds else None,
        "safe_zone_colored": round(float(np.mean(coloreds)), 4) if coloreds else None,
        "artifact_hf": round(float(np.mean(hfs)), 2) if hfs else None,
        "contrast_headroom": round(float(np.mean(headrooms)), 1) if headrooms else None,
        "relevance": relevance,
        "layout_ok": layout_ok,
        "seconds": round(seconds, 2),
        "n_zones": len(stds),
    }

def score_relevance(preset_key: str, record: Dict[str, Any], prompt: str) -> Tuple[float, float]:
    hit = prompt
    preset = PRESETS[preset_key]
    tone = TONES[preset.tone]

    kws = KEYWORD_PRESET.get(preset_key, ())
    kw_hit = 1.0 if any(k in hit for k in kws) else 0.0

    tone_rgb = np.array(tone.tint, dtype=np.float32)
    if record.get("_tone_probe") is not None:
        probe = np.array(record["_tone_probe"], dtype=np.float32)
        tone_dist = float(np.abs(probe - tone_rgb).mean())
        tone_score = float(np.clip(1.0 - tone_dist / 90.0, 0.0, 1.0))
    else:
        tone_score = 0.0

    layout_ok = 1.0 if record.get("_layout_ok") else 0.0
    return round(0.5 * kw_hit + 0.3 * tone_score + 0.2 * layout_ok, 4), layout_ok

def tone_probe(img: Image.Image) -> List[float]:
    tiny = img.convert("RGB").resize((8, 8), Image.LANCZOS)
    return [float(v) for v in np.asarray(tiny, dtype=np.float32).reshape(-1, 3).mean(axis=0)]

def layout_compliance(text_zones: List[Dict[str, Any]], target_std: float = 26.0) -> bool:
    """留白有没有留出来——量的是放字的地方够不够干净，而不是谁比谁干净。
    
    必须低于 `target_std`**。这既是可读性的直接度量，也和 `safe_zone_std` 同源、不会互相打架。"""
    want = [z for z in (text_zones or []) if z.get("region") in ("title", "body")]
    if not want:
        return False
    return all(float(z.get("contrast_std", 255.0)) <= target_std for z in want)

def infer_preset(prompt: str, seed: int) -> str:
    scored: List[Tuple[int, str]] = []
    for key, kws in KEYWORD_PRESET.items():
        n = sum(1 for k in kws if k in prompt)
        scored.append((n, key))
    scored.sort(key=lambda t: (-t[0], t[1]))
    if scored[0][0] > 0:
        return scored[0][1]
    return random.Random(seed).choice(sorted(PRESETS))

class Backend:
    name = "base"

    def setup(self, args: argparse.Namespace) -> None:  # noqa: B027
        pass

    def generate(self, prompt: str, preset_key: str, seed: int, args: argparse.Namespace) -> Tuple[Image.Image, Dict[str, Any]]:
        raise NotImplementedError

class ProceduralBackend(Backend):

    name = "procedural"

    def generate(self, prompt, preset_key, seed, args):
        pool = _load_asset_pool()
        img, meta = _mkbg.synthesize_background(
            preset_key,
            args.aspect,
            seed,
            _ASSETS_ROOT,
            pool,
            with_panel=True,
        )
        meta["_layout_ok"] = layout_compliance(meta.get("safe_zones"))
        meta["_tone_probe"] = tone_probe(img)
        meta["_safe_zone_source"] = "synthesize_background"
        return img, meta

class NoiseBackend(Backend):

    name = "noise"

    def generate(self, prompt, preset_key, seed, args):
        w, h = ASPECTS[args.aspect]
        gen = np.random.default_rng(seed)
        img = Image.fromarray(
            np.clip(gen.integers(60, 200, size=(h, w, 3)), 0, 255).astype(np.uint8), mode="RGB"
        )
        img = img.filter(ImageFilter.GaussianBlur(max(3.0, min(w, h) * 0.02)))
        arr = np.clip(np.asarray(img, dtype=np.float32) + gen.normal(0, 22, size=(h, w, 3)), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return img, {"layout": "none", "safe_zones": [], "_layout_ok": False}

class RawProceduralBackend(Backend):
    """消融：同一套影调 + 纹理 + 暗角 + 颗粒，但不做面板、不留安全区。
    
    这个后端回答一个尖锐的问题：作品里那套预设 + 版式 + 安全区约束到底值多少分？"""

    name = "procedural_raw"

    def generate(self, prompt, preset_key, seed, args):
        pool = _load_asset_pool()
        img, meta = _mkbg.synthesize_background(
            preset_key,
            args.aspect,
            seed,
            _ASSETS_ROOT,
            pool,
            with_panel=False,
        )
        meta["_layout_ok"] = layout_compliance(meta.get("safe_zones"))
        meta["_tone_probe"] = tone_probe(img)
        meta["_safe_zone_source"] = "synthesize_background(no panel)"
        return img, meta

class DiffusersBackend(Backend):
    """直接复用 `app/sd_service.SDService`，保证评测测的就是线上那条流水线。"""

    def __init__(self, name: str, use_lora: bool, base_model: str, lora_dir: Optional[Path]) -> None:
        self.name = name
        self.use_lora = use_lora
        self.base_model = base_model
        self.lora_dir = lora_dir
        self._svc = None
        self._scratch: Optional[Path] = None

    def setup(self, args):
        from sd_service import SDService
        from train.config import InferConfig

        if self.use_lora and (self.lora_dir is None or not Path(self.lora_dir).exists()):
            raise SystemExit(f"找不到 LoRA 权重目录 {self.lora_dir}，请先训练")
        cfg = InferConfig(
            base_model=str(self.base_model),
            lora_dir=str(self.lora_dir) if self.use_lora else "",
            steps=args.steps,
        )
        self._svc = SDService(cfg)
        backend = self._svc.load()
        if backend != "diffusers":
            raise SystemExit(
                f"评测需要真模型后端，但 SDService 降级成了 {backend}："
                f"{getattr(self._svc, '_load_error', '')}"
            )
        self._scratch = ROOT / "eval" / "_scratch" / self.name
        self._scratch.mkdir(parents=True, exist_ok=True)

    def generate(self, prompt, preset_key, seed, args):
        from sd_service import GenerateRequest

        req = GenerateRequest(
            preset_key=preset_key,
            subject=prompt[:200],
            aspect=args.aspect,
            count=1,
            steps=args.steps,
            seed=seed,
            enforce_safe_zone=False,
        )
        rec = self._svc.generate(req, self._scratch)[0]
        img = Image.open(self._scratch / rec["file"]).convert("RGB")
        zones = rec.get("safe_zones") or []
        meta = {
            "layout": rec.get("layout"),
            "panel_geom": rec.get("panel_geom"),
            "safe_zones": zones,
            "_layout_ok": layout_compliance(zones),
            "_prompt": rec.get("prompt"),
        }
        return img, meta

def load_prompts(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            rows.append(json.loads(line))
    return rows

def build_backend(name: str, args: argparse.Namespace, lora_dir: Optional[Path], base_model: str) -> Backend:
    if name == "procedural":
        return ProceduralBackend()
    if name == "procedural_raw":
        return RawProceduralBackend()
    if name == "noise":
        return NoiseBackend()
    if name == "lora":
        return DiffusersBackend("lora", True, base_model, lora_dir)
    if name == "sd_base":
        return DiffusersBackend("sd_base", False, base_model, None)
    raise SystemExit(f"未知后端 {name}")

def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for backend in sorted({r["backend"] for r in rows}):
        sub = [r for r in rows if r["backend"] == backend]
        agg: Dict[str, Any] = {"n": len(sub)}
        for key in METRIC_KEYS:
            vals = [r[key] for r in sub if r.get(key) is not None]
            if not vals:
                agg[key] = None
                continue
            agg[key] = {
                "mean": round(float(statistics.fmean(vals)), 3),
                "median": round(float(statistics.median(vals)), 3),
                "min": round(float(min(vals)), 3),
                "max": round(float(max(vals)), 3),
            }
        for flag in ("safe_zone_ok", "layout_compliance_rate"):
            vals = [1.0 if r.get(flag) else 0.0 for r in sub]
            agg[flag] = round(float(statistics.fmean(vals)), 4) if vals else None
        out[backend] = agg
    return out

def verdict_table(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    lower_better = {"safe_zone_std", "safe_zone_colored", "artifact_hf", "seconds"}
    rows: List[Dict[str, Any]] = []
    for key in METRIC_KEYS:
        vals = {b: v.get(key, {}).get("mean") for b, v in summary.items() if v.get(key)}
        vals = {b: v for b, v in vals.items() if v is not None}
        if not vals:
            continue
        best = min(vals, key=vals.get) if key in lower_better else max(vals, key=vals.get)
        worst = max(vals, key=vals.get) if key in lower_better else min(vals, key=vals.get)
        ratio = None
        if vals[worst] not in (0, None) and vals[best] is not None:
            ratio = round(float(vals[worst]) / float(vals[best]), 2) if vals[best] else None
        rows.append(
            {
                "metric": key,
                "better": "越低越好" if key in lower_better else "越高越好",
                "best_backend": best,
                "best_value": vals[best],
                "worst_backend": worst,
                "worst_value": vals[worst],
                "spread_x": ratio,
                "all": {b: round(float(v), 3) for b, v in vals.items()},
            }
        )
    return rows

def write_markdown(summary: Dict[str, Any], table: List[Dict[str, Any]], path: Path, meta: Dict[str, Any]) -> None:
    lines = ["# 评测结果（自动生成，勿手改）", ""]
    lines.append(f"- 生成时间：{meta['created']}")
    lines.append(f"- 需求条数：{meta['prompts']}　比例：{meta['aspect']}　步数：{meta['steps']}")
    lines.append(f"- 后端：{', '.join(sorted(summary))}")
    lines.append("")
    lines.append("## 指标对比（均值）")
    lines.append("")
    backends = sorted(summary)
    lines.append("| 指标 | 方向 | " + " | ".join(backends) + " | 最优 | 差距 |")
    lines.append("|---|---|" + "---|" * (len(backends) + 2))
    for row in table:
        cells = " | ".join(
            ("—" if summary[b].get(row["metric"]) is None else f"{summary[b][row['metric']]['mean']:g}")
            for b in backends
        )
        spread = f"{row['spread_x']}×" if row["spread_x"] else "—"
        lines.append(f"| {row['metric']} | {row['better']} | {cells} | {row['best_backend']} | {spread} |")
    lines.append("")
    lines.append("## 可用性达标率")
    lines.append("")
    lines.append("| 后端 | 安全区达标率(≤26) | 版式合规率 | 单张耗时(s) |")
    lines.append("|---|---|---|---|")
    for b in backends:
        s = summary[b]
        sec = s.get("seconds")
        lines.append(
            f"| {b} | {s.get('safe_zone_ok')} | {s.get('layout_compliance_rate')} | {sec['mean'] if sec else '—'} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def resolve_lora_dir(hint: Path) -> Path:
    def is_weights(p: Path) -> bool:
        return p.is_dir() and any((p / n).is_file() for n in ("adapter_config.json", "lora_meta.json"))

    if is_weights(hint):
        return hint

    def latest(parent: Path) -> Optional[Path]:
        if not parent.is_dir():
            return None
        cands: List[tuple] = []
        for p in parent.glob("checkpoint-*"):
            try:
                cands.append((int(p.name.rsplit("-", 1)[1]), p))
            except (ValueError, IndexError):
                continue
        for _, p in sorted(cands, reverse=True):
            if is_weights(p):
                return p
        return None

    for parent in (hint, hint.parent):
        got = latest(parent)
        if got is not None:
            return got
    return hint

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="pptbg 评测")
    ap.add_argument("--prompts", type=Path, default=ROOT / "eval" / "prompts_v1.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    ap.add_argument("--backends", nargs="*", default=list(BACKENDS), choices=list(BACKENDS))
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条需求（0=全部）")
    ap.add_argument("--aspect", default="16:9")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--base-model", default=None)
    ap.add_argument(
        "--lora-dir",
        type=Path,
        default=None,
        help="缺省自动找 train/runs/pptbg-lora-v1 下最新的 checkpoint-<step>",
    )
    ap.add_argument("--seed-base", type=int, default=20261016)
    ap.add_argument("--save-images", action="store_true", default=True)
    args = ap.parse_args(argv)

    hint = args.lora_dir if args.lora_dir is not None else (ROOT / "train" / "runs" / "pptbg-lora-v1")
    resolved = resolve_lora_dir(hint)
    if resolved != hint:
        print(f"[lora] 自动定位到权重目录 {resolved}")
    args.lora_dir = resolved

    prompts = load_prompts(args.prompts)
    if args.limit:
        prompts = prompts[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    base_model = args.base_model or os.environ.get("PPT_BASE_MODEL") or str(ROOT / "tools" / "hf" / "stable-diffusion-v1-5")
    if not Path(base_model).exists() and "/" not in base_model:
        base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    rows: List[Dict[str, Any]] = []
    meta = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompts": len(prompts),
        "aspect": args.aspect,
        "steps": args.steps,
        "base_model": base_model,
        "lora_dir": str(args.lora_dir),
    }

    for name in args.backends:
        print(f"\n=== 后端 {name} ===", flush=True)
        try:
            backend = build_backend(name, args, args.lora_dir, base_model)
            backend.setup(args)
        except SystemExit as exc:
            print(f"[跳过] {name}: {exc}", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"[跳过] {name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        img_dir = args.out / "images" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(prompts):
            prompt = item["prompt"]
            seed = args.seed_base + i
            preset_key = item.get("preset") or infer_preset(prompt, seed)
            if preset_key not in PRESETS:
                preset_key = infer_preset(prompt, seed)
            t0 = time.time()
            try:
                img, meta_i = backend.generate(prompt, preset_key, seed, args)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{i}] 失败：{type(exc).__name__}: {exc}", flush=True)
                continue
            seconds = time.time() - t0
            record = {**meta_i, "_tone_probe": tone_probe(img)}
            m = measure(img, record, prompt, preset_key, seconds)
            m.update(
                {
                    "backend": name,
                    "idx": i,
                    "prompt": prompt,
                    "preset": preset_key,
                    "scene": item.get("scene", ""),
                    "seed": seed,
                    "file": f"{name}/{i:03d}_{preset_key}.jpg",
                }
            )
            m["safe_zone_ok"] = bool(
                m["safe_zone_std"] is not None and m["safe_zone_std"] <= 26.0
            )
            m["layout_compliance_rate"] = bool(m["layout_ok"])
            if args.save_images:
                img.save(img_dir / Path(m["file"]).name, quality=92, optimize=True)
            rows.append(m)
            if (i + 1) % 10 == 0 or i == len(prompts) - 1:
                print(f"  {i + 1}/{len(prompts)} 完成，最近一张 {seconds:.1f}s", flush=True)

    if not rows:
        print("没有任何后端跑出结果，请检查模型与权重路径。")
        return 2

    summary = summarize(rows)
    table = verdict_table(summary)

    (args.out / "metrics.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    (args.out / "summary.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "verdict": table}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, table, args.out / "RESULTS.md", meta)

    print("\n=== 指标对比（均值）===")
    for row in table:
        print(f"  {row['metric']:18s} 最优={row['best_backend']:10s} {row['best_value']:<10g} "
              f"最差={row['worst_backend']:10s} {row['worst_value']:<10g} 差距={row['spread_x']}×")
    print(f"\n结果 -> {args.out / 'RESULTS.md'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
