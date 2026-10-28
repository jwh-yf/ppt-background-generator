"""推理服务核心：加载基座 SD + LoRA 适配器，按预设生成 PPT 背景并施加安全区约束。"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _extra in (PROJECT_ROOT, PROJECT_ROOT / "data", PROJECT_ROOT / "app"):
    if str(_extra) not in sys.path:
        sys.path.insert(0, str(_extra))

from presets import ASPECTS, LAYOUTS, PRESETS, TONES, compose_prompt, resolve_tone, variant_keys  # noqa: E402
from train.config import InferConfig  # noqa: E402

import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("_mkbg", PROJECT_ROOT / "data" / "make_backgrounds.py")
_mkbg = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mkbg)  # type: ignore[union-attr]

safe_zone_rects = _mkbg.safe_zone_rects
analyze_zone = _mkbg.analyze_zone
panel_geometry = _mkbg.panel_geometry
hex_to_rgb = _mkbg.hex_to_rgb

# 安全区约束

def feather_mask(w: int, h: int, feather: int) -> np.ndarray:
    m = np.ones((h, w), dtype=np.float32)
    f = max(1, int(feather))
    ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
    m[:f, :] *= ramp[:, None]
    m[-f:, :] *= ramp[::-1][:, None]
    m[:, :f] *= ramp[None, :]
    m[:, -f:] *= ramp[::-1][None, :]
    return m

def enforce_safe_zones(
    img: Image.Image,
    rects: Dict[str, Tuple[int, int, int, int]],
    radius: int,
    target_std: float,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    stats: List[Dict[str, Any]] = []

    for name, rect in rects.items():
        x, y, rw, rh = rect
        x2, y2 = min(w, x + rw), min(h, y + rh)
        x, y = max(0, x), max(0, y)
        if x2 <= x or y2 <= y:
            continue
        crop = arr[y:y2, x:x2, :]
        before = analyze_zone(arr, (x, y, x2 - x, y2 - y), name)

        r = max(2, int(radius))
        blurred = np.asarray(
            Image.fromarray(np.clip(crop, 0, 255).astype(np.uint8)).filter(
                ImageFilter.GaussianBlur(r)
            ),
            dtype=np.float32,
        )
        excess = max(0.0, before["contrast_std"] - target_std)
        alpha = float(np.clip(excess / max(1.0, target_std), 0.0, 0.92))
        if alpha <= 0.01:
            stats.append({**before, "blur_alpha": 0.0, "radius": 0})
            continue

        mask = feather_mask(x2 - x, y2 - y, feather=max(4, int(min(x2 - x, y2 - y) * 0.12)))
        mixed = crop * (1.0 - alpha * mask[..., None]) + blurred * (alpha * mask[..., None])
        arr[y:y2, x:x2, :] = mixed

        after = analyze_zone(arr, (x, y, x2 - x, y2 - y), name)
        stats.append(
            {
                **after,
                "contrast_std_before": before["contrast_std"],
                "blur_alpha": round(alpha, 3),
                "radius": r,
            }
        )

    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
    return out, stats

# 卡片后面，实测只剩一条几乎看不见的斜线（eval/compare/skins/blocks.jpg）。
COMPOSE_SKIN: Dict[str, str] = {
    "academic_blue": "rules",
    "clean_white": "rules",
    "ink_mono": "rules",
    "warm_terra": "glow",
    "forest_deep": "glow",
    "violet_tech": "glow",
    "dish_math": "glow",
    "dish_mech": "glow",
    "dish_philo": "rules",
}

COMPOSE_MOTIF: Dict[str, str] = {
    "dish_math": "math",
    "dish_mech": "mech",
    "dish_philo": "philo",
}

def layout_for_preset(preset_key: str, seed: int, width: int, height: int) -> Dict[str, Any]:
    preset = PRESETS[preset_key]
    rng = random.Random(seed)
    layout = rng.choice(preset.layouts)
    geom = panel_geometry(layout, width, height, rng)
    return {"layout": layout, "geom": geom, "rects": safe_zone_rects(layout, width, height, geom)}

@dataclass
class GenerateRequest:
    preset_key: str
    subject: str = ""
    aspect: str = "16:9"
    count: int = 1
    steps: int = 28  # TODO 降到 20 能省三分之一时间，画质差多少一直没量
    guidance: float = 6.5
    seed: int = -1
    enforce_safe_zone: bool = True
    target_std: float = 26.0
    tone_variant: str = ""
    tone_random: bool = False
    soft_focus: float = 0.0
    frost: float = 0.0
    motif_richness: str = "normal"
    frost_shape: str = "square"
    symbol_style: str = "vector"

class SDService:

    def __init__(self, cfg: Optional[InferConfig] = None) -> None:
        self.cfg = cfg or InferConfig()
        self._lock = threading.Lock()
        self._pipe = None
        self._backend = "unloaded"
        self._load_error: Optional[str] = None

    def device(self) -> str:
        want = os.environ.get("PPT_DEVICE", self.cfg.device)
        if want != "auto":
            return want
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"

    def load(self) -> str:
        if self._backend != "unloaded":
            return self._backend
        with self._lock:
            if self._backend != "unloaded":
                return self._backend
            try:
                self._load_diffusers()
                self._backend = "diffusers"
            except Exception as exc:  # noqa: BLE001 - 降级是设计的一部分
                self._load_error = f"{type(exc).__name__}: {exc}"
                self._backend = "procedural"
        return self._backend

    def _load_diffusers(self) -> None:
        import torch
        from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler

        cfg = self.cfg
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.dtype]
        device = self.device()
        if device == "cpu":
            dtype = torch.float32

        base = cfg.resolve_base_model() if hasattr(cfg, "resolve_base_model") else cfg.base_model
        pipe = AutoPipelineForText2Image.from_pretrained(
            base, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False
        )
        resolved = cfg.resolve_lora_dir() if hasattr(cfg, "resolve_lora_dir") else None
        lora_dir = Path(resolved) if resolved else Path(cfg.lora_dir)
        if not lora_dir.is_absolute():
            lora_dir = PROJECT_ROOT / lora_dir
        if (lora_dir / "adapter_model.safetensors").exists() or (
            lora_dir / "pytorch_lora_weights.safetensors"
        ).exists():
            if (lora_dir / "adapter_config.json").exists():
                from peft import PeftModel

                pipe.unet = PeftModel.from_pretrained(pipe.unet, str(lora_dir))
            else:
                pipe.load_lora_weights(str(lora_dir))
            self._lora_loaded = str(lora_dir)
        else:
            self._lora_loaded = None

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
            # 不要开 enable_vae_tiling()：实测在 768×432 → 1920×1080 这条路径上，
        self._pipe = pipe
        self._torch = torch

    def health(self) -> Dict[str, Any]:
        if self._backend == "unloaded":
            return {"ready": False, "backend": "unloaded", "reason": "尚未加载模型（首次请求时加载）"}
        if self._backend == "diffusers":
            return {
                "ready": True,
                "backend": f"diffusers/{self.device()}",
                "base_model": self.cfg.base_model,
                "lora": getattr(self, "_lora_loaded", None),
                "dtype": self.cfg.dtype,
            }
        return {
            "ready": True,
            "backend": "procedural（降级：未安装 diffusers 或模型不可用）",
            "reason": self._load_error,
            "note": "此后端复用数据合成算法，输出仍是合法可用的 PPT 背景，但不体现 LoRA 效果。",
        }

    def generate(self, req: GenerateRequest, out_dir: Path) -> List[Dict[str, Any]]:
        backend = self.load()
        out_dir.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        with self._lock:
            for i in range(req.count):
                seed = (req.seed + i) if req.seed >= 0 else random.randrange(1 << 31)
                t0 = time.time()
                if backend == "diffusers":
                    img, prompt = self._gen_diffusers(req, seed)
                    img = self._compose_diffusers(img, req, seed)
                else:
                    img, prompt = self._gen_procedural(req, seed)
                    img = self._compose_diffusers(img, req, seed, panel=False, layers=False)
                info = self._postprocess_and_save(img, req, seed, prompt, out_dir, i, time.time() - t0)
                results.append(info)
        return results

    def _prompt_for(self, req: GenerateRequest) -> str:
        preset = PRESETS[req.preset_key]
        subject = req.subject.strip() or preset.scenario_zh
        return compose_prompt(preset.key, subject)

    def _gen_diffusers(self, req: GenerateRequest, seed: int) -> Tuple[Image.Image, str]:
        torch = self._torch
        pipe = self._pipe
        preset = PRESETS[req.preset_key]
        tone = TONES[preset.tone]
        w, h = ASPECTS[req.aspect]
        # 4:3 → 680×512，9:16 → 512×912（都取 8 的整数倍，VAE 必须）。
        # 实测对比：短边 768 时整幅是电视雪花，512 时颗粒明显收敛、色块更干净。
        short = 512
        if w >= h:
            gen_h = short
            gen_w = int(round(short * w / h / 8) * 8)
        else:
            gen_w = short
            gen_h = int(round(short * h / w / 8) * 8)
        prompt = self._prompt_for(req)

        gen_device = self.device()
        generator = torch.Generator(device=gen_device).manual_seed(seed)
        img = pipe(
            prompt=prompt,
            negative_prompt=self.cfg.negative_prompt,
            num_inference_steps=req.steps,
            guidance_scale=req.guidance,
            width=gen_w,
            height=gen_h,
            generator=generator,
            output_type="pil",
        ).images[0]
        if img.size != (w, h):
            img = img.resize((w, h), Image.LANCZOS)
        return img.convert("RGB"), prompt

    def _gen_procedural(self, req: GenerateRequest, seed: int) -> Tuple[Image.Image, str]:
        preset = PRESETS[req.preset_key]
        tone = TONES[preset.tone]
        w, h = ASPECTS[req.aspect]
        rng = random.Random(seed)
        if req.preset_key in COMPOSE_MOTIF:
            layout = "vignette_plain"
        else:
            layout = rng.choice(preset.layouts)
        geom = panel_geometry(layout, w, h, rng)
        panel_style = "none" if layout == "vignette_plain" else rng.choice(
            ["frosted", "frosted", "gradient", "solid"]
        )
        recipe = {
            "layout": layout,
            "tone": tone.key,
            "panel_side": geom["side"],
            "panel_style": panel_style,
            "photo_scale": round(1.06 + rng.random() * 0.28, 3),
            "photo_blur_px": round(6.0 + rng.random() * 14.0, 2),
            "overlay_alpha": round(0.20 + rng.random() * 0.25, 3),
            "accent_hex": tone.accent_hex,
            "vignette": round(0.12 + rng.random() * 0.30, 3),
            "grain": round(rng.random() * 1.0, 3),
        }
        gen = np.random.default_rng(seed)
        small = gen.normal(0, 1, size=(h // 24, w // 24)).astype(np.float32)
        gray = Image.fromarray(np.clip(small * 32 + 128, 0, 255).astype(np.uint8)).resize(
            (w, h), Image.BICUBIC
        )
        base = np.asarray(gray.convert("RGB"), dtype=np.float32)
        canvas = _mkbg.apply_tone(base, tone)
        if recipe["overlay_alpha"] > 0:
            tint_rgb = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
            canvas = canvas * (1 - recipe["overlay_alpha"]) + tint_rgb * recipe["overlay_alpha"]
        canvas = _mkbg.draw_panel(
            canvas, geom, panel_style, tone.ink_hex, tone.accent_hex, tone, rng
        )
        canvas = _mkbg.apply_vignette(canvas, recipe["vignette"])
        canvas = _mkbg.apply_grain(canvas, recipe["grain"], rng)
        img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
        return img, self._prompt_for(req)

    def _compose_diffusers(
        self,
        img: Image.Image,
        req: GenerateRequest,
        seed: int,
        panel: bool = True,
        layers: bool = True,
    ) -> Image.Image:

        preset = PRESETS[req.preset_key]
        tone = TONES[preset.tone]
        if req.tone_variant:
            tone = resolve_tone(req.preset_key, req.tone_variant)
        elif req.tone_random:
            _vk = variant_keys(req.preset_key)
            if _vk:

                tone = resolve_tone(req.preset_key, random.Random(seed ^ 0x71A7).choice(_vk))
        w, h = ASPECTS[req.aspect]
        arr = np.asarray(img.convert("RGB").resize((w, h), Image.LANCZOS), dtype=np.float32)

        rng = random.Random(seed)
        geo = layout_for_preset(req.preset_key, seed, w, h)
        if req.preset_key in COMPOSE_MOTIF:
            panel_style = "none"
        else:
            panel_style = "none" if geo["layout"] == "vignette_plain" else rng.choice(
                ["frosted", "frosted", "gradient", "solid"]
            )

        tint_rgb = np.array(tone.tint, dtype=np.float32).reshape(1, 1, 3)
        canvas = arr * 0.88 + tint_rgb * 0.12
        _dev = float(np.abs(canvas.mean(axis=(0, 1)) - tint_rgb.reshape(3)).mean())
        _extra = min(0.18, _dev / 260.0)
        if _extra > 0.01:
            canvas = canvas * (1.0 - _extra) + tint_rgb * _extra
        # 放大到 1920 之后就是一块去不掉的污点。
        if req.preset_key in COMPOSE_MOTIF:
            # 只往预设色拉还不够：模型给的底色时亮时暗，拉完之后仍是「灰蓝」而不是
            # 黑板蓝。这里直接做均值对齐——把整幅的均值挪到预设色上，纹理起伏不动。
            _mu = canvas.mean(axis=(0, 1), keepdims=True)
            canvas = np.clip(canvas - _mu + tint_rgb, 0, 255)
            _flat = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")
            _flat = _flat.filter(ImageFilter.GaussianBlur(min(w, h) * 0.012))
            canvas = np.asarray(_flat, dtype=np.float32)

        canvas = _mkbg.suppress_blotches(canvas, strength=0.55)
        canvas, _speck_cover = _mkbg.despeckle_discs(
            canvas, k_sigma=4.0, sigma_mid=46.0, dilate=16.0
        )
        if req.soft_focus > 0:
            canvas = _mkbg.apply_soft_focus(
                canvas, w, h, random.Random(seed ^ 0x50F7), tone, req.soft_focus
            )
        if req.frost > 0:
            canvas = _mkbg.apply_frost_patches(
                canvas, w, h, geo["rects"], random.Random(seed ^ 0x9A31), tone, req.frost,
            shape=req.frost_shape,
            )
        skin = COMPOSE_SKIN.get(req.preset_key, "rules") if layers else "none"
        drng = random.Random(seed ^ 0x5EED)
        is_discipline = req.preset_key in COMPOSE_MOTIF
        if is_discipline:
            pass
        elif skin == "rules":
            canvas = _mkbg.apply_skin_rules(canvas, w, h, geo["rects"], drng, tone, 1.0)
        elif skin == "blocks":
            canvas = _mkbg.apply_skin_blocks(canvas, w, h, geo["rects"], drng, tone, 1.0)
        else:
            canvas = _mkbg.apply_skin_glow(canvas, w, h, geo["rects"], drng, tone, 1.0)
        if not is_discipline:
            canvas = _mkbg.apply_corner_glow(
                canvas, w, h, geo["rects"], drng, 0.09 + drng.random() * 0.05
            )
        if panel_style != "none" and panel and layers:
            card_strength = 0.55 if skin == "glow" else 1.0
            canvas = _mkbg.draw_panel_card(
                canvas, geo["geom"], w, h, tone, rng, card_strength
            )
        motif = COMPOSE_MOTIF.get(req.preset_key)
        if motif:
            _ink_before = canvas.copy() if req.symbol_style == "brush" else None
        if motif == "math":
            canvas = _mkbg.apply_motif_math(canvas, w, h, geo["rects"], drng, tone, 1.0)
        elif motif == "mech":
            canvas = _mkbg.apply_motif_mech(canvas, w, h, geo["rects"], drng, tone, 1.0)
        elif motif == "philo":
            canvas = _mkbg.apply_motif_philo(canvas, w, h, geo["rects"], drng, tone, 1.0)
        if motif:
            from motif_extras import apply_motif_extras as _extras
            canvas = _extras(
                canvas, w, h, geo["rects"], drng, tone, motif, req.motif_richness, 1.0
            )
            if _ink_before is not None:
                from brush import apply_ink as _apply_ink
                canvas = _apply_ink(_ink_before, canvas, motif, seed)
        canvas = _mkbg.apply_vignette(canvas, 0.28 + rng.random() * 0.18)
        # 颗粒会变雪花，投影上就是脏。实测 0.15 以内还留得住质感。
        canvas = _mkbg.apply_grain(canvas, rng.random() * 0.15, rng)
        return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")

    def _postprocess_and_save(
        self, img: Image.Image, req: GenerateRequest, seed: int, prompt: str,
        out_dir: Path, idx: int, seconds: float,
    ) -> Dict[str, Any]:
        preset = PRESETS[req.preset_key]
        w, h = ASPECTS[req.aspect]
        geo = layout_for_preset(req.preset_key, seed, w, h)
        zones_meta: List[Dict[str, Any]] = []

        if req.enforce_safe_zone:
            img, zones_meta = enforce_safe_zones(
                img, geo["rects"], radius=max(8, int(min(w, h) * 0.012)),
                target_std=req.target_std,
            )
        else:
            arr = np.asarray(img.convert("RGB"), dtype=np.float32)
            zones_meta = [analyze_zone(arr, r, n) for n, r in geo["rects"].items()]

        name = f"{req.preset_key}_{seed:010d}_{idx:02d}"
        fpath = out_dir / f"{name}.jpg"
        img.save(fpath, format="JPEG", quality=92, optimize=True, subsampling=1)

        return {
            "id": name,
            "file": fpath.name,
            "url": f"/outputs/{fpath.name}",
            "preset": preset.key,
            "preset_zh": preset.name_zh,
            "scenario_zh": preset.scenario_zh,
            "aspect": req.aspect,
            "width": w,
            "height": h,
            "seed": seed,
            "steps": req.steps,
            "guidance": req.guidance,
            "subject": req.subject,  # 重画这一张要按原主题重发，不可以只剩拼好的 prompt
            "tone_variant": req.tone_variant,
            "tone_random": bool(req.tone_random),
            "soft_focus": round(float(req.soft_focus), 3),
            "frost": round(float(req.frost), 3),
            "frost_shape": req.frost_shape,
            "motif_richness": req.motif_richness,
            "symbol_style": req.symbol_style,
            "prompt": prompt,
            "negative_prompt": self.cfg.negative_prompt,
            "layout": geo["layout"],
            "panel_geom": geo["geom"],
            "safe_zones": zones_meta,
            "safe_zone_enforced": bool(req.enforce_safe_zone),
            "backend": self._backend,
            "seconds": round(seconds, 2),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

def selftest() -> int:
    print("=== 1. prompt 拼装（训练/推理同一份预设）===")
    for key in list(PRESETS)[:2]:
        print(f"[{key}] {compose_prompt(key, 'a modern lecture hall')[:120]}…")

    print("\n=== 2. 安全区几何 ===")
    for key in list(PRESETS)[:3]:
        geo = layout_for_preset(key, 12345, 1920, 1080)
        print(f"[{key}] layout={geo['layout']} rects={ {k: v for k, v in geo['rects'].items()} }")

    print("\n=== 3. 安全区约束效果（构造一张高频噪声图）===")
    rng = np.random.default_rng(7)
    w, h = 1920, 1080
    noise = rng.normal(128, 60, size=(h, w, 3)).astype(np.float32)
    src = Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8), "RGB")

    geo = layout_for_preset("academic_blue", 999, w, h)
    out, stats = enforce_safe_zones(src, geo["rects"], radius=16, target_std=26.0)
    for s in stats:
        print(
            f"  {s['region']:8s} std {s.get('contrast_std_before', s['contrast_std']):6.1f}"
            f" -> {s['contrast_std']:6.1f}  (alpha={s['blur_alpha']})"
        )
    improved = all(
        s["contrast_std"] <= s.get("contrast_std_before", s["contrast_std"]) + 0.01 for s in stats
    )
    assert improved, "安全区处理后对比度不应升高"
    print("  低对比度方向正确 ✓")

    print("\n=== 4. 推理后端可用性 ===")
    try:
        import diffusers  # noqa: F401

        print(f"  diffusers 可用：{diffusers.__version__}")
    except Exception as exc:  # noqa: BLE001
        print(f"  diffusers 不可用（会走降级后端）：{type(exc).__name__}: {exc}")
    try:
        import torch

        print(
            f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}"
            + (f"  gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "")
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  torch 不可用：{type(exc).__name__}: {exc}")
    print("\n自检完成。")
    return 0

def demo(count: int, preset: str, aspect: str) -> int:
    svc = SDService()
    backend = svc.load()
    print(f"后端：{backend}  {json.dumps(svc.health(), ensure_ascii=False)}")
    out = PROJECT_ROOT / "app" / "static" / "outputs"
    req = GenerateRequest(preset_key=preset, subject="a modern lecture hall", aspect=aspect,
                          count=count, steps=8, guidance=6.5, seed=20261016)
    for info in svc.generate(req, out):
        print(
            f"  {info['file']}  {info['width']}×{info['height']}  {info['seconds']}s  "
            f"layout={info['layout']}  zones="
            + ", ".join(f"{z['region']}:{z['contrast_std']}" for z in info["safe_zones"])
        )
    print(f"输出目录：{out}")
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="推理服务自检 / 演示")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--demo", type=int, default=0, help="生成 N 张演示图")
    ap.add_argument("--preset", default="academic_blue")
    ap.add_argument("--aspect", default="16:9", choices=list(ASPECTS))
    args = ap.parse_args(argv)
    if args.demo:
        return demo(args.demo, args.preset, args.aspect)
    return selftest()

if __name__ == "__main__":
    raise SystemExit(main())
