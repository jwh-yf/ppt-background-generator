"""LoRA 微调脚本：在合成 PPT 背景数据集上训练 Stable Diffusion LoRA。"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TrainConfig, abspath  # noqa: E402
from dataset import (  # noqa: E402
    BackgroundDataset,
    images_root_of,
    load_manifest,
    preprocess_split,
)

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    _HAS_TORCH = False




def environment_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(getattr(torch, "cuda", None) and torch.cuda.is_available()),
    }
    if info["cuda_available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        # 否则会以 "no kernel image is available" 报错。
        try:
            arch_list = torch.cuda.get_arch_list()
            info["arch_list"] = arch_list
            want = f"sm_{info['gpu_capability'][0]}{info['gpu_capability'][1]}"
            info["arch_supported"] = want in arch_list
            if not info["arch_supported"]:
                info["arch_warning"] = (
                    f"当前 PyTorch 构建不包含 {want}；请安装对应 CUDA 版本的 torch"
                    f"（如 sm_120 需 cu128 构建），否则会报 no kernel image is available"
                )
        except Exception as exc:  # noqa: BLE001
            info["arch_warning"] = f"无法读取 arch_list：{exc}"
    return info




def load_models(cfg: TrainConfig, device: str, log=print):
    """返回 (tokenizer, noise_scheduler, text_encoder, vae, unet)。"""
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    m = cfg.model
    weight_dtype = torch.float16 if cfg.mixed_precision == "fp16" else (
        torch.bfloat16 if cfg.mixed_precision == "bf16" else torch.float32
    )

    src = m.resolve_pretrained()
    log(f"基座模型来源：{src}")

    tokenizer = CLIPTokenizer.from_pretrained(src, subfolder="tokenizer", revision=m.revision)
    text_encoder = CLIPTextModel.from_pretrained(
        src, subfolder="text_encoder", revision=m.revision
    )
    vae = AutoencoderKL.from_pretrained(src, subfolder="vae", revision=m.revision)
    unet = UNet2DConditionModel.from_pretrained(
        src, subfolder="unet", revision=m.revision, variant=m.variant
    )
    scheduler = DDPMScheduler.from_pretrained(src, subfolder="scheduler", revision=m.revision)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    unet.to(device, dtype=weight_dtype)
    text_encoder.to(device, dtype=weight_dtype)
    vae.to(device, dtype=torch.float32 if m.vae_fp32 else weight_dtype)
    log(f"基座模型 {m.pretrained} 已加载；权重 dtype = {weight_dtype}")
    return tokenizer, scheduler, text_encoder, vae, unet


def attach_lora(unet, cfg: TrainConfig, log=print):
    """给 UNet 注入 LoRA 适配器（用 peft 的通用实现）。

    target_modules 写 ["attn.to_q", ...] 会直接抛 NoMatchingPeftModuleError：peft 对字符串
    走 re.fullmatch，匹配的是完整点分路径，而 diffusers 的 UNet 里只有 attn1 / attn2。
    """
    from peft import LoraConfig as PeftLoraConfig
    from peft import get_peft_model

    lc = cfg.lora
    short = [re.escape(n) for n in lc.target_modules]
    pattern = r".*\.attn[12]\.(" + "|".join(short) + r")$"
    lora_cfg = PeftLoraConfig(
        r=lc.rank,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        init_lora_weights="gaussian",
        target_modules=pattern,
    )
    unet = get_peft_model(unet, lora_cfg)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    log(
        f"LoRA 已注入：rank={lc.rank} alpha={lc.alpha}\n"
        f"  目标正则 {pattern}\n"
        f"  可训练参数 {trainable:,} / 总参数 {total:,} = {trainable/total*100:.3f}%"
    )
    return unet, trainable




def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def constant_with_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    return base_lr * (step + 1) / max(1, warmup) if step < warmup else base_lr


def linear_with_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    return base_lr * max(0.0, 1.0 - (step - warmup) / max(1, total - warmup))


SCHEDULERS = {
    "cosine": cosine_with_warmup,
    "constant": constant_with_warmup,
    "linear": linear_with_warmup,
}


@torch.no_grad()
def run_validation(unet, vae, text_encoder, scheduler, val_loader, device, weight_dtype, amp_dtype):
    """在 val 集上算平均去噪 MSE。只用于观察过拟合，不参与任何超参搜索。"""
    unet.eval()
    losses: List[float] = []
    for batch in val_loader:
        px = batch["pixel_values"].to(device, dtype=weight_dtype)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        ctx = text_encoder(batch["input_ids"].to(device))[0]
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = unet(noisy, timesteps, ctx).sample
        target = noise if scheduler.config.prediction_type == "epsilon" else latents
        losses.append(float(F.mse_loss(pred.float(), target.float())))
    unet.train()
    return float(np.mean(losses)) if losses else float("nan")


def train(cfg: TrainConfig, args: argparse.Namespace) -> int:
    if not _HAS_TORCH:
        raise SystemExit("未检测到 torch。请先安装：见 tools/README 或 requirements.txt")
    paths = cfg.resolved()
    out_dir = paths["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    env = environment_info()
    print(json.dumps(env, ensure_ascii=False, indent=2))
    if env.get("arch_warning"):
        print(f"\n!! {env['arch_warning']}\n")

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "no": None}[cfg.mixed_precision]
    weight_dtype = amp_dtype or torch.float32
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("!! 在 CPU 上训练 LoRA 会非常慢（约 20 秒/步）。建议只在冒烟测试时用。")
        amp_dtype = None
        weight_dtype = torch.float32

    print("准备数据缓存……")
    images_root = images_root_of(cfg)
    if not images_root.is_dir():
        raise SystemExit(f"找不到图片目录 {images_root}，请先运行 data/make_backgrounds.py")
    train_rows = load_manifest(paths["train_manifest"])
    train_cache = preprocess_split(
        train_rows, images_root, cfg.model.resolution, paths["cache_dir"] / "train.npz",
        force=args.force_cache,
    )
    val_cache: Optional[Path] = None
    if paths["val_manifest"].exists():
        val_rows = load_manifest(paths["val_manifest"])
        if val_rows:
            val_cache = preprocess_split(
                val_rows, images_root, cfg.model.resolution,
                paths["cache_dir"] / "val.npz", force=args.force_cache,
            )

    tokenizer, scheduler, text_encoder, vae, unet = load_models(cfg, device)
    unet, n_trainable = attach_lora(unet, cfg)

    ds = BackgroundDataset(train_cache, tokenizer, augment=True)
    loader = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = None
    if val_cache is not None:
        val_loader = DataLoader(
            BackgroundDataset(val_cache, tokenizer, augment=False),
            batch_size=cfg.batch_size, shuffle=False, num_workers=0,
        )

    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    sched_fn = SCHEDULERS[cfg.lr_scheduler]

    global_step = 0
    if args.resume:
        st = torch.load(Path(args.resume) / "optimizer.bin", map_location="cpu")
        opt.load_state_dict(st["optimizer"])
        global_step = int(st["global_step"])
        print(f"从 {args.resume} 续训，起始 step={global_step}")

    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.mixed_precision == "fp16" and device == "cuda"))
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.save(out_dir / "config.json")
    run_meta = {
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": env,
        "trainable_params": n_trainable,
        "train_samples": len(ds),
        "val_samples": len(val_loader.dataset) if val_loader else 0,
        "effective_batch": cfg.batch_size * cfg.grad_accum,
        "command": " ".join(sys.argv),
    }
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history: List[Dict[str, Any]] = []
    history_path = out_dir / "history.jsonl"
    unet.train()
    t0 = time.time()
    micro = 0
    opt.zero_grad(set_to_none=True)
    data_iter = iter(loader)
    running: List[float] = []

    print(f"开始训练：steps={cfg.steps} batch={cfg.batch_size}×{cfg.grad_accum} lr={cfg.lr}")
    while global_step < cfg.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        lr = sched_fn(global_step, cfg.steps, cfg.warmup_steps, cfg.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        px = batch["pixel_values"].to(device, dtype=weight_dtype)
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=device).long()
        noisy = scheduler.add_noise(latents, noise, timesteps)
        ctx = text_encoder(batch["input_ids"].to(device))[0]

        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            pred = unet(noisy, timesteps, ctx).sample
            target = noise if scheduler.config.prediction_type == "epsilon" else latents
            loss = F.mse_loss(pred.float(), target.float()) / cfg.grad_accum

        scaler.scale(loss).backward()
        # 必须 detach：loss 仍挂在计算图上（requires_grad=True），直接 float() 会被
        running.append(float(loss.detach()) * cfg.grad_accum)
        micro += 1

        if micro % cfg.grad_accum == 0:
            if cfg.max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % cfg.log_every == 0 or global_step == 1:
                avg = float(np.mean(running[-cfg.log_every * cfg.grad_accum :]))
                elapsed = time.time() - t0
                per_step = elapsed / max(1, global_step)
                eta = per_step * (cfg.steps - global_step)
                msg = (
                    f"step {global_step:5d}/{cfg.steps}  loss {avg:.4f}  lr {lr:.2e}  "
                    f"{per_step:.2f}s/step  ETA {eta/60:.1f}min"
                )
                print(msg, flush=True)
                history.append({"step": global_step, "loss": round(avg, 5), "lr": lr,
                                "elapsed_s": round(elapsed, 1)})
                with history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(history[-1], ensure_ascii=False) + "\n")

            if val_loader is not None and global_step % cfg.save_every == 0:
                vloss = run_validation(
                    unet, vae, text_encoder, scheduler, val_loader, device, weight_dtype, amp_dtype
                )
                print(f"  [val] step {global_step}  val_loss {vloss:.4f}", flush=True)
                with history_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"step": global_step, "val_loss": round(vloss, 5)},
                                        ensure_ascii=False) + "\n")

            if global_step % cfg.save_every == 0 or global_step == cfg.steps:
                save_checkpoint(unet, opt, out_dir, global_step, cfg, n_trainable)
                if args.sample_every and global_step % cfg.sample_every == 0:
                    generate_samples(unet, vae, text_encoder, tokenizer, scheduler, cfg, out_dir,
                                     global_step, device, weight_dtype, amp_dtype)

    total_min = (time.time() - t0) / 60.0
    print(f"\n训练结束：{global_step} 步，用时 {total_min:.1f} 分钟")
    print(f"产物目录：{out_dir}")
    run_meta["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    run_meta["minutes"] = round(total_min, 2)
    run_meta["final_step"] = global_step
    (out_dir / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


def save_checkpoint(unet, opt, out_dir: Path, step: int, cfg: TrainConfig, n_trainable: int) -> None:
    ckpt = out_dir / f"checkpoint-{step}"
    ckpt.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(ckpt)
    torch.save({"optimizer": opt.state_dict(), "global_step": step}, ckpt / "optimizer.bin")
    meta = {
        "step": step,
        "base_model": cfg.model.pretrained,
        "lora": {"rank": cfg.lora.rank, "alpha": cfg.lora.alpha,
                 "target_modules": cfg.lora.target_modules},
        "trainable_params": n_trainable,
        "resolution": cfg.model.resolution,
    }
    (ckpt / "lora_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  已保存 {ckpt.name}", flush=True)


@torch.no_grad()
def generate_samples(unet, vae, text_encoder, tokenizer, scheduler, cfg, out_dir, step,
                     device, weight_dtype, amp_dtype, n: int = 2) -> None:
    """用固定 prompt 出图存证：训练过程中肉眼可查是否学偏。"""
    from diffusers import DDIMScheduler, StableDiffusionPipeline

    sample_dir = out_dir / "samples" / f"step-{step:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        pipe = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler.from_config(scheduler.config),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        sys.path.insert(0, str(abspath("data")))
        from presets import compose_prompt

        for preset_key in list(("academic_blue", "violet_tech"))[:n]:
            prompt = compose_prompt(preset_key, "a modern lecture hall")
            img = pipe(
                prompt, num_inference_steps=24, guidance_scale=6.5,
                height=cfg.model.resolution, width=int(cfg.model.resolution * 16 / 9) // 8 * 8,
                generator=torch.Generator(device=device).manual_seed(cfg.seed),
            ).images[0]
            img.save(sample_dir / f"{preset_key}.jpg", quality=88)
        print(f"  已生成训练中样本图 -> {sample_dir}", flush=True)
        del pipe
        torch.cuda.empty_cache() if device == "cuda" else None
        unet.train()
    except Exception as exc:  # noqa: BLE001 - 存证失败不应中断训练
        print(f"  !! 样本图生成失败（已忽略）：{type(exc).__name__}: {exc}", flush=True)




def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="LoRA 训练")
    ap.add_argument("--config", type=Path, default=None, help="JSON 配置，缺省用 config.py 默认值")
    ap.add_argument("--steps", type=int, default=None, help="覆盖总步数（冒烟测试用）")
    ap.add_argument("--output-dir", type=str, default=None)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--workers", type=int, default=0, help="Windows 上默认 0，避免多进程开销")
    ap.add_argument("--resume", type=Path, default=None, help="从 checkpoint 目录续训")
    ap.add_argument("--sample-every", type=int, default=None)
    ap.add_argument("--force-cache", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="只校验配置与数据，不加载模型")
    args = ap.parse_args(argv)

    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    if args.steps:
        cfg.steps = args.steps
        cfg.save_every = min(cfg.save_every, max(1, args.steps))
        cfg.sample_every = min(cfg.sample_every, max(1, args.steps))
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.sample_every is not None:
        cfg.sample_every = args.sample_every

    errs = cfg.validate()
    if errs:
        print("配置校验失败：")
        for e in errs:
            print(f"  - {e}")
        return 2

    paths = cfg.resolved()
    for key in ("train_manifest", "val_manifest"):
        if not paths[key].exists():
            print(f"!! 找不到 {paths[key]}，请先运行：python train/prepare_dataset.py")
            return 2
    n_train = len(load_manifest(paths["train_manifest"]))
    print(f"train={n_train}  val={len(load_manifest(paths['val_manifest']))}")
    print(f"env: {json.dumps(environment_info(), ensure_ascii=False)}")
    if args.dry_run:
        print("\n--dry-run：配置与数据检查通过，未加载模型。")
        return 0
    return train(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
