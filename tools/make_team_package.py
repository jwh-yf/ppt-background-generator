# -*- coding: utf-8 -*-
"""把「能跑起来录视频」所需的最小集合打成一个 zip，交给队友。

设计原则（为什么不直接压整个仓库）：
- `tools/hf` 3.1 GB 是 SD1.5 权重，属于第三方资产，不能随作品分发（合规清单里
  明确写了「不重分发基座」），必须由队友用 `tools/fetch_model.py` 自己下。
- 依赖（torch/diffusers）在我的机器上装在 D 盘，不在仓库里，同样不能拷。
- `data/dataset_v*`、`eval/results_v*`、`train/runs` 是几百 MB 的中间产物，
  录视频用不到。

所以包里只放：代码 + 母图素材 + 文档 + 录像用素材，再附三个脚本让队友
自己补依赖与权重。目标体积 60–90 MB。
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "给队友"
ZIP = ROOT / "submission" / "pptbg_给队友录视频.zip"

DIRS = [
    "app",
    "configs",
    "docs",
    "train",
    "submission/video/assets",
]

PICK = {
    "data": [
        "presets.py",
        "make_backgrounds.py",
        "make_placeholder_assets.py",
        "asset_manifest.py",
        "validate_dataset.py",
        "motif_extras.py",
        "brush.py",
        "assets_v2/manifest.jsonl",
    ],
    "eval": ["run_eval.py", "prompts_v1.jsonl"],
    "tools": [
        "fetch_model.py",
        "net_ipv6.py",
        "fix_sd15_filenames.py",
        "get_ffmpeg.py",
        "burn_subs.py",
        "make_video_assets.py",
        "md2pdf.py",
        "count_intro.py",
    ],
}

FILES = ["requirements.txt", "README.md", "LICENSE"]

SKIP_SUFFIX = {".pyc", ".pyo", ".npz", ".pt", ".bin", ".safetensors", ".pptx"}
SKIP_DIRNAME = {"__pycache__", ".cache", "outputs", "runs", "images"}


def copy_tree(src: Path, dst: Path) -> None:
    for p in src.rglob("*"):
        if p.is_dir():
            continue
        if any(part in SKIP_DIRNAME for part in p.parts):
            continue
        if p.suffix.lower() in SKIP_SUFFIX:
            continue
        rel = p.relative_to(src)
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for d in DIRS:
        src = ROOT / d
        if src.is_dir():
            copy_tree(src, OUT / d)

    for d, names in PICK.items():
        for name in names:
            src = ROOT / d / name
            if src.is_file():
                tgt = OUT / d / name
                tgt.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, tgt)

    raw = ROOT / "data" / "assets_v2" / "raw"
    if raw.is_dir():
        dst = OUT / "data" / "assets_v2" / "raw"
        dst.mkdir(parents=True, exist_ok=True)
        for p in raw.iterdir():
            if p.is_file() and p.suffix.lower() in (".jpg", ".png"):
                shutil.copy2(p, dst / p.name)

    # 它们是 512×512 的高频图层，4 个文件合计几百 KB，必须跟着包走，
    # 否则队友那边一选「模型笔触」就报 FileNotFoundError。
    brush = ROOT / "data" / "brush"
    if brush.is_dir():
        dst = OUT / "data" / "brush"
        dst.mkdir(parents=True, exist_ok=True)
        for p in brush.iterdir():
            if p.is_file() and p.suffix.lower() == ".png":
                shutil.copy2(p, dst / p.name)

    srt = ROOT / "submission" / "video" / "字幕.srt"
    if srt.is_file():
        (OUT / "submission" / "video").mkdir(parents=True, exist_ok=True)
        shutil.copy2(srt, OUT / "submission" / "video" / srt.name)

    for name in FILES:
        src = ROOT / name
        if src.is_file():
            shutil.copy2(src, OUT / name)

    for gen in ("安装依赖.ps1", "开始演示.ps1", "给队友的说明.md"):
        src = Path(__file__).resolve().parent / "_team_pkg" / gen
        if src.is_file():
            shutil.copy2(src, OUT / gen)

    size_mb = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    n = sum(1 for p in OUT.rglob("*") if p.is_file())
    print(f"目录 {OUT}  {size_mb:.1f} MB / {n} 个文件")

    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in OUT.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(OUT.parent))
    print(f"压缩包 {ZIP}  {ZIP.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
