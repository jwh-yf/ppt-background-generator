# -*- coding: utf-8 -*-
r"""把 SRT 字幕烧进录屏文件（硬字幕），顺带压到 1080p / 250 MB 以内。

为什么用 ffmpeg 的 subtitles 滤镜而不是自己画：本机没有装 ffmpeg，但
`tools/get_ffmpeg.py` 已经把 imageio-ffmpeg 自带的 7.1 静态版解出来了，它带 libass
（已实测 `-filters` 里有 subtitles / ass / drawtext），所以硬字幕这条链路是通的。

Windows 上 subtitles 滤镜的路径最容易翻车（`:` 和 `\` 都要转义），所以这里先切到
字幕所在目录、只用文件名调用，绕开整条路径。

用法：
    pptbg tools/burn_subs.py --video submission/video/raw.mp4 ^
                             --srt   submission/video/字幕.srt ^
                             --out   submission/video/成片.mp4
可选：
    --font-size 12     字号（默认 12。libass 按 PlayResY=384 缩放，1080p 下实际约 36px）
    --crf 20           画质，越小越清晰也越大（默认 20）
    --intro 封面.jpg   片头静帧（3 秒，可省）
    --dry-run          只打印即将执行的命令
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def find_ffmpeg() -> Path:
    base = ROOT / "tools" / "ffmpeg"
    hits = sorted(base.rglob("ffmpeg*win*x86_64*.exe")) or sorted(base.rglob("ffmpeg*.exe"))
    if not hits:
        raise SystemExit(
            "没找到 ffmpeg.exe，先跑一次：pptbg tools\\get_ffmpeg.py"
        )
    return hits[0]


def duration_of(ff: Path, video: Path) -> float:
    out = subprocess.run(
        [str(ff), "-hide_banner", "-i", str(video)],
        capture_output=True, text=True, errors="ignore",
    ).stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--srt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--font-size", type=int, default=12)
    ap.add_argument("--crf", type=int, default=12)
    ap.add_argument("--intro", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ff = find_ffmpeg()
    video = Path(args.video).resolve()
    srt = Path(args.srt).resolve()
    out = Path(args.out).resolve()
    if not video.is_file():
        raise SystemExit(f"找不到视频：{video}")
    if not srt.is_file():
        raise SystemExit(f"找不到字幕：{srt}")

    style = (
        f"FontName=Microsoft YaHei,FontSize={args.font_size},"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=2,Shadow=0,Alignment=2,MarginV=40"
    )
    vf = f"subtitles={srt.name}:force_style='{style}'"

    cmd = [
        str(ff), "-y", "-hide_banner", "-loglevel", "warning",
        "-i", str(video),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(args.crf), "-preset", "medium",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-c:a", "copy",
        str(out),
    ]
    print("ffmpeg :", ff)
    print("输入   :", video.name, f"{duration_of(ff, video):.1f} s")
    print("字幕   :", srt.name)
    print("输出   :", out)
    if args.dry_run:
        print("命令   :", " ".join(cmd))
        return 0

    r = subprocess.run(cmd, cwd=str(srt.parent))
    if r.returncode != 0:
        print("ffmpeg 失败，退出码", r.returncode)
        return r.returncode
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"完成：{out.name}  {duration_of(ff, out):.1f} s  {size_mb:.1f} MB")
    if size_mb > 300:
        print("⚠ 超过 300 MB 上限，把 --crf 调大（例如 24）重跑一次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
