# -*- coding: utf-8 -*-
"""把 imageio-ffmpeg 的 win_amd64 wheel 解到 tools/ffmpeg/，取里面自带的 ffmpeg.exe。

为什么不用系统 ffmpeg：本机没有装，也不想让用户去装。pip 在这个沙箱里装不进东西
（解包阶段被拒），但「用 urllib 拿 wheel → zipfile 解压」这条路是通的（reportlab
和 pypdfium2 都是这么装进来的）。

用法：
    pptbg tools/get_ffmpeg.py
"""
from __future__ import annotations

import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "tools" / "ffmpeg"
PKG = "imageio-ffmpeg"


def pick_wheel() -> str:
    with urllib.request.urlopen(f"https://pypi.org/pypi/{PKG}/json", timeout=60) as r:
        meta = json.load(r)
    urls = meta["releases"][meta["info"]["version"]]
    for u in urls:
        name = u["filename"]
        if "win_amd64" in name and name.endswith(".whl"):
            return u["url"]
    raise SystemExit("没有找到 win_amd64 的 wheel")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    url = pick_wheel()
    whl = DEST / url.rsplit("/", 1)[-1]
    if not whl.exists():
        print(f"下载 {whl.name} …")
        urllib.request.urlretrieve(url, whl)
    print(f"解压 -> {DEST}")
    with zipfile.ZipFile(whl) as z:
        z.extractall(DEST)
    exes = sorted(DEST.rglob("ffmpeg*.exe"))
    if not exes:
        raise SystemExit("wheel 里没有 ffmpeg.exe")
    print("ffmpeg:", exes[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
