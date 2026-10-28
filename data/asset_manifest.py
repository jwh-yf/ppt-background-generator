"""素材清单工具：给真实 CC0 素材建立 manifest.jsonl，并从素材图生成占位素材。

为什么必须这么做"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

SIDECAR_TEMPLATE: Dict[str, Any] = {
    "asset_id": "",
    "origin": "",
    "license": "",
    "author": "",
    "retrieved": "",
    "role": "photo",
    "notes": "填写要求：origin 必须是可直接访问的来源 URL；license 例如 CC0-1.0 / Pexels-License / team-owned；"
    "role 取 photo（照片母图）或 texture（纹理/抽象图）。",
    "allowed_for_training": None,
    "allowed_for_redistribution": None,
}

def sidecar_path(img: Path) -> Path:
    return img.with_suffix(img.suffix + ".json")

def make_template(img: Path) -> Dict[str, Any]:
    t = dict(SIDECAR_TEMPLATE)
    t["asset_id"] = img.stem
    return t

def cmd_template(args: argparse.Namespace) -> int:
    root: Path = args.raw
    if not root.is_dir():
        print(f"目录不存在：{root}", file=sys.stderr)
        return 2
    made = 0
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        sp = sidecar_path(img)
        if sp.exists() and not args.force:
            continue
        sp.write_text(json.dumps(make_template(img), ensure_ascii=False, indent=2), encoding="utf-8")
        made += 1
    print(f"生成侧车模板 {made} 个（目录：{root}）")
    return 0

def cmd_scan(args: argparse.Namespace) -> int:
    root: Path = args.raw
    rows: List[Dict[str, Any]] = []
    pending: List[str] = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXT:
            continue
        sp = sidecar_path(img)
        if not sp.exists():
            pending.append(str(img.relative_to(root)))
            continue
        data = json.loads(sp.read_text(encoding="utf-8"))
        missing = [k for k in ("asset_id", "origin", "license", "role") if not data.get(k)]
        if missing:
            pending.append(f"{img.relative_to(root)}（缺字段：{','.join(missing)}）")
            continue
        if data.get("allowed_for_training") is False:
            pending.append(f"{img.relative_to(root)}（许可不允许用于训练）")
            continue
        rows.append(
            {
                "asset_id": data["asset_id"],
                "file": str(img.relative_to(root)).replace("\\", "/"),
                "origin": data["origin"],
                "license": data["license"],
                "author": data.get("author", ""),
                "retrieved": data.get("retrieved", ""),
                "role": data.get("role", "photo"),
            }
        )

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"可用素材 {len(rows)} 条 -> {out}")
    if pending:
        print(f"\n以下 {len(pending)} 项未纳入（需补全来源/许可信息）：")
        for p in pending[:50]:
            print("  -", p)
        if len(pending) > 50:
            print(f"  ... 另有 {len(pending) - 50} 项")
    if not rows:
        print("\n没有任何可用素材：请先补全侧车 JSON，再运行 make_backgrounds.py。", file=sys.stderr)
        return 1
    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="素材清单工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("template", help="为 raw/ 里的图片生成侧车 JSON 模板")
    t.add_argument("--raw", type=Path, required=True)
    t.add_argument("--force", action="store_true", help="覆盖已有侧车文件")
    t.set_defaults(func=cmd_template)

    s = sub.add_parser("scan", help="扫描 raw/ 生成 manifest.jsonl")
    s.add_argument("--raw", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.set_defaults(func=cmd_scan)

    args = ap.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    raise SystemExit(main())
