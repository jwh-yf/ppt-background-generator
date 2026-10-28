"""把合成数据集的 records.jsonl 切成 train/val/test 三份清单，并做质量体检。"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import TrainConfig, abspath  # noqa: E402

SPLIT_NAMES = ("train", "val", "test")

def load_records(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    if not rows:
        raise SystemExit(f"{path} 是空的，没什么可切的")
    return rows

def bucket_of(record: Dict[str, Any], salt: str) -> float:
    """把一条记录稳定映射到 [0,1)。优先用图片哈希，其次用 id。"""
    key = str(record.get("image_sha256") or record.get("id") or "")
    h = hashlib.sha256((salt + key).encode("utf-8")).digest()
    return int.from_bytes(h[:6], "big") / float(1 << 48)

def split_records(
    records: List[Dict[str, Any]], val_ratio: float, test_ratio: float, salt: str
) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in SPLIT_NAMES}
    for r in records:
        b = bucket_of(r, salt)
        if b < test_ratio:
            out["test"].append(r)
        elif b < test_ratio + val_ratio:
            out["val"].append(r)
        else:
            out["train"].append(r)
    return out

def health_check(records: List[Dict[str, Any]]) -> Tuple[List[str], Dict[str, Any]]:
    """返回 (问题列表, 统计字典)。

    问题只警告不中止：数据本身是合成的，比例偏一点不说明训练会坏，但崩了必须先看见。
    """
    problems: List[str] = []
    stats: Dict[str, Any] = {}

    hashes = Counter(r.get("image_sha256", "") for r in records)
    dups = [h for h, n in hashes.items() if n > 1 and h]
    if dups:
        problems.append(f"存在 {len(dups)} 组重复图片（image_sha256 相同）")

    cap_len = [len(r.get("caption", "")) for r in records]
    stats["caption_chars"] = {
        "min": min(cap_len),
        "median": int(statistics.median(cap_len)),
        "max": max(cap_len),
    }
    if stats["caption_chars"]["min"] < 40:
        problems.append("最短 caption 少于 40 字符，风格描述可能不足")

    missing = [r["id"] for r in records if not r.get("sources")]
    if missing:
        problems.append(f"{len(missing)} 条记录缺少素材来源（无法满足溯源要求）")

    stds: List[float] = []
    for r in records:
        for z in r.get("safe_zones", []):
            stds.append(float(z.get("contrast_std", 0.0)))
    if stds:
        stats["safe_zone_contrast_std"] = {
            "min": round(min(stds), 2),
            "median": round(statistics.median(stds), 2),
            "max": round(max(stds), 2),
        }
        bad = sum(1 for s in stds if s > 40.0)
        if bad:
            problems.append(
                f"{bad}/{len(stds)} 个安全区对比度过高（std>40），放字会读不清；"
                f"可增大该预设的 overlay_alpha 或模糊半径"
            )

    stats["presets"] = dict(Counter(r["preset"] for r in records))
    stats["aspects"] = dict(Counter(r["aspect"] for r in records))
    stats["layouts"] = dict(Counter(r["recipe"]["layout"] for r in records))
    lic = Counter(s["license"] for r in records for s in r.get("sources", []))
    stats["licenses"] = dict(lic)
    unknown = [k for k in lic if k in ("unknown", "UNKNOWN", "")]
    if unknown:
        problems.append(f"有素材许可证未标注：{unknown}")

    thin = [k for k, n in stats["presets"].items() if n < 30]
    if thin:
        problems.append(f"以下预设样本数不足 30：{thin}（建议提高合成数量）")

    stats["total"] = len(records)
    return problems, stats

def write_jsonl(rows: Sequence[Dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(body, encoding="utf-8")
    return hashlib.sha256(body.encode("utf-8")).hexdigest()

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="切分数据集并体检")
    ap.add_argument("--records", type=Path, default=None, help="默认取 config.train.dataset_dir/records.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="默认 data/splits")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--test-ratio", type=float, default=0.05)
    ap.add_argument("--salt", type=str, default="pptbg-split-v1")
    ap.add_argument("--strict", action="store_true", help="有问题就返回非零退出码")
    args = ap.parse_args(argv)

    base = TrainConfig()
    records_path = args.records or (abspath(base.dataset_dir) / "records.jsonl")
    if not records_path.exists():
        raise SystemExit(f"找不到 {records_path}，请先运行 data/make_backgrounds.py")

    records = load_records(records_path)
    parts = split_records(records, args.val_ratio, args.test_ratio, args.salt)
    problems, stats = health_check(records)

    out_dir = args.out or abspath("data/splits")
    digests = {}
    for name in SPLIT_NAMES:
        rows = sorted(parts[name], key=lambda r: r["id"])
        digests[name] = write_jsonl(rows, out_dir / f"{name}.jsonl")

    index = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_records": str(records_path),
        "source_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "split_salt": args.salt,
        "counts": {k: len(v) for k, v in parts.items()},
        "split_sha256": digests,
        "stats": stats,
        "warnings": problems,
    }
    (out_dir / "split_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps({k: index[k] for k in ("counts", "stats")}, ensure_ascii=False, indent=2))
    if problems:
        print("\n体检警告：")
        for p in problems:
            print(f"  - {p}")
    else:
        print("\n体检通过，无警告。")
    print(f"\n切分清单 -> {out_dir}")
    return 1 if (problems and args.strict) else 0

if __name__ == "__main__":
    raise SystemExit(main())
