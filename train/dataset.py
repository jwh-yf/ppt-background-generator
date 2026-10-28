"""PPT 背景数据集（PyTorch Dataset），带磁盘缓存与确定性增强。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _HAS_TORCH = True
except Exception:  # noqa: BLE001
    torch = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    _HAS_TORCH = False

def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno} 不是合法 JSON：{exc}") from exc
    return rows

def resize_center_crop(img, size: int):
    from PIL import Image

    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - size) // 2, (nh - size) // 2
    return img.crop((left, top, left + size, top + size))

def resize_any_crop(img, size: int, seed: int):
    from PIL import Image
    import random as _random

    w, h = img.size
    scale = size / min(w, h)
    nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
    img = img.resize((nw, nh), Image.LANCZOS)
    rnd = _random.Random(seed)
    left = rnd.randint(0, max(0, nw - size))
    top = rnd.randint(0, max(0, nh - size))
    return img.crop((left, top, left + size, top + size))

def preprocess_split(
    rows: Sequence[Dict[str, Any]],
    images_root: Path,
    size: int,
    cache_file: Path,
    force: bool = False,
) -> Path:
    from PIL import Image

    key_src = "|".join(
        f"{r['id']}:{r.get('image_sha256', '')}" for r in rows
    ) + f"|size={size}|crop=any-v2"
    key = hashlib.sha256(key_src.encode("utf-8")).hexdigest()[:16]
    cache_file = cache_file.with_name(f"{cache_file.stem}-{key}{cache_file.suffix}")

    if cache_file.exists() and not force:
        return cache_file

    imgs = np.zeros((len(rows), size, size, 3), dtype=np.uint8)
    captions: List[str] = []
    ids: List[str] = []
    for i, r in enumerate(rows):
        path = images_root / r["file"]
        if not path.exists():
            raise SystemExit(f"找不到图片 {path}（清单 {r['id']} 指向它）")
        with Image.open(path) as im:
            im = im.convert("RGB")
            seed = int(str(r.get("image_sha256", r["id"]))[:12], 16)
            im = resize_any_crop(im, size, seed)
            imgs[i] = np.asarray(im, dtype=np.uint8)
        captions.append(r["caption"])
        ids.append(r["id"])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_file,
        images=imgs,
        captions=np.array(captions, dtype=object),
        ids=np.array(ids, dtype=object),
    )
    return cache_file

if _HAS_TORCH:

    class BackgroundDataset(Dataset):
        """从 .npz 缓存读取样本，输出 (pixel_values, input_ids)。"""

        def __init__(
            self,
            cache_file: Path,
            tokenizer,
            augment: bool = False,
            max_caption_tokens: int = 77,
        ) -> None:
            data = np.load(cache_file, allow_pickle=True)
            self.images: np.ndarray = data["images"]
            self.captions: List[str] = list(data["captions"])
            self.ids: List[str] = list(data["ids"])
            self.tokenizer = tokenizer
            self.augment = augment
            self.max_caption_tokens = max_caption_tokens

        def __len__(self) -> int:
            return len(self.images)

        def __getitem__(self, idx: int) -> Dict[str, Any]:
            img = self.images[idx]
            if self.augment and np.random.random() < 0.5:
                img = img[:, ::-1, :]  # 水平翻转：不改变留白语义
            px = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
            px = px / 127.5 - 1.0
            tok = self.tokenizer(
                self.captions[idx],
                max_length=self.max_caption_tokens,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            return {
                "pixel_values": px,
                "input_ids": tok.input_ids[0],
                "attention_mask": tok.attention_mask[0],
                "sample_id": self.ids[idx],
            }

def images_root_of(cfg) -> "Path":
    from pathlib import Path as _Path

    return _Path(cfg.resolved()["dataset_dir"]) / "images"

def describe_cache(cache_file: Path) -> Dict[str, Any]:
    data = np.load(cache_file, allow_pickle=True)
    imgs = data["images"]
    caps = list(data["captions"])
    return {
        "file": str(cache_file),
        "n": int(imgs.shape[0]),
        "shape": list(imgs.shape[1:]),
        "dtype": str(imgs.dtype),
        "bytes": cache_file.stat().st_size,
        "caption_chars_min": min(len(c) for c in caps),
        "caption_chars_max": max(len(c) for c in caps),
        "first_caption": caps[0],
    }

def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    from config import TrainConfig, abspath

    ap = argparse.ArgumentParser(description="预处理数据集缓存")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="忽略已有缓存重建")
    args = ap.parse_args(argv)

    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    paths = cfg.resolved()
    size = cfg.model.resolution
    images_root = images_root_of(cfg)
    if not images_root.is_dir():
        raise SystemExit(f"找不到图片目录 {images_root}，请先运行 data/make_backgrounds.py")

    cache_root = paths["cache_dir"]
    report: Dict[str, Any] = {"resolution": size, "splits": {}}
    for name, manifest in (("train", paths["train_manifest"]), ("val", paths["val_manifest"])):
        if not manifest.exists():
            print(f"跳过 {name}：找不到 {manifest}")
            continue
        rows = load_manifest(manifest)
        cf = preprocess_split(rows, images_root_of(cfg), size, cache_root / f"{name}.npz", force=args.force)
        info = describe_cache(cf)
        report["splits"][name] = info
        print(f"{name}: {info['n']} 张 @ {info['shape']} -> {cf.name} ({info['bytes']/1e6:.1f} MB)")

    (cache_root / "cache_index.json").parent.mkdir(parents=True, exist_ok=True)
    (cache_root / "cache_index.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n缓存索引 -> {cache_root / 'cache_index.json'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
