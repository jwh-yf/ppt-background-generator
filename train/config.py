"""训练与推理配置的唯一事实来源。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def abspath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)

def _read_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"配置文件不存在：{path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                f"{path} 是 YAML，但当前环境没有 pyyaml。"
                f"请改用等价的 JSON 配置，或 pip install pyyaml。（{exc}）"
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"{path} 顶层必须是一个字典（键 = 配置字段名），实际是 {type(data).__name__}")
    return data

@dataclass
class ModelConfig:

    pretrained: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    local_dir: Optional[str] = "tools/hf/stable-diffusion-v1-5"
    variant: Optional[str] = None
    revision: Optional[str] = None
    resolution: int = 512
    vae_fp32: bool = True

    def resolve_pretrained(self) -> str:
        """把 pretrained 解析成 from_pretrained 真正能加载的路径。

        本项目实测踩到：只给 Hub id 时 transformers 取不到词表，报的却是 protobuf 缺失，
        栈里连一句找不到 vocab.json 都没有。本地有 model_index.json 就用绝对路径，否则回退成原 id。
        """
        if not self.local_dir:
            return self.pretrained
        root = PROJECT_ROOT / self.local_dir
        if (root / "model_index.json").is_file():
            return str(root)
        return self.pretrained

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.resolution % 64 != 0:
            errs.append(f"resolution={self.resolution} 必须是 64 的倍数（含无 VAE 下采样要求）")
        if self.resolution < 256 or self.resolution > 1024:
            errs.append(f"resolution={self.resolution} 超出合理区间 256..1024")
        return errs

@dataclass
class LoraConfig:

    rank: int = 16
    alpha: int = 16
    dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["to_q", "to_k", "to_v", "to_out.0"]
    )
    train_text_encoder: bool = False

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.rank <= 0:
            errs.append(f"lora.rank={self.rank} 必须为正")
        if self.alpha <= 0:
            errs.append(f"lora.alpha={self.alpha} 必须为正")
        if not 0.0 <= self.dropout < 1.0:
            errs.append(f"lora.dropout={self.dropout} 必须落在 [0, 1)")
        if not self.target_modules:
            errs.append("lora.target_modules 不能为空")
        return errs

@dataclass
class TrainConfig:

    dataset_dir: str = "data/dataset_v1"
    train_manifest: str = "data/splits/train.jsonl"
    val_manifest: str = "data/splits/val.jsonl"
    cache_dir: str = "train/.cache"
    caption_field: str = "caption"
    preprocess_to_cache: bool = True

    steps: int = 1000
    batch_size: int = 1
    grad_accum: int = 4
    lr: float = 1e-4
    lr_scheduler: str = "cosine"
    warmup_steps: int = 50
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    seed: int = 20261016
    gradient_checkpointing: bool = True
    attention_backend: str = "sdpa"

    output_dir: str = "train/runs/pptbg-lora-v1"
    save_every: int = 250
    log_every: int = 10
    sample_every: int = 250

    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)

    def validate(self) -> List[str]:
        errs: List[str] = []
        errs += [f"model: {e}" for e in self.model.validate()]
        errs += [f"lora: {e}" for e in self.lora.validate()]
        if self.steps <= 0:
            errs.append(f"steps={self.steps} 必须为正")
        if self.batch_size <= 0 or self.grad_accum <= 0:
            errs.append("batch_size / grad_accum 必须为正")
        if self.mixed_precision not in ("fp16", "bf16", "no"):
            errs.append(f"mixed_precision={self.mixed_precision} 必须是 fp16/bf16/no")
        if self.lr_scheduler not in ("cosine", "constant", "linear"):
            errs.append(f"lr_scheduler={self.lr_scheduler} 不是已实现的调度器")
        if self.warmup_steps > self.steps:
            errs.append(f"warmup_steps={self.warmup_steps} 大于 steps={self.steps}")
        if self.attention_backend not in ("sdpa", "xformers", "none"):
            errs.append(f"attention_backend={self.attention_backend} 无效")
        if self.mixed_precision == "fp16" and not self.model.vae_fp32:
            errs.append("fp16 训练时建议 model.vae_fp32=True，否则 VAE 易出 NaN")
        return errs

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "TrainConfig":
        raw = _read_config_file(Path(path))
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise SystemExit(f"{path} 里有无法识别的字段：{sorted(unknown)}")
        cfg = cls()
        for k, v in raw.items():
            if k == "model":
                cfg.model = ModelConfig(**v)
            elif k == "lora":
                cfg.lora = LoraConfig(**v)
            else:
                setattr(cfg, k, v)
        return cfg

    def resolved(self) -> Dict[str, Path]:
        return {
            "dataset_dir": abspath(self.dataset_dir),
            "train_manifest": abspath(self.train_manifest),
            "val_manifest": abspath(self.val_manifest),
            "cache_dir": abspath(self.cache_dir),
            "output_dir": abspath(self.output_dir),
        }

@dataclass
class InferConfig:

    base_model: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    lora_dir: str = "train/runs/pptbg-lora-v5-nobokeh/checkpoint-1000"
    device: str = "auto"
    dtype: str = "fp16"
    steps: int = 28
    guidance: float = 6.5
    aspect: str = "16:9"
    negative_prompt: str = (
        "text, letters, words, watermark, signature, logo, ui, lowres, jpeg artifacts, "
        "blurry mess, cluttered, busy background, high contrast noise, distorted, "
        "bokeh, out of focus discs, light spots, circular blotch, stain, splotch"
    )
    # 安全区约束：出图后对候选放字区域做对比度自适应压制
    enforce_safe_zone: bool = True
    safe_zone_target_std: float = 26.0
    seed: int = -1

    def resolve_base_model(self) -> str:
        """把 base_model 解析成优先本地目录、否则退回 Hub id。
        
        是同一个坑，所以共用同一套解析顺序。"""
        cand = Path(self.base_model)
        if not cand.is_absolute():
            cand = PROJECT_ROOT / cand
        if (cand / "model_index.json").is_file():
            return str(cand)
        fallback = PROJECT_ROOT / "tools" / "hf" / "stable-diffusion-v1-5"
        if (fallback / "model_index.json").is_file():
            return str(fallback)
        return self.base_model

    def resolve_lora_dir(self) -> Optional[str]:
        def _latest(dirpath: Path) -> Optional[Path]:
            if not dirpath.is_dir():
                return None
            if (dirpath / "adapter_model.safetensors").is_file():
                return dirpath
            ckpts = []
            for child in dirpath.glob("checkpoint-*"):
                if not (child / "adapter_model.safetensors").is_file():
                    continue
                try:
                    ckpts.append((int(child.name.split("-")[-1]), child))
                except ValueError:
                    continue
            return max(ckpts)[1] if ckpts else None

        if not self.lora_dir:
            return None

        cand = Path(self.lora_dir)
        if not cand.is_absolute():
            cand = PROJECT_ROOT / cand
        hit = _latest(cand)
        if hit is not None:
            return str(hit)
        runs = PROJECT_ROOT / "train" / "runs"
        found: List[Tuple[int, float, Path]] = []
        if runs.is_dir():
            for run in runs.iterdir():
                hit = _latest(run)
                if hit is None:
                    continue
                try:
                    step = int(hit.name.split("-")[-1])
                except ValueError:
                    continue
                try:
                    mtime = (hit / "adapter_model.safetensors").stat().st_mtime
                except OSError:
                    mtime = 0.0
                found.append((step, mtime, hit))
        if found:
            return str(max(found)[2])
        return None

    def validate(self) -> List[str]:
        errs: List[str] = []
        if self.steps <= 0:
            errs.append(f"steps={self.steps} 必须为正")
        if not 1.0 <= self.guidance <= 20.0:
            errs.append(f"guidance={self.guidance} 超出 [1, 20]")
        if self.dtype not in ("fp16", "bf16", "fp32"):
            errs.append(f"dtype={self.dtype} 无效")
        if self.safe_zone_target_std <= 0:
            errs.append("safe_zone_target_std 必须为正")
        return errs

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="配置自检")
    ap.add_argument("--check", action="store_true", help="打印配置并校验")
    ap.add_argument("--config", type=Path, default=None, help="从 JSON 载入后再校验")
    args = ap.parse_args(argv)

    cfg = TrainConfig.load(args.config) if args.config else TrainConfig()
    errs = cfg.validate() + [f"infer: {e}" for e in InferConfig().validate()]
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))
    print(f"\n解析后的路径：")
    for k, v in cfg.resolved().items():
        print(f"  {k:16s} {v}")
    if errs:
        print("\n!! 配置校验失败：")
        for e in errs:
            print(f"   - {e}")
        return 1
    print("\n配置校验通过。")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
