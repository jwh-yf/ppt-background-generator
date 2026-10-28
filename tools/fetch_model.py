"""把 HuggingFace 权重拉到项目内固定目录，带断点续传与镜像回退。

为什么不直接用 huggingface-cli download：。"""

from __future__ import annotations

import argparse

import fnmatch

import hashlib

import http.client

import json

import os

import socket

import sys

import time

import urllib.error

import urllib.request

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent

HF_HOME = ROOT / "tools" / "hf"

STALL_TIMEOUT = 90.0

MAX_TRIES = 8

# 却决定了"这份权重能不能用在参赛作品里"这个必须先回答的问题。

_PRIORITY = {

    "": 0,

    "tokenizer": 1,

    "scheduler": 1,

    "feature_extractor": 1,

    "text_encoder": 2,

    "vae": 3,

    "unet": 4,

    "safety_checker": 9,

}

def wrote_so_far(dest: Path, before: int) -> int:


    now = dest.stat().st_size if dest.is_file() else 0

    return max(0, now - before)

def socket_family_of(resp: Any) -> str:

    """从响应里挖出底层 socket 的地址族，用来在日志里证明到底走了 IPv6 还是 IPv4。"""

    try:

        sock = resp.fp.raw._sock  # type: ignore[attr-defined]

        return "IPv6" if sock.family == socket.AF_INET6 else "IPv4"

    except Exception:  # noqa: BLE001

        return "?"

CATALOG: Dict[str, Dict[str, Any]] = {

    "tiny": {

        "key": "tiny",

        "repo_id": "hf-internal-testing/tiny-stable-diffusion-pipe",

        "local": "tiny-sd",

        "purpose": "训练环冒烟测试：结构完整但权重极小，用来验证训练/推理代码能跑通",

        "license": "MIT（测试用途，不用于任何生成结果）",

        "approx_mb": 20,

    },

    "sd15": {

        "key": "sd15",

        "repo_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",

        "local": "stable-diffusion-v1-5",

        "purpose": "LoRA 训练基座与推理基座",

        "license": "CreativeML OpenRAIL-M（允许商用与再分发，需随附许可证并传递使用限制）",

        "approx_mb": 4200,

    },

}

SUBDIRS = {

    "unet": ["unet"],

    "vae": ["vae"],

    "text_encoder": ["text_encoder"],

    "tokenizer": ["tokenizer"],

    "scheduler": ["scheduler"],

    "safety_checker": ["safety_checker", "feature_extractor"],

    "top": ["model_index.json", "README.md", "LICENSE.md", "LICENSE", "*.json", "*.txt"],

}

# 为什么需要它：huggingface_hub 在**远端不可达但 local_dir 已存在**时，

# 所以判定成功不能只看有没有异常，必须逐项核对必要文件。

REQUIRED: Dict[str, List[str]] = {

    "tiny": [

        "model_index.json",

        "unet/diffusion_pytorch_model.bin",

        "vae/diffusion_pytorch_model.bin",

        "text_encoder/pytorch_model.bin",

        "tokenizer/vocab.json",

    ],

    "sd15": [

        "model_index.json",

        "unet/diffusion_pytorch_model.safetensors | unet/diffusion_pytorch_model.fp16.safetensors",

        "vae/diffusion_pytorch_model.safetensors | vae/diffusion_pytorch_model.fp16.safetensors",

        "text_encoder/model.safetensors | text_encoder/model.fp16.safetensors",

        "tokenizer/vocab.json",

        "scheduler/scheduler_config.json",

    ],

}

PROBE_FILES: Dict[str, str] = {

    "tiny": "model_index.json",

    "sd15": "model_index.json",

}

def check_complete(spec: Dict[str, Any], local_dir: Path) -> List[str]:

    """返回缺失的必要文件列表；空列表代表完整。

    一项写成多条路径时表示「任一存在即可」——这正是权重 fp16/fp32 两种命名的处理方式。

    """

    missing: List[str] = []

    for rel in REQUIRED.get(spec["key"], []):

        alternatives = [a.strip() for a in rel.split("|")]

        if not any(

            p.is_file() and p.stat().st_size > 0

            for a in alternatives

            for p in local_dir.glob(a)

        ):

            missing.append(rel)

    return missing

def download_one_direct(spec: Dict[str, Any], patterns: Optional[List[str]], quiet: bool = False) -> Dict[str, Any]:

    """不依赖 huggingface_hub 的直连下载：先取文件清单，再逐个下载（支持断点续传）。

    什么时候需要它：huggingface_hub 对「远端超时」的处理是回退到本地目录并静默返回，

    无法据此判断失败。直连实现里每个文件都自己发请求、自己核对字节数，失败就是失败。

    """

    local_dir = HF_HOME / spec["local"]

    local_dir.mkdir(parents=True, exist_ok=True)

    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")

    repo = spec["repo_id"]

    failed: List[str] = []

    def fetch(url: str, tries: int = 3) -> bytes:

        last: Optional[BaseException] = None

        for attempt in range(tries):

            try:

                req = urllib.request.Request(url, headers={"User-Agent": "pptbg/1.0"})

                with urllib.request.urlopen(req, timeout=60) as resp:

                    return resp.read()

            except Exception as exc:  # noqa: BLE001

                last = exc

                time.sleep(1.5 * (attempt + 1))

        raise RuntimeError(f"{url} 失败：{type(last).__name__}: {last}")

    def remote_size(url: str) -> Optional[int]:

        """取远端文件大小。

        优先用文件清单里给的 `size`（HF API 的 siblings 里就有，最可靠）；

        没有时退化为 HEAD，并同时看 Content-Length 与 HuggingFace LFS 用的 X-Linked-Size。

        这个值只用来做两件事：判断本地那份是否已完整、判断能否安全地发 Range 续传。

        拿不到就一律整文件重下——「猜」出来的 Range 会得到 416，

        而把 416 当成"已下载完"正是本脚本踩过的坑（真实文件只有几十字节却报跳过）。

        """

        try:

            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "pptbg/1.0"})

            with urllib.request.urlopen(req, timeout=30) as resp:

                for key in ("X-Linked-Size", "Content-Length"):

                    value = resp.headers.get(key)

                    if value and value.isdigit():

                        return int(value)

        except Exception:  # noqa: BLE001 —— 有些镜像不支持 HEAD

            return None

        return None

    info = json.loads(fetch(f"{endpoint}/api/models/{repo}"))

    files = [s["rfilename"] for s in info.get("siblings", [])]

    remote_sizes: Dict[str, int] = {}

    for s in info.get("siblings", []):

        size = s.get("size")

        if isinstance(size, int) and size > 0:

            remote_sizes[s["rfilename"]] = size

    try:

        tree = json.loads(fetch(f"{endpoint}/api/models/{repo}/tree/main?recursive=1"))

        for item in tree:

            if item.get("type") == "file" and isinstance(item.get("size"), int):

                remote_sizes.setdefault(item["path"], item["size"])

    except Exception as exc:  # noqa: BLE001 —— 拿不到就退回 HEAD 逐个问

        print(f"  （文件大小清单不可用：{type(exc).__name__}，将逐个 HEAD）", flush=True)

    if remote_sizes:

        known = sum(1 for f in files if f in remote_sizes)

        print(f"[{spec['local']}] 远端文件大小清单：{known}/{len(files)} 项已知", flush=True)

    if patterns:

        files = [f for f in files if any(fnmatch.fnmatch(f, p) or f.startswith(p.rstrip("/") + "/") for p in patterns)]

    stems = {f.rsplit(".", 1)[0] for f in files if f.endswith(".safetensors")}

    SKIP_SUFFIX = (".msgpack", ".onnx", ".ckpt")

    selected: List[str] = []

    for f in files:

        if f.endswith(SKIP_SUFFIX):

            continue

        if f.endswith(".bin"):

            continue

        if f.endswith(".safetensors") and "fp16" not in f:

            if any(g == f[: -len(".safetensors")] + ".fp16.safetensors" for g in files):

                continue

        if f.startswith("safety_checker/") and "fp16" in f and any(

            g.startswith("safety_checker/") and g.endswith(".safetensors") and "fp16" not in g for g in files

        ):

            continue

        # 实测这一条省 15.4 GB（v1-5-pruned.safetensors 7.7GB + v1-5-pruned-emaonly 4.27GB。

        if f.count("/") == 0 and f.endswith(".safetensors"):

            continue

        if f.endswith("diffusion_pytorch_model.non_ema.safetensors"):

            continue

        selected.append(f)

    needs_alias = [

        f for f in selected

        if f.endswith(".fp16.safetensors")

        and f[: -len(".fp16.safetensors")] + ".safetensors" not in selected

        and "/" in f

    ]

    if needs_alias:

        print(

            f"  提示：{len(needs_alias)} 个组件权重只有 fp16 名字，下载后请运行 "

            f"`python tools/fix_sd15_filenames.py` 补标准文件名",

            flush=True,

        )

    selected.sort(key=lambda n: (0 if not n.startswith("safety_checker/") else 1, _PRIORITY.get(n.split("/")[0], 5), n))

    print(f"[{spec['local']}] 直连模式：远端列出 {len(files)} 个文件，选中 {len(selected)} 个", flush=True)

    total_new = 0

    skipped_done = 0

    for i, rel in enumerate(selected, 1):

        dest = local_dir / rel

        url = f"{endpoint}/{repo}/resolve/main/{rel}"

        dest.parent.mkdir(parents=True, exist_ok=True)

        have = dest.stat().st_size if dest.is_file() else 0

        want = remote_sizes.get(rel) or remote_size(url)

        if have and want and have >= want:

            skipped_done += 1

            print(f"  [{i}/{len(selected)}] {rel} 已完整（{have / 1e6:.1f}MB），跳过", flush=True)

            continue

        if have == 0 and dest.is_file():

            dest.unlink(missing_ok=True)

        if want:

            print(f"  [{i}/{len(selected)}] {rel} 需要下载 {want / 1e6:.1f}MB（已有 {have / 1e6:.1f}MB）", flush=True)

        def stream(resume_from: int) -> Tuple[int, str]:


            headers = {"User-Agent": "pptbg/1.0"}

            if resume_from:

                headers["Range"] = f"bytes={resume_from}-"

            req = urllib.request.Request(url, headers=headers)

            with urllib.request.urlopen(req, timeout=STALL_TIMEOUT) as resp:

                append = bool(resume_from) and resp.status == 206

                wrote = 0

                next_report = 0

                family = socket_family_of(resp)

                with open(dest, "ab" if append else "wb") as fh:

                    while True:

                        chunk = resp.read(1 << 20)

                        if not chunk:

                            break

                        fh.write(chunk)

                        wrote += len(chunk)

                        # 大文件（unet/vae 几百 MB）必须能看到"在动"，

                        # 否则外面无法区分"在慢速下载"和"卡死了"。每 64MB 报一次。

                        if not quiet and wrote >= next_report:

                            next_report = wrote + 64 * 1024 * 1024

                            done = resume_from + wrote

                            total = f"/{want / 1e6:.0f}" if want else ""

                            print(f"    … {rel} {done / 1e6:.0f}{total}MB [{family}]", flush=True)

            return wrote, family

        def download_with_retry() -> Tuple[int, str]:

            """一轮一轮地续传，直到字节数够了或者重试用尽。

            为什么要重试而不是直接抛：镜像对大文件经常在几百 MB 处断流，

            每次断流都从头下会把用户流量烧光。续传 + 指数退避是唯一省流量的做法。

            已知远端长度时按长度对齐；长度未知时「本次把流读到了 EOF」就算成功——

            这一点必须显式判断，否则小文件会被反复重下 8 遍（实测过：README.md 下了 8 次）。

            """

            nonlocal have

            total_written = 0

            family = "?"

            completed = False

            for attempt in range(1, MAX_TRIES + 1):

                have = dest.stat().st_size if dest.is_file() else 0

                if want and have >= want:

                    break

                try:

                    wrote, family = stream(have if (want and 0 < have < want) else 0)

                except urllib.error.HTTPError as exc:

                    if exc.code == 404:

                        return -1, family

                    if exc.code == 416:

                        # 416 = 本地字节数已不小于远端长度。不能当成功，

                        # 但也不能无限重下：清掉再整份来一次。

                        print(f"    {rel} Range 被拒（416），整份重下", flush=True)

                        dest.unlink(missing_ok=True)

                        continue

                    if exc.code in (429, 500, 502, 503, 504):

                        wait = min(30.0, 2.0 * attempt)

                        print(f"    {rel} HTTP {exc.code}，{wait:.0f}s 后重试（{attempt}/{MAX_TRIES}）", flush=True)

                        time.sleep(wait)

                        continue

                    raise

                except (TimeoutError, socket.timeout, OSError, http.client.IncompleteRead) as exc:

                    total_written += wrote_so_far(dest, have)

                    if attempt >= MAX_TRIES:

                        raise

                    wait = min(20.0, 2.0 * attempt)

                    kept = dest.stat().st_size if dest.is_file() else 0

                    print(

                        f"    {rel} 断流（{type(exc).__name__}: {str(exc)[:80]}），"

                        f"已保留 {kept / 1e6:.0f}MB，{wait:.0f}s 后续传（{attempt}/{MAX_TRIES}）",

                        flush=True,

                    )

                    time.sleep(wait)

                    continue

                total_written += wrote

                now = dest.stat().st_size if dest.is_file() else 0

                if want:

                    if now >= want:

                        completed = True

                        break

                    if wrote == 0:

                        break

                    continue

                completed = True

                break

            return total_written, family, completed

        try:

            written, family, completed = download_with_retry()

        except BaseException as exc:  # noqa: BLE001

            kept = dest.stat().st_size if dest.is_file() else 0

            print(

                f"  [{i}/{len(selected)}] {rel} 下载失败（已保留 {kept / 1e6:.1f}MB，下次运行会续传）："

                f"{type(exc).__name__}: {str(exc)[:120]}",

                flush=True,

            )

            failed.append(rel)

            continue

        if written < 0:

            print(f"  [{i}/{len(selected)}] {rel} 跳过（远端 404）", flush=True)

            continue

        if not completed:

            kept = dest.stat().st_size if dest.is_file() else 0

            want_txt = f"/{want / 1e6:.0f}" if want else ""

            print(

                f"  [{i}/{len(selected)}] {rel} 没下完（{kept / 1e6:.1f}{want_txt}MB），"

                f"下次运行会从这个断点继续",

                flush=True,

            )

            failed.append(rel)

            continue

        total_new += written

        final = dest.stat().st_size

        flag = ""

        if want and final != want:

            flag = f"  <-- 警告：本地 {final}B / 远端 {want}B"

        print(f"  [{i}/{len(selected)}] {rel}  +{written / 1e6:.1f}MB (共 {final / 1e6:.1f}MB){flag}", flush=True)

    size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())

    return {

        "ok": not failed,

        "endpoint": endpoint,

        "method": "direct",

        "path": str(local_dir),

        "bytes": size,

        "mb": round(size / 1024 / 1024, 1),

        "new_bytes": total_new,

        "files": len(selected),

        "files_already_complete": skipped_done,

        "failed": failed,

    }

def build_patterns(files: Optional[List[str]]) -> Optional[List[str]]:

    if not files:

        return None

    pats: List[str] = []

    for f in files:

        pats.extend(SUBDIRS.get(f, [f]))

    return pats

def endpoint_candidates() -> List[Optional[str]]:

    """返回要依次尝试的 HF_ENDPOINT 列表。

    默认顺序是 **镜像优先，再直连**：本作品开发环境实测 huggingface.co 的 443 端口

    直接超时（DNS 能解析），而 hf-mirror.com 137ms 可达；反过来排会先白白等一轮超时。

    """

    forced = os.environ.get("PPT_HF_ENDPOINT")

    if forced:

        return [forced]

    if os.environ.get("HF_ENDPOINT"):

        return [os.environ["HF_ENDPOINT"]]

    return ["https://hf-mirror.com", None]

def download_one(spec: Dict[str, Any], patterns: Optional[List[str]], direct: bool = False,

                 quiet: bool = False) -> Dict[str, Any]:

    local_dir = HF_HOME / spec["local"]

    local_dir.mkdir(parents=True, exist_ok=True)

    missing = check_complete(spec, local_dir)

    if not missing:

        size = sum(f.stat().st_size for f in local_dir.rglob("*") if f.is_file())

        print(f"[{spec['local']}] 本地已完整（{round(size / 1024 / 1024, 1)}MB），跳过下载", flush=True)

        return {

            "ok": True,

            "endpoint": "（本地已有）",

            "method": "cache",

            "path": str(local_dir),

            "bytes": size,

            "mb": round(size / 1024 / 1024, 1),

            "seconds": 0.0,

        }

    last_err: Optional[BaseException] = None

    for endpoint in endpoint_candidates():

        if endpoint:

            os.environ["HF_ENDPOINT"] = endpoint

        else:

            os.environ.pop("HF_ENDPOINT", None)

        label = endpoint or "https://huggingface.co（默认）"

        print(f"\n[{spec['local']}] 尝试 {label}（还缺 {len(missing)} 个必要文件）", flush=True)

        if direct:

            t0 = time.time()

            try:

                result = download_one_direct(spec, patterns, quiet=quiet)

            except BaseException as exc:  # noqa: BLE001

                last_err = exc

                print(f"[{spec['local']}] 直连失败：{type(exc).__name__}: {str(exc)[:300]}", flush=True)

                continue

            missing2 = check_complete(spec, local_dir)

            if missing2:

                last_err = RuntimeError(f"缺少必要文件：{missing2}")

                print(f"[{spec['local']}] 直连完成但仍缺：{missing2}", flush=True)

                continue

            result["endpoint"] = label

            result["seconds"] = round(time.time() - t0, 1)

            result["missing_before"] = missing

            return result

        try:

            # 必须在设置 HF_ENDPOINT 之后再 import，否则常量已经定死

            from huggingface_hub import snapshot_download

            t0 = time.time()

            path = snapshot_download(

                repo_id=spec["repo_id"],

                local_dir=str(local_dir),

                allow_patterns=patterns,

                max_workers=4,

                resume_download=True,

            )

        except BaseException as exc:  # noqa: BLE001 —— 网络异常种类太多，一律回退重试

            last_err = exc

            print(f"[{spec['local']}] 失败：{type(exc).__name__}: {str(exc)[:300]}", flush=True)

            continue

        # 关键校验：远端不可达时 huggingface_hub 会打印警告后把已有 local_dir 当结果返回，

        # 不抛异常。所以必须自己核对必要文件，否则会把半份权重记成"下载完成"。

        missing2 = check_complete(spec, Path(path))

        if missing2:

            last_err = RuntimeError(f"缺少必要文件：{missing2}")

            print(f"[{spec['local']}] snapshot_download 返回但本地不全，缺：{missing2}", flush=True)

            continue

        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())

        return {

            "ok": True,

            "endpoint": label,

            "method": "snapshot_download",

            "path": str(path),

            "bytes": size,

            "mb": round(size / 1024 / 1024, 1),

            "seconds": round(time.time() - t0, 1),

            "missing_before": missing,

        }

    return {"ok": False, "error": f"{type(last_err).__name__}: {last_err}", "missing": missing}

def write_manifest(entries: Dict[str, Dict[str, Any]]) -> Path:

    out = ROOT / "artifacts" / "model_manifest.json"

    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {

        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),

        "hf_home": str(HF_HOME),

        "entries": entries,

    }

    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return out

def main(argv: Optional[List[str]] = None) -> int:

    ap = argparse.ArgumentParser(description="下载 HuggingFace 权重到项目内")

    ap.add_argument("which", nargs="+", choices=sorted(CATALOG) + ["all"])

    ap.add_argument("--files", nargs="*", default=None,

                    help=f"只要部分子目录，可选：{', '.join(sorted(SUBDIRS))}")

    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--direct", action="store_true",

                    help="不用 huggingface_hub，直接按文件清单逐个下载（远端超时的环境更可靠）")

    ap.add_argument("--quiet", action="store_true", help="不打印每个文件的增量进度")

    ap.add_argument("--v6", action="store_true",

                    help="优先走 IPv6（省流量：镜像的 CDN 有 AAAA，API 仍只能走 IPv4）")

    ap.add_argument("--v4", action="store_true", help="强制只走 IPv4（对照用）")

    args = ap.parse_args(argv)

    if args.v6 or args.v4:

        sys.path.insert(0, str(ROOT / "tools"))

        import net_ipv6

        if args.v6:

            print(f"[net] {net_ipv6.describe()}", flush=True)

            net_ipv6.install_urllib_ipv6()

        else:

            net_ipv6._IPV6_STATUS = False

            print("[net] 强制 IPv4", flush=True)

    targets = sorted(CATALOG) if "all" in args.which else args.which

    patterns = build_patterns(args.files)

    plan = {k: CATALOG[k] for k in targets}

    print(json.dumps(plan, ensure_ascii=False, indent=2))

    if patterns:

        print(f"只下载：{patterns}")

    if args.dry_run:

        return 0

    manifest_path = ROOT / "artifacts" / "model_manifest.json"

    entries: Dict[str, Dict[str, Any]] = {}

    if manifest_path.is_file():

        try:

            entries = json.loads(manifest_path.read_text(encoding="utf-8")).get("entries", {})

        except json.JSONDecodeError:

            entries = {}

    rc = 0

    for key in targets:

        spec = CATALOG[key]

        result = download_one(spec, patterns, direct=args.direct, quiet=args.quiet)

        entries[key] = {**spec, **result, "files_filter": patterns}

        if not result["ok"]:

            rc = 1

            print(f"[{key}] 未完成：{result.get('error')}", flush=True)

        else:

            print(f"[{key}] 完成：{result['mb']}MB / {result.get('seconds')}s -> {result['path']}", flush=True)

    out = write_manifest(entries)

    print(f"\n清单 -> {out}")

    return rc

if __name__ == "__main__":

    raise SystemExit(main())

