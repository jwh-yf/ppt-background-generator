"""pptbg Web 应用：输入需求 → 输出成套 PPT 背景 → 一键导出 .pptx。"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import random
import sys
import threading
import time
import traceback
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
for extra in (str(APP_DIR), str(ROOT), str(ROOT / "data"), str(ROOT / "train")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from formspec import render_page  # noqa: E402
from pptx_exporter import export as export_pptx  # noqa: E402

from presets import ASPECTS, PRESETS, variant_catalog  # noqa: E402
from sd_service import GenerateRequest, SDService  # noqa: E402

from formspec import SCENARIOS  # noqa: E402

PRESET_ORDER: list[str] = []
for _s in SCENARIOS:
    if _s["preset"] in PRESETS and _s["preset"] not in PRESET_ORDER:
        PRESET_ORDER.append(_s["preset"])
for _k in PRESETS:
    if _k not in PRESET_ORDER:
        PRESET_ORDER.append(_k)

class JobStore:
    def __init__(self, root: Path, keep: int = 30) -> None:
        self.root = root
        self.keep = keep
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._order: list[str] = []

    def create(self) -> str:
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        with self._lock:
            self._jobs[job_id] = {"created": time.time(), "records": [], "dir": self.root / job_id}
            self._order.append(job_id)
            while len(self._order) > self.keep:
                old = self._order.pop(0)
                self._jobs.pop(old, None)
        (self.root / job_id).mkdir(parents=True, exist_ok=True)
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def set_records(self, job_id: str, records: list[Dict[str, Any]]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["records"] = records

    def add_records(self, job_id: str, records: list[Dict[str, Any]]) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["records"].extend(records)

SERVICE = SDService()
JOBS: Optional[JobStore] = None

def _public_record(job_id: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(rec)
    out["url"] = f"/files/{job_id}/{rec.get('file', '')}"
    out.pop("prompt_full", None)
    return out

FROST_SHAPES = ("square", "ellipse", "mix")
RICHNESS_LEVELS = ("sparse", "normal", "rich")
SYMBOL_STYLES = ("vector", "brush")

def _choice(data: Dict[str, Any], key: str, default: str, allowed: tuple) -> str:
    v = str(data.get(key) or default).strip().lower()
    return v if v in allowed else default

class Handler(BaseHTTPRequestHandler):
    server_version = "pptbg/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload: Any, code: int = HTTPStatus.OK) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"))

    def _error(self, code: int, message: str, detail: str = "") -> None:
        self._json({"ok": False, "error": message, "detail": detail}, code)

    def _body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def log_message(self, fmt: str, *args: Any) -> None:
        if not str(args[0] if args else "").startswith("2"):
            sys.stderr.write("[pptbg] " + (fmt % args) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, render_page().encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json({"ok": True, **SERVICE.health(), "presets": len(PRESETS)})
            elif path == "/api/scenarios":
                self._json(
                    {
                        "ok": True,
                        "scenarios": SCENARIOS,
                        "presets": _preset_payload(),
                        "variants": variant_catalog(),
                    }
                )
            elif path == "/api/presets":
                self._json({"ok": True, "order": PRESET_ORDER, "presets": _preset_payload(), "aspects": list(ASPECTS)})
            elif path.startswith("/files/"):
                self._serve_file(path[len("/files/") :])
            else:
                self._error(HTTPStatus.NOT_FOUND, f"未知路径 {path}")
        except Exception:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "服务端异常", traceback.format_exc())

    def do_HEAD(self) -> None:  # noqa: N802
        self.do_GET()

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        try:
            if path == "/api/generate":
                self._api_generate()
            elif path == "/api/regen":
                self._api_regen()
            elif path == "/api/deck":
                self._api_deck()
            elif path == "/api/export":
                self._api_export()
            else:
                self._error(HTTPStatus.NOT_FOUND, f"未知路径 {path}")
        except Exception:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "服务端异常", traceback.format_exc())

    def _serve_file(self, rel: str) -> None:
        assert JOBS is not None
        parts = rel.split("/")
        if len(parts) != 2:
            self._error(HTTPStatus.BAD_REQUEST, "路径格式应为 /files/<job>/<name>")
            return
        job_id, name = parts
        if JOBS.get(job_id) is None:
            self._error(HTTPStatus.NOT_FOUND, f"任务不存在或已过期：{job_id}")
            return
        target = (JOBS.root / job_id / name).resolve()
        if target.parent != (JOBS.root / job_id).resolve() or not target.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"文件不存在：{name}")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "image/svg+xml"):
            ctype += "; charset=utf-8"
        self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    def _api_generate(self) -> None:
        assert JOBS is not None
        data = self._body()
        preset_key = str(data.get("preset") or "academic_blue")
        if preset_key not in PRESETS:
            self._error(HTTPStatus.BAD_REQUEST, f"未知预设：{preset_key}")
            return
        aspect = str(data.get("aspect") or "16:9")
        if aspect not in ASPECTS:
            self._error(HTTPStatus.BAD_REQUEST, f"未知比例：{aspect}")
            return

        def _int(key: str, default: int, lo: int, hi: int) -> int:
            try:
                v = int(data.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _float(key: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(data.get(key, default))
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        def _richness(key: str, default: str) -> str:
            v = str(data.get(key) or default).strip().lower()
            return v if v in ("sparse", "normal", "rich") else default

        def _shape(key: str, default: str) -> str:
            # 白名单而不是直接透传：这个值会决定走哪套轮廓代码，脏值必须挡在门口
            v = str(data.get(key) or default).strip().lower()
            return v if v in ("square", "ellipse", "mix") else default

        req = GenerateRequest(
            preset_key=preset_key,
            subject=str(data.get("subject") or ""),
            aspect=aspect,
            count=_int("count", 1, 1, 4),
            steps=_int("steps", SERVICE.cfg.steps, 8, 60),
            guidance=float(data.get("guidance", SERVICE.cfg.guidance) or SERVICE.cfg.guidance),
            seed=_int("seed", -1, -1, 2**31 - 1),
            enforce_safe_zone=bool(data.get("enforce_safe_zone", True)),
            tone_variant=str(data.get("tone_variant") or ""),
            tone_random=bool(data.get("tone_random", False)),
            soft_focus=_float("soft_focus", 0.0, 0.0, 1.0),
            frost=_float("frost", 0.0, 0.0, 1.0),
            frost_shape=_shape("frost_shape", "square"),
            motif_richness=_richness("motif_richness", "normal"),
            symbol_style=_choice(data, "symbol_style", "vector", SYMBOL_STYLES),
        )

        job_id = JOBS.create()
        t0 = time.time()
        records = SERVICE.generate(req, JOBS.root / job_id)
        JOBS.add_records(job_id, records)
        self._json(
            {
                "ok": True,
                "job_id": job_id,
                "elapsed": round(time.time() - t0, 2),
                "backend": SERVICE.health().get("backend"),
                "request": asdict(req),
                "images": [_public_record(job_id, r) for r in records],
            }
        )

    def _api_regen(self) -> None:
        """把某一张重画，结果写回同一个任务目录。
        
        为什么不做在前端拼一个新请求：导出 pptx 是按任务目录取记录的，如果新图落在"""
        assert JOBS is not None
        data = self._body()
        job_id = str(data.get("job_id") or "")
        job = JOBS.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "任务不存在或已过期，请先生成")
            return
        records = job.get("records") or []
        try:
            index = int(data.get("index", -1))
        except (TypeError, ValueError):
            index = -1
        if not (0 <= index < len(records)):
            self._error(HTTPStatus.BAD_REQUEST, "index 超出范围")
            return

        old = records[index]
        new_seed = int(data.get("seed", -1))
        if new_seed < 0:
            new_seed = random.randrange(1 << 31)
        req = GenerateRequest(
            preset_key=str(old.get("preset") or ""),
            subject=str(old.get("subject") or ""),
            aspect=str(old.get("aspect") or "16:9"),
            count=1,
            steps=int(old.get("steps") or SERVICE.cfg.steps),
            guidance=float(old.get("guidance") or SERVICE.cfg.guidance),
            seed=new_seed,
            enforce_safe_zone=bool(old.get("safe_zone_enforced", True)),
            # 重画要沿用原标题的配色与模糊设置，否则重画一张会悄悄换风格
            tone_variant=str(old.get("tone_variant") or ""),
            tone_random=bool(old.get("tone_random", False)),
            soft_focus=float(old.get("soft_focus") or 0.0),
            frost=float(old.get("frost") or 0.0),
            frost_shape=str(old.get("frost_shape") or "square"),
            motif_richness=str(old.get("motif_richness") or "normal"),
            symbol_style=str(old.get("symbol_style") or "vector"),
        )
        t0 = time.time()
        fresh = SERVICE.generate(req, JOBS.root / job_id)[0]
        records[index] = fresh
        old_path = JOBS.root / job_id / str(old.get("file") or "")
        try:
            if old_path.is_file() and old_path.name != fresh["file"]:
                old_path.unlink()
        except OSError:
            pass
        self._json(
            {
                "ok": True,
                "job_id": job_id,
                "index": index,
                "elapsed": round(time.time() - t0, 2),
                "backend": SERVICE.health().get("backend"),
                "image": _public_record(job_id, fresh),
            }
        )

    def _api_deck(self) -> None:
        """一次生成一整套 PPT：少量背景图循环复用，配上封面/目录/正文/尾页的文字。
        
        为什么不是每页生成一张背景：这张卡限功率时一张要 9–45 秒，8 页就是几分钟；"""
        assert JOBS is not None
        data = self._body()
        preset = str(data.get("preset") or "")
        if preset not in PRESETS:
            self._error(HTTPStatus.BAD_REQUEST, f"未知预设：{preset}")
            return
        topic = str(data.get("title") or data.get("topic") or "").strip()
        subject = str(data.get("subject") or "").strip()
        slides = max(3, min(20, int(data.get("slides") or 8)))
        # TODO 这里只画 3 张底图，导出时按 idx % 3 循环铺。答辩要求「每页都不一样」的话，
        #      把下面的 3 改成 slides 就行，代价是生成时间线性涨。
        aspect = str(data.get("aspect") or "16:9")
        if aspect not in ASPECTS:
            self._error(HTTPStatus.BAD_REQUEST, f"未知画面比例：{aspect}")
            return

        req = GenerateRequest(
            preset_key=preset,
            subject=subject or topic,
            aspect=aspect,
            count=3,
            steps=int(data.get("steps") or SERVICE.cfg.steps),
            guidance=float(data.get("guidance") or SERVICE.cfg.guidance),
            seed=int(data.get("seed", -1)),
            enforce_safe_zone=bool(data.get("enforce_safe_zone", True)),
            tone_variant=str(data.get("tone_variant") or ""),
            tone_random=bool(data.get("tone_random", False)),
            soft_focus=max(0.0, min(1.0, float(data.get("soft_focus") or 0.0))),
            frost=max(0.0, min(1.0, float(data.get("frost") or 0.0))),
            frost_shape=_choice(data, "frost_shape", "square", FROST_SHAPES),
            motif_richness=_choice(data, "motif_richness", "normal", RICHNESS_LEVELS),
            symbol_style=_choice(data, "symbol_style", "vector", SYMBOL_STYLES),
        )
        job_id = JOBS.create()
        t0 = time.time()
        records = SERVICE.generate(req, JOBS.root / job_id)
        out_path = JOBS.root / job_id / "deck.pptx"
        info = export_pptx(
            records, out_path, topic or "演示主题", JOBS.root / job_id, slides=slides
        )
        self._json(
            {
                "ok": True,
                "job_id": job_id,
                "elapsed": round(time.time() - t0, 2),
                "backend": SERVICE.health().get("backend"),
                "images": [_public_record(job_id, r) for r in records],
                "pptx_url": f"/files/{job_id}/{out_path.name}",
                "size_kb": round(out_path.stat().st_size / 1024, 1),
                "slides": info,
                "topic": topic or "演示主题",
                "requested_slides": slides,
            }
        )

    def _api_export(self) -> None:
        assert JOBS is not None
        data = self._body()
        job_id = str(data.get("job_id") or "")
        job = JOBS.get(job_id)
        if job is None:
            self._error(HTTPStatus.NOT_FOUND, "任务不存在或已过期，请先生成")
            return
        records = job.get("records") or []
        if not records:
            self._error(HTTPStatus.BAD_REQUEST, "该任务没有可用结果")
            return
        topic = str(data.get("topic") or "").strip() or "演示主题"
        count = max(1, min(20, int(data.get("slides") or len(records))))
        out_path = JOBS.root / job_id / "deck.pptx"
        info = export_pptx(records, out_path, topic, JOBS.root / job_id, slides=count)
        self._json(
            {
                "ok": True,
                "job_id": job_id,
                "pptx_url": f"/files/{job_id}/{out_path.name}",
                "size_kb": round(out_path.stat().st_size / 1024, 1),
                "slides": info,
                "topic": topic,
                "requested_slides": count,
            }
        )

def _preset_payload() -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    for key in PRESET_ORDER:
        if key not in PRESETS:
            continue
        p = PRESETS[key]
        payload[key] = {
            "key": key,
            "name_zh": getattr(p, "name_zh", key),
            "scenario_zh": getattr(p, "scenario_zh", ""),
            "layouts": list(getattr(p, "layouts", ())),
            "tone": getattr(p, "tone", ""),
            "caption": getattr(p, "caption", ""),
        }
    return payload

def build_server(host: str, port: int, out_root: Path) -> ThreadingHTTPServer:
    global JOBS
    JOBS = JobStore(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd

def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="pptbg Web 服务")
    ap.add_argument("--host", default=os.environ.get("PPT_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PPT_PORT", "8760")))
    ap.add_argument("--out", type=Path, default=ROOT / "app" / "outputs")
    ap.add_argument("--preload", action="store_true", help="启动时即加载模型（默认首次请求时加载）")
    args = ap.parse_args(argv)

    if args.preload:
        print(f"[pptbg] 预加载模型：{SERVICE.load()}", flush=True)

    httpd = build_server(args.host, args.port, args.out)
    url = f"http://{args.host}:{args.port}/"
    print(f"[pptbg] 服务已启动：{url}", flush=True)
    print(f"[pptbg] 后端：{SERVICE.health().get('backend')}（首次生成时可能需等待模型加载）", flush=True)
    print("[pptbg] Ctrl+C 停止", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[pptbg] 已停止", flush=True)
    finally:
        httpd.server_close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
