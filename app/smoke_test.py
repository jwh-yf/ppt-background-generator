"""端到端冒烟测试"""



from __future__ import annotations

import argparse
import io
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"

class Result:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, test: str, status: str, detail: str, evidence: Optional[Dict[str, Any]] = None) -> None:
        self.rows.append({"test": test, "status": status, "detail": detail, "evidence": evidence or {}})
        mark = {PASS: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]"}[status]
        print(f"{mark} {test}: {detail}", flush=True)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.rows if r["status"] == FAIL)

def http(
    base: str, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None, timeout: float = 600.0
) -> Tuple[int, bytes, Dict[str, str]]:
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers or {})

def http_json(base: str, path: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Tuple[int, Any]:
    code, body, _ = http(base, path, method, payload)
    try:
        return code, json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return code, {"_raw": body[:400].decode("utf-8", "replace")}

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])

def wait_ready(base: str, timeout: float = 60.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code, _ = http_json(base, "/api/health")
            if code == 200:
                return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.5)
    return False

def t1_page(base: str, res: Result) -> None:
    code, body, headers = http(base, "/")
    html = body.decode("utf-8", "replace")
    needed = ["<title", "生成", "安全区", "导出"]
    missing = [k for k in needed if k not in html]
    if code == 200 and not missing:
        res.add("T1 页面", PASS, f"HTTP 200，{len(html)} 字符，关键元素齐全", {"chars": len(html)})
    else:
        res.add("T1 页面", FAIL, f"HTTP {code}，缺元素 {missing}")

def t2_health(base: str, res: Result) -> None:
    code, data = http_json(base, "/api/health")
    if code == 200 and data.get("ok"):
        res.add("T2 健康检查", PASS, f"后端={data.get('backend')} 已加载={data.get('ready')}", data)
    else:
        res.add("T2 健康检查", FAIL, f"HTTP {code} {data}")

def t3_presets(base: str, res: Result) -> None:
    code, data = http_json(base, "/api/presets")
    order = data.get("order") or []
    aspects = data.get("aspects") or []
    if code == 200 and len(order) >= 3 and "16:9" in aspects:
        res.add("T3 预设清单", PASS, f"{len(order)} 个预设 / 比例 {aspects}", {"order": order, "aspects": aspects})
    else:
        res.add("T3 预设清单", FAIL, f"HTTP {code} {data}")

def t4_generate(base: str, res: Result) -> Optional[Dict[str, Any]]:
    payload = {
        "preset": "academic_blue",
        "subject": "城市轨道交通客流量预测建模与仿真",
        "aspect": "16:9",
        "count": 2,
        "seed": 20261016,
    }
    t0 = time.time()
    code, data = http_json(base, "/api/generate", "POST", payload)
    dt = time.time() - t0
    if code != 200 or not data.get("ok"):
        res.add("T4 生成", FAIL, f"HTTP {code} {str(data)[:300]}")
        return None
    images = data.get("images") or []
    bad = [i for i in images if not i.get("safe_zones")]
    if not images or bad:
        res.add("T4 生成", FAIL, f"返回 {len(images)} 张，其中 {len(bad)} 张缺安全区元数据")
        return None
    zones = {z["region"]: round(float(z["contrast_std"]), 1) for z in images[0]["safe_zones"]}
    res.add(
        "T4 生成",
        PASS,
        f"{len(images)} 张 / {dt:.1f}s / 后端 {data.get('backend')} / 安全区 {zones}",
        {"job_id": data["job_id"], "elapsed": round(dt, 2), "backend": data.get("backend"),
         "first_zones": zones, "files": [i["file"] for i in images]},
    )
    return data

def t5_files(base: str, gen: Dict[str, Any], res: Result, out_dir: Path) -> None:
    job = gen["job_id"]
    first = gen["images"][0]
    code, body, headers = http(base, first["url"])
    ok_jpeg = body[:2] == b"\xff\xd8" and body[-2:] == b"\xff\xd9"
    if code == 200 and ok_jpeg:
        (out_dir / Path(first["file"]).name).write_bytes(body)
        res.add("T5 图片下载", PASS, f"{first['file']} {len(body) // 1024}KB，JPEG 头尾合法",
                {"content_type": headers.get("Content-Type"), "bytes": len(body)})
    else:
        res.add("T5 图片下载", FAIL, f"HTTP {code}，{len(body)} 字节，JPEG 头尾合法={ok_jpeg}")

def t6_export(base: str, gen: Dict[str, Any], res: Result, out_dir: Path) -> None:
    job = gen["job_id"]
    code, data = http_json(base, "/api/export", "POST", {"job_id": job, "topic": "城市轨道交通客流量预测"})
    if code != 200 or not data.get("ok"):
        res.add("T6 导出 pptx", FAIL, f"HTTP {code} {str(data)[:300]}")
        return
    code, blob, _ = http(base, data["pptx_url"])
    deck = out_dir / "deck.pptx"
    deck.write_bytes(blob)
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = zf.namelist()
            slides = [n for n in names if n.startswith("ppt/slides/slide") and n.endswith(".xml")]
            media = [n for n in names if n.startswith("ppt/media/")]
            bad = zf.testzip()
    except zipfile.BadZipFile as exc:
        res.add("T6 导出 pptx", FAIL, f"下载到的不是合法 zip：{exc}")
        return
    expect = len(gen["images"])
    status = PASS if (code == 200 and len(slides) == expect and not bad) else FAIL
    res.add(
        "T6 导出 pptx",
        status,
        f"{len(blob) // 1024}KB，{len(slides)} 张幻灯片（期望 {expect}），内嵌 {len(media)} 张图，"
        f"zip 校验={'通过' if not bad else bad}",
        {"engine": (data.get("slides") or {}).get("engine"), "slides": len(slides), "media": len(media),
         "saved": str(deck)},
    )

def t7_negative(base: str, gen: Optional[Dict[str, Any]], res: Result) -> None:
    checks: List[Tuple[str, int, Any]] = []
    code, data = http_json(base, "/api/generate", "POST", {"preset": "not_a_preset"})
    checks.append(("未知预设被拒", code, data.get("error")))
    code, data = http_json(base, "/api/export", "POST", {"job_id": "20000101-000000-deadbe"})
    checks.append(("不存在的任务被拒", code, data.get("error")))
    code, data = http_json(base, "/api/nope")
    checks.append(("未知路由返回 404", code, data.get("error")))
    if gen:
        code, _, _ = http(base, f"/files/{gen['job_id']}/../../server.py")
        checks.append(("路径穿越被拒", code, None))
    bad = [c for c in checks if not (400 <= int(c[1]) < 500)]
    if bad:
        res.add("T7 负路径", FAIL, f"这些本该 4xx：{bad}")
    else:
        res.add("T7 负路径", PASS, "；".join(f"{n}→{c}" for n, c, _ in checks),
                {"checks": [{"name": n, "code": c} for n, c, _ in checks]})

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="pptbg 端到端冒烟测试")
    ap.add_argument("--base-url", default=None, help="不填则自动起一个临时服务")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts" / "smoke")
    args = ap.parse_args(argv)

    args.artifacts.mkdir(parents=True, exist_ok=True)
    proc: Optional[subprocess.Popen] = None
    base = args.base_url

    if base is None:
        port = args.port or free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = [sys.executable, str(APP_DIR / "server.py"), "--port", str(port),
               "--out", str(args.artifacts / "outputs")]
        print(f"启动临时服务：{' '.join(cmd)}", flush=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        if not wait_ready(base):
            print("服务未能在 60s 内就绪，输出如下：", flush=True)
            if proc.poll() is not None and proc.stdout:
                print(proc.stdout.read(), flush=True)
            return 3

    res = Result()
    try:
        t1_page(base, res)
        t2_health(base, res)
        t3_presets(base, res)
        gen = t4_generate(base, res)
        if gen:
            t5_files(base, gen, res, args.artifacts)
            t6_export(base, gen, res, args.artifacts)
        t7_negative(base, gen, res)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()

    report = {
        "base_url": base,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "passed": sum(1 for r in res.rows if r["status"] == PASS),
        "failed": res.failed,
        "results": res.rows,
    }
    (args.artifacts / "smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n通过 {report['passed']} / 失败 {report['failed']}；报告 -> {args.artifacts / 'smoke_report.json'}")
    return 0 if res.failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
