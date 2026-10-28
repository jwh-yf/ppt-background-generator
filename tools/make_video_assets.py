# -*- coding: utf-8 -*-

"""把演示视频要用的几张「讲解图」一次性生成到 submission/video/assets/。

为什么要预先生成：视频是**无声录屏 + 字幕**的方案，讲解靠的是画面上出现的图。"""
from __future__ import annotations

import json

import sys

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent

OUT = ROOT / "submission" / "video" / "assets"

OUT.mkdir(parents=True, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"

FONT_B = r"C:\Windows\Fonts\msyhbd.ttc"

W, H = 1920, 1080

BG = (18, 20, 24)

FG = (238, 240, 244)

DIM = (150, 156, 168)

ACC = (90, 150, 255)

def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:

    return ImageFont.truetype(FONT_B if bold else FONT, size)

def canvas() -> Image.Image:

    return Image.new("RGB", (W, H), BG)

def text(d: ImageDraw.ImageDraw, xy, s, size=40, color=FG, bold=False, anchor="la"):

    d.text(xy, s, font=f(size, bold), fill=color, anchor=anchor)

def tw(font: ImageFont.FreeTypeFont, s: str) -> int:

    """量一段文字有多宽，用来把注释行接在数值右边。"""

    l, _, r, _ = font.getbbox(s)

    return r - l

def footer(d: ImageDraw.ImageDraw, s: str) -> None:

    text(d, (W - 64, H - 46), s, 24, DIM, anchor="ra")

def fit(path: Path) -> Image.Image:

    return Image.open(path).convert("RGB")

def paste_fit(base: Image.Image, img: Image.Image, box) -> None:

    x, y, w, h = box

    r = min(w / img.width, h / img.height)

    im = img.resize((int(img.width * r), int(img.height * r)), Image.LANCZOS)

    base.paste(im, (x + (w - im.width) // 2, y + (h - im.height) // 2))

def asset_01() -> Path:

    """左：真实图库照片压标题（糊）；右：本作品背景压同一标题（清楚）。

    左边用的是**基座模型自己的出图**（不带 LoRA），这比拿网图更有说服力：

    同一个模型、同一个需求，只是缺了「留白」这一课。

    """

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 40), "同一个标题，两种背景", 46, FG, bold=True)

    text(d, (64, 104), "左边：通用文生图模型的输出（实景、高对比）　右边：本作品（先留白，再压字）",

         28, DIM)

    src = fit(ROOT / "eval/compare/compare_sheet.jpg")

    left = src.crop((0, 30, 392, 244))

    right = fit(ROOT / "eval/compare/skins2/academic_blue_0020261016_00.jpg")

    box_w, box_h = 860, 780

    for i, (img, x) in enumerate(((left, 64), (right, 64 + box_w + 72))):

        base.paste(img.resize((box_w, box_h), Image.LANCZOS), (x, 160))

        dd = ImageDraw.Draw(base)

        dd.text((x + box_w // 2, 160 + 210), "校园垃圾分类调研报告",

                font=f(74, True), fill=(255, 255, 255), anchor="ma")

        dd.text((x + box_w // 2, 160 + 330), "本页用于验证标题压上去是否还读得清",

                font=f(34), fill=(255, 255, 255), anchor="ma")

        dd.text((x + box_w // 2, 160 + box_h + 24),

                "读出吃力" if i == 0 else "清楚可读", font=f(32, True),

                fill=(255, 120, 120) if i == 0 else (120, 230, 160), anchor="ma")

    footer(d, "pptbg · 同一个模型 · 同一个 seed · 只差「有没有为留白训练过」")

    p = OUT / "01_痛点对比.jpg"

    base.save(p, quality=92)

    return p

def asset_02() -> Path:

    """数据集：600 张全程序生成，安全区标注在渲染那一刻就定下来了。"""

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 40), "训练数据：600 张，全部由脚本渲染，不抓图库", 46, FG, bold=True)

    text(d, (64, 104), "安全区坐标与图像在同一次渲染中产生 —— 所以没有标注噪声，也不涉及第三方图片版权",

         28, DIM)

    sheet = fit(ROOT / "eval/compare/dataset_v4_preview.jpg")

    paste_fit(base, sheet, (64, 160, 1200, 700))

    rep = json.loads((ROOT / "data/dataset_v3/report.json").read_text(encoding="utf-8"))

    v = rep.get("verdict", {})

    agg = rep.get("aggregate", {})

    rows = [

        ("图片数", "600 张 / 1920×1080", ""),

        ("体检结论", "可训练", "硬问题 0"),

        ("软问题", f"{v.get('soft_issues', 0)} 条", "安全区偏亮 / 少量近似重复"),

        ("近似重复",

         f"{float(v.get('dup_ratio', agg.get('dup_ratio', 0))) * 100:.2f}%",

         f"{agg.get('dup_count', 0)} 对"),

        ("版式", "7 种",

         "最热一种占 "

         f"{float(rep['distributions']['layout']['max_share']) * 100:.1f}%"),

        ("生成器版本", "1.4.0", ""),

    ]

    y = 196

    for k, val, note in rows:

        text(d, (1300, y), k, 26, DIM)

        text(d, (1300, y + 32), val, 40, FG, bold=True)

        if note:

            text(d, (1300 + tw(f(40, True), val) + 16, y + 44), note, 24, DIM)

        y += 132

    footer(d, "data/dataset_v3 · 体检报告 report.json")

    p = OUT / "02_数据集.jpg"

    base.save(p, quality=92)

    return p

def asset_03() -> Path:

    """两版 LoRA 的 val_loss 曲线：说明「加步数」不是解法。"""

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 40), "训练：1000 步 / 21 分钟 / 可训练参数只占 0.37%", 46, FG, bold=True)

    text(d, (64, 104), "验证损失在第 500 步就见底，之后回升——所以「继续训练」不是提升画质的办法",

         28, DIM)

    hist = {}

    for tag, path in (("v1", "train/runs/pptbg-lora-v1/history.jsonl"),

                      ("v4", "train/runs/pptbg-lora-v4-layouts/history.jsonl")):

        rows = [json.loads(l) for l in (ROOT / path).open(encoding="utf-8")]

        hist[tag] = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]

    x0, y0, x1, y1 = 200, 220, 1380, 820

    d.rectangle([x0, y0, x1, y1], outline=(60, 64, 74), width=2)

    allv = [v for pts in hist.values() for _, v in pts]

    vmax = max(allv) * 1.15

    for frac in (0.25, 0.5, 0.75):

        yy = int(y1 - (y1 - y0) * frac)

        d.line([x0, yy, x1, yy], fill=(40, 44, 52), width=1)

    colors = {"v1": (110, 170, 255), "v4": (255, 180, 90)}

    above = {"v1": True, "v4": False}

    for tag, pts in hist.items():

        xy = [(x0 + int((x1 - x0) * st / 1000), y1 - int((y1 - y0) * v / vmax)) for st, v in pts]

        d.line(xy, fill=colors[tag], width=5, joint="curve")

        for p_, (st, v) in zip(xy, pts):

            d.ellipse([p_[0] - 7, p_[1] - 7, p_[0] + 7, p_[1] + 7], fill=colors[tag])

            dy = -34 if above[tag] else 16

            d.text((p_[0], p_[1] + dy), f"{v:.4f}", font=f(24, True), fill=colors[tag],

                   anchor="md" if above[tag] else "ma")

    for st, _ in hist["v1"]:

        d.text((x0 + int((x1 - x0) * st / 1000), y1 + 18), str(st), font=f(26), fill=DIM, anchor="ma")

    text(d, (200, 170), "val_loss", 28, DIM)

    text(d, (1450, 300), "v1", 34, colors["v1"], bold=True)

    text(d, (1520, 300), "旧数据集（4 种版式）", 28, DIM)

    text(d, (1450, 360), "v4", 34, colors["v4"], bold=True)

    text(d, (1520, 360), "7 种版式 + 装饰层", 28, DIM)

    for i, line in enumerate(("两版 val_loss 同水位：", "数据换了，拟合能力没变。", "瓶颈不在训练。")):

        text(d, (1450, 470 + i * 46), line, 28, FG if i == 2 else DIM, bold=(i == 2))

    footer(d, "train/runs/*/history.jsonl")

    p = OUT / "03_训练曲线.jpg"

    base.save(p, quality=92)

    return p

def asset_04() -> Path:

    src = ROOT / "eval/results_v1/sheets/backends_with_text.jpg"

    img = fit(src)

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 34), "五个后端 · 同一需求 · 同一 seed · 同一条代码流水线", 44, FG, bold=True)

    paste_fit(base, img, (64, 110, W - 128, H - 190))

    footer(d, "eval/results_v1 · 40 条需求 × 5 后端 = 200 张")

    p = OUT / "04_五后端对比.jpg"

    base.save(p, quality=92)

    return p

def asset_05() -> Path:

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 40), "负结果也照写：换数据集没换来画质提升", 46, FG, bold=True)

    text(d, (64, 104), "左：v1（4 种版式）　右：v4（7 种版式 + 装饰层进数据）—— 成图几乎无法区分",

         28, DIM)

    paste_fit(base, fit(ROOT / "eval/compare/ab/sheet.jpg"), (64, 170, 900, H - 210))

    y = 200

    for line, col, bold in (

        ("为什么没有差别？", FG, True),

        ("版式不是模型画的，", DIM, False),

        ("是合成层用确定性代码画的。", DIM, False),

        ("", DIM, False),

        ("模型在 512 分辨率上", DIM, False),

        ("只贡献色场和色调；", DIM, False),

        ("卡片、装饰线、暗角", DIM, False),

        ("全部由代码补上。", DIM, False),

        ("", DIM, False),

        ("所以「让模型学会构图」", ACC, True),

        ("在架构上不会生效。", ACC, True),

    ):

        if line:

            text(d, (1010, y), line, 34 if not bold else 36, col, bold=bold)

        y += 52

    footer(d, "技术报告 §4.7 / §4.8 · eval/compare/ab/sheet.jpg")

    p = OUT / "05_负结果.jpg"

    base.save(p, quality=92)

    return p

def asset_06() -> Path:

    base = canvas()

    d = ImageDraw.Draw(base)

    text(d, (64, 40), "开源与复现：每个数字都能重跑出来", 46, FG, bold=True)

    text(d, (64, 104), "代码 MIT · 数据集 CC BY 4.0 · LoRA 适配器沿用基座 CreativeML OpenRAIL-M",

         28, DIM)

    cmds = [

        ("生成数据", "pptbg data\\make_backgrounds.py --assets data\\assets_v2\\manifest.jsonl --out data\\dataset_v3 --count 600 --seed 20261016"),

        ("数据体检", "pptbg data\\validate_dataset.py --records data\\dataset_v3\\records.jsonl --images-root data\\dataset_v3\\images --out data\\dataset_v3\\report"),

        ("切分", "pptbg train\\prepare_dataset.py --records data\\dataset_v3\\records.jsonl --out data\\splits_v3"),

        ("训练 LoRA", "pptbg train\\train_lora.py --config configs\\lora_v1.yaml"),

        ("评测", "pptbg eval\\run_eval.py --prompts eval\\prompts_v1.jsonl --out eval\\results_v1"),

        ("端到端冒烟", "pptbg app\\smoke_test.py"),

    ]

    y = 190

    for name, cmd in cmds:

        text(d, (64, y), name, 26, ACC, bold=True)

        text(d, (64, y + 34), cmd, 25, FG)

        d.line([64, y + 74, W - 64, y + 74], fill=(38, 42, 50), width=1)

        y += 118

    text(d, (64, y + 20), "数据集 600 张 / 母图 288 张 / 评测 200 张 / 权重全部随仓库或可重训", 28, DIM)

    footer(d, "README §8 · docs/开源及第三方资源使用清单.md")

    p = OUT / "06_开源复现.jpg"

    base.save(p, quality=92)

    return p

def main() -> int:

    made = []

    for fn in (asset_01, asset_02, asset_03, asset_04, asset_05, asset_06):

        try:

            p = fn()

            made.append(p)

            print(f"OK  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")

        except Exception as exc:  # noqa: BLE001

            print(f"FAIL {fn.__name__}: {type(exc).__name__}: {exc}")

    print(f"共 {len(made)} 张 -> {OUT.relative_to(ROOT)}")

    return 0 if len(made) == 6 else 1

if __name__ == "__main__":

    sys.exit(main())

