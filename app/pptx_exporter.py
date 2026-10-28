"""把生成的背景图导出成可直接编辑的 .pptx。

为什么这一步值得单独做"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "data"))
from presets import ASPECTS, PRESETS, TONES  # noqa: E402

EMU_PER_INCH = 914400

def _has_pptx() -> bool:
    try:
        import pptx  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False

def zone_luminance(img_path: Path, zone: Dict[str, Any]) -> Optional[float]:
    """安全区平均亮度（0-255），取不到图就返回 None。

    导出时字色要跟着背景走，不能只信预设里的 ink_hex，否则预览承诺的可读性在导出后就没了。
    """
    try:
        from PIL import Image

        with Image.open(img_path) as im:
            gray = im.convert("L")
            box = (
                max(0, int(zone["x"])),
                max(0, int(zone["y"])),
                min(gray.width, int(zone["x"] + zone["w"])),
                min(gray.height, int(zone["y"] + zone["h"])),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            data = list(gray.crop(box).resize((16, 16), Image.BILINEAR).getdata())
        return sum(data) / len(data)
    except Exception:  # noqa: BLE001
        return None

@dataclass
class SlideText:
    kind: str
    title: str
    bullets: List[str]
    zone_name: str
    title_zone: str = "title"

def plan_slides(topic: str, count: int, sections: Sequence[str] = ()) -> List[SlideText]:
    """把主题 + 张数规划成一套页面骨架。"""
    topic = topic.strip() or "演示主题"
    secs = [s for s in sections if s.strip()]
    if not secs:
        secs = [f"第 {i} 部分" for i in range(1, max(2, count - 2) + 1)]

    slides: List[SlideText] = [SlideText("cover", topic, ["副标题 / 汇报人 / 日期"], "body")]
    if count >= 4:
        slides.append(SlideText("agenda", "目录", secs[:6], "body"))
    body_kinds = ["内容页", "要点页", "图表页", "对比页"]
    while len(slides) < max(1, count - 1):
        i = len(slides) - (2 if count >= 4 else 1)
        sec = secs[i % len(secs)] if secs else "内容"
        slides.append(
            SlideText(
                "content",
                sec,
                [
                    f"{body_kinds[i % len(body_kinds)]}占位：在此填入要点",
                    "建议每页不超过 5 行、每行不超过 20 字",
                    "背景安全区已避开面板与高对比区域",
                ],
                "body",
            )
        )
    slides.append(SlideText("end", "谢谢", ["Q & A / 联系方式"], "body"))
    return slides[:count] if count > 0 else slides

def export_with_pptx(
    records: Sequence[Dict[str, Any]],
    out_path: Path,
    topic: str,
    images_root: Path,
    font_zh: str = "微软雅黑",
    slides: int = 0,
) -> Dict[str, Any]:
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Emu, Pt

    first = records[0]
    w_px, h_px = int(first["width"]), int(first["height"])
    total = int(slides) if slides and slides > 0 else len(records)
    slides_plan = plan_slides(topic, total)
    preset = PRESETS[first["preset"]]
    tone = TONES[preset.tone]
    ink = RGBColor.from_string(tone.ink_hex.lstrip("#"))

    prs = Presentation()
    prs.slide_width = Emu(int(w_px / 96 * EMU_PER_INCH))
    prs.slide_height = Emu(int(h_px / 96 * EMU_PER_INCH))
    blank = prs.slide_layouts[6]

    for idx, plan in enumerate(slides_plan):
        rec = records[idx % len(records)]
        slide = prs.slides.add_slide(blank)
        img_path = images_root / rec["file"]
        slide.shapes.add_picture(
            str(img_path), 0, 0, width=prs.slide_width, height=prs.slide_height
        )

        zones = {z["region"]: z for z in rec.get("safe_zones", [])}
        tz = zones.get(plan.title_zone) or zones.get("title") or zones.get("body")
        bz = zones.get(plan.zone_name) or zones.get("body") or tz

        def add_text(text: str, zone: Optional[Dict[str, Any]], size: int, bold: bool,
                     align=PP_ALIGN.LEFT, color=None, line_spacing: float = 1.25):
            if not zone:
                return
            pad = int(min(w_px, h_px) * 0.02)
            if color is None:
                lum = zone_luminance(images_root / rec["file"], zone)
                color = RGBColor(0xFF, 0xFF, 0xFF) if (lum is not None and lum < 140) else ink
            box = slide.shapes.add_textbox(
                Emu(int((zone["x"] + pad) / 96 * EMU_PER_INCH)),
                Emu(int((zone["y"] + pad) / 96 * EMU_PER_INCH)),
                Emu(int((zone["w"] - 2 * pad) / 96 * EMU_PER_INCH)),
                Emu(int((zone["h"] - 2 * pad) / 96 * EMU_PER_INCH)),
            )
            tf = box.text_frame
            tf.word_wrap = True
            for i, line in enumerate(text.split("\n")):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.text = line
                p.alignment = align
                p.line_spacing = line_spacing
                for run in p.runs:
                    run.font.size = Pt(size)
                    run.font.bold = bold
                    run.font.name = font_zh
                    run.font.color.rgb = color
            return box

        if plan.kind == "cover":
            add_text(plan.title, tz, size=40, bold=True)
            add_text("\n".join(plan.bullets), bz, size=16, bold=False)
        elif plan.kind == "end":
            add_text(plan.title, tz, size=40, bold=True)
            add_text("\n".join(plan.bullets), bz, size=16, bold=False)
        else:
            add_text(plan.title, tz, size=30, bold=True)
            add_text("\n".join(f"• {b}" for b in plan.bullets), bz, size=17, bold=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    return {
        "engine": "python-pptx",
        "slides": len(prs.slides),
        "size_px": [w_px, h_px],
        "font": font_zh,
        "note": "背景铺满整页，文本框基于安全区元数据定位，可自由编辑。",
    }

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="jpeg" ContentType="image/jpeg"/>
<Default Extension="jpg" ContentType="image/jpeg"/>
<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
{slide_overrides}
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""

def _presentation_xml(slide_ids: Sequence[int], cx: int, cy: int) -> str:
    items = "".join(
        f'<p:sldId id="{256 + i}" r:id="rId{i + 1}"/>' for i, _ in enumerate(slide_ids)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:sldIdLst>{items}</p:sldIdLst>
<p:sldSz cx="{cx}" cy="{cy}"/>
<p:notesSz cx="{cy}" cy="{cx}"/>
</p:presentation>"""

def _presentation_rels(n: int) -> str:
    rels = "".join(
        f'<Relationship Id="rId{i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i + 1}.xml"/>'
        for i in range(n)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{rels}</Relationships>"""

def _slide_rels(media_name: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{media_name}"/>
</Relationships>"""

def _slide_xml(
    media_rel: str, cx: int, cy: int,
    title: str, body: str, title_col: str, box: Tuple[int, int, int, int],
    title_size: int = 3000,
) -> str:
    """一个 slides/slideN.xml：整页图片 + 一个标题框 + 一个正文框。"""
    bx, by, bw, bh = box
    def esc(s: str) -> str:
        return (
            s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        )

    body_paras = "".join(
        f'<a:p><a:r><a:rPr lang="zh-CN" sz="1700" dirty="0"><a:solidFill><a:srgbClr val="{title_col}"/></a:solidFill>'
        f'<a:latin typeface="微软雅黑"/><a:ea typeface="微软雅黑"/></a:rPr><a:t>{esc(line)}</a:t></a:r></a:p>'
        for line in body.split("\n") if line
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
 xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>
<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>
<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>
<p:pic>
  <p:nvPicPr><p:cNvPr id="2" name="背景"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>
  <p:blipFill><a:blip r:embed="{media_rel}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
  <p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr>
</p:pic>
<p:sp>
  <p:nvSpPr><p:cNvPr id="3" name="标题"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{int(cx*0.07)}" y="{int(cy*0.10)}"/><a:ext cx="{int(cx*0.86)}" cy="{int(cy*0.16)}"/></a:xfrm>
  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>
  <a:p><a:pPr algn="l"/><a:r><a:rPr lang="zh-CN" sz="{title_size}" b="1" dirty="0">
  <a:solidFill><a:srgbClr val="{title_col}"/></a:solidFill>
  <a:latin typeface="微软雅黑"/><a:ea typeface="微软雅黑"/></a:rPr><a:t>{esc(title)}</a:t></a:r></a:p>
  </p:txBody>
</p:sp>
<p:sp>
  <p:nvSpPr><p:cNvPr id="4" name="正文"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>
  <p:spPr><a:xfrm><a:off x="{bx}" y="{by}"/><a:ext cx="{bw}" cy="{bh}"/></a:xfrm>
  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom><a:noFill/></p:spPr>
  <p:txBody><a:bodyPr wrap="square"/><a:lstStyle/>{body_paras}</p:txBody>
</p:sp>
</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>"""

def export_minimal_pptx(
    records: Sequence[Dict[str, Any]], out_path: Path, topic: str, images_root: Path
) -> Dict[str, Any]:
    first = records[0]
    w_px, h_px = int(first["width"]), int(first["height"])
    cx, cy = int(w_px / 96 * EMU_PER_INCH), int(h_px / 96 * EMU_PER_INCH)
    plan = plan_slides(topic, len(records))
    tone = TONES[PRESETS[first["preset"]].tone]
    ink = tone.ink_hex.lstrip("#").upper()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            _CONTENT_TYPES.format(
                slide_overrides="".join(
                    f'<Override PartName="/ppt/slides/slide{i+1}.xml" '
                    f'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                    for i in range(len(records))
                )
            ),
        )
        z.writestr("_rels/.rels", _ROOT_RELS)
        z.writestr("ppt/presentation.xml", _presentation_xml(range(len(records)), cx, cy))
        z.writestr("ppt/_rels/presentation.xml.rels", _presentation_rels(len(records)))
        for i, (rec, sp) in enumerate(zip(records, plan), 1):
            media_name = f"image{i}.jpg"
            z.write(images_root / rec["file"], f"ppt/media/{media_name}")
            z.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", _slide_rels(media_name))
            zones = {zz["region"]: zz for zz in rec.get("safe_zones", [])}
            bz = zones.get(sp.zone_name) or zones.get("body") or zones.get("title")
            box = (
                int((bz["x"] + 20) / 96 * EMU_PER_INCH),
                int((bz["y"] + 20) / 96 * EMU_PER_INCH),
                int((bz["w"] - 40) / 96 * EMU_PER_INCH),
                int((bz["h"] - 40) / 96 * EMU_PER_INCH),
            ) if bz else (int(cx * 0.07), int(cy * 0.35), int(cx * 0.86), int(cy * 0.55))
            body = "\n".join(
                (f"• {b}" for b in sp.bullets) if sp.kind not in ("cover", "end") else sp.bullets
            )
            z.writestr(
                f"ppt/slides/slide{i}.xml",
                _slide_xml(f"rId1", cx, cy, sp.title, body, ink, box,
                           title_size=4000 if sp.kind in ("cover", "end") else 3000),
            )
    return {
        "engine": "stdlib-zipfile（降级：未安装 python-pptx）",
        "slides": len(records),
        "size_px": [w_px, h_px],
        "note": "最小合法 pptx：背景铺满 + 可编辑文本框。安装 python-pptx 后样式更丰富。",
    }

def export(
    records: Sequence[Dict[str, Any]],
    out_path: Path,
    topic: str,
    images_root: Path,
    force_minimal: bool = False,
    slides: int = 0,
) -> Dict[str, Any]:
    if not records:
        raise SystemExit("没有可导出的记录")
    if not force_minimal and _has_pptx():
        return export_with_pptx(records, out_path, topic, images_root, slides=slides)
    return export_minimal_pptx(records, out_path, topic, images_root)

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="把生成结果导出为 .pptx")
    ap.add_argument("--records", type=Path, required=True, help="含 file/preset/safe_zones 的 JSONL")
    ap.add_argument("--images-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topic", default="演示主题")
    ap.add_argument("--minimal", action="store_true", help="强制走标准库降级路径")
    args = ap.parse_args(argv)

    records = [json.loads(l) for l in args.records.read_text(encoding="utf-8").splitlines() if l.strip()]
    info = export(records, args.out, args.topic, args.images_root, force_minimal=args.minimal)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"已导出 -> {args.out}（{args.out.stat().st_size/1024:.0f} KB）")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
