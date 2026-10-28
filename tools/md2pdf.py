"""把 docs/技术报告.md 渲染成 PDF，不依赖 Office / LaTeX / 浏览器。

为什么自己写一个渲染器而不是调现成工具（本机实测结论，记录下来免得后人重走）：。"""

from __future__ import annotations

import argparse

import re

import sys

from pathlib import Path

# 本机把 reportlab 解压在 tools/pypdf 下，不装进 site-packages
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pypdf'))

from reportlab.lib import colors

from reportlab.lib.enums import TA_CENTER, TA_LEFT

from reportlab.lib.pagesizes import A4

from reportlab.lib.styles import ParagraphStyle

from reportlab.lib.units import mm

from reportlab.pdfbase import pdfmetrics

from reportlab.pdfbase.ttfonts import TTFont

from reportlab.pdfgen import canvas

from reportlab.platypus import (

    BaseDocTemplate,

    Frame,

    HRFlowable,

    Image,

    KeepTogether,

    PageTemplate,

    Paragraph,

    Preformatted,

    Spacer,

    Table,

    TableStyle,

)

FONT_DIR = Path(r"C:\Windows\Fonts")

FONT_REG = FONT_DIR / "Deng.ttf"

FONT_BOLD = FONT_DIR / "Dengb.ttf"

FONT_MONO = FONT_DIR / "consola.ttf"

FONT_MONO_CJK = FONT_DIR / "simsun.ttc"

INK = colors.HexColor("#1A1A1F")

INK_SOFT = colors.HexColor("#4A4A55")

ACCENT = colors.HexColor("#2F4F8F")

ACCENT_LIGHT = colors.HexColor("#E8EEF8")

RULE = colors.HexColor("#C9D2E0")

CODE_BG = "#EEF2F7"

QUOTE_BG = colors.HexColor("#F4F6FA")

MONO_SIZE = 8.6

BODY_SIZE = 9.6

BODY_LEAD = 15.0

_CJK = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")

_BOX = re.compile(r"[\u2500-\u257f\u25b2\u25bc\u2190-\u21ff]")

def register_fonts() -> None:

    for path in (FONT_REG, FONT_BOLD, FONT_MONO, FONT_MONO_CJK):

        if not path.is_file():

            raise SystemExit(f"缺字体文件：{path}")

    pdfmetrics.registerFont(TTFont("Deng", str(FONT_REG)))

    pdfmetrics.registerFont(TTFont("Deng-Bold", str(FONT_BOLD)))

    pdfmetrics.registerFont(TTFont("Mono", str(FONT_MONO)))

    pdfmetrics.registerFont(TTFont("MonoCJK", str(FONT_MONO_CJK), subfontIndex=0))

    pdfmetrics.registerFontFamily(

        "Deng", normal="Deng", bold="Deng-Bold", italic="Deng", boldItalic="Deng-Bold"

    )

_XML = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}

def _esc(text: str) -> str:

    for k, v in _XML.items():

        text = text.replace(k, v)

    return text

def inline(text: str) -> str:

    """把一行 Markdown 行内标记转成 reportlab Paragraph 的迷你标记。

    先转义 XML 实体，再打标记——顺序不能反，否则 `a < b` 里的 `<` 会被当成标签。

    """

    text = _esc(text)

    def code_span(m: re.Match) -> str:

        body = m.group(1)

        # 等宽字体没有汉字：代码里一旦出现中文，退回中文字体，否则会渲染成豆腐块

        face = "Deng" if _CJK.search(body) else "Mono"

        size = BODY_SIZE - 0.6 if face == "Deng" else MONO_SIZE

        return f'<font face="{face}" size="{size}" backColor="{CODE_BG}">{body}</font>'

    text = re.sub(r"`([^`]+)`", code_span, text)

    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)

    text = re.sub(

        r"\[([^\]]+)\]\(([^)]+)\)",

        r'<link href="\2"><font color="#1D4ED8">\1</font></link>',

        text,

    )

    return text

# 表格行切分：`|` 出现在行内代码或转义里时不能当分隔符

def split_row(line: str) -> list[str]:

    line = line.strip()

    if line.startswith("|"):

        line = line[1:]

    if line.endswith("|") and not line.endswith("\\|"):

        line = line[:-1]

    cells: list[str] = []

    buf = ""

    in_code = False

    i = 0

    while i < len(line):

        ch = line[i]

        if ch == "\\" and i + 1 < len(line) and line[i + 1] == "|":

            buf += "|"

            i += 2

            continue

        if ch == "`":

            in_code = not in_code

            buf += ch

            i += 1

            continue

        if ch == "|" and not in_code:

            cells.append(buf.strip())

            buf = ""

            i += 1

            continue

        buf += ch

        i += 1

    cells.append(buf.strip())

    return cells

_SEP = re.compile(r"^\|?[\s:|-]+\|[\s:|-]*$")

def _join_cont(lines: list[str], i: int, first: str) -> tuple[str, int]:


    parts = [first.strip()]

    j = i + 1

    while j < len(lines):

        cur = lines[j]

        if not cur.strip():

            break

        if not cur.startswith(("  ", "\t")):

            break

        if re.match(r"^\s*(?:[-*+]|\d+[.)])\s+", cur):

            break

        parts.append(cur.strip())

        j += 1

    return " ".join(parts), j

def _pop_heading(story: list) -> list:


    if not story:

        return []

    head: list = []

    if isinstance(story[-1], HRFlowable) and len(story) >= 2:

        cand = story[-2]

        if isinstance(cand, Paragraph) and cand.style.name in ("h1", "h2", "h3"):

            head = [cand, story[-1]]

            del story[-2:]

            return head

    if isinstance(story[-1], Paragraph) and story[-1].style.name in ("h2", "h3"):

        head = [story.pop()]

    return head

def _col_widths(rows: list[list[str]], avail: float) -> list[float]:


    ncol = max(len(r) for r in rows)

    pad = 13.0

    floors: list[float] = []

    weights: list[float] = []

    for c in range(ncol):

        cells = [

            re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", re.sub(r"[`*]", "", r[c]))

            for r in rows

            if c < len(r)

        ] or [""]

        measured = max(pdfmetrics.stringWidth(t, "Deng", 8.4) for t in cells)

        longest_tok = max(

            (

                pdfmetrics.stringWidth(tok, "Deng", 8.4)

                for t in cells

                for tok in re.findall(r"[A-Za-z0-9_./:%+\-]{3,}", t)

            ),

            default=0.0,

        )

        floors.append(max(longest_tok + pad, 34.0))

        weights.append(max(min(measured, avail * 0.32), 12.0) ** 0.75)

    total = sum(weights) or 1.0

    widths = [avail * w / total for w in weights]

    deficit = sum(max(0.0, f - w) for f, w in zip(floors, widths))

    if deficit > 0.5:

        surplus = [max(0.0, w - f) for f, w in zip(floors, widths)]

        s_total = sum(surplus)

        if s_total > 0:

            widths = [

                f if w < f else w - deficit * s / s_total

                for f, w, s in zip(floors, widths, surplus)

            ]

    return widths

class Renderer:

    def __init__(self, base_dir: Path | None = None) -> None:

        self.base_dir = base_dir or Path(".")

        self.s_h1 = ParagraphStyle(

            "h1", fontName="Deng-Bold", fontSize=19, leading=26, textColor=INK,

            spaceBefore=0, spaceAfter=6, keepWithNext=1,

        )

        self.s_h2 = ParagraphStyle(

            "h2", fontName="Deng-Bold", fontSize=13.5, leading=19, textColor=ACCENT,

            spaceBefore=15, spaceAfter=5, keepWithNext=1,

        )

        self.s_h3 = ParagraphStyle(

            "h3", fontName="Deng-Bold", fontSize=11, leading=16, textColor=INK,

            spaceBefore=10, spaceAfter=3, keepWithNext=1,

        )

        self.s_body = ParagraphStyle(

            "body", fontName="Deng", fontSize=BODY_SIZE, leading=BODY_LEAD,

            textColor=INK, alignment=TA_LEFT, spaceBefore=2, spaceAfter=4,

        )

        self.s_bullet = ParagraphStyle(

            "bullet", parent=self.s_body, leftIndent=13, bulletIndent=3,

            spaceBefore=1, spaceAfter=1,

        )

        self.s_cell = ParagraphStyle(

            "cell", fontName="Deng", fontSize=8.5, leading=12, textColor=INK,

        )

        self.s_cell_head = ParagraphStyle(

            "cellhead", parent=self.s_cell, fontName="Deng-Bold", textColor=INK,

        )

        self.s_code = ParagraphStyle(

            "code", fontName="Mono", fontSize=8.2, leading=11.6,

            textColor=colors.HexColor("#22303F"),

        )

        self.s_caption = ParagraphStyle(

            "caption", parent=self.s_body, fontName="Deng", fontSize=8.3,

            leading=12, textColor=colors.HexColor("#5A6675"),

            alignment=TA_CENTER, spaceBefore=4, spaceAfter=8,

        )

        self.s_code_cjk = ParagraphStyle(

            "codecjk", parent=self.s_code, fontName="MonoCJK", fontSize=7.6,

            leading=12.4,

        )

    def blocks(self, text: str) -> list[tuple[str, str]]:

        out: list[tuple[str, str]] = []

        lines = text.splitlines()

        i = 0

        while i < len(lines):

            line = lines[i]

            stripped = line.strip()

            if not stripped:

                i += 1

                continue

            if stripped.startswith("```"):

                j = i + 1

                buf: list[str] = []

                while j < len(lines) and not lines[j].strip().startswith("```"):

                    buf.append(lines[j])

                    j += 1

                out.append(("code", "\n".join(buf)))

                i = j + 1

                continue

            if stripped.startswith(">"):

                j = i

                buf = []

                while j < len(lines) and lines[j].strip().startswith(">"):

                    raw = lines[j].strip()

                    raw = raw[1:]

                    if raw.startswith(" "):

                        raw = raw[1:]

                    buf.append(raw)

                    j += 1

                out.append(("quote", "\n".join(buf)))

                i = j

                continue

            if stripped.startswith("|") and i + 1 < len(lines) and _SEP.match(lines[i + 1].strip()):

                j = i

                rows: list[str] = []

                while j < len(lines) and lines[j].strip().startswith("|"):

                    rows.append(lines[j])

                    j += 1

                out.append(("table", "\n".join(rows)))

                i = j

                continue

            m = re.match(r"^(#{1,6})\s+(.*)$", stripped)

            if m:

                out.append((f"h{len(m.group(1))}", m.group(2).strip()))

                i += 1

                continue

            if re.match(r"^(-{3,}|\*{3,})$", stripped):

                out.append(("hr", ""))

                i += 1

                continue

            m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped)

            if m:

                out.append(("img", f"{m.group(1)}\x00{m.group(2)}"))

                i += 1

                continue

            m = re.match(r"^\s*[-*+]\s+(.*)$", line)

            if m:

                body, j = _join_cont(lines, i, m.group(1))

                out.append(("li", body))

                i = j

                continue

            m = re.match(r"^(\s*)(\d+)[.)]\s+(.*)$", line)

            if m:

                body, j = _join_cont(lines, i, m.group(3))

                out.append(("oli", f"{m.group(2)}\x00{body}"))

                i = j

                continue

            j = i

            buf = []

            while j < len(lines):

                cur = lines[j]

                s = cur.strip()

                if not s:

                    break

                if (

                    s.startswith(("```", ">", "|", "#"))

                    or re.match(r"^\s*[-*+]\s+", cur)

                    or re.match(r"^\s*\d+[.)]\s+", cur)

                ):

                    break

                buf.append(s)

                j += 1

            if buf:

                out.append(("p", " ".join(buf)))

                i = j

            else:

                i += 1

        return out

    def flow(self, text: str, avail: float) -> list:

        story: list = []

        ordered: dict[int, int] = {}

        for kind, payload in self.blocks(text):

            if kind == "h1":

                story.append(Paragraph(inline(payload), self.s_h1))

                story.append(

                    HRFlowable(width="100%", thickness=1.6, color=ACCENT,

                               spaceBefore=1, spaceAfter=8)

                )

                ordered.clear()

            elif kind == "h2":

                story.append(Paragraph(inline(payload), self.s_h2))

                story.append(

                    HRFlowable(width="100%", thickness=0.6, color=RULE,

                               spaceBefore=0, spaceAfter=5)

                )

                ordered.clear()

            elif kind == "h3":

                story.append(Paragraph(inline(payload), self.s_h3))

                ordered.clear()

            elif kind in ("h4", "h5", "h6"):

                story.append(Paragraph(inline(payload), self.s_h3))

            elif kind == "p":

                story.append(Paragraph(inline(payload), self.s_body))

                ordered.clear()

            elif kind == "li":

                story.append(Paragraph(inline(payload), self.s_bullet, bulletText="•"))

            elif kind == "oli":

                num, _, body = payload.partition("\x00")

                story.append(

                    Paragraph(inline(body), self.s_bullet, bulletText=f"{num}.")

                )

            elif kind == "hr":

                story.append(Spacer(1, 4))

                story.append(HRFlowable(width="100%", thickness=0.5, color=RULE))

                story.append(Spacer(1, 4))

            elif kind == "code":

                block = self._code_block(payload, avail)

                head = _pop_heading(story)

                story.append(KeepTogether(head + [block]) if head else block)

            elif kind == "img":

                figure = self._figure(payload, avail)

                head = _pop_heading(story)

                story.append(KeepTogether(head + [figure]) if head else figure)

            elif kind == "table":

                story.append(self._table(payload, avail))

            elif kind == "quote":

                story.append(self._quote(payload, avail))

        return story

    def _figure(self, payload: str, avail: float):

        caption, _, ref = payload.partition("\x00")

        path = Path(ref.strip())

        if not path.is_absolute():

            path = (self.base_dir / ref.strip()).resolve()

        if not path.is_file():

            return Paragraph(f"［缺图：{inline(ref)}］", self.s_body)

        from reportlab.lib.utils import ImageReader

        iw, ih = ImageReader(str(path)).getSize()

        width = avail

        height = width * ih / iw

        if height > 128 * mm:

            height = 128 * mm

            width = height * iw / ih

        pic = Image(str(path), width=width, height=height)

        pic.hAlign = "CENTER"

        flows = [Spacer(1, 3), pic]

        if caption:

            flows.append(Paragraph(inline(caption), self.s_caption))

        return KeepTogether(flows)

    def _code_block(self, body: str, avail: float):

        if _CJK.search(body) or _BOX.search(body):

            style = self.s_code_cjk

        else:

            style = self.s_code

        pre = Preformatted(body, style)

        t = Table([[pre]], colWidths=[avail])

        t.setStyle(TableStyle([

            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(CODE_BG)),

            ("BOX", (0, 0), (-1, -1), 0.4, RULE),

            ("LEFTPADDING", (0, 0), (-1, -1), 8),

            ("RIGHTPADDING", (0, 0), (-1, -1), 8),

            ("TOPPADDING", (0, 0), (-1, -1), 5),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),

        ]))

        return t

    def _table(self, payload: str, avail: float):

        lines = [ln for ln in payload.splitlines() if ln.strip()]

        header = split_row(lines[0])

        body = [split_row(ln) for ln in lines[2:]]

        ncol = len(header)

        norm = [header] + [(r + [""] * ncol)[:ncol] for r in body]

        widths = _col_widths(norm, avail)

        data = []

        for r_i, row in enumerate(norm):

            style = self.s_cell_head if r_i == 0 else self.s_cell

            data.append([Paragraph(inline(c), style) for c in row])

        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")

        t.setStyle(TableStyle([

            ("BACKGROUND", (0, 0), (-1, 0), ACCENT_LIGHT),

            ("TEXTCOLOR", (0, 0), (-1, -1), INK),

            ("GRID", (0, 0), (-1, -1), 0.4, RULE),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("LEFTPADDING", (0, 0), (-1, -1), 5),

            ("RIGHTPADDING", (0, 0), (-1, -1), 5),

            ("TOPPADDING", (0, 0), (-1, -1), 3.5),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),

            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#FAFBFD")]),

        ]))

        return t

    def _quote(self, payload: str, avail: float):

        inner = self.flow(payload, avail - 20)

        if not inner:

            return Spacer(1, 0)

        t = Table([[inner]], colWidths=[avail], hAlign="LEFT")

        t.setStyle(TableStyle([

            ("BACKGROUND", (0, 0), (-1, -1), QUOTE_BG),

            ("LINEBEFORE", (0, 0), (0, -1), 2.6, ACCENT),

            ("VALIGN", (0, 0), (-1, -1), "TOP"),

            ("LEFTPADDING", (0, 0), (-1, -1), 9),

            ("RIGHTPADDING", (0, 0), (-1, -1), 8),

            ("TOPPADDING", (0, 0), (-1, -1), 6),

            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),

        ]))

        return t

class NumberedCanvas(canvas.Canvas):

    def __init__(self, *args, footer: str = "", **kwargs):

        self._footer = footer

        self._states: list[dict] = []

        super().__init__(*args, **kwargs)

    def showPage(self):

        self._states.append(dict(self.__dict__))

        self._startPage()

    def save(self):

        total = len(self._states)

        for state in self._states:

            self.__dict__.update(state)

            self._delayed_footer(total)

            canvas.Canvas.showPage(self)

        canvas.Canvas.save(self)

    def _delayed_footer(self, total: int):

        self.saveState()

        width, _ = A4

        self.setStrokeColor(RULE)

        self.setLineWidth(0.5)

        self.line(16 * mm, 13 * mm, width - 16 * mm, 13 * mm)

        self.setFont("Deng", 8)

        self.setFillColor(INK_SOFT)

        self.drawString(16 * mm, 8.4 * mm, self._footer)

        self.drawRightString(width - 16 * mm, 8.4 * mm, f"第 {self._pageNumber} 页 / 共 {total} 页")

        self.restoreState()

def build(src: Path, out: Path, footer: str) -> tuple[int, int]:

    text = src.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s+(.*)$", text, re.M)

    title = title_match.group(1).strip() if title_match else src.stem

    if not footer:

        footer = title

    register_fonts()

    renderer = Renderer(base_dir=src.parent)

    left = right = 16 * mm

    top = 16 * mm

    bottom = 17 * mm

    avail = A4[0] - left - right

    doc = BaseDocTemplate(

        str(out), pagesize=A4,

        leftMargin=left, rightMargin=right, topMargin=top, bottomMargin=bottom,

        title=title, author="pptbg", subject=title,

    )

    frame = Frame(left, bottom, avail, A4[1] - top - bottom, id="main",

                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    doc.addPageTemplates([PageTemplate(id="page", frames=[frame])])

    story = renderer.flow(text, avail)

    doc.build(story, canvasmaker=lambda *a, **k: NumberedCanvas(*a, footer=footer, **k))

    pages = doc.page

    return pages, out.stat().st_size

def main(argv: list[str] | None = None) -> int:

    ap = argparse.ArgumentParser(description="Markdown → PDF（reportlab，CJK 安全）")

    ap.add_argument("--src", default="docs/技术报告.md")

    ap.add_argument("--out", default="docs/技术报告.pdf")

    ap.add_argument("--footer", default="", help="页脚左侧文字，默认取一级标题")

    args = ap.parse_args(argv)

    src = Path(args.src)

    out = Path(args.out)

    if not src.is_file():

        print(f"找不到源文件：{src}", file=sys.stderr)

        return 2

    out.parent.mkdir(parents=True, exist_ok=True)

    pages, size = build(src, out, args.footer)

    print(f"OK  {out}  {pages} 页  {size/1024/1024:.2f} MB")

    if size > 10 * 1024 * 1024:

        print("!! 超过 10 MB 上限", file=sys.stderr)

        return 1

    return 0

if __name__ == "__main__":

    raise SystemExit(main())

