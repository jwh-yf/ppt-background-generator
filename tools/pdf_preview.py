"""把 PDF 每页渲染成 PNG，用来肉眼检查排版（字体、表格断行、页脚）。

本机没有 Office 导出、没有 Ghostscript、没有 poppler，也没有可用的 headless。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PDF 逐页导出 PNG")
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--out", default="artifacts/pdf/preview")
    ap.add_argument("--dpi", type=int, default=110)
    ap.add_argument("--pages", default="", help="如 1,2,5；留空=全部")
    args = ap.parse_args(argv)

    try:
        import pypdfium2 as pdfium
    except ImportError:
        print(
            "缺 pypdfium2。先解包 wheel 到 tools/pyrender，再把 PYTHONPATH 指过去。",
            file=sys.stderr,
        )
        return 2

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"找不到 PDF：{pdf_path}", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    doc = pdfium.PdfDocument(str(pdf_path))
    total = len(doc)
    wanted = (
        [int(x) for x in args.pages.split(",") if x.strip()]
        if args.pages
        else list(range(1, total + 1))
    )
    scale = args.dpi / 72.0

    for page_no in wanted:
        if not 1 <= page_no <= total:
            print(f"跳过越界页 {page_no}", file=sys.stderr)
            continue
        page = doc[page_no - 1]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil()
        dst = out_dir / f"page{page_no:02d}.png"
        image.save(dst)
        print(f"  {dst}  {image.size[0]}x{image.size[1]}")

    print(f"OK  {len(wanted)}/{total} 页 -> {out_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
