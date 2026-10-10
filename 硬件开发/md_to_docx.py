# -*- coding: utf-8 -*-
"""
Markdown → DOCX 转换器（项目专用）

用途：把《项目开发计划书.md》转成可交付的 .docx。

为什么自己写而不用 pandoc：
  1. 本项目文档含大量 **中文 + emoji 标记（✅⚠️❌★）+ ASCII 架构图**，
     通用转换器对这三者的处理都不理想；
  2. 需要控制中文字体（正文宋体/等线、标题黑体）与表格样式；
  3. 避免引入额外依赖（pandoc 未安装）。

支持：
  标题 (#~####)、表格（含对齐行）、代码块（```，等宽字体、不换行）、
  引用块（>）、无序/有序列表、水平线（---）、行内加粗 **x**、行内代码 `x`。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

# ----------------------------------------------------------------- 样式常量
FONT_ZH_BODY = "等线"
FONT_ZH_HEAD = "黑体"
FONT_MONO = "Consolas"

#: 代码块里需要保持等宽的 ASCII 图同样走 MONO
MONO_SIZE = Pt(7.5)


def set_run_font(run, name: str, size=None, bold=None, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = size
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def set_cell_bg(cell, hex_color: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


# ----------------------------------------------------------------- 行内解析
#: 先匹配 **加粗** / `代码` / [链接](url)；** 优先于 * 与 _
INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\[[^\]]+\]\([^)]*\))")


def add_inline(par, text: str, base_font=FONT_ZH_BODY, size=Pt(10.5), bold=False):
    """把含 **加粗** / `代码` / [链接](url) 的文本写入段落。"""
    text = text.replace("\\|", "|")
    for piece in INLINE_RE.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**") and len(piece) > 4:
            r = par.add_run(piece[2:-2])
            set_run_font(r, base_font, size, bold=True)
        elif piece.startswith("`") and piece.endswith("`") and len(piece) > 2:
            r = par.add_run(piece[1:-1])
            set_run_font(r, FONT_MONO, Pt(size.pt - 1.0))
            r.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
        elif piece.startswith("[") and "](" in piece:
            # Markdown 链接 → 只显示文字（Word 里给文字加超链接需要额外 XML，
            # 对本项目文档（内部目录 + 少量外链）意义不大，保持可读即可）
            label = piece[1:piece.index("](")]
            r = par.add_run(label)
            set_run_font(r, base_font, size, bold=bold)
        else:
            r = par.add_run(piece)
            set_run_font(r, base_font, size, bold=bold)


def add_left_border(par, color="5B9BD5", size="18"):
    p_pr = par._p.get_or_add_pPr()
    pbdr = OxmlElement("w:pBdr")
    left = OxmlElement("w:left")
    left.set(qn("w:val"), "single")
    left.set(qn("w:sz"), size)
    left.set(qn("w:color"), color)
    left.set(qn("w:space"), "8")
    pbdr.append(left)
    p_pr.append(pbdr)


# ----------------------------------------------------------------- 表格
def split_row(line: str):
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    # 保护转义竖线
    parts, buf, i = [], "", 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] == "|":
            buf += "|"
            i += 2
            continue
        if line[i] == "|":
            parts.append(buf.strip())
            buf = ""
        else:
            buf += line[i]
        i += 1
    parts.append(buf.strip())
    return parts


def is_sep_row(line: str) -> bool:
    s = line.strip()
    if not s.startswith("|"):
        return False
    return bool(re.fullmatch(r"\|[\s:\-\|]+\|?", s)) and "-" in s


def add_table(doc, rows):
    header, body = rows[0], rows[1:]
    ncol = len(header)
    table = doc.add_table(rows=1, cols=ncol)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for j, cell_text in enumerate(header):
        cell = table.rows[0].cells[j]
        cell.text = ""
        par = cell.paragraphs[0]
        par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(par, cell_text, FONT_ZH_HEAD, Pt(9.5), bold=True)
        set_cell_bg(cell, "EAF1F8")

    for row in body:
        cells = table.add_row().cells
        for j in range(ncol):
            val = row[j] if j < len(row) else ""
            cells[j].text = ""
            par = cells[j].paragraphs[0]
            add_inline(par, val, FONT_ZH_BODY, Pt(9.5))

    doc.add_paragraph()
    return table


# ----------------------------------------------------------------- 主转换
def render(lines, doc, in_quote_deep=False):
    """渲染一组 markdown 行。

    Args:
        in_quote_deep: 若为 True，本层所有段落都加引用左边框。
            引用块内部可能含标题与表格（本项目 v2 修订说明即是），
            因此**不能**把引用块整体当成一个段落 ——
            必须剥掉 ``>`` 标记后递归渲染，否则会出现"整块挤在一段里、
            换行变成字面 \\n"的问题。
    """
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i].rstrip()
        stripped = line.strip()

        # ---------------- 表格 ----------------
        if stripped.startswith("|") and i + 1 < n and is_sep_row(lines[i + 1]):
            rows = [split_row(stripped)]
            i += 2
            while i < n and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            add_table(doc, rows)
            continue

        # ---------------- 标题 ----------------
        m = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            sizes = {1: Pt(18), 2: Pt(15), 3: Pt(12.5), 4: Pt(11)}
            par = doc.add_paragraph()
            par.paragraph_format.space_before = Pt(14 if level <= 2 else 9)
            par.paragraph_format.space_after = Pt(6 if level <= 2 else 4)
            if level == 1 and not in_quote_deep:
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = par.add_run(text)
            set_run_font(r, FONT_ZH_HEAD, sizes[level], bold=True)
            if level <= 2:
                r.font.color.rgb = RGBColor(0x1F, 0x3B, 0x63)
            i += 1
            continue

        # ---------------- 引用块（递归） ----------------
        if stripped.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip()[1:].strip())
                i += 1
            render(buf, doc, in_quote_deep=True)
            continue

        # ---------------- 水平线 ----------------
        if re.fullmatch(r"-{3,}", stripped):
            par = doc.add_paragraph()
            p_pr = par._p.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:color"), "BBBBBB")
            pbdr.append(bottom)
            p_pr.append(pbdr)
            i += 1
            continue

        # ---------------- 列表 ----------------
        m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if m:
            indent, marker, text = m.group(1), m.group(2), m.group(3)
            par = doc.add_paragraph()
            depth = min(len(indent) // 2, 3)
            par.paragraph_format.left_indent = Cm(0.6 + 0.6 * depth)
            par.paragraph_format.space_after = Pt(2)
            bullet = "•" if marker in ("-", "*") else marker
            add_inline(par, f"{bullet} {text}", FONT_ZH_BODY, Pt(10.5))
            if in_quote_deep:
                add_left_border(par, "AAB7C4", "8")
            i += 1
            continue

        # ---------------- 空行 ----------------
        if not stripped:
            i += 1
            continue

        # ---------------- 普通段落 ----------------
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(5)
        par.paragraph_format.line_spacing = 1.25
        add_inline(par, stripped, FONT_ZH_BODY, Pt(10.5))
        if in_quote_deep:
            add_left_border(par)
        i += 1


def convert(md_path: Path, docx_path: Path):
    lines = md_path.read_text(encoding="utf-8").splitlines()
    doc = Document()

    # 全局默认字体
    normal = doc.styles["Normal"]
    normal.font.name = FONT_ZH_BODY
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), FONT_ZH_BODY)

    # 页边距（文档含宽表格，适当收窄）
    for section in doc.sections:
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)
        section.top_margin = Cm(2.2)
        section.bottom_margin = Cm(2.2)

    # 代码块需要在此层单独处理（跨行），因此先在外层切成段
    idx = 0
    n = len(lines)
    chunk = []
    while idx < n:
        stripped = lines[idx].strip()
        if stripped.startswith("```"):
            if chunk:
                render(chunk, doc)
                chunk = []
            buf = []
            idx += 1
            while idx < n and not lines[idx].strip().startswith("```"):
                buf.append(lines[idx])
                idx += 1
            idx += 1
            if buf:
                par = doc.add_paragraph()
                par.paragraph_format.space_before = Pt(3)
                par.paragraph_format.space_after = Pt(8)
                par.paragraph_format.left_indent = Cm(0.3)
                for k, cl in enumerate(buf):
                    r = par.add_run(cl)
                    set_run_font(r, FONT_MONO, MONO_SIZE)
                    if k != len(buf) - 1:
                        r.add_break()
                p_pr = par._p.get_or_add_pPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:fill"), "F5F5F5")
                p_pr.append(shd)
            continue
        chunk.append(lines[idx])
        idx += 1
    if chunk:
        render(chunk, doc)

    doc.save(str(docx_path))
    return docx_path


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python md_to_docx.py <input.md> <output.docx>")
        raise SystemExit(1)
    out = convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"[OK] {out}  ({out.stat().st_size} bytes)")
