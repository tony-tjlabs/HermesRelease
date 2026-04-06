"""
Hermes weekly report PDF generator v2.0.
Dynamic Layout Engine: blocks measure height, LayoutEngine packs pages, gaps distributed.
Dependencies: fpdf2, matplotlib, numpy. No Streamlit imports.
"""
from __future__ import annotations

import io
import math
import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF

# ── Brand colours (RGB 0-255) ────────────────────────────────────────────────
NAVY  = (18, 40, 76)
GOLD  = (212, 160, 23)
SLATE = (100, 116, 139)
WHITE = (255, 255, 255)
LBG   = (248, 249, 252)
BDR   = (215, 222, 233)
GREEN = (34, 197, 94)
RED   = (239, 68, 68)
AMBER = (245, 158, 11)
AI_BG = (252, 249, 240)

def _m(t: tuple) -> tuple:
    """RGB 0-255 → matplotlib 0-1."""
    return (t[0] / 255.0, t[1] / 255.0, t[2] / 255.0)

# ── Unicode → Latin-1 cleaner (FPDF Helvetica) ──────────────────────────────
_CM = {
    "\u2014": "--", "\u2013": "-",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2022": "-", "\u2192": "->",
    "\u2026": "...",
}

def cl(s: str, max_chars: Optional[int] = None) -> str:
    """Clean string for PDF. If NanumGothic available, keep Korean; otherwise Latin-1 only."""
    if not s:
        return ""
    s = str(s)
    for k, v in _CM.items():
        s = s.replace(k, v)
    if not _HAS_NANUM:
        # Fallback: Latin-1 safe only
        s = "".join(c if ord(c) < 256 else "?" for c in s)
    if max_chars is not None and len(s) > max_chars:
        s = s[:max_chars].rsplit(" ", 1)[0] + " ..." if " " in s[:max_chars] else s[:max_chars] + " ..."
    return s

# ── Temp PNG helper ─────────────────────────────────────────────────────────
def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def _tmp_path(data: bytes) -> str:
    fd, p = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    with open(p, "wb") as f:
        f.write(data)
    return p

# ── Page geometry (A4, mm) ───────────────────────────────────────────────────
MARG     = 14
AUTO_BRK = 14
PBT      = 297 - MARG
CHART_W  = 182
P1_START = 52
P2_START = 12
P1_AVAIL = PBT - P1_START
P2_AVAIL = PBT - P2_START

CHART_FIG_W  = 9.0
CHART_FIG_H  = 1.8
CHART_PDF_H  = 37
AI_BOX_H     = 18
SECTION_TITLE_H = 8.5

DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _fmt_d(date_val) -> str:
    """Format date as 'Feb 15' (no leading zero on day for compatibility)."""
    if hasattr(date_val, "strftime"):
        return date_val.strftime("%b ") + str(date_val.day)
    return str(date_val)[:10]


# ── Block system ─────────────────────────────────────────────────────────────
class Block:
    def __init__(self, height: float, soft: bool = False, keep_with_next: bool = False):
        self.height = height
        self.soft = soft
        self.keep_with_next = keep_with_next

    def draw(self, pdf: "HermesPDF"):
        raise NotImplementedError


class SpacerBlock(Block):
    def __init__(self, h: float):
        super().__init__(h, soft=True)
        self._h = h

    def draw(self, pdf: "HermesPDF"):
        pdf.ln(self._h)


class SectionTitleBlock(Block):
    def __init__(self, n: int, title: str):
        super().__init__(SECTION_TITLE_H, keep_with_next=True)
        self.n = n
        self.title = title

    def draw(self, pdf: "HermesPDF"):
        pdf.fc(NAVY)
        pdf.tc(WHITE)
        pdf.set_font(pdf._font_name, "B", 10.5)
        pdf.cell(0, 7, f"  {self.n}. {cl(self.title)}", fill=True, ln=True)
        pdf.tc((0, 0, 0))
        pdf.ln(1.5)


class SubHeadBlock(Block):
    def __init__(self, text: str):
        super().__init__(6, keep_with_next=True)
        self.text = text

    def draw(self, pdf: "HermesPDF"):
        pdf.tc(NAVY)
        pdf.set_font(pdf._font_name, "B", 8.5)
        pdf.cell(0, 5, cl(self.text), ln=True)
        pdf.tc((0, 0, 0))
        pdf.ln(0.5)


class TextBlock(Block):
    LH = 4.8

    def __init__(self, text: str, indent: float = 0, lh: float = None):
        self.text = text
        self.indent = indent
        self.lh = lh or self.LH
        effective_width = 182 - indent
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in (text or ""))
        factor = 0.28 if has_korean else 0.49
        chars_per_line = max(1, int(effective_width * factor))
        lines = 0
        for para in (text or "").split("\n"):
            lines += max(1, math.ceil(len(para) / chars_per_line))
        super().__init__(lines * self.lh + 0.5)

    def draw(self, pdf: "HermesPDF"):
        pdf.set_x(pdf.l_margin + self.indent)
        pdf.set_font(pdf._font_name, "", 8.5)
        pdf.tc(SLATE)
        pdf.multi_cell(0, self.lh, cl(self.text))
        pdf.tc((0, 0, 0))


class BulletBlock(Block):
    LH = 4.8

    def __init__(self, text: str, num: Optional[int] = None):
        self.text = text
        self.num = num
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in (text or ""))
        factor = 0.28 if has_korean else 0.49
        chars_per_line = int(176 * factor)
        lines = max(1, math.ceil(len(text) / chars_per_line))
        super().__init__(lines * self.LH + 0.5)

    def draw(self, pdf: "HermesPDF"):
        pfx = f"{self.num}." if self.num is not None else "-"
        pdf.set_x(pdf.l_margin + 3)
        pdf.set_font(pdf._font_name, "", 8.5)
        pdf.tc(SLATE)
        pdf.multi_cell(0, self.LH, f"{pfx}  {cl(self.text)}")
        pdf.tc((0, 0, 0))


class SmallTextBlock(Block):
    def __init__(self, text: str, italic: bool = True, color: tuple = NAVY):
        super().__init__(5)
        self.text = text
        self.italic = italic
        self.color = color

    def draw(self, pdf: "HermesPDF"):
        style = "I" if self.italic else ""
        pdf.set_font(pdf._font_name, style, 7.5)
        pdf.tc(self.color)
        pdf.cell(0, 4.5, cl(self.text), ln=True)
        pdf.tc((0, 0, 0))


def _fmt_dw(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and (seconds != seconds)):
        return "N/A"
    s = int(seconds)
    return f"{s // 60}m {s % 60}s"


def _fmt_kpi_value(metric: str, value: Any, delta: Any, unit: str) -> Tuple[str, str]:
    """Format KPI value and delta for display. Returns (value_str, delta_str)."""
    if metric == "fp" or metric == "vis":
        try:
            v_str = f"{int(round(float(value or 0))):,}"
        except (TypeError, ValueError):
            v_str = "0"
        d_str = f"+{float(delta):.1f}%" if delta is not None else "--"
    elif metric == "cvr":
        v_str = f"{float(value):.1f}%" if value is not None else "0%"
        d_str = f"{float(delta):+.1f}pp" if delta is not None else "--"
    elif metric == "dw":
        v_str = _fmt_dw(value)
        d_str = f"{float(delta):+.0f}s" if delta is not None else "--"
    else:
        v_str = str(value) if value is not None else ""
        d_str = str(delta) if delta is not None else ""
    return (v_str, d_str)


class KPICardsBlock(Block):
    CARD_H = 33

    def __init__(self, kpi: dict, summary: str):
        self.kpi = kpi
        self.summary = summary
        super().__init__(self.CARD_H + 2 + 5.5 + 4.8 * 3.5)

    def draw(self, pdf: "HermesPDF"):
        CW4, CH, GAP = 43.5, self.CARD_H, 2
        sx, y = pdf.l_margin, pdf.get_y()

        v_fp, d_fp = _fmt_kpi_value("fp", self.kpi["fp"]["v"], self.kpi["fp"]["d"], "%")
        v_vis, d_vis = _fmt_kpi_value("vis", self.kpi["vis"]["v"], self.kpi["vis"]["d"], "%")
        v_cvr, d_cvr = _fmt_kpi_value("cvr", self.kpi["cvr"]["v"], self.kpi["cvr"]["d"], "pp")
        v_dw, d_dw = _fmt_kpi_value("dw", self.kpi["dw"]["v"], self.kpi["dw"]["d"], "s")
        cards = [
            (self.kpi["fp"], v_fp, d_fp, GREEN),
            (self.kpi["vis"], v_vis, d_vis, GREEN),
            (self.kpi["cvr"], v_cvr, d_cvr, RED),
            (self.kpi["dw"], v_dw, d_dw, RED),
        ]
        for i, (meta, val, delta, dcol) in enumerate(cards):
            x = sx + i * (CW4 + GAP)
            pdf.fc(LBG)
            pdf.rect(x, y, CW4, CH, "F")
            pdf.fc(NAVY)
            pdf.rect(x, y, CW4, 1.5, "F")
            pdf.set_xy(x + 2, y + 2.5)
            pdf.set_font(pdf._font_name, "B", 6.5)
            pdf.tc(SLATE)
            pdf.cell(CW4 - 4, 4, cl(meta["label"]), ln=True)
            pdf.set_xy(x + 2, y + 8)
            pdf.set_font(pdf._font_name, "B", 14)
            pdf.tc(NAVY)
            pdf.cell(CW4 - 4, 7, cl(val), ln=True)
            pdf.set_xy(x + 2, y + 16.5)
            pdf.set_font(pdf._font_name, "B", 7.5)
            pdf.tc(dcol)
            pdf.cell(CW4 - 4, 4.5, cl(delta + " vs prev week"), ln=True)
            pdf.set_xy(x + 2, y + 22.5)
            pdf.set_font(pdf._font_name, "I", 6.5)
            pdf.tc(SLATE)
            pdf.multi_cell(CW4 - 4, 3.5, cl(meta.get("sub", "")))
        pdf.tc((0, 0, 0))
        pdf.set_y(y + CH + 2)
        pdf.fc(LBG)
        pdf.set_x(pdf.l_margin)
        pdf.set_font(pdf._font_name, "B", 8)
        pdf.tc(NAVY)
        pdf.cell(0, 5.5, "  Weekly Performance Summary:", fill=True, ln=True)
        pdf.set_font(pdf._font_name, "", 8)
        pdf.tc(SLATE)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 4.8, "  " + cl(self.summary))
        pdf.tc((0, 0, 0))


class ChartBlock(Block):
    def __init__(self, img_bytes: bytes, gap_after: float = 1.5):
        super().__init__(CHART_PDF_H + gap_after)
        self.img = img_bytes
        self.gap = gap_after
        self._path: Optional[str] = None

    def draw(self, pdf: "HermesPDF"):
        if not self.img:
            pdf.ln(self.gap)
            return
        if self._path is None:
            self._path = _tmp_path(self.img)
            pdf._tmp.append(self._path)
        pdf.image(self._path, x=pdf.l_margin, w=CHART_W)
        pdf.ln(self.gap)


class AIBoxBlock(Block):
    """AI Analysis box with dynamic height from content. No duplicate label+content."""
    PADDING = 4
    LINE_HEIGHT_MM = 4.4
    LABEL_H = 6.5
    CHARS_PER_LINE_EN = 95   # English: ~95 chars/line at 8pt
    CHARS_PER_LINE_KR = 50   # Korean: wider glyphs, ~50 chars/line
    MIN_H = 20
    MAX_H = 120  # Increased for Korean (was 55)

    def __init__(self, content: str, max_chars: int = 2000):
        self.content = (content or "").strip()
        if not self.content:
            self.content = "(AI analysis unavailable)"
        self.content = cl(self.content, max_chars=max_chars)
        # Detect Korean content → use wider char estimate
        has_korean = any('\uac00' <= c <= '\ud7a3' for c in self.content)
        cpl = self.CHARS_PER_LINE_KR if has_korean else self.CHARS_PER_LINE_EN
        lines = 0
        for para in self.content.split("\n"):
            if not para.strip():
                lines += 1
                continue
            lines += max(1, math.ceil(len(para) / cpl))
        content_h = lines * self.LINE_HEIGHT_MM
        box_h = max(self.MIN_H, min(self.LABEL_H + self.PADDING * 2 + content_h, self.MAX_H))
        super().__init__(box_h + 2)
        self._box_h = box_h

    def draw(self, pdf: "HermesPDF"):
        y = pdf.get_y()
        w = pdf.epw
        h = self._box_h
        pdf.fc(AI_BG)
        pdf.dc(GOLD)
        pdf.set_line_width(0.25)
        pdf.rect(pdf.l_margin, y, w, h, "FD")
        pdf.fc(GOLD)
        pdf.rect(pdf.l_margin, y, 2.5, h, "F")
        pdf.fc(NAVY)
        pdf.tc(WHITE)
        pdf.set_font(pdf._font_name, "B", 6)
        pdf.set_xy(pdf.l_margin + 4, y + 1.8)
        pdf.cell(14, 3.5, "  AI Analysis", fill=True)
        pdf.set_xy(pdf.l_margin + self.PADDING, y + self.LABEL_H)
        pdf.set_font(pdf._font_name, "", 8)
        pdf.tc(SLATE)
        pdf.multi_cell(w - self.PADDING * 2, self.LINE_HEIGHT_MM, self.content)
        pdf.tc((0, 0, 0))
        pdf.dc((0, 0, 0))
        pdf.set_line_width(0.2)
        pdf.set_y(y + h)
        pdf.ln(2)


class WeatherTableBlock(Block):
    ROW_H = [7, 6, 6, 6]

    def __init__(self, days: list):
        super().__init__(sum(self.ROW_H))
        self.days = days

    def draw(self, pdf: "HermesPDF"):
        LW = 16
        dw = (pdf.epw - LW) / 7
        labels = ["Date", "Weekday", "Weather", "Temp"]
        rows = [
            [_fmt_d(d["date"]) for d in self.days],
            [("*" if d.get("hol") else "") + d.get("wd", "") for d in self.days],
            [d.get("wx", "N/A") for d in self.days],
            [f"{d.get('tmax', '')}C / {d.get('tmin', '')}C" for d in self.days],
        ]
        hol_cols = {i for i, d in enumerate(self.days) if d.get("hol")}
        for ri, (lbl, vals, rh) in enumerate(zip(labels, rows, self.ROW_H)):
            y = pdf.get_y()
            pdf.fc(NAVY)
            pdf.tc(WHITE)
            pdf.set_font(pdf._font_name, "B", 7.5)
            pdf.rect(pdf.l_margin, y, LW, rh, "F")
            pdf.set_xy(pdf.l_margin + 1, y + (rh - 4) / 2)
            pdf.cell(LW - 1, 4, lbl)
            for ci, val in enumerate(vals):
                x = pdf.l_margin + LW + ci * dw
                if ci in hol_cols:
                    pdf.fc((255, 248, 220))
                    pdf.tc(NAVY)
                else:
                    pdf.fc(LBG if ri % 2 == 0 else WHITE)
                    pdf.tc((50, 50, 50))
                pdf.rect(x, y, dw, rh, "F")
                pdf.dc(BDR)
                pdf.rect(x, y, dw, rh)
                pdf.set_font(pdf._font_name, "B" if ri == 0 else "", 7 if ri >= 2 else 7.5)
                pdf.set_xy(x + 0.5, y + (rh - 4) / 2)
                pdf.cell(dw - 1, 4, cl(str(val)), align="C")
            pdf.set_y(y + rh)
        pdf.tc((0, 0, 0))
        pdf.dc((0, 0, 0))


class CompTableBlock(Block):
    def __init__(self, headers: list, rows: list):
        self.headers = headers
        self.rows = rows
        super().__init__(7 + len(rows) * 6)

    def draw(self, pdf: "HermesPDF"):
        cw = pdf.epw / len(self.headers)
        y = pdf.get_y()
        pdf.fc(NAVY)
        pdf.tc(WHITE)
        pdf.set_font(pdf._font_name, "B", 8.5)
        for ci, h in enumerate(self.headers):
            x = pdf.l_margin + ci * cw
            pdf.rect(x, y, cw, 7, "F")
            pdf.set_xy(x + 2, y + 1.5)
            pdf.cell(cw - 2, 4, cl(h))
        pdf.set_y(y + 7)
        for ri, row in enumerate(self.rows):
            y2 = pdf.get_y()
            pdf.fc(LBG if ri % 2 == 0 else WHITE)
            pdf.tc((40, 40, 40))
            pdf.set_font(pdf._font_name, "", 8)
            for ci, cell in enumerate(row):
                x2 = pdf.l_margin + ci * cw
                pdf.rect(x2, y2, cw, 6, "F")
                pdf.dc(BDR)
                pdf.rect(x2, y2, cw, 6)
                pdf.set_xy(x2 + 2, y2 + 1)
                pdf.cell(cw - 2, 4, cl(str(cell)))
            pdf.set_y(y2 + 6)
        pdf.tc((0, 0, 0))
        pdf.dc((0, 0, 0))


class ForecastGridBlock(Block):
    GH = 24

    def __init__(self, next_days: list):
        super().__init__(self.GH)
        self.nxt = next_days

    def draw(self, pdf: "HermesPDF"):
        cw = pdf.epw / 7
        y = pdf.get_y()
        for i, day in enumerate(self.nxt):
            x = pdf.l_margin + i * cw
            wk = day.get("wd") in ("Sat", "Sun")
            pdf.fc(NAVY if wk else SLATE)
            pdf.tc(WHITE)
            pdf.rect(x, y, cw, 7, "F")
            pdf.set_font(pdf._font_name, "B", 7.5)
            pdf.set_xy(x + 1, y + 1.5)
            dt = day.get("date")
            lbl = _fmt_d(dt) if hasattr(dt, "day") else str(dt)[:10]
            pdf.cell(cw - 1, 4, cl(lbl), align="C")
            pdf.fc(LBG)
            pdf.tc(NAVY)
            pdf.rect(x, y + 7, cw, 5, "F")
            pdf.set_font(pdf._font_name, "B", 7)
            pdf.set_xy(x + 1, y + 8.5)
            pdf.cell(cw - 1, 4, cl(day.get("wd", "")), align="C")
            pdf.fc(WHITE)
            pdf.tc((50, 50, 50))
            pdf.rect(x, y + 12, cw, 6, "F")
            pdf.set_font(pdf._font_name, "", 6.5)
            pdf.set_xy(x + 1, y + 13.5)
            pdf.cell(cw - 1, 4, cl(f"FP: ~{day.get('predicted_fp', 0):,}"), align="C")
            pdf.fc(LBG)
            pdf.tc(NAVY)
            pdf.rect(x, y + 18, cw, 6, "F")
            pdf.set_font(pdf._font_name, "B", 7)
            pdf.set_xy(x + 1, y + 19.5)
            pdf.cell(cw - 1, 4, cl(f"CVR: {day.get('predicted_cvr', 0):.1f}%"), align="C")
            pdf.dc(BDR)
            pdf.rect(x, y, cw, self.GH)
        pdf.tc((0, 0, 0))
        pdf.dc((0, 0, 0))
        pdf.set_y(y + self.GH)


class ClosingBlock(Block):
    MIN_H = 14
    MAX_H = 42

    def __init__(self, space: str, start, end):
        super().__init__(self.MIN_H)
        self.space = space
        self.start = start
        self.end = end

    def draw(self, pdf: "HermesPDF"):
        rem = pdf.page_break_trigger - pdf.get_y()
        bh = max(self.MIN_H, min(rem - 1, self.MAX_H))
        y = pdf.get_y()
        pdf.fc(LBG)
        pdf.dc(NAVY)
        pdf.set_line_width(0.3)
        pdf.rect(pdf.l_margin, y, pdf.epw, bh, "FD")
        pdf.fc(NAVY)
        pdf.rect(pdf.l_margin, y, pdf.epw, 1, "F")
        pdf.set_xy(pdf.l_margin + 4, y + 3)
        pdf.tc(NAVY)
        pdf.set_font(pdf._font_name, "B", 8)
        pdf.cell(0, 5, "About This Report", ln=True)
        pdf.set_x(pdf.l_margin + 4)
        pdf.tc(SLATE)
        pdf.set_font(pdf._font_name, "", 7.5)
        start_s = self.start.strftime("%Y-%m-%d") if hasattr(self.start, "strftime") else str(self.start)
        end_s = self.end.strftime("%Y-%m-%d") if hasattr(self.end, "strftime") else str(self.end)
        pdf.multi_cell(pdf.epw - 8, 4.3, cl(
            f"Space: {self.space}  |  Period: {start_s} to {end_s}  |  Source: Hermes BLE Sensor Network\n"
            "Floating Population = unique MACs at entrance above RSSI threshold. "
            "Quality Visitors = strict-entry visitors (3+ signals/min) who stayed 3+ min. "
            "Quality CVR = Quality Visitors / Floating Population x 100. "
            "Dwell segments: Short < 3min, Medium 3-10min, Long 10+ min. "
            "Predictions improve as more historical data accumulates.\n"
            "This report is produced and provided by TJLABS Co., Ltd. "
            "All content and analysis are the exclusive property of TJLABS Co., Ltd. "
            "For inquiries, contact pepper@tjlabscorp.com"
        ))
        pdf.tc((0, 0, 0))
        pdf.set_line_width(0.2)


# ── Page & LayoutEngine ──────────────────────────────────────────────────────
class Page:
    def __init__(self, avail: float):
        self.avail = avail
        self.blocks: List[Block] = []
        self.gaps: List[Tuple[int, float]] = []

    @property
    def content_h(self) -> float:
        return sum(b.height for b in self.blocks if not b.soft)

    @property
    def spare(self) -> float:
        return self.avail - self.content_h

    def fits(self, block: Block) -> bool:
        return self.content_h + block.height <= self.avail

    def compute_gaps(self, min_gap: float = 6, max_gap: float = 45):
        sec_indices = [i for i, b in enumerate(self.blocks) if isinstance(b, SectionTitleBlock)]
        n_slots = len(sec_indices) + 1
        raw_gap = self.spare / n_slots if n_slots > 0 else 0
        gap = max(min_gap, min(raw_gap, max_gap))
        slots: List[Tuple[int, float]] = [(0, gap)]
        for idx in sec_indices[1:]:
            slots.append((idx, gap))
        self.gaps = slots


class LayoutEngine:
    MIN_GAP = 6
    MAX_GAP = 45
    SECTION_FIT_RATIO = 0.70

    def __init__(self, blocks: List[Block], p1_avail: float = P1_AVAIL, p2_avail: float = P2_AVAIL):
        self.blocks = blocks
        self.p1_avail = p1_avail
        self.p2_avail = p2_avail

    def _new_page(self, pages: List[Page]) -> Page:
        avail = self.p1_avail if not pages else self.p2_avail
        p = Page(avail)
        pages.append(p)
        return p

    def _section_total_height(self, start_idx: int) -> float:
        total = 0.0
        for j in range(start_idx, len(self.blocks)):
            b = self.blocks[j]
            if j > start_idx and isinstance(b, SectionTitleBlock):
                break
            if not b.soft:
                total += b.height
        return total

    def pack(self) -> List[Page]:
        pages: List[Page] = []
        cur = self._new_page(pages)
        i = 0
        while i < len(self.blocks):
            blk = self.blocks[i]
            if blk.soft:
                if cur.fits(blk):
                    cur.blocks.append(blk)
                i += 1
                continue
            if isinstance(blk, SectionTitleBlock):
                section_h = self._section_total_height(i)
                remaining = cur.avail - cur.content_h
                fits_ratio = remaining / section_h if section_h > 0 else 1.0
                if fits_ratio < self.SECTION_FIT_RATIO and section_h <= self.p2_avail:
                    cur = self._new_page(pages)
            elif isinstance(blk, ClosingBlock):
                cur.blocks.append(blk)
                i += 1
                continue
            else:
                if not cur.fits(blk):
                    cur = self._new_page(pages)
            cur.blocks.append(blk)
            i += 1
        for p in pages:
            p.compute_gaps(self.MIN_GAP, self.MAX_GAP)
        return pages


# ── Chart functions (matplotlib, figsize 9 x 1.8) ─────────────────────────────
def _chart_traffic(week_df: pd.DataFrame) -> bytes:
    if week_df.empty or len(week_df) == 0:
        fig, ax = plt.subplots(figsize=(9.0, 1.8))
        ax.set_facecolor("#f8f9fc")
        ax.text(0.5, 0.5, "No traffic data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _png(fig)
    fig, ax1 = plt.subplots(figsize=(9.0, 1.8))
    fig.patch.set_facecolor("#f8f9fc")
    ax1.set_facecolor("#f8f9fc")
    fp = week_df["floating_unique"].tolist()
    vis = week_df["quality_visitor_count"].tolist()
    dates = [d.strftime("%b ") + str(getattr(d, "day", d)) for d in week_df["date"]]
    x = list(range(len(dates)))
    bars = ax1.bar(x, fp, color=_m(NAVY), alpha=0.85, width=0.55, label="Floating Pop")
    hol_col = week_df.get("hol")
    if hol_col is not None:
        hol_idx = [i for i, h in enumerate(hol_col) if h]
        for i in hol_idx:
            bars[i].set_color(_m(GOLD))
            bars[i].set_alpha(0.9)
    else:
        hol_idx = []
    ax2 = ax1.twinx()
    ax2.plot(x, vis, color=_m(AMBER), marker="o", lw=2.0, ms=5, label="Quality Visitors", zorder=5)
    if hol_idx and "hol" in week_df.columns:
        hol_label = week_df["hol"].iloc[hol_idx[0]] if hol_idx else ""
        ax1.axvspan(hol_idx[0] - 0.45, hol_idx[-1] + 0.45, alpha=0.06, color=_m(GOLD))
        ax1.text((hol_idx[0] + hol_idx[-1]) / 2, max(fp) * 1.16 if fp else 1,
                 cl(str(hol_label)), ha="center", fontsize=7, color=_m(GOLD), style="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, fontsize=8.5)
    ax1.set_ylabel("Floating Pop", fontsize=8, color=_m(NAVY))
    ax2.set_ylabel("Quality Visitors", fontsize=8, color=_m(AMBER))
    ax1.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)
    if fp:
        ax1.set_ylim(0, max(fp) * 1.28)
    if vis:
        ax2.set_ylim(0, max(vis) * 1.4)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=7.5, framealpha=0.7)
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top", "left"]].set_visible(False)
    fig.tight_layout(pad=0.5)
    return _png(fig)


def _chart_dwell(week_df: pd.DataFrame, dwell_ratios: dict, quality_cvr: float) -> bytes:
    """5-tier dwell funnel chart: stacked bar + donut.

    dwell_ratios keys:
      - "1_3min": 1~3 min
      - "3_6min": 3~6 min
      - "6_10min": 6~10 min
      - "10_15min": 10~15 min
      - "15plus": 15+ min

    Colors:
      - 1~3min: SLATE (#64748b)
      - 3~6min: FP_COLOR (#4A90D9)
      - 6~10min: TEAL (#64ffda)
      - 10~15min: GOLD (#c49a3a)
      - 15+min: AMBER (#d97706)
    """
    # Color definitions (RGB 0-255)
    C_1_3 = (100, 116, 139)   # slate gray
    C_3_6 = (74, 144, 217)    # blue
    C_6_10 = (100, 255, 218)  # teal
    C_10_15 = (196, 154, 58)  # gold
    C_15plus = (217, 119, 6)  # amber

    if week_df.empty or len(week_df) == 0:
        fig, ax = plt.subplots(figsize=(9.0, 1.8))
        ax.set_facecolor("#f8f9fc")
        ax.text(0.5, 0.5, "No dwell data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _png(fig)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 1.8))
    fig.patch.set_facecolor("#f8f9fc")
    for ax in (ax1, ax2):
        ax.set_facecolor("#f8f9fc")

    vis = week_df["quality_visitor_count"].tolist() if "quality_visitor_count" in week_df.columns else [0] * len(week_df)
    dates = [str(getattr(d, "day", d)) for d in week_df["date"]]
    x = list(range(len(dates)))

    # Extract ratios
    r_1_3 = dwell_ratios.get("1_3min", 0) or 0
    r_3_6 = dwell_ratios.get("3_6min", 0) or 0
    r_6_10 = dwell_ratios.get("6_10min", 0) or 0
    r_10_15 = dwell_ratios.get("10_15min", 0) or 0
    r_15plus = dwell_ratios.get("15plus", 0) or 0

    # Calculate bar heights
    d_1_3 = [v * r_1_3 / 100 for v in vis]
    d_3_6 = [v * r_3_6 / 100 for v in vis]
    d_6_10 = [v * r_6_10 / 100 for v in vis]
    d_10_15 = [v * r_10_15 / 100 for v in vis]
    d_15plus = [v * r_15plus / 100 for v in vis]

    # Stacked bar chart
    bottom = [0] * len(x)
    ax1.bar(x, d_1_3, color=_m(C_1_3), alpha=0.85, label="1-3min", width=0.6, bottom=bottom)
    bottom = [a + b for a, b in zip(bottom, d_1_3)]
    ax1.bar(x, d_3_6, color=_m(C_3_6), alpha=0.85, label="3-6min", width=0.6, bottom=bottom)
    bottom = [a + b for a, b in zip(bottom, d_3_6)]
    ax1.bar(x, d_6_10, color=_m(C_6_10), alpha=0.85, label="6-10min", width=0.6, bottom=bottom)
    bottom = [a + b for a, b in zip(bottom, d_6_10)]
    ax1.bar(x, d_10_15, color=_m(C_10_15), alpha=0.90, label="10-15min", width=0.6, bottom=bottom)
    bottom = [a + b for a, b in zip(bottom, d_10_15)]
    ax1.bar(x, d_15plus, color=_m(C_15plus), alpha=0.95, label="15+min", width=0.6, bottom=bottom)

    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, fontsize=8.5)
    ax1.set_ylabel("Visitors", fontsize=8)
    ax1.legend(fontsize=6, loc="upper right", ncol=2)
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title("Daily Dwell Funnel (5-tier)", fontsize=9, fontweight="bold", pad=4)
    ax1.tick_params(labelsize=8)

    # Donut chart
    sizes = [r_1_3, r_3_6, r_6_10, r_10_15, r_15plus]
    colors = [_m(C_1_3), _m(C_3_6), _m(C_6_10), _m(C_10_15), _m(C_15plus)]
    # Filter out zero slices for cleaner donut
    nonzero = [(s, c) for s, c in zip(sizes, colors) if s > 0]
    if nonzero:
        sizes_nz, colors_nz = zip(*nonzero)
    else:
        sizes_nz, colors_nz = [1], [_m(SLATE)]

    wedges, _ = ax2.pie(sizes_nz, colors=colors_nz, startangle=90,
                        wedgeprops=dict(width=0.45, edgecolor="white", lw=1.5))
    ax2.text(0, 0, f"Quality\nCVR\n{quality_cvr:.1f}%",
             ha="center", va="center", fontsize=9, fontweight="bold", color=_m(NAVY))

    # Legend labels
    labels = [
        f"1-3m {r_1_3:.1f}%", f"3-6m {r_3_6:.1f}%", f"6-10m {r_6_10:.1f}%",
        f"10-15m {r_10_15:.1f}%", f"15+m {r_15plus:.1f}%"
    ]
    # Filter labels to match nonzero wedges
    labels_nz = [labels[i] for i, s in enumerate(sizes) if s > 0] or ["N/A"]
    ax2.legend(wedges, labels_nz, loc="lower center", fontsize=6, bbox_to_anchor=(0.5, -0.10), ncol=3)
    ax2.set_title("Dwell Distribution", fontsize=9, fontweight="bold", pad=4)
    fig.tight_layout(pad=0.6)
    return _png(fig)


def _chart_nextweek(next_df: pd.DataFrame) -> bytes:
    if next_df.empty or len(next_df) == 0:
        fig, ax = plt.subplots(figsize=(9.0, 1.8))
        ax.set_facecolor("#f8f9fc")
        ax.text(0.5, 0.5, "No forecast data", ha="center", va="center", fontsize=10)
        ax.axis("off")
        return _png(fig)
    fig, ax1 = plt.subplots(figsize=(9.0, 1.8))
    fig.patch.set_facecolor("#f8f9fc")
    ax1.set_facecolor("#f8f9fc")
    fp = next_df["predicted_fp"].tolist()
    cvr = next_df["predicted_cvr"].tolist()
    dates = [str(getattr(d, "day", d)) + "\n" + next_df["wd"].iloc[i] for i, d in enumerate(next_df["date"])]
    x = list(range(len(dates)))
    wds = next_df["wd"].tolist()
    bar_col = [_m((10, 28, 70)) if wd in ("Sat", "Sun") else _m(NAVY) for wd in wds]
    ax1.bar(x, fp, color=bar_col, alpha=0.85, width=0.55)
    ax1.set_xticks(x)
    ax1.set_xticklabels(dates, fontsize=8.5)
    ax1.set_ylabel("Est. Floating Pop", fontsize=8, color=_m(NAVY))
    if fp:
        ax1.set_ylim(0, max(fp) * 1.35)
    ax1.tick_params(labelsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, cvr, color=_m(AMBER), marker="D", lw=2.0, ms=5.5, ls="--", label="Est. CVR %")
    ax2.set_ylabel("Est. CVR (%)", fontsize=8, color=_m(AMBER))
    if cvr:
        ax2.set_ylim(min(cvr) - 0.4, max(cvr) + 0.8)
    ax2.tick_params(labelsize=8)
    for i, (v, c) in enumerate(zip(fp, cvr)):
        ax1.text(i, v + (max(fp) * 0.025 if fp else 1), f"{v:,}",
                 ha="center", fontsize=7.5, color=_m(NAVY), fontweight="bold")
        ax2.text(i, c + 0.07, f"{c:.1f}%",
                 ha="center", fontsize=7.5, color=_m(AMBER), fontweight="bold")
    ax1.grid(axis="y", ls="--", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top", "left"]].set_visible(False)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.text(0.01, 0.95, "Darker = weekend", transform=ax1.transAxes,
             fontsize=7, va="top", color=_m(SLATE), style="italic")
    fig.tight_layout(pad=0.5)
    return _png(fig)


# ── HermesPDF ──────────────────────────────────────────────────────────────
_FONT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "fonts")
_NANUM_PATH = os.path.join(_FONT_DIR, "NanumGothic.ttf")
_HAS_NANUM = os.path.exists(_NANUM_PATH)


class HermesPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_auto_page_break(auto=True, margin=AUTO_BRK)
        self.set_margins(MARG, MARG, MARG)
        self._tmp: List[str] = []
        self._space = ""
        self._rpt_start = ""
        self._rpt_end = ""
        # Register Korean font if available
        if _HAS_NANUM:
            self.add_font("Nanum", "", _NANUM_PATH, uni=True)
            self.add_font("Nanum", "B", _NANUM_PATH, uni=True)
            self.add_font("Nanum", "I", _NANUM_PATH, uni=True)
            self.add_font("Nanum", "BI", _NANUM_PATH, uni=True)
            self._font_name = "Nanum"
        else:
            self._font_name = "Helvetica"

    def fc(self, c): self.set_fill_color(*c)
    def tc(self, c): self.set_text_color(*c)
    def dc(self, c): self.set_draw_color(*c)

    def header(self):
        if self.page_no() == 1:
            return
        self.fc(NAVY)
        self.rect(0, 0, 210, 9, "F")
        self.set_y(1.5)
        self.tc(WHITE)
        self.set_font(self._font_name, "B", 7.5)
        self.cell(0, 5, f"  HERMES  |  {self._space}  |  {self._rpt_start} - {self._rpt_end}", ln=True)
        self.tc((0, 0, 0))
        self.set_y(P2_START)

    def footer(self):
        self.set_y(-11)
        self.set_font(self._font_name, "I", 6.5)
        self.tc(SLATE)
        self.cell(0, 4,
                  "(c) TJLABS Co., Ltd. All rights reserved.  |  "
                  "Inquiries: pepper@tjlabscorp.com  |  "
                  f"Powered by Hermes  |  Page {self.page_no()}",
                  align="C")
        self.tc((0, 0, 0))

    def draw_cover(self, space: str, rpt_start, rpt_end, generated_at: str):
        self._space = cl(space)
        self._rpt_start = rpt_start.strftime("%b %d") if hasattr(rpt_start, "strftime") else str(rpt_start)[:10]
        self._rpt_end = rpt_end.strftime("%b %d") if hasattr(rpt_end, "strftime") else str(rpt_end)[:10]
        self.add_page()
        self.fc(NAVY)
        self.rect(0, 0, 210, 48, "F")
        self.fc(GOLD)
        self.rect(0, 46, 210, 2, "F")
        self.set_y(9)
        self.tc(GOLD)
        self.set_font(self._font_name, "B", 22)
        self.cell(0, 10, "HERMES Weekly Traffic Report", align="C", ln=True)
        self.tc(WHITE)
        self.set_font(self._font_name, "", 11)
        start_fmt = rpt_start.strftime("%b %d (%a)") if hasattr(rpt_start, "strftime") else str(rpt_start)
        end_fmt = rpt_end.strftime("%b %d (%a)") if hasattr(rpt_end, "strftime") else str(rpt_end)
        self.cell(0, 7, f"{space}  |  {start_fmt} - {end_fmt}, {getattr(rpt_end, 'year', '')}", align="C", ln=True)
        self.tc((175, 185, 210))
        self.set_font(self._font_name, "I", 8.5)
        self.cell(0, 5, f"Generated: {generated_at}", align="C", ln=True)
        self.set_y(40)
        self.tc((140, 155, 185))
        self.set_font(self._font_name, "", 7)
        self.cell(0, 4, "Provided by TJLABS Co., Ltd.  |  pepper@tjlabscorp.com", align="C", ln=True)
        self.set_y(P1_START)
        self.tc((0, 0, 0))

    def render_pages(self, pages: List[Page]):
        for pi, page in enumerate(pages):
            if pi > 0:
                self.add_page()
            gap_map = {idx: g for idx, g in page.gaps}
            for bi, block in enumerate(page.blocks):
                if bi in gap_map:
                    self.ln(gap_map[bi])
                block.draw(self)

    def cleanup(self):
        for p in self._tmp:
            try:
                os.remove(p)
            except Exception:
                pass


# ── build_blocks & generate ─────────────────────────────────────────────────
def build_blocks(
    week_df: pd.DataFrame,
    next_df: pd.DataFrame,
    kpi: dict,
    dwell: dict,
    ai_texts: dict,
    summary: str,
    market_ctx: str,
    next_outlook: str,
    space: str,
    rpt_start,
    rpt_end,
    holiday_label: str = "",
    season: str = "Winter",
    comp_table_rows: Optional[List[list]] = None,
) -> List[Block]:
    blocks: List[Block] = []
    n = 1
    quality_cvr = kpi["cvr"]["v"]

    blocks.append(SectionTitleBlock(n, "This Week Key Metrics"))
    n += 1
    blocks.append(KPICardsBlock(kpi, summary))

    blocks.append(SectionTitleBlock(n, "Weekly Traffic Flow"))
    n += 1
    try:
        traffic_img = _chart_traffic(week_df)
    except Exception:
        traffic_img = b""
    blocks.append(ChartBlock(traffic_img))
    blocks.append(AIBoxBlock(ai_texts.get("traffic", ""), max_chars=600))

    blocks.append(SectionTitleBlock(n, "Dwell Funnel (5-tier)"))
    n += 1
    # 5분류 텍스트
    blocks.append(SmallTextBlock(
        f"1-3m: {dwell.get('1_3min', 0):.1f}%  |  "
        f"3-6m: {dwell.get('3_6min', 0):.1f}%  |  "
        f"6-10m: {dwell.get('6_10min', 0):.1f}%  |  "
        f"10-15m: {dwell.get('10_15min', 0):.1f}%  |  "
        f"15+m: {dwell.get('15plus', 0):.1f}%  |  Quality CVR: {quality_cvr:.1f}%",
        italic=False, color=SLATE))
    try:
        dwell_img = _chart_dwell(week_df, dwell, quality_cvr)
    except Exception:
        dwell_img = b""
    blocks.append(ChartBlock(dwell_img))
    blocks.append(AIBoxBlock(ai_texts.get("dwell", ""), max_chars=600))

    blocks.append(SectionTitleBlock(n, "This Week Context"))
    n += 1
    days_list = _week_df_to_days(week_df)
    blocks.append(WeatherTableBlock(days_list))
    hol_legend = f"  * Highlighted = Public Holiday  |  Season: {season}" + (f"  |  {holiday_label}" if holiday_label else "")
    blocks.append(SmallTextBlock(hol_legend, italic=True, color=NAVY))
    blocks.append(SubHeadBlock("Market Context"))
    blocks.append(AIBoxBlock(ai_texts.get("context", ""), max_chars=800))

    blocks.append(SectionTitleBlock(n, "AI Insights -- Weekly Synthesis"))
    n += 1
    blocks.append(TextBlock(ai_texts.get("synthesis", "")))
    blocks.append(SpacerBlock(2))
    blocks.append(SubHeadBlock("DIAGNOSIS -- Key Patterns"))
    for pat in ai_texts.get("patterns", []):
        blocks.append(BulletBlock(pat))
        blocks.append(SpacerBlock(1))
    blocks.append(SpacerBlock(2))
    blocks.append(SubHeadBlock("ACTIONS -- Recommended Next Steps"))
    for i, act in enumerate(ai_texts.get("actions", []), 1):
        blocks.append(BulletBlock(act, num=i))
        blocks.append(SpacerBlock(1))

    blocks.append(SectionTitleBlock(n, "Next Week Outlook"))
    n += 1
    if comp_table_rows is None:
        comp_table_rows = []
    blocks.append(CompTableBlock(["Metric", "This Week", "Next Week (est.)"], comp_table_rows))
    blocks.append(SpacerBlock(2))
    blocks.append(SubHeadBlock("Outlook"))
    blocks.append(SpacerBlock(2))
    blocks.append(SubHeadBlock("Daily Forecast Grid"))
    next_records = _next_df_to_records(next_df)
    blocks.append(ForecastGridBlock(next_records))
    blocks.append(SpacerBlock(2))
    blocks.append(SubHeadBlock("Traffic Forecast Chart"))
    try:
        next_img = _chart_nextweek(next_df)
    except Exception:
        next_img = b""
    blocks.append(ChartBlock(next_img))
    blocks.append(AIBoxBlock(ai_texts.get("nextweek", ""), max_chars=600))
    blocks.append(SmallTextBlock(
        "* Predictions based on historical day-type patterns. Weather forecast may be unavailable.",
        italic=True, color=SLATE))

    blocks.append(ClosingBlock(space, rpt_start, rpt_end))
    return blocks


def _week_df_to_days(week_df: pd.DataFrame) -> list:
    """Convert week_df to list of dicts for WeatherTableBlock."""
    days = []
    for r in week_df.to_dict("records"):
        d = r.get("date")
        if hasattr(d, "weekday"):
            wd = DAY_ABBR[d.weekday()] if d.weekday() < 7 else ""
        else:
            try:
                dt = pd.to_datetime(d)
                wd = DAY_ABBR[dt.weekday()]
            except Exception:
                wd = ""
        days.append({
            "date": d,
            "wd": wd,
            "wx": r.get("weather", "N/A"),
            "tmax": int(r.get("temp_max", 0)) if pd.notna(r.get("temp_max")) else "",
            "tmin": int(r.get("temp_min", 0)) if pd.notna(r.get("temp_min")) else "",
            "hol": r.get("holiday_name") or (r.get("holiday_period") if r.get("is_holiday") else None),
        })
    return days


def _next_df_to_records(next_df: pd.DataFrame) -> list:
    """Convert next_df to list of dicts for ForecastGridBlock."""
    recs = []
    for r in next_df.to_dict("records"):
        d = r.get("date")
        wd = r.get("wd", "")
        if hasattr(d, "strftime"):
            pass
        else:
            try:
                d = pd.to_datetime(d)
                wd = DAY_ABBR[d.weekday()]
            except Exception:
                wd = ""
        recs.append({
            "date": d,
            "wd": wd,
            "predicted_fp": int(r.get("predicted_fp", 0)),
            "predicted_cvr": float(r.get("predicted_cvr", 0)),
        })
    return recs


def generate(
    out_path: Optional[str] = None,
    generated_at: str = "",
    **kwargs,
) -> bytes:
    """Build blocks, pack pages, render PDF. Returns PDF bytes. If out_path given, also writes file."""
    blocks = build_blocks(**kwargs)
    engine = LayoutEngine(blocks)
    pages = engine.pack()
    pdf = HermesPDF()
    pdf.draw_cover(
        space=kwargs["space"],
        rpt_start=kwargs["rpt_start"],
        rpt_end=kwargs["rpt_end"],
        generated_at=generated_at or datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    pdf.render_pages(pages)
    try:
        data = pdf.output(dest="S")
        if isinstance(data, str):
            data = data.encode("latin-1")
    except TypeError:
        buf = io.BytesIO()
        pdf.output(buf)
        data = buf.getvalue()
    pdf.cleanup()
    if out_path:
        with open(out_path, "wb") as f:
            f.write(data)
    return data


def _report_data_to_v2_args(
    report_data: Dict[str, Any],
    space_name: str,
    date_range: tuple,
    ai_insight: str,
) -> dict:
    """Map existing report_data (from pages) to build_blocks() kwargs."""
    daily = report_data.get("daily", pd.DataFrame())
    if daily.empty:
        daily = pd.DataFrame(columns=["date", "floating_unique", "quality_visitor_count", "weather", "temp_max", "temp_min", "is_holiday", "holiday_name", "holiday_period"])
    week_df = daily.tail(7).copy()
    if not week_df.empty:
        if "date" in week_df.columns:
            week_df["date"] = pd.to_datetime(week_df["date"])
            if hasattr(week_df["date"].dt, "tz_localize") and week_df["date"].dt.tz is not None:
                week_df["date"] = week_df["date"].dt.tz_localize(None)
            week_df["wd"] = week_df["date"].apply(lambda d: DAY_ABBR[d.weekday()] if d.weekday() < 7 else "")
            if "holiday_name" in week_df.columns or "is_holiday" in week_df.columns:
                week_df["hol"] = week_df.apply(
                    lambda r: r.get("holiday_name") or (r.get("holiday_period") if r.get("is_holiday") else None),
                    axis=1,
                )
            else:
                week_df["hol"] = None
        if "quality_visitor_count" not in week_df.columns:
            week_df["quality_visitor_count"] = week_df["visitor_count"] if "visitor_count" in week_df.columns else 0

    preds = report_data.get("predictions", [])
    next_rows = []
    for p in preds[:7]:
        d = p.get("date_obj") or p.get("date")
        try:
            dt = pd.to_datetime(d)
            wd = DAY_ABBR[dt.weekday()]
        except Exception:
            wd = ""
        next_rows.append({
            "date": d,
            "wd": wd,
            "predicted_fp": int(p.get("floating_mean", 0)),
            "predicted_cvr": float(p.get("quality_cvr_mean", 0)),
        })
    next_df = pd.DataFrame(next_rows) if next_rows else pd.DataFrame(columns=["date", "wd", "predicted_fp", "predicted_cvr"])

    kpi_data = report_data.get("kpi", {})
    tw = kpi_data.get("this_week", {})
    delta = kpi_data.get("delta", {})
    kpi = {
        "fp": {"v": tw.get("floating_unique", 0), "d": delta.get("floating_pct") or 0,
               "label": "FLOATING POP", "sub": "People who passed by the store entrance"},
        "vis": {"v": tw.get("quality_visitor_count", 0), "d": delta.get("quality_visitor_pct") or 0,
                "label": "QUALITY VISITORS", "sub": "Visitors who stayed 3+ minutes (engaged)"},
        "cvr": {"v": tw.get("quality_cvr", 0), "d": delta.get("quality_cvr_pp") or 0,
                "label": "QUALITY CVR", "sub": "% of passers-by who became engaged visitors"},
        "dw": {"v": tw.get("dwell_median_seconds", 0), "d": delta.get("dwell_median") or 0,
               "label": "MEDIAN DWELL", "sub": "Typical time spent inside the store"},
    }

    funnel = report_data.get("funnel", {})
    # 5분류 dwell ratios
    dwell = {
        "1_3min": funnel.get("1_3min_pct", 0),
        "3_6min": funnel.get("3_6min_pct", 0),
        "6_10min": funnel.get("6_10min_pct", 0),
        "10_15min": funnel.get("10_15min_pct", 0),
        "15plus": funnel.get("15plus_pct", 0),
    }

    synthesis = ai_insight or ""
    patterns = []
    actions = []
    for line in (ai_insight or "").split("\n"):
        line = line.strip()
        if "PATTERN" in line.upper():
            continue
        if "ACTIONS" in line.upper() or "ACTION" in line.upper():
            continue
        if line.startswith("-") or line.startswith(">>"):
            content = line.lstrip("- >>").strip()
            if content and len(patterns) < 5:
                patterns.append(content)
            elif content and len(actions) < 5:
                actions.append(content)
    if not patterns and synthesis:
        patterns = [synthesis[:300]]
    context_comment = (report_data.get("context_comment") or "").strip()
    prediction_comment = (report_data.get("prediction_comment") or "").strip()
    traffic_text = (report_data.get("traffic_comment") or synthesis[:600] or "").strip()
    dwell_text = (report_data.get("dwell_comment") or synthesis[600:1200] or synthesis[:600] or "").strip()
    ai_texts = {
        "traffic": traffic_text or "",
        "dwell": dwell_text or "",
        "context": context_comment,
        "nextweek": prediction_comment,
        "synthesis": synthesis,
        "patterns": patterns,
        "actions": actions if actions else ["Review next week forecast and adjust staffing."],
    }

    this_week = report_data.get("this_week", {})
    this_fp_avg = (this_week.get("floating") or 0) / 7
    this_cvr = this_week.get("quality_cvr") or 0
    next_fp_avg = next_df["predicted_fp"].mean() if not next_df.empty and "predicted_fp" in next_df.columns else 0
    next_cvr_avg = next_df["predicted_cvr"].mean() if not next_df.empty and "predicted_cvr" in next_df.columns else 0
    comp_table_rows = [
        ["Avg Daily Floating Pop", f"{this_fp_avg:,.0f}", f"{next_fp_avg:,.0f}"],
        ["Quality CVR", f"{this_cvr:.1f}%", f"{next_cvr_avg:.1f}%"],
    ]

    start, end = date_range[0], date_range[1]
    try:
        start_d = datetime.strptime(str(start)[:10], "%Y-%m-%d").date()
        end_d = datetime.strptime(str(end)[:10], "%Y-%m-%d").date()
    except Exception:
        start_d = start
        end_d = end

    ctx = report_data.get("context", {})
    holiday_label = ctx.get("holiday_period", "") or ""
    season = ctx.get("season", "Winter") or "Winter"

    return {
        "week_df": week_df,
        "next_df": next_df,
        "kpi": kpi,
        "dwell": dwell,
        "ai_texts": ai_texts,
        "summary": report_data.get("kpi_summary", "") or "",
        "market_ctx": report_data.get("context_comment", "") or "",
        "next_outlook": report_data.get("prediction_comment", "") or "",
        "space": space_name,
        "rpt_start": start_d,
        "rpt_end": end_d,
        "holiday_label": holiday_label,
        "season": season,
        "comp_table_rows": comp_table_rows,
    }


def generate_weekly_report_pdf(
    report_data: Dict[str, Any],
    chart_figures: Dict[str, Any],
    space_name: str,
    date_range: tuple,
    ai_insight: str,
) -> bytes:
    """Entry point for Streamlit Report tab. Returns PDF bytes. v2.0 uses Dynamic Layout Engine."""
    try:
        kwargs = _report_data_to_v2_args(report_data, space_name, date_range, ai_insight)
        return generate(generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"), **kwargs)
    except Exception as e:
        pdf_err = FPDF()
        pdf_err.set_margins(MARG, 20, MARG)
        pdf_err.add_page()
        pdf_err.set_font("Helvetica", "B", 14)
        pdf_err.cell(0, 10, "PDF Generation Error", ln=True)
        pdf_err.set_font("Helvetica", "", 10)
        err_msg = cl(f"{type(e).__name__}: {str(e)}")
        pdf_err.multi_cell(0, 6, err_msg[:500])
        try:
            return pdf_err.output(dest="S") or b""
        except Exception:
            buf = io.BytesIO()
            pdf_err.output(buf)
            return buf.getvalue()
