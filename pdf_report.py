"""
pdf_report.py - one-page PDF tearsheet for the valuation app (reportlab, no internet needed).
"""
import io
from datetime import date

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

BLACK = colors.HexColor("#111111")
ORANGE = colors.HexColor("#F25C05")
GREY = colors.HexColor("#8A8A8A")
LGREY = colors.HexColor("#C9C9C9")
LIGHT = colors.HexColor("#F3F3F3")
W, H = A4
LM = 15 * mm


def _fmt(x, d=2):
    try:
        return f"{x:,.{d}f}"
    except (TypeError, ValueError):
        return "n/a"


def _table(c, x, y, rows, widths, header=True, size=8, bold_last=False):
    """Draws a simple table; returns y below the table."""
    lh = size + 4.2
    for i, row in enumerate(rows):
        is_head = header and i == 0
        if not is_head and i % 2 == 0:
            c.setFillColor(LIGHT)
            c.rect(x, y - lh + 3, sum(widths), lh, stroke=0, fill=1)
        bold = is_head or (bold_last and i == len(rows) - 1)
        c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        c.setFillColor(BLACK)
        cx = x
        for j, (val, w) in enumerate(zip(row, widths)):
            txt = str(val)
            if j == 0:
                c.drawString(cx + 2, y - lh + 6, txt)
            else:
                c.drawRightString(cx + w - 2, y - lh + 6, txt)
            cx += w
        if is_head:
            c.setStrokeColor(ORANGE)
            c.setLineWidth(1.1)
            c.line(x, y - lh + 3, x + sum(widths), y - lh + 3)
        y -= lh
    c.setStrokeColor(LGREY)
    c.setLineWidth(0.5)
    c.line(x, y + 3, x + sum(widths), y + 3)
    return y - 4


def _heading(c, x, y, text):
    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.setFillColor(ORANGE)
    c.drawString(x, y, text)
    return y - 6


def build_tearsheet(d):
    """d: dict with keys name, ticker, rating, target, price, upside, div_yield, total_return, dcf, ri, comps, fair_value,
    ke, wacc, tg, scenario, preset, scen_rows (list of [name, dcf, ri, prob]), ff_rows (list of [label, low, mid, high]),
    assumptions (list of [label, value]), mc (dict p10, p50, p90, p_above), notes (str), currency, data_source."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle(f"{d['name']} - valuation tearsheet")
    # header band
    c.setFillColor(BLACK)
    c.rect(0, H - 34 * mm, W, 34 * mm, stroke=0, fill=1)
    c.setFillColor(ORANGE)
    c.rect(0, H - 35.5 * mm, W, 1.5 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(LM, H - 15 * mm, f"{d['name']} ({d['ticker']})")
    c.setFont("Helvetica", 8.5)
    c.drawString(LM, H - 21 * mm, f"Valuation tearsheet · {date.today():%d %B %Y} · {d['currency']}")
    c.drawString(LM, H - 25.5 * mm, f"Assumptions: {d['preset']} · scenario: {d['scenario']}")
    c.drawString(LM, H - 30 * mm, f"Data: {d['data_source']}")
    # rating box
    col = {"BUY": colors.HexColor("#1E8E3E"), "SELL": colors.HexColor("#B00020")}.get(d["rating"], ORANGE)
    c.setFillColor(col)
    c.rect(W - LM - 42 * mm, H - 29 * mm, 42 * mm, 20 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W - LM - 21 * mm, H - 18 * mm, d["rating"])
    c.setFont("Helvetica", 8.5)
    c.drawCentredString(W - LM - 21 * mm, H - 24.5 * mm, f"Target {_fmt(d['target'])}")

    # KPI strip
    y = H - 46 * mm
    kpis = [("Share price", _fmt(d["price"])), ("12-month target", _fmt(d["target"])), ("Upside", f"{d['upside']:+.1%}"),
            ("Dividend yield", f"{d['div_yield']:.1%}"), ("Total return", f"{d['total_return']:+.1%}"),
            ("WACC / ke", f"{d['wacc']:.2%} / {d['ke']:.2%}")]
    bw = (W - 2 * LM) / len(kpis)
    for i, (lab, val) in enumerate(kpis):
        x = LM + i * bw
        c.setFillColor(LIGHT)
        c.rect(x + 1, y - 4 * mm, bw - 2, 13 * mm, stroke=0, fill=1)
        c.setFillColor(ORANGE if i == 1 else BLACK)
        c.rect(x + 1, y - 4 * mm, 1.2 * mm, 13 * mm, stroke=0, fill=1)
        c.setFillColor(GREY)
        c.setFont("Helvetica", 7)
        c.drawString(x + 4 * mm, y + 5 * mm, lab.upper())
        c.setFillColor(BLACK)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x + 4 * mm, y - 1 * mm, val)

    # left column: valuation + scenarios + assumptions
    colw = (W - 2 * LM - 8 * mm) / 2
    xl, xr = LM, LM + colw + 8 * mm
    y = H - 60 * mm
    y = _heading(c, xl, y, "Target price derivation")
    rows = [["Step", "Value"], [f"DCF ({d['scenario']})", _fmt(d["dcf"])], ["Trading comps", _fmt(d["comps"])],
            ["Blended fair value", _fmt(d["fair_value"])], [f"x (1 + ke {d['ke']:.2%}) - dividends", ""],
            ["12-month target price", _fmt(d["target"])]]
    y = _table(c, xl, y, rows, [colw * 0.7, colw * 0.3], bold_last=True)
    y -= 4
    y = _heading(c, xl, y, "Scenarios")
    rows = [["Scenario", "DCF", "Residual income", "Probability"]] + \
           [[r[0], _fmt(r[1]), _fmt(r[2]), f"{r[3]:.0%}"] for r in d["scen_rows"]]
    y = _table(c, xl, y, rows, [colw * 0.28, colw * 0.18, colw * 0.34, colw * 0.2])
    y -= 4
    y = _heading(c, xl, y, "Key assumptions")
    rows = [["Assumption", "Value"]] + [[a, v] for a, v in d["assumptions"]]
    y = _table(c, xl, y, rows, [colw * 0.62, colw * 0.38])
    y -= 4
    y = _heading(c, xl, y, "Monte Carlo (DCF, active scenario)")
    mc = d["mc"]
    rows = [["Statistic", "Value"], ["10th percentile", _fmt(mc["p10"])], ["Median", _fmt(mc["p50"])],
            ["90th percentile", _fmt(mc["p90"])], ["P(value > share price)", f"{mc['p_above']:.0%}"]]
    _table(c, xl, y, rows, [colw * 0.62, colw * 0.38])

    # right column: football field drawing
    y = H - 60 * mm
    y = _heading(c, xr, y, "Football field (value per share)")
    ff = d["ff_rows"]
    vals = [v for r in ff for v in r[1:4]] + [d["price"], d["target"]]
    lo, hi = min(vals) * 0.9, max(vals) * 1.05
    label_w, bar_w = 33 * mm, colw - 33 * mm - 10 * mm
    sx = lambda v: xr + label_w + (v - lo) / (hi - lo) * bar_w
    row_h = 11 * mm
    top = y - 3 * mm
    for i, (lab, l, m, h_) in enumerate(ff):
        yy = top - i * row_h
        c.setFont("Helvetica", 7.2)
        c.setFillColor(BLACK)
        c.drawString(xr, yy - 4, lab[:28])
        colr = BLACK if lab.startswith("DCF") else ORANGE if lab.startswith("Comps") else GREY if (
            lab.startswith("Residual") or lab.startswith("Monte")) else LGREY
        c.setFillColor(colr)
        c.rect(sx(l), yy - 7, max(sx(h_) - sx(l), 1), 9, stroke=0, fill=1)
        c.setStrokeColor(colors.white if colr == BLACK else BLACK)
        c.setLineWidth(1.4)
        c.line(sx(m), yy - 8, sx(m), yy + 3)
        c.setFillColor(BLACK)
        c.setFont("Helvetica", 6.3)
        c.drawString(sx(h_) + 2, yy - 4, f"{l:,.0f}-{h_:,.0f}")
    bottom = top - len(ff) * row_h
    for v, colr, lab, dash in [(d["price"], GREY, f"Price {d['price']:,.2f}", True), (d["target"], ORANGE, f"Target {d['target']:,.2f}", False)]:
        c.setStrokeColor(colr)
        c.setLineWidth(1)
        c.setDash(3, 2) if dash else c.setDash()
        c.line(sx(v), top + 8, sx(v), bottom + 2)
        c.setDash()
        c.setFillColor(colr)
        c.setFont("Helvetica-Bold", 7)
        if dash:
            c.drawRightString(sx(v) - 2, bottom - 6, lab)
        else:
            c.drawString(sx(v) + 2, bottom - 6, lab)
    y = bottom - 18
    y = _heading(c, xr, y, "Notes")
    c.setFont("Helvetica", 7)
    c.setFillColor(BLACK)
    words, line = d["notes"].split(), ""
    for wd in words:
        if c.stringWidth(line + " " + wd, "Helvetica", 7) > colw:
            c.drawString(xr, y - 4, line)
            y -= 9
            line = wd
        else:
            line = (line + " " + wd).strip()
    c.drawString(xr, y - 4, line)

    # footer
    c.setStrokeColor(LGREY)
    c.line(LM, 14 * mm, W - LM, 14 * mm)
    c.setFont("Helvetica", 6.8)
    c.setFillColor(GREY)
    c.drawString(LM, 10 * mm, "Team 10 · WHU PT-MiM · generated by the Equity Valuation App · academic project, not investment advice")
    c.showPage()
    c.save()
    return buf.getvalue()
