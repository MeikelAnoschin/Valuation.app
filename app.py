"""
app.py - Equity Valuation App (Team 10, WHU PT-MiM)

DCF + Residual Income + Trading Comps with Bear/Base/Bull scenarios, 12-month target price & rating,
Monte Carlo, reverse DCF, tornado chart, analyst consensus, peer suggestions, company comparison, Excel export.

Run with:   streamlit run app.py
Needs:      valuation.py in the same folder, packages from requirements.txt, internet access (Yahoo Finance).

Data note: Yahoo's company-profile endpoint ("info") is often blocked on cloud servers. The app therefore
builds every key figure from sources that work reliably (price history / fast_info and the financial
statements) and only uses "info" when it is available.
"""
import io
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import yfinance as yf

from valuation import (
    dcf_valuation,
    implied_prices_from_comps,
    monte_carlo,
    residual_income_valuation,
    sensitivity_grid,
    solve_for_price,
    tornado,
    wacc_calc,
)
from pdf_report import build_tearsheet

st.set_page_config(page_title="Equity Valuation App | Team 10", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# 0. LOOK & FEEL
# --------------------------------------------------------------------------
BLACK, ORANGE, GREY, LGREY, LORANGE = "#111111", "#F25C05", "#8A8A8A", "#C9C9C9", "#F7A96B"
SCENARIOS = ["Bear", "Base", "Bull"]
SCEN_COLORS = {"Bear": ORANGE, "Base": BLACK, "Bull": GREY}
pio.templates["team10"] = go.layout.Template(layout=dict(colorway=[BLACK, ORANGE, GREY, LGREY, LORANGE],
                                                         font=dict(family="Helvetica, Arial, sans-serif", size=13),
                                                         margin=dict(t=50, l=10, r=10, b=10)))
pio.templates.default = "plotly_white+team10"
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.hero {background:#111; color:#fff; padding:18px 24px 16px; border-radius:8px; border-bottom:6px solid #F25C05; margin-bottom:10px;}
.hero h1 {color:#fff; margin:0; padding:0; font-size:2rem; font-weight:800; letter-spacing:-0.5px;}
.hero .sub {color:#BDBDBD; font-size:0.85rem; margin:2px 0 12px;}
.hero .row {display:flex; flex-wrap:wrap; gap:10px 28px; align-items:center;}
.hero .badge {background:#F25C05; color:#fff; font-weight:800; padding:6px 14px; border-radius:4px; font-size:1.05rem;}
.hero .badge.sell {background:#B00020;} .hero .badge.hold {background:#F25C05;} .hero .badge.buy {background:#1E8E3E;}
.hero .kpi {font-size:0.8rem; color:#BDBDBD; line-height:1.15;}
.hero .kpi b {display:block; color:#fff; font-size:1.35rem;}
.card {border:1px solid #E3E3E3; border-left:5px solid #111; border-radius:8px; padding:12px 14px; background:#fff; height:100%;}
.card.o {border-left-color:#F25C05;}
.card .lbl {font-size:0.78rem; color:#666; text-transform:uppercase; letter-spacing:0.4px;}
.card .val {font-size:1.55rem; font-weight:800; color:#111; margin-top:2px;}
.card .dlt {font-size:0.82rem; color:#555;}
.card .pos {color:#1E8E3E;} .card .neg {color:#B00020;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 8px 14px; font-weight:600;}
.stTabs [aria-selected="true"] p {color:#F25C05 !important;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#F25C05 !important;}
section[data-testid="stSidebar"] .stButton button {width:100%;}
</style>
""", unsafe_allow_html=True)


def card(label, value, delta=None, orange=False):
    cls = ""
    if delta is not None and isinstance(delta, str) and delta[:1] in "+-":
        cls = "pos" if delta.startswith("+") else "neg"
    d = f'<div class="dlt {cls}">{delta}</div>' if delta else ""
    return f'<div class="card {"o" if orange else ""}"><div class="lbl">{label}</div><div class="val">{value}</div>{d}</div>'


# --------------------------------------------------------------------------
# 1. CONSTANTS, PRESETS, EXPLANATIONS, PEER MAP
# --------------------------------------------------------------------------
DEFAULT_PEERS = {"NKE": "ADS.DE, PUMA.DE, DECK, ONON, LULU, AS"}
KNOWN_NAMES = {"NKE": ("NIKE, Inc.", "Consumer Cyclical", "Footwear & Accessories")}
PHASE2 = {"growth": [-1.59, 4.70, 4.70, 3.26, 3.26],
          "margin": [7.321017182, 8.626545316, 8.778371510, 9.087953208, 9.390908032],
          "rev0": 46398.0, "bv0": 14865.0, "shares": 1479.8, "payout": 62.32,
          "rf": 4.44, "beta": 1.11, "erp": 4.18, "target_price": 36.97}
REPORT = {"growth": {"Bear": [-10.0, -2.0, 2.0, 3.0, 3.0],
                     "Base": [-8.19132290184922, 1.36610412046372, 4.65742243009708, 4.88446334351913, 4.05815365507638],
                     "Bull": [-6.0, 5.0, 7.0, 6.0, 5.0]},
          "ebit": {"Bear": [5.0, 5.5, 7.0, 8.0, 9.0], "Base": [6.0, 7.5, 9.5, 11.0, 12.0], "Bull": [6.5, 9.0, 11.5, 13.0, 14.0]},
          "tg": {"Bear": 1.5, "Base": 2.15, "Bull": 2.75},
          "tax": [25.0, 23.0, 23.0, 23.0, 23.0], "one_off": [-300.0, -250.0, -200.0, -150.0, -100.0],
          "rev0": 46398.0, "bv0": 14865.0, "shares": 1484.699, "debt": 7890.0, "cash": 8370.0, "price": 34.71,
          "payout": 62.32, "rf": 4.44, "beta": 1.11, "erp": 4.18, "kd": 3.2, "tax_lr": 23.0,
          "da": 1.6, "capex": 1.7, "nwc": 5884 / 46398 * 100, "dps": 1.64, "comps": 31.7809,
          "values": {"Bear": 25.42, "Base": 40.64, "Bull": 56.87}, "tp": 38}
EXPLAIN = {
    "overview": "**Target price** = blended fair value today (DCF and trading comps), rolled forward 12 months at the cost "
                "of equity, minus dividends paid during those 12 months. **Total return** = price upside + dividend yield. "
                "The rating compares the total return with the Buy / Sell thresholds (sidebar).",
    "scenarios": "Three internally consistent sets of assumptions: **Bear** (things go wrong), **Base** (most likely) and "
                 "**Bull** (things go right). The probability-weighted value combines them.",
    "dcf": "A **DCF** values the business as the present value of future **free cash flows to the firm (FCFF)** - cash "
           "after taxes and investment, before paying lenders and shareholders - discounted at the **WACC** (weighted "
           "cost of equity and debt). The **terminal value** covers all years after the forecast. Enterprise value + net "
           "cash = equity value.",
    "ri": "**Residual income** = net income minus a charge for the equity used (cost of equity x opening book value). "
          "Value = book value today + present value of future residual income. Growth only adds value if the return on "
          "equity exceeds the cost of equity.",
    "sens": "Two-way tables move two inputs together. The **tornado chart** moves one driver at a time and ranks drivers "
            "by impact - the same idea as the Phase II sensitivity ranking.",
    "mc": "**Monte Carlo** draws thousands of random combinations of growth, margin, WACC and terminal growth around the "
          "selected scenario and shows the distribution of values instead of a single number.",
    "reverse": "A **reverse DCF** asks the opposite question: which margin, growth, WACC or terminal growth would justify "
               "today's share price? It shows what the market is pricing in.",
    "comps": "**Trading comparables** value the company with the multiples investors pay for similar listed companies. "
             "EV multiples are converted to equity value via net cash and diluted shares.",
    "compare": "Two companies side by side: key metrics, a quick automatic DCF and relative share-price performance.",
}
PEER_MAP = {
    "footwear & accessories": "NKE, ADS.DE, PUMA.DE, DECK, ONON, CROX, BIRK",
    "apparel manufacturing": "LULU, UAA, VFC, PVH, RL, AS",
    "apparel retail": "TJX, ROST, GAP, ANF, URBN, AEO",
    "leisure": "AS, HAS, MAT, PII, YETI",
    "consumer electronics": "AAPL, SONY, 005930.KS",
    "software - infrastructure": "MSFT, ORCL, ADBE, CRM, NOW",
    "software - application": "CRM, ADBE, INTU, SAP, NOW, WDAY",
    "semiconductors": "NVDA, AMD, AVGO, QCOM, INTC, TXN",
    "internet content & information": "GOOGL, META, PINS, SNAP, RDDT",
    "internet retail": "AMZN, BABA, JD, EBAY, ETSY, MELI",
    "drug manufacturers - general": "PFE, MRK, JNJ, ABBV, LLY, BMY, NOVN.SW",
    "beverages - non-alcoholic": "KO, PEP, KDP, MNST",
    "restaurants": "MCD, SBUX, YUM, CMG, QSR",
    "discount stores": "WMT, COST, TGT, DG, DLTR",
    "banks - diversified": "JPM, BAC, C, WFC",
    "auto manufacturers": "TSLA, TM, GM, F, VOW3.DE, BMW.DE, MBG.DE",
    "specialty retail": "HD, LOW, TSCO, BBY, ULTA",
    "household & personal products": "PG, CL, KMB, EL, CHD",
    "packaged foods": "NESN.SW, MDLZ, GIS, KHC, HSY",
}


def explain(key):
    if st.session_state.get("explain_mode"):
        st.info(EXPLAIN[key], icon="💡")


# --------------------------------------------------------------------------
# 2. DATA - robust snapshot (works even when Yahoo 'info' is empty)
# --------------------------------------------------------------------------
def _row(df, names):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index(ascending=False)
    return None


def _div(a, b):
    try:
        return a / b if (a is not None and b not in (None, 0) and not pd.isna(a) and not pd.isna(b)) else np.nan
    except TypeError:
        return np.nan


def _first(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and v != 0:
            return v
    return None


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def _ltm(qdf, adf, names):
    """Last-twelve-months value: sum of last 4 quarters if available, else latest fiscal year."""
    q = _row(qdf, names)
    if q is not None and len(q) >= 4:
        return float(q.iloc[:4].sum())
    a = _row(adf, names)
    return float(a.iloc[0]) if a is not None else None


def build_snapshot(tk, t, info, inc, bs, cf, qinc, prices):
    """Standardised company figures (units, not millions). 'info' is optional."""
    info = info or {}
    fi = _safe(lambda: t.fast_info, None)
    fget = lambda attr: _safe(lambda: getattr(fi, attr), None) if fi is not None else None
    price = _first(info.get("currentPrice"), info.get("regularMarketPrice"), fget("last_price"),
                   float(prices.iloc[-1]) if len(prices) else None)
    sh_cands = [info.get("impliedSharesOutstanding"), info.get("sharesOutstanding"), fget("shares")]
    for r in (_row(bs, ["Ordinary Shares Number", "Share Issued"]), _row(inc, ["Diluted Average Shares", "Basic Average Shares"])):
        if r is not None:
            sh_cands.append(float(r.iloc[0]))
    mcap_info = _first(info.get("marketCap"), fget("market_cap"))
    if mcap_info and price:
        sh_cands.append(mcap_info / price)
    sh_cands = [x for x in sh_cands if x and x > 0]
    shares = max(sh_cands) if sh_cands else None
    mcap = _first(info.get("marketCap"), fget("market_cap"), (price * shares) if (price and shares) else None)
    debt_bs = _row(bs, ["Total Debt"])
    cash_bs = _row(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"])
    debt = _first(info.get("totalDebt"), float(debt_bs.iloc[0]) if debt_bs is not None else None) or 0.0
    lease_bs = _row(bs, ["Capital Lease Obligations"])
    if lease_bs is None:
        parts = [_row(bs, ["Long Term Capital Lease Obligation"]), _row(bs, ["Current Capital Lease Obligation"])]
        lease_val = sum(float(x.iloc[0]) for x in parts if x is not None) if any(x is not None for x in parts) else 0.0
    else:
        lease_val = float(lease_bs.iloc[0])
    mino = _row(bs, ["Minority Interest"])
    cash = _first(info.get("totalCash"), float(cash_bs.iloc[0]) if cash_bs is not None else None) or 0.0
    rev_ltm = _first(info.get("totalRevenue"), _ltm(qinc, inc, ["Total Revenue", "Operating Revenue"]))
    ebitda_ltm = _first(info.get("ebitda"), _ltm(qinc, inc, ["EBITDA", "Normalized EBITDA"]))
    ni_ltm = _first(info.get("netIncomeToCommon"), _ltm(qinc, inc, ["Net Income Common Stockholders", "Net Income"]))
    eq = _row(bs, ["Stockholders Equity", "Common Stock Equity"])
    book = float(eq.iloc[0]) if eq is not None else None
    divs = _row(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"])
    dps = _first(info.get("dividendRate"), info.get("trailingAnnualDividendRate"),
                 _div(abs(float(divs.iloc[0])), shares) if (divs is not None and shares) else None) or 0.0
    known = KNOWN_NAMES.get(tk, (None, None, None))
    yr = prices[prices.index >= prices.index[-1] - pd.Timedelta(days=365)] if len(prices) else prices
    return {
        "ticker": tk, "info_ok": bool(info.get("marketCap") or info.get("longName")),
        "name": info.get("longName") or info.get("shortName") or known[0] or tk,
        "sector": info.get("sector") or known[1] or "", "industry": info.get("industry") or known[2] or "",
        "currency": info.get("currency") or fget("currency") or "USD",
        "fin_currency": info.get("financialCurrency") or info.get("currency") or fget("currency") or "USD",
        "price": price, "shares": shares, "mcap": mcap, "debt": debt, "cash": cash, "offline": False,
        "leases": lease_val, "debt_includes_leases": bool(lease_val) and debt >= lease_val,
        "minority": float(mino.iloc[0]) if mino is not None else 0.0, "eps_for_pe": None,
        "rev_ltm": rev_ltm, "ebitda_ltm": ebitda_ltm, "ni_ltm": ni_ltm, "book": book, "dps": dps,
        "beta": info.get("beta"),
        "low52": _first(info.get("fiftyTwoWeekLow"), fget("year_low"), float(yr.min()) if len(yr) else None),
        "high52": _first(info.get("fiftyTwoWeekHigh"), fget("year_high"), float(yr.max()) if len(yr) else None),
        "t_mean": info.get("targetMeanPrice"), "t_low": info.get("targetLowPrice"), "t_high": info.get("targetHighPrice"),
        "n_analysts": info.get("numberOfAnalystOpinions"), "rec_key": info.get("recommendationKey"),
        "fwd_eps": info.get("forwardEps"),
        "ratio_ev_rev": info.get("enterpriseToRevenue"), "ratio_ev_ebitda": info.get("enterpriseToEbitda"),
        "ratio_pe": info.get("trailingPE"), "ratio_pb": info.get("priceToBook"),
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(tk):
    t = yf.Ticker(tk)
    info = _safe(lambda: t.info, {}) or {}
    inc, bs, cf = _safe(lambda: t.income_stmt), _safe(lambda: t.balance_sheet), _safe(lambda: t.cashflow)
    qinc = _safe(lambda: t.quarterly_income_stmt)
    hist = _safe(lambda: t.history(period="5y"), pd.DataFrame())
    prices = hist["Close"].dropna() if isinstance(hist, pd.DataFrame) and "Close" in hist else pd.Series(dtype=float)
    data = {
        "revenue": _row(inc, ["Total Revenue", "Operating Revenue"]),
        "net_income": _row(inc, ["Net Income", "Net Income Common Stockholders"]),
        "ebit": _row(inc, ["EBIT", "Operating Income"]),
        "pretax": _row(inc, ["Pretax Income"]), "tax": _row(inc, ["Tax Provision"]),
        "equity": _row(bs, ["Stockholders Equity", "Common Stock Equity"]),
        "receivables": _row(bs, ["Accounts Receivable", "Receivables"]), "inventory": _row(bs, ["Inventory"]),
        "payables": _row(bs, ["Accounts Payable", "Payables"]),
        "dividends": _row(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
        "da": _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]),
        "capex": _row(cf, ["Capital Expenditure", "Purchase Of PPE"]),
        "cfo": _row(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]),
        "prices": prices,
    }
    data["snap"] = build_snapshot(tk, t, info, inc, bs, cf, qinc, prices)
    recs = _safe(lambda: t.recommendations)
    data["recs"] = recs if isinstance(recs, pd.DataFrame) and not recs.empty else None
    return data


OFFLINE_FILE = Path(__file__).parent / "nke_offline.json"
OFFLINE_TICKERS = {"NKE"} if OFFLINE_FILE.exists() else set()


@st.cache_data(show_spinner=False)
def load_offline(tk):
    """Offline data set (same structure as load_company) - used when Yahoo is unavailable or on request."""
    raw = json.loads(OFFLINE_FILE.read_text())
    st_ = raw["statements"]
    ser = lambda key: pd.Series({pd.Timestamp(k): float(v) * 1e6 for k, v in st_[key].items()}).sort_index(ascending=False)
    data = {k: ser(k) for k in ["revenue", "net_income", "ebit", "pretax", "tax", "equity", "receivables", "inventory",
                                "payables", "dividends", "da", "capex", "cfo"]}
    data["prices"] = pd.Series({pd.Timestamp(k): float(v) for k, v in raw["prices"].items()}).sort_index()
    sn, M_ = raw["snap"], 1e6
    data["snap"] = {
        "ticker": tk, "info_ok": True, "offline": True, "name": sn["name"], "sector": sn["sector"], "industry": sn["industry"],
        "currency": sn["currency"], "fin_currency": sn["fin_currency"], "price": sn["price"], "shares": sn["shares_m"] * M_,
        "mcap": sn["mcap_m"] * M_, "debt": sn["debt_m"] * M_, "cash": sn["cash_m"] * M_, "leases": sn["leases_m"] * M_,
        "debt_includes_leases": False, "minority": 0.0, "rev_ltm": sn["rev_ltm_m"] * M_, "ebitda_ltm": sn["ebitda_ltm_m"] * M_,
        "ni_ltm": sn["ni_ltm_m"] * M_, "eps_for_pe": sn["eps_for_pe"], "book": sn["book_m"] * M_, "dps": sn["dps"],
        "beta": sn["beta"], "low52": sn["low52"], "high52": sn["high52"], "t_mean": sn["t_mean"], "t_low": sn["t_low"],
        "t_high": sn["t_high"], "n_analysts": sn["n_analysts"], "rec_key": sn["rec_key"], "fwd_eps": sn["fwd_eps"],
        "ratio_ev_rev": None, "ratio_ev_ebitda": None, "ratio_pe": None, "ratio_pb": None}
    data["recs"] = None
    peers = pd.DataFrame(raw["peers"])
    for c in ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book", "Market cap (bn)", "EV (bn)"]:
        peers[c] = pd.to_numeric(peers[c], errors="coerce")
    data["offline_peers"] = peers
    data["as_of"], data["sources"], data["snap_notes"] = raw["as_of"], raw["sources"], sn.get("notes", {})
    return data


def multiples_from_snapshot(s):
    """EV = market cap + debt - cash; multiples on LTM figures. Uses Yahoo ratios if currencies differ (e.g. ADRs)."""
    ev = (s["mcap"] + s["debt"] - s["cash"]) if s["mcap"] else None
    if s["fin_currency"] != s["currency"]:
        return {"EV/Revenue": s["ratio_ev_rev"], "EV/EBITDA": s["ratio_ev_ebitda"], "P/E": s["ratio_pe"],
                "P/Book": s["ratio_pb"]}, ev
    return {"EV/Revenue": _div(ev, s["rev_ltm"]), "EV/EBITDA": _div(ev, s["ebitda_ltm"]),
            "P/E": _div(s["mcap"], s["ni_ltm"]), "P/Book": _div(s["mcap"], s["book"])}, ev


def peer_row(s):
    mult, ev = multiples_from_snapshot(s)
    return {"Ticker": s["ticker"], "Name": s["name"], "Currency": s["currency"], "Market cap (bn)": _div(s["mcap"], 1e9),
            "EV (bn)": _div(ev, 1e9), "Revenue LTM (bn)": _div(s["rev_ltm"], 1e9),
            "EBITDA LTM (bn)": _div(s["ebitda_ltm"], 1e9), **mult}


@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_multiples(tickers):
    rows, failed = [], []
    for tk in tickers:
        try:
            s = load_company(tk)["snap"]
            if not s["mcap"] or not s["rev_ltm"]:
                failed.append(tk)
                continue
            rows.append(peer_row(s))
        except Exception:
            failed.append(tk)
    df = pd.DataFrame(rows)
    for c in ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= 0, c] = np.nan
    return df, failed


def clip(x, lo, hi):
    x = 0.0 if x is None or pd.isna(x) else x
    return float(min(max(x, lo), hi))


def fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def money(x, d=2):
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:,.{d}f}" if abs(x) < 1e6 else f"{x:,.0f}"


def hist_ratios(d):
    """Historical ratios from the last fiscal year ($m), plus snapshot data."""
    M = 1e6
    s = d["snap"]
    rev, ni = d["revenue"] / M, d["net_income"] / M
    r0 = float(rev.iloc[0])
    divs = d["dividends"].abs() / M if d["dividends"] is not None else None
    if d["receivables"] is not None and d["payables"] is not None:
        inv0 = d["inventory"].iloc[0] if d["inventory"] is not None else 0
        nwc = _div((d["receivables"].iloc[0] + inv0 - d["payables"].iloc[0]) / M, r0)
    else:
        nwc = 0.10
    return {
        "rev0": r0, "bv0": float(d["equity"].iloc[0] / M), "base_year": int(rev.index[0].year), "price": s["price"],
        "shares": (s["shares"] / M) if s["shares"] else np.nan,
        "growth": rev.iloc[0] / rev.iloc[1] - 1 if len(rev) > 1 else 0.03, "margin": ni.iloc[0] / r0,
        "ebit": (d["ebit"].iloc[0] / M / r0) if d["ebit"] is not None else ni.iloc[0] / r0 / 0.79,
        "tax": _div(d["tax"].iloc[0], d["pretax"].iloc[0]) if (d["tax"] is not None and d["pretax"] is not None) else 0.21,
        "payout": _div(divs.iloc[0], ni.iloc[0]) if (divs is not None and ni.iloc[0] > 0) else 0.0,
        "da": _div(d["da"].iloc[0] / M, r0) if d["da"] is not None else 0.03,
        "capex": _div(abs(d["capex"].iloc[0]) / M, r0) if d["capex"] is not None else 0.03,
        "nwc": nwc, "cash": s["cash"] / M, "beta": s["beta"] or 1.0, "dps": s["dps"],
        "debt": (s["debt"] - (s.get("leases") or 0) if s.get("debt_includes_leases") else s["debt"]) / M,
        "leases": (s.get("leases") or 0) / M, "minority": (s.get("minority") or 0) / M,
    }


def quick_dcf(h, rf, erp, kd, years=5, tg=0.0215):
    ke_ = (rf + h["beta"] * erp) / 100
    t_ = clip(h["tax"], 0.05, 0.35)
    w_ = wacc_calc(ke_, kd / 100, t_, h["price"] * h["shares"], h["debt"])["wacc"]
    r = dcf_valuation(rev0=h["rev0"], growth=[clip(h["growth"], -0.10, 0.15)] * years,
                      ebit_margin=[clip(h["ebit"], -0.2, 0.5)] * years, tax=[t_] * years, da_pct=h["da"],
                      capex_pct=h["capex"], nwc_pct=clip(h["nwc"], -0.2, 0.6), one_off=[0.0] * years, wacc=w_, g=tg,
                      net_cash=h["cash"] - h["debt"], shares=h["shares"], base_year=h["base_year"])
    return r["per_share"], w_


# --------------------------------------------------------------------------
# 3. SIDEBAR - ticker & presets first
# --------------------------------------------------------------------------
ss = st.session_state
st.sidebar.markdown("## 📈 Equity Valuation")
st.sidebar.caption("Team 10 · WHU PT-MiM")
qp = st.query_params
ticker = st.sidebar.text_input("Stock ticker (Yahoo Finance)", value=qp.get("ticker", "NKE"),
                               help="e.g. NKE, AAPL, MSFT, ADS.DE").strip().upper()
ss.setdefault("offline", qp.get("offline") == "1")
if ticker in OFFLINE_TICKERS:
    st.sidebar.toggle("📦 Offline data set (9 Oct 2026)", key="offline",
                      help="Uses the data of the Team 10 report instead of live Yahoo data. "
                           "The app switches to it automatically if Yahoo is unavailable.")

fallback_used, load_error = False, None
if ticker in OFFLINE_TICKERS and ss.get("offline"):
    data = load_offline(ticker)
else:
    try:
        with st.spinner(f"Loading {ticker} from Yahoo Finance ..."):
            data = load_company(ticker)
    except Exception as e:
        data, load_error = None, e
    incomplete = data is None or data["revenue"] is None or data["equity"] is None or data["net_income"] is None
    if incomplete and ticker in OFFLINE_TICKERS:
        data, fallback_used = load_offline(ticker), True
    elif data is None:
        st.error(f"Could not load **{ticker}** ({load_error}). Yahoo sometimes limits requests - wait a minute and reload.")
        st.stop()
    elif incomplete:
        st.error(f"Yahoo Finance returned no annual financial statements for **{ticker}**. Check the ticker or try again later.")
        st.stop()

M = 1e6
SNAP = data["snap"]
H = hist_ratios(data)
rev_hist, ni_hist, eq_hist = data["revenue"] / M, data["net_income"] / M, data["equity"] / M
div_hist = data["dividends"].abs() / M if data["dividends"] is not None else None
base_year = H["base_year"]
live_price = SNAP["price"]
currency = SNAP["fin_currency"]
industry = (SNAP["industry"] or "").replace("—", " - ").strip()
suggested_peers = PEER_MAP.get(industry.lower(), "")
if suggested_peers:
    suggested_peers = ", ".join(p for p in [x.strip() for x in suggested_peers.split(",")] if p.upper() != ticker)


def make_scenarios(years):
    g0, m0 = clip(H["growth"] * 100, -10, 15), clip(H["margin"] * 100, -20, 40)
    e0, t0 = clip(H["ebit"] * 100, -20, 50), clip(H["tax"] * 100, 5, 35)
    return {s: pd.DataFrame({"Revenue growth %": [round(g0 + d, 2)] * years, "Net margin %": [round(m0 + d * 0.75, 2)] * years,
                             "EBIT margin %": [round(e0 + d, 2)] * years, "Tax rate %": [round(t0, 1)] * years,
                             "One-off cash costs ($m)": [0.0] * years}) for s, d in zip(SCENARIOS, [-2.0, 0.0, 2.0])}


def resize(df, years):
    if len(df) == years:
        return df
    if len(df) > years:
        return df.iloc[:years].reset_index(drop=True)
    extra = pd.DataFrame([df.iloc[-1].to_dict()] * (years - len(df)))
    extra["One-off cash costs ($m)"] = 0.0
    return pd.concat([df, extra], ignore_index=True)


WIDGET_KEYS = ["payout", "rf", "erp", "beta", "kd", "tax_lr", "da", "capex", "nwc", "years", "scenario",
               "rev0", "bv0", "shares", "debt", "cash", "ref_price", "peers", "w_bear", "w_base", "w_bull",
               "w_dcf", "tg_Bear", "tg_Base", "tg_Bull", "w_tp", "dps", "buy_thr", "sell_thr", "comps_override",
               "round_tp", "mc_n", "mc_sg", "mc_sm", "mc_sw", "mc_st", "explain_mode", "cmp_ticker",
               "view_val", "view_risk", "view_mkt", "leases", "minority", "pensions", "nonop", "lease_on",
               "mid_year", "stub_on", "val_date"]
for k in WIDGET_KEYS:
    if k in ss:
        ss[k] = ss[k]

load_key = f"{ticker}|{SNAP.get('offline')}"
if ss.get("loaded_ticker") != load_key:
    ss.update({
        "payout": round(clip(H["payout"] * 100, 0, 100), 1), "rf": 4.44, "erp": 4.18,
        "beta": round(clip(H["beta"], 0, 3), 2), "kd": 3.2 if ticker == "NKE" else 5.0,
        "tax_lr": round(clip(H["tax"] * 100, 5, 35), 1), "da": round(clip(H["da"] * 100, 0, 20), 2),
        "capex": round(clip(H["capex"] * 100, 0, 30), 2), "nwc": round(clip(H["nwc"] * 100, -20, 60), 2),
        "years": 5, "scenario": "Base", "rev0": H["rev0"], "bv0": H["bv0"],
        "shares": float(H["shares"]) if not pd.isna(H["shares"]) else 0.0,
        "debt": float(H["debt"]), "cash": float(H["cash"]), "ref_price": float(live_price) if live_price else 0.0,
        "peers": DEFAULT_PEERS.get(ticker, suggested_peers), "w_bear": 25, "w_base": 50, "w_bull": 25, "w_dcf": 50,
        "tg_Bear": 1.5, "tg_Base": 2.15, "tg_Bull": 2.75, "preset": "Automatic (last fiscal year)",
        "w_tp": 50, "dps": float(round(H["dps"], 2)), "buy_thr": 15.0, "sell_thr": 0.0, "comps_override": 0.0,
        "round_tp": True, "mc_n": 3000, "mc_sg": 2.0, "mc_sm": 1.5, "mc_sw": 0.75, "mc_st": 0.5,
        "cmp_ticker": "ADS.DE" if ticker == "NKE" else (suggested_peers.split(",")[0].strip() if suggested_peers else ""),
        "view_val": "DCF", "view_risk": "Tornado & sensitivity", "view_mkt": "Football field",
        "leases": float(H["leases"]), "minority": float(H["minority"]), "pensions": 0.0, "nonop": 0.0, "lease_on": False,
        "mid_year": False, "stub_on": False, "val_date": date.today(), "preset_key": "auto"})
    ss.setdefault("explain_mode", False)
    ss["scen"] = make_scenarios(5)
    ss["scen_ver"] = ss.get("scen_ver", 0) + 1
    ss["loaded_ticker"] = load_key
    ss["just_initialised"] = True


def load_report():
    c = REPORT
    ss.update({k: c[k] for k in ["rev0", "bv0", "shares", "debt", "cash", "payout", "rf", "beta", "erp", "kd",
                                 "tax_lr", "da", "capex", "nwc", "dps"]})
    ss.update({"ref_price": c["price"], "years": 5, "scenario": "Base", "comps_override": c["comps"], "w_tp": 50,
               "round_tp": True, "buy_thr": 15.0, "sell_thr": 0.0, "preset": "Team 10 report (9 Oct 2026)",
               "leases": 3091.0, "minority": 0.0, "pensions": 0.0, "nonop": 0.0, "lease_on": False, "mid_year": False,
               "stub_on": False, "preset_key": "report"})
    for s in SCENARIOS:
        ss[f"tg_{s}"] = c["tg"][s]
    ss["scen"] = {s: pd.DataFrame({"Revenue growth %": c["growth"][s],
                                   "Net margin %": [round(e * (1 - t / 100), 4) for e, t in zip(c["ebit"][s], c["tax"])],
                                   "EBIT margin %": c["ebit"][s], "Tax rate %": c["tax"],
                                   "One-off cash costs ($m)": c["one_off"]}) for s in SCENARIOS}
    ss["scen_ver"] = ss.get("scen_ver", 0) + 1


def load_phase2():
    c = PHASE2
    ss.update({"rev0": c["rev0"], "bv0": c["bv0"], "shares": c["shares"], "payout": c["payout"], "rf": c["rf"],
               "beta": c["beta"], "erp": c["erp"], "years": 5, "scenario": "Base", "tg_Base": 2.15,
               "preset": "Course Phase II check", "view_val": "Residual income", "preset_key": "phase2"})
    base = resize(ss["scen"]["Base"], 5).copy()
    base["Revenue growth %"], base["Net margin %"] = c["growth"], c["margin"]
    ss["scen"]["Base"] = base
    ss["scen_ver"] = ss.get("scen_ver", 0) + 1


def reset_auto():
    ss["loaded_ticker"] = None
    ss["force_auto"] = True


def use_suggested():
    ss["peers"] = suggested_peers


first_run_for_ticker = ss.pop("just_initialised", False)
force_auto = ss.pop("force_auto", False) if first_run_for_ticker else False
if not ss.get("qp_applied"):
    ss["qp_applied"] = True
    if ticker == "NKE" and qp.get("preset") == "phase2":
        load_phase2()
    elif ticker == "NKE" and qp.get("preset") != "auto":
        load_report()          # Nike opens with the Team 10 report assumptions by default
    if qp.get("scenario") in SCENARIOS:
        ss["scenario"] = qp.get("scenario")
elif first_run_for_ticker and ticker == "NKE" and not force_auto:
    load_report()


def export_config():
    cfg = {"app": "Team 10 Equity Valuation", "ticker": ticker, "preset": ss.get("preset"), "saved": str(date.today()),
           "settings": {k: (ss[k].isoformat() if isinstance(ss.get(k), date) else ss[k]) for k in WIDGET_KEYS if k in ss},
           "scenarios": {s: ss["scen"][s].to_dict(orient="list") for s in SCENARIOS}}
    return json.dumps(cfg, indent=1, default=str).encode()


def import_config():
    f = ss.get("cfg_upload")
    if f is None:
        return
    try:
        cfg = json.loads(f.getvalue())
        if cfg.get("ticker") != ticker:
            ss["cfg_msg"] = f"File is for {cfg.get('ticker')} - enter that ticker first, then load the file again."
            return
        for k, v in cfg.get("settings", {}).items():
            if k == "val_date":
                v = date.fromisoformat(v)
            if k in WIDGET_KEYS:
                ss[k] = v
        ss["scen"] = {s: pd.DataFrame(cfg["scenarios"][s]) for s in SCENARIOS}
        ss["scen_ver"] = ss.get("scen_ver", 0) + 1
        ss["preset"] = f"{cfg.get('preset')} (loaded from file)"
        ss["cfg_msg"] = "Assumptions loaded."
    except Exception as e:
        ss["cfg_msg"] = f"Could not read the file: {e}"


def make_link():
    st.query_params.update({"ticker": ticker, "preset": ss.get("preset_key", "auto"), "scenario": ss.scenario,
                            "offline": "1" if ss.get("offline") else "0"})
    ss["link_msg"] = True


if ticker == "NKE":
    st.sidebar.markdown("**Load a Team 10 model**")
    b1, b2 = st.sidebar.columns(2)
    b1.button("📄 Report", on_click=load_report, help="Sell-side report: DCF Bear 25.42 / Base 40.64 / Bull 56.87, target $38")
    b2.button("✅ Phase II", on_click=load_phase2, help="Course residual income model: $36.97")
st.sidebar.button("↺ Reset to automatic assumptions", on_click=reset_auto)
st.sidebar.radio("Scenario", SCENARIOS, horizontal=True, key="scenario")
years = st.sidebar.slider("Forecast period (years)", 3, 10, key="years")
for s in SCENARIOS:
    ss["scen"][s] = resize(ss["scen"][s], years)
st.sidebar.toggle("💡 Explain mode", key="explain_mode", help="Short explanations of every method")

with st.sidebar.expander("Discount rates & terminal growth"):
    st.number_input("Risk-free rate (%)", 0.0, 15.0, step=0.01, key="rf")
    st.number_input("Beta", 0.0, 3.0, step=0.01, key="beta",
                    help="Yahoo beta not available - default 1.00, please check" if SNAP["beta"] is None else f"Yahoo beta {SNAP['beta']}")
    st.number_input("Equity risk premium (%)", 0.0, 15.0, step=0.01, key="erp")
    st.number_input("Pre-tax cost of debt (%)", 0.0, 20.0, step=0.1, key="kd")
    st.number_input("Tax rate for WACC (%)", 0.0, 50.0, step=0.5, key="tax_lr")
    st.caption("Terminal growth (%)")
    tg_cols = st.columns(3)
    for col, s in zip(tg_cols, SCENARIOS):
        col.number_input(s, -2.0, 5.0, step=0.05, key=f"tg_{s}")
with st.sidebar.expander("Operating assumptions"):
    st.slider("Dividend payout ratio (%)", 0.0, 100.0, step=0.1, key="payout", help=f"Last FY: {fmt_pct(H['payout'])}")
    st.number_input("D&A (% of revenue)", 0.0, 30.0, step=0.1, key="da", help=f"Last FY: {fmt_pct(H['da'])}")
    st.number_input("Capex (% of revenue)", 0.0, 40.0, step=0.1, key="capex", help=f"Last FY: {fmt_pct(H['capex'])}")
    st.number_input("Net working capital (% of revenue)", -30.0, 80.0, step=0.1, key="nwc",
                    help="Receivables + inventory - payables")
with st.sidebar.expander("Target price & rating"):
    st.slider("DCF weight in target (%)", 0, 100, key="w_tp", help="Remainder = trading comps")
    st.number_input("Dividends per share, next 12m", 0.0, 100.0, step=0.01, key="dps")
    st.number_input("BUY if total return above (%)", -50.0, 100.0, step=1.0, key="buy_thr")
    st.number_input("SELL if total return below (%)", -50.0, 100.0, step=1.0, key="sell_thr")
    st.number_input("Comps value override (0 = live)", 0.0, 100000.0, step=0.01, key="comps_override")
    st.checkbox("Round target to whole number", key="round_tp")
with st.sidebar.expander("Peers for trading comps"):
    st.text_input("Peer tickers (comma separated)", key="peers")
    if suggested_peers:
        st.caption(f"Suggested ({SNAP['industry']}): {suggested_peers}")
        st.button("Use suggested peers", on_click=use_suggested)
with st.sidebar.expander("Starting values (editable)"):
    st.number_input(f"Revenue FY{base_year} (m)", key="rev0", step=100.0)
    st.number_input(f"Book equity FY{base_year} (m)", key="bv0", step=100.0)
    st.number_input("Diluted shares (m)", key="shares", step=1.0)
    st.number_input("Total debt (m)", key="debt", step=100.0, help="Yahoo total debt may include lease liabilities")
    st.number_input("Cash & short-term investments (m)", key="cash", step=100.0)
    st.number_input("Share price", key="ref_price", step=0.01, help="Default = latest Yahoo close")
with st.sidebar.expander("Valuation conventions & equity bridge"):
    st.checkbox("Mid-year discounting", key="mid_year", help="Cash flows arrive in the middle of each year")
    st.checkbox("Discount from valuation date (stub period)", key="stub_on",
                help="Year-1 cash flow is scaled to the part of the fiscal year still ahead")
    st.date_input("Valuation date", key="val_date", disabled=not ss.get("stub_on"))
    st.checkbox("Treat lease liabilities as debt", key="lease_on",
                help="Only consistent if lease costs are excluded from EBIT. Default: leases are operating costs.")
    st.number_input("Lease liabilities (m)", key="leases", step=100.0)
    st.number_input("Minority interests (m)", key="minority", step=10.0)
    st.number_input("Pensions & other debt-like items (m)", key="pensions", step=10.0)
    st.number_input("Non-operating assets (m)", key="nonop", step=10.0, help="e.g. investments in associates")
with st.sidebar.expander("Save, load & share"):
    st.download_button("💾 Save assumptions (.json)", data=export_config(), file_name=f"{ticker}_assumptions.json",
                       mime="application/json")
    st.file_uploader("Load assumptions (.json)", type=["json"], key="cfg_upload", on_change=import_config)
    if ss.get("cfg_msg"):
        st.caption(ss["cfg_msg"])
    st.button("🔗 Put settings in page link", on_click=make_link)
    if ss.get("link_msg"):
        st.caption("Done - copy the address bar URL. It reopens this ticker, preset and scenario. "
                   "Individual edits are kept via the .json file.")

# --------------------------------------------------------------------------
# 4. INPUT SANITY CHECKS
# --------------------------------------------------------------------------
problems = []
if not ss.shares or ss.shares <= 0:
    problems.append("the number of shares")
if not ss.ref_price or ss.ref_price <= 0:
    problems.append("the share price")
if problems:
    st.error(f"Yahoo did not return {' and '.join(problems)} for {ticker}. Please enter the value under "
             "'Starting values' in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------
# 5. HERO PLACEHOLDER + SCENARIO EDITOR (before calculations)
# --------------------------------------------------------------------------
hero = st.empty()
notice = st.empty()
with st.expander(f"✏️ Edit assumptions - {ss.scenario} case  ·  set: {ss.get('preset')}", expanded=False):
    st.caption("Values in %. Net margin drives residual income; EBIT margin and tax rate drive the DCF. One-off cash costs "
               "(negative, e.g. restructuring) reduce cash flow but are excluded from the terminal value. Without a preset, "
               "Bear / Bull = Base -/+ 2pp. Switch the scenario in the sidebar to edit another case.")
    view_df = ss["scen"][ss.scenario].copy()
    view_df.index = [f"FY{base_year + i + 1}" for i in range(len(view_df))]
    edited = st.data_editor(view_df, key=f"ed_{ss.scenario}_{years}_{ss.scen_ver}_{ticker}")
    if edited.isna().any().any():
        st.error("Please fill in every cell of the table.")
        st.stop()
    ss["scen"][ss.scenario] = edited.reset_index(drop=True).astype(float)

# --------------------------------------------------------------------------
# 6. CALCULATIONS
# --------------------------------------------------------------------------
ke = (ss.rf + ss.beta * ss.erp) / 100
price = ss.ref_price
lease_debt = ss.leases if ss.lease_on else 0.0
w = wacc_calc(ke, ss.kd / 100, ss.tax_lr / 100, price * ss.shares, ss.debt + lease_debt)
wacc = w["wacc"]
net_cash = ss.cash - ss.debt - lease_debt - ss.minority - ss.pensions + ss.nonop   # equity bridge for the DCF
fin_net_cash = ss.cash - ss.debt                                                    # comps (EV definition of peers)
stub = 1.0
if ss.stub_on:
    next_fye = rev_hist.index[0] + pd.DateOffset(years=1)
    stub = clip((next_fye - pd.Timestamp(ss.val_date)).days / 365.0, 0.01, 1.0)
res = {}
try:
    for s in SCENARIOS:
        df = ss["scen"][s]
        tg = ss[f"tg_{s}"] / 100
        ri_kwargs = dict(rev0=ss.rev0, bv0=ss.bv0, shares=ss.shares, growth=(df["Revenue growth %"] / 100).tolist(),
                         margin=(df["Net margin %"] / 100).tolist(), payout=ss.payout / 100, ke=ke, g=tg, base_year=base_year)
        dcf_kwargs = dict(rev0=ss.rev0, growth=(df["Revenue growth %"] / 100).tolist(),
                          ebit_margin=(df["EBIT margin %"] / 100).tolist(), tax=(df["Tax rate %"] / 100).tolist(),
                          da_pct=ss.da / 100, capex_pct=ss.capex / 100, nwc_pct=ss.nwc / 100,
                          one_off=df["One-off cash costs ($m)"].tolist(), wacc=wacc, g=tg, net_cash=net_cash,
                          shares=ss.shares, base_year=base_year, stub=stub, mid_year=bool(ss.mid_year))
        res[s] = {"ri": residual_income_valuation(**ri_kwargs), "dcf": dcf_valuation(**dcf_kwargs),
                  "ri_kwargs": ri_kwargs, "dcf_kwargs": dcf_kwargs}
except ValueError as e:
    st.error(f"{e} Adjust WACC or terminal growth in the sidebar.")
    st.stop()

act = res[ss.scenario]
ri, dcf = act["ri"], act["dcf"]
ri_value, dcf_value = ri["per_share"], dcf["per_share"]

# Trading comps (main company metrics from the same robust snapshot)
peer_list = tuple(p.strip().upper() for p in ss.peers.split(",") if p.strip() and p.strip().upper() != ticker)
peers, failed_peers, comps_prices = pd.DataFrame(), [], pd.DataFrame()
if SNAP.get("offline"):
    peers = data["offline_peers"]
    peer_list = tuple(peers["Ticker"])
elif peer_list:
    with st.spinner("Loading peers ..."):
        peers, failed_peers = load_peer_multiples(peer_list)
if not peers.empty:
    comps_prices = implied_prices_from_comps(
        peers, revenue=(SNAP["rev_ltm"] or 0) / M, ebitda=(SNAP["ebitda_ltm"] or 0) / M,
        eps=SNAP.get("eps_for_pe") or _div(SNAP["ni_ltm"], ss.shares * M), bvps=ss.bv0 / ss.shares,
        net_debt=-fin_net_cash, shares=ss.shares)
live_comps = comps_prices["Mid"].mean() if not comps_prices.empty else np.nan
comps_value = ss.comps_override if ss.comps_override > 0 else live_comps

# Target price & rating
if pd.isna(comps_value):
    fair_value, comps_note, w_dcf_eff = dcf_value, "no comps - DCF only", 100
else:
    fair_value = ss.w_tp / 100 * dcf_value + (1 - ss.w_tp / 100) * comps_value
    comps_note, w_dcf_eff = ("override" if ss.comps_override > 0 else "live peer median"), ss.w_tp
tp_raw = fair_value * (1 + ke) - ss.dps
target = round(tp_raw) if ss.round_tp else tp_raw
upside = target / price - 1
div_yield = ss.dps / price
total_return = upside + div_yield
rating = "BUY" if total_return * 100 > ss.buy_thr else ("SELL" if total_return * 100 < ss.sell_thr else "HOLD")

# Risk analytics (active scenario)
k_act = act["dcf_kwargs"]
mc = monte_carlo(k_act, n=int(ss.mc_n), sd_growth=ss.mc_sg / 100, sd_margin=ss.mc_sm / 100,
                 sd_wacc=ss.mc_sw / 100, sd_g=ss.mc_st / 100)
pc = np.percentile(mc, [5, 10, 25, 50, 75, 90, 95])


def _val(**over):
    try:
        return dcf_valuation(**{**k_act, **over})["per_share"]
    except ValueError:
        return np.nan


rev_margin = solve_for_price(lambda x: _val(ebit_margin=[m + x for m in k_act["ebit_margin"]]), price, -0.30, 0.40)
rev_growth = solve_for_price(lambda x: _val(growth=[g + x for g in k_act["growth"]]), price, -0.30, 0.40)
rev_wacc = solve_for_price(lambda x: _val(wacc=x), price, k_act["g"] + 0.002, 0.30)
rev_tg = solve_for_price(lambda x: _val(g=x), price, -0.05, k_act["wacc"] - 0.002)
torn = tornado(k_act, dcf_value)
final_m = k_act["ebit_margin"][-1]

# --------------------------------------------------------------------------
# 7. HERO, DATA NOTICE, KPI CARDS
# --------------------------------------------------------------------------
data_source = (f"Offline data set (as of {data.get('as_of')})" if SNAP.get("offline")
               else "Yahoo Finance (live)" if SNAP["info_ok"] else "Yahoo Finance (price feed & statements)")
title = SNAP["name"] if SNAP["name"] != ticker else ticker
sub = " · ".join(x for x in [SNAP["sector"], SNAP["industry"], currency, f"Base year FY{base_year}",
                             f"Assumptions: {ss.get('preset')}", f"Data: {data_source}"] if x)
hero.markdown(f"""
<div class="hero">
  <h1>{title} <span style="color:#F25C05">({ticker})</span></h1>
  <div class="sub">{sub}</div>
  <div class="row">
    <span class="badge {rating.lower()}">{rating}</span>
    <span class="kpi">12-month target<b>{money(target)}</b></span>
    <span class="kpi">Share price<b>{money(price)}</b></span>
    <span class="kpi">Upside<b>{upside:+.1%}</b></span>
    <span class="kpi">Dividend yield<b>{div_yield:.1%}</b></span>
    <span class="kpi">Total return<b>{total_return:+.1%}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

notes = []
if fallback_used:
    notes.append("Yahoo returned no data, so the app switched to the offline data set of the Team 10 report (9 Oct 2026).")
if not SNAP["info_ok"]:
    notes.append("Yahoo's company profile was not available, so shares, market value, debt and cash come from the price "
                 "feed and balance sheet. Beta defaults to 1.00 and analyst data is not shown - check the sidebar inputs.")
if failed_peers:
    notes.append(f"No usable data for peer(s): {', '.join(failed_peers)}.")
if dcf_value > 20 * price or dcf_value < 0:
    notes.append(f"The DCF value ({money(dcf_value)}) looks implausible versus the share price - check shares, debt and margins.")
if notes:
    notice.warning(" ".join(notes), icon="⚠️")
if ss.bv0 <= 0:
    st.warning("Book equity is negative - the residual income model is not meaningful here.")
if ticker == "NKE" and ss.get("preset", "").startswith("Team 10 report"):
    chk = " · ".join(f"{s} {res[s]['dcf']['per_share']:.2f} (Excel {REPORT['values'][s]:.2f})" for s in SCENARIOS)
    tp_chk = (f" · Target price {target:,.0f} (report {REPORT['tp']})" if ss.scenario == "Base"
              else f" · target shown for the {ss.scenario} case (report target uses Base)")
    st.success(f"Matches the Team 10 report - DCF: {chk}{tp_chk}", icon="✅")
if ticker == "NKE" and ss.get("preset", "").startswith("Course Phase II"):
    v = res["Base"]["ri"]["per_share"]
    st.success(f"Phase II check - residual income (Base): app {v:.2f} vs. Excel {PHASE2['target_price']:.2f} "
               f"({v / PHASE2['target_price'] - 1:+.2%})", icon="✅")

k1, k2, k3, k4 = st.columns(4)
k1.markdown(card(f"DCF value · {ss.scenario}", money(dcf_value), f"{dcf_value / price - 1:+.1%} vs. price"), unsafe_allow_html=True)
k2.markdown(card(f"Residual income · {ss.scenario}", money(ri_value), f"{ri_value / price - 1:+.1%} vs. price"), unsafe_allow_html=True)
k3.markdown(card("Trading comps", money(comps_value), None if pd.isna(comps_value) else f"{comps_value / price - 1:+.1%} vs. price"),
            unsafe_allow_html=True)
k4.markdown(card("Monte Carlo: P(value > price)", f"{(mc > price).mean():.0%}",
                 f"median {money(pc[3])} · WACC {wacc:.2%}", orange=True), unsafe_allow_html=True)
st.write("")


# --------------------------------------------------------------------------
# 8. EXCEL EXPORT (sidebar)
# --------------------------------------------------------------------------
def build_excel():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        pd.DataFrame({"Value": [ticker, SNAP["name"], str(date.today()), ss.get("preset"), ss.scenario, price, dcf_value,
                                ri_value, comps_value, fair_value, ke, ss.dps, target, upside, div_yield, total_return,
                                rating, wacc, float(pc[3]), float((mc > price).mean())]},
                     index=["Ticker", "Company", "Export date", "Assumption set", "Active scenario", "Share price",
                            "DCF value", "Residual income value", "Trading comps value", "Blended fair value",
                            "Cost of equity", "Dividends next 12m", "12-month target price", "Price upside",
                            "Dividend yield", "Total return", "Rating", "WACC", "Monte Carlo median",
                            "P(value > price)"]).to_excel(xw, sheet_name="Summary")
        pd.DataFrame({"Value": [ss.rf, ss.beta, ss.erp, ss.kd, ss.tax_lr, ss.payout, ss.da, ss.capex, ss.nwc, ss.rev0,
                                ss.bv0, ss.shares, ss.debt, ss.cash, ss.w_tp, ss.buy_thr, ss.sell_thr, ss.tg_Bear,
                                ss.tg_Base, ss.tg_Bull, ss.leases, ss.lease_on, ss.minority, ss.pensions, ss.nonop,
                                ss.mid_year, stub, data_source]},
                     index=["Risk-free rate %", "Beta", "Equity risk premium %", "Cost of debt %", "Tax rate (WACC) %",
                            "Payout ratio %", "D&A % revenue", "Capex % revenue", "NWC % revenue", "Revenue base year",
                            "Book equity base year", "Diluted shares (m)", "Debt", "Cash", "DCF weight in target %",
                            "Buy threshold %", "Sell threshold %", "Terminal g Bear %", "Terminal g Base %",
                            "Terminal g Bull %", "Lease liabilities", "Leases treated as debt", "Minority interests",
                            "Pensions & other", "Non-operating assets", "Mid-year discounting", "Stub (years)",
                            "Data source"]).to_excel(xw, sheet_name="Inputs")
        for s in SCENARIOS:
            a = ss["scen"][s].copy()
            a.index = [f"FY{base_year + i + 1}" for i in range(len(a))]
            a.to_excel(xw, sheet_name=f"Assumptions {s}")
            res[s]["dcf"]["table"].to_excel(xw, sheet_name=f"DCF {s}", index=False)
            res[s]["ri"]["table"].to_excel(xw, sheet_name=f"RI {s}", index=False)
        if not peers.empty:
            peers.to_excel(xw, sheet_name="Comps peers", index=False)
            comps_prices.to_excel(xw, sheet_name="Comps implied", index=False)
        torn.to_excel(xw, sheet_name="Tornado")
        pd.DataFrame({"Percentile": [5, 10, 25, 50, 75, 90, 95], "Value per share": pc}).to_excel(
            xw, sheet_name="Monte Carlo", index=False)
    return buf.getvalue()


st.sidebar.divider()
try:
    st.sidebar.download_button("⬇️ Export to Excel", data=build_excel(), file_name=f"{ticker}_valuation_{date.today():%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
except Exception as e:
    st.sidebar.caption(f"Excel export unavailable: {e}")


# --------------------------------------------------------------------------
# 9. HELPERS FOR TABS
# --------------------------------------------------------------------------
def fmt_table(df, pct_rows):
    tbl = df.set_index("Year").T
    show = tbl.copy().astype(object)
    for r in show.index:
        show.loc[r] = [f"{v:.2%}" if r in pct_rows else (f"{v:.4f}" if r == "Discount factor" else f"{v:,.0f}") for v in tbl.loc[r]]
    show.columns = [f"FY{c}" for c in show.columns]
    return show


def heat(grid, xlab, ylab, xs, ys, title):
    fig = go.Figure(go.Heatmap(z=grid, x=[f"{xlab} {v:.2%}" for v in xs], y=[f"{ylab} {v:.2%}" for v in ys],
                               text=[[f"{v:,.2f}" if not pd.isna(v) else "–" for v in row] for row in grid],
                               texttemplate="%{text}", colorscale="RdYlGn", zmid=price, showscale=False))
    fig.update_layout(title=title, height=380, yaxis_autorange="reversed")
    return fig


def football_field():
    dv = [res[s]["dcf"]["per_share"] for s in SCENARIOS]
    rv = [res[s]["ri"]["per_share"] for s in SCENARIOS]
    bars = [("DCF: Bear - Base - Bull", min(dv), dv[1], max(dv), BLACK),
            ("Monte Carlo: 10th - 90th pct", pc[1], pc[3], pc[5], "#444444"),
            ("Residual income: Bear - Base - Bull", min(rv), rv[1], max(rv), GREY)]
    for _, r in comps_prices.iterrows():
        bars.append((f"Comps: {r['Multiple']}", r["Low"], r["Mid"], r["High"], ORANGE))
    if SNAP["t_low"] and SNAP["t_high"]:
        bars.append(("Analyst price targets", SNAP["t_low"], SNAP["t_mean"] or SNAP["t_low"], SNAP["t_high"], LORANGE))
    if SNAP["low52"] and SNAP["high52"]:
        bars.append(("52-week trading range", SNAP["low52"], (SNAP["low52"] + SNAP["high52"]) / 2, SNAP["high52"], LGREY))
    ff = pd.DataFrame(bars, columns=["Method", "Low", "Mid", "High", "Color"])
    fig = go.Figure()
    fig.add_bar(y=ff["Method"], x=ff["High"] - ff["Low"], base=ff["Low"], orientation="h", marker_color=ff["Color"],
                text=[f"{l:,.0f} – {h:,.0f}" for l, h in zip(ff["Low"], ff["High"])], textposition="outside",
                hovertemplate="%{y}<extra></extra>")
    fig.add_scatter(y=ff["Method"], x=ff["Mid"], mode="markers",
                    marker=dict(color="white", symbol="line-ns-open", size=16, line_width=3), hoverinfo="skip")
    fig.add_vline(x=price, line_dash="dash", line_color=GREY, annotation_text=f"Price {price:,.2f}")
    fig.add_vline(x=target, line_color=ORANGE, line_width=2, annotation_text=f"Target {target:,.2f}",
                  annotation_position="bottom right")
    fig.update_layout(height=120 + 52 * len(ff), showlegend=False, yaxis_autorange="reversed", xaxis_title="Value per share")
    return fig, ff


def build_pdf():
    _, ffd = football_field()
    short = {"DCF: Bear - Base - Bull": "DCF scenarios", "Monte Carlo: 10th - 90th pct": "Monte Carlo 10-90%",
             "Residual income: Bear - Base - Bull": "Residual income", "52-week trading range": "52-week range"}
    df_a = ss["scen"][ss.scenario]
    cagr = np.prod(1 + df_a["Revenue growth %"] / 100) ** (1 / len(df_a)) - 1
    tot_w = (ss.w_bear + ss.w_base + ss.w_bull) or 1
    d = {"name": SNAP["name"], "ticker": ticker, "rating": rating, "target": target, "price": price, "upside": upside,
         "div_yield": div_yield, "total_return": total_return, "dcf": dcf_value, "ri": ri_value, "comps": comps_value,
         "fair_value": fair_value, "ke": ke, "wacc": wacc, "tg": k_act["g"], "scenario": ss.scenario,
         "preset": ss.get("preset"), "currency": currency, "data_source": data_source,
         "scen_rows": [[s, res[s]["dcf"]["per_share"], res[s]["ri"]["per_share"],
                        {"Bear": ss.w_bear, "Base": ss.w_base, "Bull": ss.w_bull}[s] / tot_w] for s in SCENARIOS],
         "ff_rows": [[short.get(r.Method, r.Method), r.Low, r.Mid, r.High] for r in ffd.itertuples()],
         "assumptions": [("Revenue CAGR (forecast)", f"{cagr:+.1%}"), ("Final-year EBIT margin", f"{final_m:.1%}"),
                         ("Terminal growth", f"{k_act['g']:.2%}"), ("Payout ratio (RI)", f"{ss.payout:.1f}%"),
                         ("Capex / D&A (% revenue)", f"{ss.capex:.1f}% / {ss.da:.1f}%"),
                         ("Net working capital (% revenue)", f"{ss.nwc:.1f}%"),
                         ("DCF weight in target", f"{w_dcf_eff}%")],
         "mc": {"p10": pc[1], "p50": pc[3], "p90": pc[5], "p_above": float((mc > price).mean())},
         "notes": (f"Valuation date convention: {'mid-year' if ss.mid_year else 'end-of-year'} discounting"
                   f"{f', stub {stub:.2f} years' if ss.stub_on else ', year 1 discounted one full year'}. "
                   f"Leases {'treated as debt' if ss.lease_on else 'treated as operating costs'}. "
                   f"Market-implied final-year EBIT margin: {'n/a' if rev_margin is None else f'{final_m + rev_margin:.1%}'}. "
                   f"Most important driver: {torn.iloc[0]['Driver'] if not torn.empty else 'n/a'}.")}
    return build_tearsheet(d)


try:
    st.sidebar.download_button("📄 One-page PDF summary", data=build_pdf(), file_name=f"{ticker}_tearsheet_{date.today():%Y%m%d}.pdf",
                               mime="application/pdf")
except Exception as e:
    st.sidebar.caption(f"PDF summary unavailable: {e}")

# --------------------------------------------------------------------------
# 10. TABS
# --------------------------------------------------------------------------
tab_ov, tab_sc, tab_val, tab_risk, tab_mkt, tab_data, tab_meth = st.tabs(
    ["🏁 Overview", "🎭 Scenarios", "💵 Valuation models", "🎲 Risk analysis", "🏢 Market & peers", "📁 Data",
     "📘 Methodology"])

# ---- Overview -------------------------------------------------------------
with tab_ov:
    explain("overview")
    left, right = st.columns([5, 4])
    with left:
        st.markdown("#### Target price derivation")
        st.table(pd.DataFrame([
            (f"DCF value ({ss.scenario} case)", money(dcf_value), f"{w_dcf_eff}%"),
            (f"Trading comps ({comps_note})", money(comps_value), f"{100 - w_dcf_eff}%"),
            ("Blended fair value today", money(fair_value), ""),
            (f"x (1 + cost of equity {ke:.2%})", money(fair_value * (1 + ke)), ""),
            ("- Dividends next 12 months", money(-ss.dps), ""),
            ("12-month target price" + (" (rounded)" if ss.round_tp else ""), money(target), ""),
            ("Expected total return", f"{total_return:+.1%}", ""),
            (f"Rating (Buy > {ss.buy_thr:.0f}%, Sell < {ss.sell_thr:.0f}%)", rating, "")],
            columns=["Step", "Value", "Weight"]).set_index("Step"))
    with right:
        st.markdown("#### Key insights")
        st.markdown(card("Market-implied final-year EBIT margin",
                         "n/a" if rev_margin is None else f"{final_m + rev_margin:.1%}", f"our {ss.scenario} case: {final_m:.1%}"),
                    unsafe_allow_html=True)
        st.write("")
        st.markdown(card("Monte Carlo range (10th - 90th percentile)", f"{money(pc[1], 0)} – {money(pc[5], 0)}",
                         f"median {money(pc[3])}"), unsafe_allow_html=True)
        st.write("")
        st.markdown(card("Most important value driver", torn.iloc[0]["Driver"].split(" (")[0] if not torn.empty else "n/a",
                         f"±{torn.iloc[0]['Range'] / 2:,.2f} per share" if not torn.empty else None, orange=True),
                    unsafe_allow_html=True)
    st.markdown("#### Valuation summary (football field)")
    fig, ff = football_field()
    st.plotly_chart(fig)

# ---- Scenarios ------------------------------------------------------------
with tab_sc:
    explain("scenarios")
    wc = st.columns(4)
    wc[0].number_input("Bear probability (%)", 0, 100, key="w_bear")
    wc[1].number_input("Base probability (%)", 0, 100, key="w_base")
    wc[2].number_input("Bull probability (%)", 0, 100, key="w_bull")
    wc[3].slider("DCF vs. RI weight (%)", 0, 100, key="w_dcf", help="Weight of DCF in the blended scenario value")
    wts = {"Bear": ss.w_bear, "Base": ss.w_base, "Bull": ss.w_bull}
    tot = sum(wts.values()) or 1
    rows = []
    for s in SCENARIOS:
        df = ss["scen"][s]
        cagr = np.prod(1 + df["Revenue growth %"] / 100) ** (1 / len(df)) - 1
        dv_, rv_ = res[s]["dcf"]["per_share"], res[s]["ri"]["per_share"]
        bl = ss.w_dcf / 100 * dv_ + (1 - ss.w_dcf / 100) * rv_
        rows.append({"Scenario": s, "Revenue CAGR": cagr, "Final EBIT margin": df["EBIT margin %"].iloc[-1] / 100,
                     "Terminal g": ss[f"tg_{s}"] / 100, "DCF": dv_, "Residual income": rv_, "Blended": bl,
                     "vs. price": bl / price - 1, "Probability": wts[s] / tot})
    sc = pd.DataFrame(rows).set_index("Scenario")
    pw = {c: (sc[c] * sc["Probability"]).sum() for c in ["DCF", "Residual income", "Blended"]}
    sc.loc["Probability-weighted"] = [np.nan, np.nan, np.nan, pw["DCF"], pw["Residual income"], pw["Blended"],
                                      pw["Blended"] / price - 1, 1.0]
    st.dataframe(sc.style.format({"Revenue CAGR": "{:+.1%}", "Final EBIT margin": "{:.1%}", "Terminal g": "{:.2%}",
                                  "DCF": "{:,.2f}", "Residual income": "{:,.2f}", "Blended": "{:,.2f}",
                                  "vs. price": "{:+.1%}", "Probability": "{:.0%}"}, na_rep="–"))
    bear_v, base_v = res["Bear"]["dcf"]["per_share"], res["Base"]["dcf"]["per_share"]
    if base_v != bear_v:
        st.caption(f"The share price sits {(price - bear_v) / (base_v - bear_v):.0%} of the way from the Bear to the Base DCF value.")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for s in SCENARIOS:
            v = res[s]["dcf"]["per_share"]
            fig.add_scatter(x=["Share price", "Scenario value"], y=[price, v], mode="lines+markers+text", name=s,
                            line=dict(color=SCEN_COLORS[s], width=3), text=["", f"{s} {v:,.2f}"], textposition="middle right")
        fig.add_scatter(x=["Share price", "Scenario value"], y=[price, target], mode="lines+markers+text", name="Target",
                        line=dict(color=LORANGE, width=2, dash="dash"), text=["", f"Target {target:,.2f}"],
                        textposition="bottom right")
        fig.update_layout(title="Share price vs. DCF value by scenario", height=380, showlegend=False,
                          xaxis=dict(range=[-0.3, 1.7]))
        st.plotly_chart(fig)
    with right:
        fig = go.Figure()
        for s in SCENARIOS:
            t = res[s]["dcf"]["table"]
            fig.add_scatter(x=t["Year"], y=t["EBIT margin"] * 100, name=s, mode="lines+markers",
                            line=dict(color=SCEN_COLORS[s], width=3))
        fig.update_layout(title="EBIT margin by scenario (%)", height=380, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig)

# ---- Valuation models -----------------------------------------------------
with tab_val:
    view = st.radio("Model", ["DCF", "Residual income"], horizontal=True, key="view_val", label_visibility="collapsed")
    if view == "DCF":
        explain("dcf")
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"#### Free cash flow to firm - {ss.scenario} case ({currency} m)")
            st.dataframe(fmt_table(dcf["table"], ["Revenue growth", "EBIT margin"]))
        with right:
            st.markdown("#### WACC")
            st.table(pd.DataFrame({"Value": [f"{ke:.2%}", f"{ss.kd:.2f}% / {w['kd_after_tax']:.2%}",
                                             f"{w['we']:.1%} / {w['wd']:.1%}", f"{wacc:.2%}", f"{ss[f'tg_{ss.scenario}']:.2f}%"]},
                                  index=["Cost of equity (CAPM)", "Cost of debt pre / after tax", "Equity / debt weight",
                                         "WACC", "Terminal growth"]))
            st.markdown("#### Equity bridge")
            bridge = [("PV of cash flows", dcf["pv_fcff"]), ("PV of terminal value", dcf["pv_tv"]),
                      ("Enterprise value", dcf["ev"]), ("+ Cash & ST investments", ss.cash), ("- Financial debt", -ss.debt)]
            if ss.lease_on:
                bridge.append(("- Lease liabilities", -ss.leases))
            for lab, v in [("- Minority interests", -ss.minority), ("- Pensions & other", -ss.pensions),
                           ("+ Non-operating assets", ss.nonop)]:
                if v:
                    bridge.append((lab, v))
            bridge.append(("Equity value", dcf["equity_value"]))
            st.table(pd.DataFrame({f"{currency} m": [v for _, v in bridge], "Per share": [v / ss.shares for _, v in bridge]},
                                  index=[k for k, _ in bridge]).style.format("{:,.2f}"))
            st.caption(f"Terminal value = {dcf['tv_share']:.0%} of enterprise value. Convention: "
                       f"{'mid-year' if ss.mid_year else 'end-of-year'} discounting"
                       f"{f', stub period {stub:.2f} years from {ss.val_date:%d %b %Y}' if ss.stub_on else ', year 1 = one full year'}.")
        t = dcf["table"]
        fig = go.Figure()
        fig.add_bar(x=t["Year"], y=t["NOPAT"], name="NOPAT", marker_color=GREY)
        fig.add_bar(x=t["Year"], y=t["FCFF"], name="Free cash flow", marker_color=BLACK)
        fig.add_bar(x=t["Year"], y=t["PV of FCFF"], name="Present value", marker_color=ORANGE)
        fig.update_layout(title=f"NOPAT, free cash flow and present value ({currency} m)", barmode="group", height=330,
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)
    else:
        explain("ri")
        left, right = st.columns([3, 2])
        with left:
            st.markdown(f"#### Residual income - {ss.scenario} case ({currency} m)")
            st.dataframe(fmt_table(ri["table"], ["Revenue growth", "Net margin", "ROE"]))
        with right:
            sh = ss.shares
            fig = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "total"],
                                         x=["Book value", "PV of RI", "PV of TV", "Value"],
                                         y=[ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, 0],
                                         text=[f"{v:,.2f}" for v in [ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, ri_value]],
                                         textposition="outside", increasing=dict(marker_color=BLACK),
                                         decreasing=dict(marker_color=ORANGE), totals=dict(marker_color=GREY)))
            fig.add_hline(y=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Price {price:,.2f}")
            fig.update_layout(title="Value bridge per share", height=380, showlegend=False)
            st.plotly_chart(fig)
            st.caption(f"Cost of equity {ke:.2%}; payout ratio {ss.payout:.1f}%.")

# ---- Risk analysis --------------------------------------------------------
with tab_risk:
    view = st.radio("Analysis", ["Tornado & sensitivity", "Monte Carlo", "Reverse DCF"], horizontal=True,
                    key="view_risk", label_visibility="collapsed")
    if view == "Tornado & sensitivity":
        explain("sens")
        tt = torn.iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=tt["Driver"], x=tt["Downside value"] - dcf_value, base=dcf_value, orientation="h", name="Downside",
                    marker_color=ORANGE, text=[f"{v:,.2f}" for v in tt["Downside value"]], textposition="outside")
        fig.add_bar(y=tt["Driver"], x=tt["Upside value"] - dcf_value, base=dcf_value, orientation="h", name="Upside",
                    marker_color=BLACK, text=[f"{v:,.2f}" for v in tt["Upside value"]], textposition="outside")
        fig.add_vline(x=dcf_value, line_color=GREY)
        fig.update_layout(title=f"Tornado chart - DCF value drivers ({ss.scenario}, base {dcf_value:,.2f})",
                          barmode="overlay", height=380, legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig)
        st.dataframe(torn.style.format({"Downside value": "{:,.2f}", "Upside value": "{:,.2f}", "Downside %": "{:+.1%}",
                                        "Upside %": "{:+.1%}", "Range": "{:,.2f}"}))
        tg_act = ss[f"tg_{ss.scenario}"] / 100
        g_vals = [round(tg_act + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
        w_vals = [round(wacc + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
        k_vals = [round(ke + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
        l, r = st.columns(2)
        l.plotly_chart(heat([[_val(wacc=wv, g=gv) for gv in g_vals] for wv in w_vals], "g", "WACC", g_vals, w_vals,
                            "DCF value: WACC x terminal growth"))
        r.plotly_chart(heat(sensitivity_grid(act["ri_kwargs"], k_vals, g_vals).values, "g", "ke", g_vals, k_vals,
                            "Residual income: cost of equity x terminal growth"))
        st.caption(f"Green = above the share price ({price:,.2f}), red = below.")
    elif view == "Monte Carlo":
        explain("mc")
        mcc = st.columns(5)
        mcc[0].select_slider("Simulations", options=[1000, 2000, 3000, 5000, 10000], key="mc_n")
        mcc[1].number_input("SD growth (pp)", 0.0, 10.0, step=0.25, key="mc_sg")
        mcc[2].number_input("SD EBIT margin (pp)", 0.0, 10.0, step=0.25, key="mc_sm")
        mcc[3].number_input("SD WACC (pp)", 0.0, 5.0, step=0.05, key="mc_sw")
        mcc[4].number_input("SD terminal g (pp)", 0.0, 3.0, step=0.05, key="mc_st")
        c_ = st.columns(4)
        c_[0].markdown(card("Median value", money(pc[3])), unsafe_allow_html=True)
        c_[1].markdown(card("10th - 90th percentile", f"{money(pc[1], 0)} – {money(pc[5], 0)}"), unsafe_allow_html=True)
        c_[2].markdown(card("P(value > price)", f"{(mc > price).mean():.0%}", orange=True), unsafe_allow_html=True)
        c_[3].markdown(card("P(value > target)", f"{(mc > target).mean():.0%}"), unsafe_allow_html=True)
        fig = go.Figure(go.Histogram(x=np.clip(mc, np.percentile(mc, 0.5), np.percentile(mc, 99.5)), nbinsx=60,
                                     marker_color=BLACK, opacity=0.85))
        fig.add_vline(x=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Price {price:,.2f}")
        fig.add_vline(x=target, line_color=ORANGE, annotation_text=f"Target {target:,.2f}", annotation_position="top left")
        fig.add_vline(x=pc[3], line_color=GREY, annotation_text=f"Median {pc[3]:,.2f}", annotation_position="bottom right")
        fig.update_layout(title=f"Distribution of DCF values ({len(mc):,} simulations, {ss.scenario} case)", height=380,
                          xaxis_title="Value per share", yaxis_title="Simulations", bargap=0.03)
        st.plotly_chart(fig)
        st.caption("Shifts apply to every forecast year. Draws with WACC - g < 0.5pp are discarded. Fixed random seed.")
    else:
        explain("reverse")
        st.markdown(f"#### What does a share price of {price:,.2f} imply? ({ss.scenario} case as starting point)")
        nf = "outside the tested range"
        st.table(pd.DataFrame([
            ("EBIT margin (all years shifted)", f"final year {final_m:.1%}",
             nf if rev_margin is None else f"final year {final_m + rev_margin:.1%} ({rev_margin * 100:+.1f}pp)"),
            ("Revenue growth (all years shifted)", f"avg {np.mean(k_act['growth']):.1%}",
             nf if rev_growth is None else f"avg {np.mean(k_act['growth']) + rev_growth:.1%} ({rev_growth * 100:+.1f}pp)"),
            ("WACC", f"{wacc:.2%}", nf if rev_wacc is None else f"{rev_wacc:.2%}"),
            ("Terminal growth", f"{k_act['g']:.2%}", nf if rev_tg is None else f"{rev_tg:.2%}")],
            columns=["Driver (others unchanged)", "Our assumption", "Implied by share price"]).set_index("Driver (others unchanged)"))
        if rev_margin is not None:
            st.info(f"The share price is consistent with a final-year EBIT margin of **{final_m + rev_margin:.1%}**, "
                    f"{abs(rev_margin) * 100:.1f}pp {'below' if rev_margin < 0 else 'above'} our {ss.scenario} case - "
                    f"{'our case is more optimistic than the market.' if rev_margin < 0 else 'our case is more conservative than the market.'}")
        margins = np.linspace(max(final_m - 0.08, -0.05), final_m + 0.08, 33)
        vals = [_val(ebit_margin=[m + (mm - final_m) for m in k_act["ebit_margin"]]) for mm in margins]
        fig = go.Figure(go.Scatter(x=margins * 100, y=vals, mode="lines", line=dict(color=BLACK, width=3)))
        fig.add_hline(y=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Share price {price:,.2f}")
        fig.add_vline(x=final_m * 100, line_color=GREY, annotation_text=f"Our {final_m:.1%}")
        fig.update_layout(title="DCF value vs. final-year EBIT margin", xaxis_title="Final-year EBIT margin (%)",
                          yaxis_title="Value per share", height=360)
        st.plotly_chart(fig)

# ---- Market & peers -------------------------------------------------------
with tab_mkt:
    view = st.radio("View", ["Football field", "Trading comps", "Analyst consensus", "Compare companies"], horizontal=True,
                    key="view_mkt", label_visibility="collapsed")
    if view == "Football field":
        fig, ff = football_field()
        st.plotly_chart(fig)
        st.dataframe(ff.drop(columns="Color").set_index("Method").style.format("{:,.2f}"))
    elif view == "Trading comps":
        explain("comps")
        st.caption(f"Peers: {', '.join(peer_list) if peer_list else 'none'}" +
                   (" (offline data set, multiples as of 7-9 Oct 2026; P/E = forward)" if SNAP.get("offline") else
                    " (edit in the sidebar)"))
        if peers.empty:
            st.info("No peer data yet - enter peer tickers in the sidebar under 'Peers for trading comps'.")
        else:
            stats = peers[["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]].agg(["median", "mean"])
            stats.index = ["Peer median", "Peer mean"]
            ref_row = pd.DataFrame([peer_row(SNAP)]).set_index("Ticker") if not SNAP.get("offline") else pd.DataFrame()
            st.dataframe(pd.concat([peers.set_index("Ticker"), stats, ref_row])
                         .style.format({"Market cap (bn)": "{:,.1f}", "EV (bn)": "{:,.1f}", "Revenue LTM (bn)": "{:,.1f}",
                                        "EBITDA LTM (bn)": "{:,.1f}", "EV/Revenue": "{:.2f}x", "EV/EBITDA": "{:.1f}x",
                                        "P/E": "{:.1f}x", "P/Book": "{:.2f}x"}, na_rep="–"))
            st.caption(f"Last row = {ticker} (reference). EV = market cap + total debt - cash; LTM = last four quarters.")
            if not comps_prices.empty:
                st.markdown(f"#### Implied value per share for {ticker}")
                st.dataframe(comps_prices.set_index("Multiple").style.format(
                    {"Peer 25th pct": "{:.2f}x", "Peer median": "{:.2f}x", "Peer 75th pct": "{:.2f}x",
                     "Low": "{:,.2f}", "Mid": "{:,.2f}", "High": "{:,.2f}"}))
                st.caption(f"Comps value = average of the medians: {money(live_comps)}. Range = peer 25th to 75th percentile.")
    elif view == "Analyst consensus":
        if not SNAP["t_mean"] and data.get("recs") is None:
            st.info("Yahoo did not return analyst data for this ticker on this server. For NKE the report uses: "
                    "consensus Hold (42 analysts), average target $38.04, range $19-94 (stockanalysis.com, 9 Oct 2026).")
        else:
            st.table(pd.DataFrame({"Value": [money(SNAP["t_mean"]),
                                             f"{money(SNAP['t_low'])} - {money(SNAP['t_high'])}" if SNAP["t_low"] else "n/a",
                                             str(SNAP["n_analysts"] or "n/a"),
                                             (SNAP["rec_key"] or "n/a").replace("_", " ").title(), money(SNAP["fwd_eps"]),
                                             f"{target / SNAP['t_mean'] - 1:+.1%}" if SNAP["t_mean"] else "n/a"]},
                                  index=["Mean target price", "Target range", "Number of analysts", "Consensus recommendation",
                                         "Forward EPS", "Our target vs. consensus"]))
            recs = data.get("recs")
            if recs is not None and {"strongBuy", "buy", "hold", "sell", "strongSell"}.issubset(recs.columns):
                r0 = recs.iloc[0]
                fig = go.Figure(go.Bar(x=["Strong buy", "Buy", "Hold", "Sell", "Strong sell"],
                                       y=[r0["strongBuy"], r0["buy"], r0["hold"], r0["sell"], r0["strongSell"]],
                                       marker_color=[BLACK, GREY, LGREY, LORANGE, ORANGE]))
                fig.update_layout(title="Analyst recommendations (current month)", height=300)
                st.plotly_chart(fig)
    else:
        explain("compare")
        st.text_input("Compare with ticker", key="cmp_ticker")
        other = ss.cmp_ticker.strip().upper()
        if other:
            try:
                d2 = load_company(other)
                ok = (d2["revenue"] is not None and d2["equity"] is not None and d2["net_income"] is not None
                      and bool(d2["snap"]["shares"]) and bool(d2["snap"]["price"]))
            except Exception as e:
                ok = False
                st.warning(f"Could not load {other}: {e}")
            if ok:
                H2, S2 = hist_ratios(d2), d2["snap"]
                v1, w1 = quick_dcf(H, ss.rf, ss.erp, ss.kd)
                v2, w2 = quick_dcf(H2, ss.rf, ss.erp, ss.kd)
                m1, _ = multiples_from_snapshot(SNAP)
                m2, _ = multiples_from_snapshot(S2)

                def colv(h, s_, m_, v, wv):
                    return [s_["name"], s_["currency"], money(h["price"]), money(_div(s_["mcap"], 1e9), 1),
                            f"{h['growth']:+.1%}", f"{h['ebit']:.1%}", f"{h['margin']:.1%}",
                            fmt_pct(_div(h["margin"] * h["rev0"], h["bv0"])), fmt_pct(_div(h["dps"], h["price"])),
                            *[("–" if m_[k] is None or pd.isna(m_[k]) else f"{m_[k]:.1f}x")
                              for k in ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]],
                            f"{h['beta']:.2f}", f"{wv:.2%}", money(v), f"{_div(v, h['price']) - 1:+.1%}"]
                idx = ["Name", "Currency", "Share price", "Market cap (bn)", "Revenue growth (last FY)", "EBIT margin (last FY)",
                       "Net margin (last FY)", "ROE (last FY)", "Dividend yield", "EV/Revenue", "EV/EBITDA", "P/E", "P/Book",
                       "Beta", "WACC (auto)", "Quick DCF value", "Quick DCF vs. price"]
                st.table(pd.DataFrame({ticker: colv(H, SNAP, m1, v1, w1), other: colv(H2, S2, m2, v2, w2)}, index=idx))
                st.caption("Quick DCF = automatic Base case for both: last fiscal year's growth and EBIT margin for 5 years, "
                           "terminal growth 2.15%, same risk-free rate, equity risk premium and cost of debt.")
                if len(data["prices"]) and len(d2["prices"]):
                    fig = go.Figure()
                    for name_, s_, c_ in [(ticker, data["prices"], BLACK), (other, d2["prices"], ORANGE)]:
                        fig.add_scatter(x=s_.index, y=s_ / s_.iloc[0] * 100, name=name_, line=dict(color=c_, width=2))
                    fig.update_layout(title="Share price, 5 years (rebased to 100, local currency)", height=360,
                                      legend=dict(orientation="h", y=-0.15))
                    st.plotly_chart(fig)
            else:
                st.warning(f"Yahoo returned incomplete data for {other}.")

# ---- Data -----------------------------------------------------------------
with tab_data:
    st.markdown(f"#### History vs. forecast ({ss.scenario} case)")
    hy = sorted(rev_hist.index)
    hlab = [f"FY{d.year}" for d in hy]
    ft = dcf["table"]
    flab = [f"FY{int(y)}E" for y in ft["Year"]]
    g1, g2 = st.columns(2)
    fig = go.Figure()
    fig.add_bar(x=hlab, y=[rev_hist[d] for d in hy], name="Reported", marker_color=GREY)
    fig.add_bar(x=flab, y=ft["Revenue"], name="Forecast", marker_color=BLACK)
    fig.update_layout(title=f"Revenue ({currency} m)", height=320, legend=dict(orientation="h", y=-0.2))
    g1.plotly_chart(fig)
    fig = go.Figure()
    if data["ebit"] is not None:
        fig.add_scatter(x=hlab, y=[_div(data["ebit"].get(d), data["revenue"].get(d)) * 100 for d in hy], name="Reported",
                        mode="lines+markers", line=dict(color=GREY, width=3))
    fig.add_scatter(x=flab, y=ft["EBIT margin"] * 100, name="Forecast", mode="lines+markers", line=dict(color=ORANGE, width=3))
    fig.update_layout(title="EBIT margin (%)", height=320, legend=dict(orientation="h", y=-0.2))
    g2.plotly_chart(fig)
    if data.get("cfo") is not None and data["capex"] is not None:
        fcf_h = [(_safe(lambda: data["cfo"][d], np.nan) - abs(_safe(lambda: data["capex"][d], np.nan))) / M for d in hy]
        fig = go.Figure()
        fig.add_bar(x=hlab, y=fcf_h, name="Reported FCF (CFO - capex)", marker_color=GREY)
        fig.add_bar(x=flab, y=ft["FCFF"], name="Forecast FCFF", marker_color=BLACK)
        fig.update_layout(title=f"Free cash flow ({currency} m)", height=320, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig)
    st.markdown("#### Historical financials (annual)")
    hist = pd.DataFrame({"Revenue": rev_hist, "Net income": ni_hist, "Book value of equity": eq_hist})
    if data["ebit"] is not None:
        hist["EBIT"] = data["ebit"] / M
    if div_hist is not None:
        hist["Dividends"] = div_hist
    hist = hist.sort_index()
    hist["Revenue growth"] = hist["Revenue"].pct_change()
    hist["Net margin"] = hist["Net income"] / hist["Revenue"]
    if "EBIT" in hist:
        hist["EBIT margin"] = hist["EBIT"] / hist["Revenue"]
    hist["ROE"] = hist["Net income"] / hist["Book value of equity"]
    if "Dividends" in hist:
        hist["Payout ratio"] = hist["Dividends"] / hist["Net income"]
    hist.index = [f"FY{d.year}" for d in hist.index]
    pct_cols = [c for c in ["Revenue growth", "Net margin", "EBIT margin", "ROE", "Payout ratio"] if c in hist]
    st.dataframe(hist.T.style.format("{:,.0f}", na_rep="–").format("{:.1%}", subset=pd.IndexSlice[pct_cols, :], na_rep="–"))
    st.markdown("#### Market data used")
    st.table(pd.DataFrame({"Value": [money(SNAP["price"]), money(_div(SNAP["shares"], 1e6), 1), money(_div(SNAP["mcap"], 1e9), 1),
                                     money(_div(SNAP["debt"], 1e6), 0), money(_div(SNAP["cash"], 1e6), 0),
                                     money(_div(SNAP["rev_ltm"], 1e6), 0), money(_div(SNAP["ebitda_ltm"], 1e6), 0),
                                     "Yahoo profile" if SNAP["info_ok"] else "price feed & statements (profile unavailable)"]},
                          index=["Latest price", "Shares (m)", "Market cap (bn)", "Total debt (m)", "Cash & ST investments (m)",
                                 "Revenue LTM (m)", "EBITDA LTM (m)", "Source"]))
    if len(data["prices"]):
        fig = go.Figure(go.Scatter(x=data["prices"].index, y=data["prices"].values, mode="lines", line=dict(color=BLACK)))
        fig.add_hline(y=target, line_dash="dot", line_color=ORANGE, annotation_text=f"Target {target:,.2f}")
        fig.update_layout(title="Share price - last 5 years", height=340)
        st.plotly_chart(fig)

# ---- Methodology ----------------------------------------------------------
with tab_meth:
    st.markdown("### Methodology & sources")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown("**1. Discounted cash flow (FCFF)**")
        st.latex(r"FCFF_t = EBIT_t(1-\tau_t) + D\&A_t - Capex_t - \Delta NWC_t + \text{one-offs}_t")
        st.latex(r"EV = \sum_{t=1}^{N}\frac{FCFF_t}{(1+WACC)^{t}} + \frac{FCFF_N^{*}(1+g)}{(WACC-g)(1+WACC)^{N}}")
        st.markdown("FCFF* excludes one-off cash costs. Equity value = EV + cash - financial debt "
                    "(- leases, minorities, pensions if selected) + non-operating assets; divided by diluted shares.")
        st.markdown("**2. Cost of capital**")
        st.latex(r"k_e = r_f + \beta \cdot ERP \qquad WACC = \tfrac{E}{D+E}k_e + \tfrac{D}{D+E}k_d(1-\tau)")
        st.markdown("Equity at market value (price x diluted shares), debt at book value.")
        st.markdown("**3. Residual income**")
        st.latex(r"RI_t = NI_t - k_e \cdot BV_{t-1} \qquad V_0 = BV_0 + \sum \frac{RI_t}{(1+k_e)^t} + \frac{RI_N(1+g)}{(k_e-g)(1+k_e)^N}")
        st.markdown("Clean-surplus roll-forward: BV_t = BV_{t-1} + NI_t - dividends (payout ratio).")
    with m2:
        st.markdown("**4. Trading comparables**")
        st.markdown("Peer EV = market cap + total debt - cash. Multiples: EV/Revenue and EV/EBITDA (LTM), P/E, P/Book. "
                    "Median, 25th and 75th percentile applied to the company; EV multiples converted via financial net cash. "
                    "Negative multiples are excluded.")
        st.markdown("**5. Target price & rating**")
        st.latex(r"TP_{12m} = \big(w\cdot DCF + (1-w)\cdot Comps\big)(1+k_e) - DPS_{12m}")
        st.markdown("Total return = TP / price - 1 + dividend yield. Buy / Sell thresholds set in the sidebar (default 15% / 0%).")
        st.markdown("**6. Risk analysis**")
        st.markdown("Tornado: one driver at a time. Monte Carlo: normal shocks to growth, EBIT margin (all years), WACC and "
                    "terminal growth; fixed seed. Reverse DCF: bisection search for the value that equals the share price.")
        st.markdown("**7. Conventions & limitations**")
        st.markdown("- Default: end-of-year discounting, year 1 = one full year (same as the Excel model). Mid-year and "
                    "valuation-date stub period are optional.\n"
                    "- Leases are treated as operating costs unless 'treat leases as debt' is selected.\n"
                    "- Automatic scenarios start from the last fiscal year; Bear / Bull = -/+ 2pp.\n"
                    "- Yahoo data can be incomplete or delayed; the profile endpoint is often unavailable on cloud servers. "
                    "Key figures are therefore rebuilt from the price feed and financial statements.")
    st.markdown("**Data sources**")
    src = ["Live: Yahoo Finance via the yfinance library (prices, financial statements, profile where available)."]
    if ticker in OFFLINE_TICKERS:
        src += [f"Offline data set ({load_offline(ticker)['as_of']}): " + "; ".join(load_offline(ticker)["sources"])]
    st.markdown("\n".join(f"- {x}" for x in src))
    st.markdown("**Validation:** the app reproduces the Team 10 Excel model - Phase II residual income $36.97, report DCF "
                "Bear $25.42 / Base $40.64 / Bull $56.87 and target price $38 (see test_valuation.py).")
    st.caption("Version 6 · Team 10 · WHU PT-MiM · built with AI assistance (Claude)")

st.caption("Team 10 · WHU PT-MiM · Built with AI assistance (Claude) · Academic project, not investment advice.")
