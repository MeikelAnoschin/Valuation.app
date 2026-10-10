"""
app.py - Equity Valuation App (Team 10, WHU PT-MiM)

Residual Income + DCF + Trading Comps, Bear/Base/Bull scenarios, target price & rating, Monte Carlo,
reverse DCF, tornado chart, analyst consensus, peer suggestions, company comparison and Excel export.

Run with:   streamlit run app.py
Needs:      valuation.py in the same folder, packages from requirements.txt, internet access (Yahoo Finance).
"""
import io
from datetime import date

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

st.set_page_config(page_title="Equity Valuation App", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------
# 0. LOOK & FEEL (black / orange)
# --------------------------------------------------------------------------
BLACK, ORANGE, GREY, LGREY = "#111111", "#F25C05", "#8A8A8A", "#C9C9C9"
SCENARIOS = ["Bear", "Base", "Bull"]
SCEN_COLORS = {"Bear": ORANGE, "Base": BLACK, "Bull": GREY}
pio.templates["team10"] = go.layout.Template(layout=dict(colorway=[BLACK, ORANGE, GREY, LGREY, "#F7A96B"],
                                                         font=dict(family="Helvetica, Arial, sans-serif")))
pio.templates.default = "plotly_white+team10"
st.markdown("""
<style>
div[data-testid="stMetric"] {background:#111111; padding:12px 14px; border-radius:6px; border-left:5px solid #F25C05;}
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p, div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:#FFFFFF !important;}
.stTabs [aria-selected="true"] p {color:#F25C05 !important; font-weight:700;}
.stTabs [data-baseweb="tab-highlight"] {background-color:#F25C05 !important;}
.hero {background:#111111; color:#FFFFFF; padding:18px 24px 14px 24px; border-radius:6px; border-bottom:6px solid #F25C05; margin-bottom:14px;}
.hero h1 {color:#FFFFFF; margin:0; padding:0; font-size:2.1rem; font-weight:800;}
.hero .sub {color:#CFCFCF; font-size:0.9rem; margin-top:2px;}
.hero .badge {display:inline-block; background:#F25C05; color:#FFFFFF; font-weight:800; padding:4px 12px; border-radius:4px; margin-right:10px;}
.hero .kpi {display:inline-block; margin-right:22px; font-size:0.95rem;}
.hero .kpi b {font-size:1.25rem;}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# 1. CONSTANTS, PRESETS, EXPLANATIONS, PEER MAP
# --------------------------------------------------------------------------
DEFAULT_PEERS = {"NKE": "ADS.DE, PUMA.DE, DECK, ONON, LULU, AS"}
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
    "summary": "**Target price** = blended fair value today (DCF and trading comps), rolled forward 12 months at the cost of "
               "equity, minus the dividends paid during those 12 months. **Total return** = price upside + dividend yield. "
               "The rating compares the total return with the Buy/Sell thresholds.",
    "scenarios": "Three internally consistent sets of assumptions. **Bear** = things go wrong, **Base** = most likely case, "
                 "**Bull** = things go right. The probability-weighted value combines them.",
    "dcf": "A **DCF** values the business as the present value of its future **free cash flows to the firm (FCFF)** - "
           "cash after taxes and investments, before paying lenders and shareholders - discounted at the **WACC**. "
           "The **terminal value** captures all years after the forecast. Enterprise value + net cash = equity value.",
    "wacc": "**WACC** = weighted average of the cost of equity (CAPM: risk-free rate + beta x equity risk premium) and the "
            "after-tax cost of debt, weighted by market value of equity and debt.",
    "ri": "**Residual income** = net income minus a charge for the equity capital used (cost of equity x opening book "
          "value). Value = book value today + present value of future residual income. It rewards growth only if "
          "return on equity exceeds the cost of equity.",
    "sens": "Sensitivity tables show how the value changes when two inputs move together. The **tornado chart** moves one "
            "driver at a time and ranks drivers by their impact - the same idea as the Phase II sensitivity ranking.",
    "mc": "**Monte Carlo** draws thousands of random combinations of growth, margin, WACC and terminal growth around the "
          "selected scenario (normal distribution, standard deviations set in the sliders) and shows the resulting "
          "distribution of values instead of a single point.",
    "reverse": "A **reverse DCF** asks the opposite question: which margin, growth, WACC or terminal growth would justify "
               "today's share price? It shows what the market is pricing in.",
    "comps": "**Trading comparables** value the company with the multiples investors pay for similar listed companies. "
             "EV multiples are converted to equity value via net cash and diluted shares.",
    "ff": "The **football field** compares the value ranges from each method with the share price and target price.",
    "compare": "Side-by-side comparison of two companies with key metrics, a quick automatic DCF (Base case from the last "
               "fiscal year) and relative share-price performance.",
}
PEER_MAP = {
    "footwear & accessories": "NKE, ADS.DE, PUMA.DE, DECK, ONON, CROX, BIRK",
    "apparel manufacturing": "LULU, UAA, VFC, PVH, RL, AS",
    "apparel retail": "TJX, ROST, GPS, ANF, URBN, AEO",
    "leisure": "AS, HAS, MAT, PII, YETI",
    "consumer electronics": "AAPL, SONY, 005930.KS, XIAOMI",
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
# 2. DATA FROM YAHOO FINANCE
# --------------------------------------------------------------------------
def _row(df, names):
    if df is None or df.empty:
        return None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index(ascending=False)
    return None


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(ticker):
    t = yf.Ticker(ticker)
    info = t.info or {}
    inc, bs, cf = t.income_stmt, t.balance_sheet, t.cashflow
    data = {
        "info": info,
        "revenue": _row(inc, ["Total Revenue", "Operating Revenue"]),
        "net_income": _row(inc, ["Net Income", "Net Income Common Stockholders"]),
        "ebit": _row(inc, ["EBIT", "Operating Income"]),
        "pretax": _row(inc, ["Pretax Income"]),
        "tax": _row(inc, ["Tax Provision"]),
        "equity": _row(bs, ["Stockholders Equity", "Common Stock Equity"]),
        "receivables": _row(bs, ["Accounts Receivable", "Receivables"]),
        "inventory": _row(bs, ["Inventory"]),
        "payables": _row(bs, ["Accounts Payable", "Payables"]),
        "dividends": _row(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
        "da": _row(cf, ["Depreciation And Amortization", "Depreciation Amortization Depletion"]),
        "capex": _row(cf, ["Capital Expenditure", "Purchase Of PPE"]),
    }
    hist = t.history(period="5y")
    data["prices"] = hist["Close"] if not hist.empty else pd.Series(dtype=float)
    try:
        recs = t.recommendations
        data["recs"] = recs if isinstance(recs, pd.DataFrame) and not recs.empty else None
    except Exception:
        data["recs"] = None
    return data


def _div(a, b):
    return a / b if (a is not None and b not in (None, 0) and not pd.isna(a) and not pd.isna(b)) else np.nan


def clip(x, lo, hi):
    x = 0.0 if x is None or pd.isna(x) else x
    return float(min(max(x, lo), hi))


def fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


def peer_row(tk, i):
    """Multiples from Yahoo raw data: EV = market cap + debt - cash."""
    mcap = i.get("marketCap")
    ev = (mcap + (i.get("totalDebt") or 0) - (i.get("totalCash") or 0)) if mcap else i.get("enterpriseValue")
    rev, ebitda, ni = i.get("totalRevenue"), i.get("ebitda"), i.get("netIncomeToCommon")
    px = i.get("currentPrice") or i.get("regularMarketPrice")
    same_ccy = (i.get("financialCurrency") or i.get("currency")) == i.get("currency")
    if same_ccy:
        ev_rev, ev_ebitda, pe = _div(ev, rev), _div(ev, ebitda), _div(mcap, ni)
    else:
        ev_rev, ev_ebitda, pe = i.get("enterpriseToRevenue"), i.get("enterpriseToEbitda"), i.get("trailingPE")
    return {"Ticker": tk, "Name": i.get("shortName", tk), "Currency": i.get("currency"),
            "Market cap (bn)": _div(mcap, 1e9), "EV (bn)": _div(ev, 1e9),
            "Revenue LTM (bn)": _div(rev, 1e9), "EBITDA LTM (bn)": _div(ebitda, 1e9),
            "EV/Revenue": ev_rev, "EV/EBITDA": ev_ebitda, "P/E": pe,
            "P/Book": _div(px, i.get("bookValue")) if same_ccy else i.get("priceToBook")}


@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_multiples(tickers):
    rows = []
    for tk in tickers:
        try:
            i = yf.Ticker(tk).info or {}
        except Exception:
            continue
        if i:
            rows.append(peer_row(tk, i))
    df = pd.DataFrame(rows)
    for c in ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= 0, c] = np.nan
    return df


def hist_ratios(d):
    """Key historical ratios from the last fiscal year (used for defaults and the comparison tab)."""
    M = 1e6
    info = d["info"]
    rev, ni = d["revenue"] / M, d["net_income"] / M
    r0 = float(rev.iloc[0])
    price = info.get("currentPrice") or info.get("regularMarketPrice") or (float(d["prices"].iloc[-1]) if len(d["prices"]) else np.nan)
    cands = [info.get("impliedSharesOutstanding"), info.get("sharesOutstanding")]
    if info.get("marketCap") and price:
        cands.append(info["marketCap"] / price)
    cands = [x for x in cands if x]
    divs = d["dividends"].abs() / M if d["dividends"] is not None else None
    if d["receivables"] is not None and d["payables"] is not None:
        inv0 = d["inventory"].iloc[0] if d["inventory"] is not None else 0
        nwc = _div((d["receivables"].iloc[0] + inv0 - d["payables"].iloc[0]) / M, r0)
    else:
        nwc = 0.10
    return {
        "rev0": r0, "bv0": float(d["equity"].iloc[0] / M), "base_year": int(rev.index[0].year), "price": price,
        "shares": max(cands) / M if cands else np.nan,
        "growth": rev.iloc[0] / rev.iloc[1] - 1 if len(rev) > 1 else 0.03,
        "margin": ni.iloc[0] / r0,
        "ebit": (d["ebit"].iloc[0] / M / r0) if d["ebit"] is not None else ni.iloc[0] / r0 / 0.79,
        "tax": _div(d["tax"].iloc[0], d["pretax"].iloc[0]) if (d["tax"] is not None and d["pretax"] is not None) else 0.21,
        "payout": _div(divs.iloc[0], ni.iloc[0]) if (divs is not None and ni.iloc[0] > 0) else 0.0,
        "da": _div(d["da"].iloc[0] / M, r0) if d["da"] is not None else 0.03,
        "capex": _div(abs(d["capex"].iloc[0]) / M, r0) if d["capex"] is not None else 0.03,
        "nwc": nwc,
        "debt": (info.get("totalDebt") or 0) / M, "cash": (info.get("totalCash") or 0) / M,
        "beta": info.get("beta") or 1.0,
        "dps": info.get("dividendRate") or info.get("trailingAnnualDividendRate") or 0.0,
    }


def quick_dcf(h, rf, erp, kd, years=5, tg=0.0215):
    """Automatic Base-case DCF from historical ratios (comparison tab)."""
    ke_ = (rf + h["beta"] * erp) / 100
    t_ = clip(h["tax"], 0.05, 0.35)
    w_ = wacc_calc(ke_, kd / 100, t_, h["price"] * h["shares"], h["debt"])["wacc"]
    r = dcf_valuation(rev0=h["rev0"], growth=[clip(h["growth"], -0.10, 0.15)] * years,
                      ebit_margin=[clip(h["ebit"], -0.2, 0.5)] * years, tax=[t_] * years, da_pct=h["da"],
                      capex_pct=h["capex"], nwc_pct=clip(h["nwc"], -0.2, 0.6), one_off=[0.0] * years, wacc=w_, g=tg,
                      net_cash=h["cash"] - h["debt"], shares=h["shares"], base_year=h["base_year"])
    return r["per_share"], w_


# --------------------------------------------------------------------------
# 3. LOAD TICKER
# --------------------------------------------------------------------------
ss = st.session_state
st.sidebar.title("📈 Valuation inputs")
ticker = st.sidebar.text_input("Stock ticker", value="NKE", help="Yahoo Finance symbol, e.g. NKE, AAPL, ADS.DE").strip().upper()

try:
    with st.spinner(f"Loading {ticker} from Yahoo Finance ..."):
        data = load_company(ticker)
except Exception as e:
    st.error(f"Could not load data for **{ticker}**: {e}. Yahoo sometimes rate-limits requests - wait a minute and reload.")
    st.stop()

info = data["info"]
if data["revenue"] is None or data["equity"] is None or data["net_income"] is None:
    st.error(f"Yahoo Finance has no complete annual financials for **{ticker}**. Check the ticker.")
    st.stop()

M = 1e6
H = hist_ratios(data)
rev_hist, ni_hist, eq_hist = data["revenue"] / M, data["net_income"] / M, data["equity"] / M
div_hist = data["dividends"].abs() / M if data["dividends"] is not None else None
base_year = H["base_year"]
live_price = H["price"]
currency = info.get("financialCurrency") or info.get("currency") or "USD"
industry = (info.get("industry") or "").replace("—", " - ").replace("—", " - ").strip()
suggested_peers = PEER_MAP.get(industry.lower(), "")
if suggested_peers:
    suggested_peers = ", ".join(p for p in [x.strip() for x in suggested_peers.split(",")] if p.upper() != ticker)


def make_scenarios(years):
    """Bear/Base/Bull from the last fiscal year: Base = history, Bear/Bull = -/+ 2pp."""
    g0, m0 = clip(H["growth"] * 100, -10, 15), clip(H["margin"] * 100, -20, 40)
    e0, t0 = clip(H["ebit"] * 100, -20, 50), clip(H["tax"] * 100, 5, 35)
    return {s: pd.DataFrame({"Revenue growth %": [round(g0 + d, 2)] * years,
                             "Net margin %": [round(m0 + d * 0.75, 2)] * years,
                             "EBIT margin %": [round(e0 + d, 2)] * years,
                             "Tax rate %": [round(t0, 1)] * years,
                             "One-off cash costs ($m)": [0.0] * years})
            for s, d in zip(SCENARIOS, [-2.0, 0.0, 2.0])}


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
               "round_tp", "mc_n", "mc_sg", "mc_sm", "mc_sw", "mc_st", "explain_mode", "cmp_ticker"]
for k in WIDGET_KEYS:
    if k in ss:
        ss[k] = ss[k]

if ss.get("loaded_ticker") != ticker:
    ss.update({
        "payout": round(clip(H["payout"] * 100, 0, 100), 1), "rf": 4.44, "erp": 4.18,
        "beta": round(clip(H["beta"], 0, 3), 2), "kd": 3.2 if ticker == "NKE" else 5.0,
        "tax_lr": round(clip(H["tax"] * 100, 5, 35), 1),
        "da": round(clip(H["da"] * 100, 0, 20), 2), "capex": round(clip(H["capex"] * 100, 0, 30), 2),
        "nwc": round(clip(H["nwc"] * 100, -20, 60), 2), "years": 5, "scenario": "Base",
        "rev0": H["rev0"], "bv0": H["bv0"], "shares": float(H["shares"]) if not np.isnan(H["shares"]) else 1.0,
        "debt": float(H["debt"]), "cash": float(H["cash"]),
        "ref_price": float(live_price) if live_price else 1.0,
        "peers": DEFAULT_PEERS.get(ticker, suggested_peers),
        "w_bear": 25, "w_base": 50, "w_bull": 25, "w_dcf": 50,
        "tg_Bear": 1.5, "tg_Base": 2.15, "tg_Bull": 2.75, "preset": "Auto (from last fiscal year)",
        "w_tp": 50, "dps": float(round(H["dps"], 2)), "buy_thr": 15.0, "sell_thr": 0.0, "comps_override": 0.0,
        "round_tp": True, "mc_n": 3000, "mc_sg": 2.0, "mc_sm": 1.5, "mc_sw": 0.75, "mc_st": 0.5,
        "cmp_ticker": "ADS.DE" if ticker == "NKE" else (suggested_peers.split(",")[0].strip() if suggested_peers else "")})
    ss.setdefault("explain_mode", False)
    ss["scen"] = make_scenarios(5)
    ss["scen_ver"] = ss.get("scen_ver", 0) + 1
    ss["loaded_ticker"] = ticker


def load_report():
    c = REPORT
    ss.update({k: c[k] for k in ["rev0", "bv0", "shares", "debt", "cash", "payout", "rf", "beta", "erp", "kd",
                                 "tax_lr", "da", "capex", "nwc", "dps"]})
    ss.update({"ref_price": c["price"], "years": 5, "scenario": "Base", "comps_override": c["comps"], "w_tp": 50,
               "round_tp": True, "buy_thr": 15.0, "sell_thr": 0.0, "preset": "Team 10 report (guidance-adjusted)"})
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
               "preset": "Course Phase II check (residual income)"})
    base = resize(ss["scen"]["Base"], 5).copy()
    base["Revenue growth %"] = c["growth"]
    base["Net margin %"] = c["margin"]
    ss["scen"]["Base"] = base
    ss["scen_ver"] = ss.get("scen_ver", 0) + 1


def use_suggested():
    ss["peers"] = suggested_peers


# --------------------------------------------------------------------------
# 4. SIDEBAR
# --------------------------------------------------------------------------
st.sidebar.toggle("💡 Explain mode", key="explain_mode", help="Shows short explanations of every method")
st.sidebar.radio("Scenario", SCENARIOS, horizontal=True, key="scenario",
                 help="Switch between Bear / Base / Bull. Edit the assumptions in the table on the main page.")
years = st.sidebar.slider("Forecast period (years)", 3, 10, key="years")
for s in SCENARIOS:
    ss["scen"][s] = resize(ss["scen"][s], years)

with st.sidebar.expander("Discount rates & terminal growth", expanded=True):
    st.number_input("Risk-free rate (%)", 0.0, 15.0, step=0.01, key="rf")
    st.number_input("Beta", 0.0, 3.0, step=0.01, key="beta", help=f"Yahoo beta: {info.get('beta', 'n/a')}")
    st.number_input("Equity risk premium (%)", 0.0, 15.0, step=0.01, key="erp")
    st.number_input("Pre-tax cost of debt (%)", 0.0, 20.0, step=0.1, key="kd")
    st.number_input("Long-run tax rate for WACC (%)", 0.0, 50.0, step=0.5, key="tax_lr")
    st.caption("Terminal growth by scenario (%)")
    tg_cols = st.columns(3)
    for col, s in zip(tg_cols, SCENARIOS):
        col.number_input(s, -2.0, 5.0, step=0.05, key=f"tg_{s}")
ke = (ss.rf + ss.beta * ss.erp) / 100

with st.sidebar.expander("Operating assumptions"):
    st.slider("Dividend payout ratio (%)", 0.0, 100.0, step=0.1, key="payout", help=f"Last FY: {fmt_pct(H['payout'])}")
    st.number_input("D&A (% of revenue)", 0.0, 30.0, step=0.1, key="da", help=f"Last FY: {fmt_pct(H['da'])}")
    st.number_input("Capex (% of revenue)", 0.0, 40.0, step=0.1, key="capex", help=f"Last FY: {fmt_pct(H['capex'])}")
    st.number_input("Net working capital (% of revenue)", -30.0, 80.0, step=0.1, key="nwc",
                    help="Approx. receivables + inventory - payables")

with st.sidebar.expander("Target price & rating"):
    st.slider("DCF weight in target price (%)", 0, 100, key="w_tp", help="Remainder = trading comps")
    st.number_input("Dividends per share, next 12 months", 0.0, 100.0, step=0.01, key="dps")
    st.number_input("BUY if total return above (%)", -50.0, 100.0, step=1.0, key="buy_thr")
    st.number_input("SELL if total return below (%)", -50.0, 100.0, step=1.0, key="sell_thr")
    st.number_input("Comps value override (0 = use live comps)", 0.0, 10000.0, step=0.01, key="comps_override")
    st.checkbox("Round target price to whole number", key="round_tp")

with st.sidebar.expander("Trading comps peers"):
    st.text_input("Peer tickers (comma separated)", key="peers")
    if suggested_peers:
        st.caption(f"Suggested for '{info.get('industry', '')}': {suggested_peers}")
        st.button("Use suggested peers", on_click=use_suggested)
    else:
        st.caption(f"No suggestion list for industry '{info.get('industry', 'n/a')}' - enter peers manually.")

with st.sidebar.expander("Starting values (from Yahoo, editable)"):
    st.number_input(f"Revenue FY{base_year} ({currency} m)", key="rev0", step=100.0)
    st.number_input(f"Book value of equity FY{base_year} ({currency} m)", key="bv0", step=100.0)
    st.number_input("Diluted shares (m)", key="shares", step=1.0)
    st.number_input("Total debt (m)", key="debt", step=100.0, help="Yahoo 'totalDebt' may include lease liabilities")
    st.number_input("Cash & short-term investments (m)", key="cash", step=100.0)
    st.number_input("Reference share price", key="ref_price", step=0.01, help="Default = live Yahoo price")

if ticker == "NKE":
    st.sidebar.subheader("Load Team 10 models")
    st.sidebar.button("📄 Team 10 report (DCF scenarios)", on_click=load_report,
                      help="Expected: Bear $25.42 / Base $40.64 / Bull $56.87, target price $38")
    st.sidebar.button("✅ Course Phase II check (RI)", on_click=load_phase2, help="Expected residual income value: $36.97")

# --------------------------------------------------------------------------
# 5. SCENARIO EDITOR (before calculations)
# --------------------------------------------------------------------------
hero = st.empty()
with st.expander(f"✏️ Scenario assumptions - editing: {ss.scenario} case", expanded=False):
    st.caption("Values in %. Unless a preset is loaded, Bear and Bull are generated as -/+ 2pp around the Base case. "
               "Net margin drives the residual income model; EBIT margin and tax rate drive the DCF. One-off cash costs "
               "(negative numbers) reduce FCFF but are excluded from the terminal value. Switch the scenario in the sidebar "
               "to edit another case.")
    view = ss["scen"][ss.scenario].copy()
    view.index = [f"FY{base_year + i + 1}" for i in range(len(view))]
    edited = st.data_editor(view, key=f"ed_{ss.scenario}_{years}_{ss.scen_ver}_{ticker}")
    if edited.isna().any().any():
        st.error("Please fill in every cell of the table.")
        st.stop()
    ss["scen"][ss.scenario] = edited.reset_index(drop=True).astype(float)

# --------------------------------------------------------------------------
# 6. CALCULATIONS
# --------------------------------------------------------------------------
price = ss.ref_price
w = wacc_calc(ke, ss.kd / 100, ss.tax_lr / 100, price * ss.shares, ss.debt)
wacc = w["wacc"]
net_cash = ss.cash - ss.debt
res = {}
try:
    for s in SCENARIOS:
        df = ss["scen"][s]
        tg = ss[f"tg_{s}"] / 100
        ri_kwargs = dict(rev0=ss.rev0, bv0=ss.bv0, shares=ss.shares, growth=(df["Revenue growth %"] / 100).tolist(),
                         margin=(df["Net margin %"] / 100).tolist(), payout=ss.payout / 100, ke=ke, g=tg,
                         base_year=base_year)
        dcf_kwargs = dict(rev0=ss.rev0, growth=(df["Revenue growth %"] / 100).tolist(),
                          ebit_margin=(df["EBIT margin %"] / 100).tolist(), tax=(df["Tax rate %"] / 100).tolist(),
                          da_pct=ss.da / 100, capex_pct=ss.capex / 100, nwc_pct=ss.nwc / 100,
                          one_off=df["One-off cash costs ($m)"].tolist(), wacc=wacc, g=tg, net_cash=net_cash,
                          shares=ss.shares, base_year=base_year)
        res[s] = {"ri": residual_income_valuation(**ri_kwargs), "dcf": dcf_valuation(**dcf_kwargs),
                  "ri_kwargs": ri_kwargs, "dcf_kwargs": dcf_kwargs}
except ValueError as e:
    st.error(str(e))
    st.stop()

act = res[ss.scenario]
ri, dcf = act["ri"], act["dcf"]
ri_value, dcf_value = ri["per_share"], dcf["per_share"]

# Trading comps
peer_list = tuple(p.strip().upper() for p in ss.peers.split(",") if p.strip() and p.strip().upper() != ticker)
peers, comps_prices = pd.DataFrame(), pd.DataFrame()
if peer_list:
    with st.spinner("Loading peer multiples ..."):
        peers = load_peer_multiples(peer_list)
    if not peers.empty:
        comps_prices = implied_prices_from_comps(
            peers, revenue=(info.get("totalRevenue") or 0) / M, ebitda=(info.get("ebitda") or 0) / M,
            eps=_div(info.get("netIncomeToCommon"), ss.shares * M) if info.get("netIncomeToCommon") else info.get("trailingEps"),
            bvps=ss.bv0 / ss.shares, net_debt=-net_cash, shares=ss.shares)
live_comps = comps_prices["Mid"].mean() if not comps_prices.empty else np.nan
comps_value = ss.comps_override if ss.comps_override > 0 else live_comps

# Target price & rating
if pd.isna(comps_value):
    fair_value, comps_note = dcf_value, "no comps available - DCF only"
else:
    fair_value = ss.w_tp / 100 * dcf_value + (1 - ss.w_tp / 100) * comps_value
    comps_note = "override" if ss.comps_override > 0 else "live peer median"
tp_raw = fair_value * (1 + ke) - ss.dps
target = round(tp_raw) if ss.round_tp else tp_raw
upside = target / price - 1
div_yield = ss.dps / price
total_return = upside + div_yield
rating = "BUY" if total_return * 100 > ss.buy_thr else ("SELL" if total_return * 100 < ss.sell_thr else "HOLD")

# Monte Carlo, reverse DCF, tornado (active scenario)
mc = monte_carlo(act["dcf_kwargs"], n=int(ss.mc_n), sd_growth=ss.mc_sg / 100, sd_margin=ss.mc_sm / 100,
                 sd_wacc=ss.mc_sw / 100, sd_g=ss.mc_st / 100)
k_act = act["dcf_kwargs"]


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
pc = np.percentile(mc, [5, 10, 25, 50, 75, 90, 95])

# --------------------------------------------------------------------------
# 7. HERO BANNER + KPI ROW
# --------------------------------------------------------------------------
hero.markdown(f"""
<div class="hero">
  <h1>{info.get('longName', ticker)} ({ticker})</h1>
  <div class="sub">{info.get('sector', '')} · {info.get('industry', '')} · {currency} · Base year FY{base_year} ·
  Assumption set: {ss.get('preset')} · Data: Yahoo Finance, {date.today():%d %b %Y}</div>
  <div style="margin-top:10px;">
    <span class="badge">{rating}</span>
    <span class="kpi">Target price <b>{target:,.2f}</b></span>
    <span class="kpi">Share price <b>{price:,.2f}</b></span>
    <span class="kpi">Upside <b>{upside:+.1%}</b></span>
    <span class="kpi">Total return <b>{total_return:+.1%}</b></span>
  </div>
</div>""", unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"DCF value ({ss.scenario})", f"{dcf_value:,.2f}", f"{dcf_value / price - 1:+.1%} vs. price")
c2.metric(f"Residual income ({ss.scenario})", f"{ri_value:,.2f}", f"{ri_value / price - 1:+.1%} vs. price")
c3.metric("Trading comps", "n/a" if pd.isna(comps_value) else f"{comps_value:,.2f}",
          None if pd.isna(comps_value) else f"{comps_value / price - 1:+.1%} vs. price")
c4.metric("WACC", f"{wacc:.2%}", f"cost of equity {ke:.2%}", delta_color="off")
c5.metric("P(value > price), Monte Carlo", f"{(mc > price).mean():.0%}", f"{len(mc):,} simulations", delta_color="off")

if ss.bv0 <= 0:
    st.warning("Book value of equity is negative - the residual income model is not meaningful here.")
if ticker == "NKE" and ss.get("preset", "").startswith("Team 10 report"):
    checks = " · ".join(f"{s}: {res[s]['dcf']['per_share']:.2f} (Excel {REPORT['values'][s]:.2f})" for s in SCENARIOS)
    st.info(f"**Report check:** DCF {checks} · Target price {target:,.0f} (report {REPORT['tp']})")
if ticker == "NKE" and ss.get("preset", "").startswith("Course Phase II"):
    v = res["Base"]["ri"]["per_share"]
    st.info(f"**Excel check (Phase II RI, Base case):** app {v:.2f} vs. manual Excel model {PHASE2['target_price']:.2f} "
            f"→ difference {v / PHASE2['target_price'] - 1:+.2%}")


# --------------------------------------------------------------------------
# 8. EXCEL EXPORT
# --------------------------------------------------------------------------
def build_excel():
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summ = pd.DataFrame({"Value": [ticker, info.get("longName", ""), str(date.today()), ss.get("preset"), ss.scenario,
                                       price, dcf_value, ri_value, comps_value, fair_value, ke, ss.dps, target, upside,
                                       div_yield, total_return, rating, wacc, float(np.median(mc)), float((mc > price).mean())]},
                            index=["Ticker", "Company", "Export date", "Assumption set", "Active scenario", "Share price",
                                   "DCF value", "Residual income value", "Trading comps value", "Blended fair value",
                                   "Cost of equity", "Dividends next 12m", "12-month target price", "Price upside",
                                   "Dividend yield", "Total return", "Rating", "WACC", "Monte Carlo median",
                                   "P(value > price)"])
        summ.to_excel(xw, sheet_name="Summary")
        inp = pd.DataFrame({"Value": [ss.rf, ss.beta, ss.erp, ss.kd, ss.tax_lr, ss.payout, ss.da, ss.capex, ss.nwc,
                                      ss.rev0, ss.bv0, ss.shares, ss.debt, ss.cash, ss.w_tp, ss.buy_thr, ss.sell_thr,
                                      ss.tg_Bear, ss.tg_Base, ss.tg_Bull]},
                           index=["Risk-free rate %", "Beta", "Equity risk premium %", "Cost of debt %", "Tax rate (WACC) %",
                                  "Payout ratio %", "D&A % revenue", "Capex % revenue", "NWC % revenue", "Revenue base year",
                                  "Book value base year", "Diluted shares (m)", "Debt", "Cash", "DCF weight in target %",
                                  "Buy threshold %", "Sell threshold %", "Terminal g Bear %", "Terminal g Base %",
                                  "Terminal g Bull %"])
        inp.to_excel(xw, sheet_name="Inputs")
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
        pd.DataFrame({"Percentile": [5, 10, 25, 50, 75, 90, 95],
                      "Value per share": np.percentile(mc, [5, 10, 25, 50, 75, 90, 95])}).to_excel(
            xw, sheet_name="Monte Carlo", index=False)
    return buf.getvalue()


st.sidebar.divider()
try:
    st.sidebar.download_button("⬇️ Export results to Excel", data=build_excel(),
                               file_name=f"{ticker}_valuation_{date.today():%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
except Exception as e:
    st.sidebar.caption(f"Excel export unavailable: {e}")

# --------------------------------------------------------------------------
# 9. TABS
# --------------------------------------------------------------------------
(tab_sum, tab_sc, tab_dcf, tab_ri, tab_sens, tab_mc, tab_rev, tab_comps, tab_ff, tab_cmp, tab_data) = st.tabs(
    ["🏁 Summary", "🎭 Scenarios", "💵 DCF", "📊 Residual Income", "🎯 Sensitivity", "🎲 Monte Carlo",
     "🔁 Reverse DCF", "🏢 Trading Comps", "🏈 Football Field", "⚖️ Compare", "📁 Company data"])


def fmt_table(df, pct_rows):
    tbl = df.set_index("Year").T
    show = tbl.copy().astype(object)
    for r in show.index:
        show.loc[r] = [f"{v:.2%}" if r in pct_rows else (f"{v:.4f}" if r == "Discount factor" else f"{v:,.0f}")
                       for v in tbl.loc[r]]
    show.columns = [f"FY{c}" for c in show.columns]
    return show


# ---- Summary --------------------------------------------------------------
with tab_sum:
    explain("summary")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Target price derivation")
        tp_rows = [("DCF value (" + ss.scenario + " case)", f"{dcf_value:,.2f}", f"{ss.w_tp}%"),
                   ("Trading comps value (" + comps_note + ")", "n/a" if pd.isna(comps_value) else f"{comps_value:,.2f}",
                    f"{100 - ss.w_tp}%" if not pd.isna(comps_value) else "0%"),
                   ("Blended fair value today", f"{fair_value:,.2f}", ""),
                   (f"x (1 + cost of equity {ke:.2%})", f"{fair_value * (1 + ke):,.2f}", ""),
                   ("- dividends next 12 months", f"{-ss.dps:,.2f}", ""),
                   ("12-month target price" + (" (rounded)" if ss.round_tp else ""), f"{target:,.2f}", ""),
                   ("Price upside / dividend yield", f"{upside:+.1%} / {div_yield:.1%}", ""),
                   ("Expected 12-month total return", f"{total_return:+.1%}", ""),
                   (f"Rating (Buy > {ss.buy_thr:.0f}%, Sell < {ss.sell_thr:.0f}%)", rating, "")]
        st.table(pd.DataFrame(tp_rows, columns=["Step", "Value", "Weight"]).set_index("Step"))
    with right:
        st.subheader("Analyst consensus (Yahoo Finance)")
        tmean, tlow, thigh = info.get("targetMeanPrice"), info.get("targetLowPrice"), info.get("targetHighPrice")
        n_an = info.get("numberOfAnalystOpinions")
        cons = pd.DataFrame({"Value": [f"{tmean:,.2f}" if tmean else "n/a",
                                       f"{tlow:,.2f} - {thigh:,.2f}" if (tlow and thigh) else "n/a",
                                       str(n_an or "n/a"), (info.get("recommendationKey") or "n/a").replace("_", " ").title(),
                                       f"{info.get('forwardEps'):,.2f}" if info.get("forwardEps") else "n/a",
                                       f"{target / tmean - 1:+.1%}" if tmean else "n/a"]},
                            index=["Mean target price", "Target range (low - high)", "Number of analysts",
                                   "Consensus recommendation", "Forward EPS", "Our target vs. consensus mean"])
        st.table(cons)
        recs = data.get("recs")
        if recs is not None and {"strongBuy", "buy", "hold", "sell", "strongSell"}.issubset(recs.columns):
            r0 = recs.iloc[0]
            fig = go.Figure(go.Bar(x=["Strong buy", "Buy", "Hold", "Sell", "Strong sell"],
                                   y=[r0["strongBuy"], r0["buy"], r0["hold"], r0["sell"], r0["strongSell"]],
                                   marker_color=[BLACK, GREY, LGREY, "#F7A96B", ORANGE]))
            fig.update_layout(title="Analyst recommendations (current month)", height=260, margin=dict(t=40, b=10))
            st.plotly_chart(fig)
        else:
            st.caption("Recommendation breakdown not available from Yahoo for this ticker.")
    st.subheader("Key insights")
    i1, i2, i3 = st.columns(3)
    i1.metric("Market-implied final EBIT margin", "n/a" if rev_margin is None else f"{final_m + rev_margin:.1%}",
              None if rev_margin is None else f"ours {final_m:.1%}", delta_color="off")
    i2.metric("Monte Carlo 10th - 90th percentile", f"{np.percentile(mc, 10):,.0f} - {np.percentile(mc, 90):,.0f}",
              f"median {np.median(mc):,.2f}", delta_color="off")
    i3.metric("Most important value driver", torn.iloc[0]["Driver"].split(" (")[0] if not torn.empty else "n/a",
              f"range {torn.iloc[0]['Range']:,.2f} per share" if not torn.empty else None, delta_color="off")

# ---- Scenarios ------------------------------------------------------------
with tab_sc:
    explain("scenarios")
    st.subheader("Bear / Base / Bull comparison")
    wc = st.columns(4)
    wc[0].number_input("Bear probability (%)", 0, 100, key="w_bear")
    wc[1].number_input("Base probability (%)", 0, 100, key="w_base")
    wc[2].number_input("Bull probability (%)", 0, 100, key="w_bull")
    wc[3].slider("DCF vs. RI weight in blended value (%)", 0, 100, key="w_dcf", help="Remainder = residual income")
    wts = {"Bear": ss.w_bear, "Base": ss.w_base, "Bull": ss.w_bull}
    tot = sum(wts.values()) or 1
    rows = []
    for s in SCENARIOS:
        df = ss["scen"][s]
        cagr = np.prod(1 + df["Revenue growth %"] / 100) ** (1 / len(df)) - 1
        dv, rv = res[s]["dcf"]["per_share"], res[s]["ri"]["per_share"]
        bl = ss.w_dcf / 100 * dv + (1 - ss.w_dcf / 100) * rv
        rows.append({"Scenario": s, "Revenue CAGR": cagr, "Final-year EBIT margin": df["EBIT margin %"].iloc[-1] / 100,
                     "Terminal growth": ss[f"tg_{s}"] / 100, "DCF value": dv, "RI value": rv, "Blended value": bl,
                     "vs. price": bl / price - 1, "Probability": wts[s] / tot})
    sc = pd.DataFrame(rows).set_index("Scenario")
    pw = {c: (sc[c] * sc["Probability"]).sum() for c in ["DCF value", "RI value", "Blended value"]}
    sc.loc["Probability-weighted"] = [np.nan, np.nan, np.nan, pw["DCF value"], pw["RI value"], pw["Blended value"],
                                      pw["Blended value"] / price - 1, 1.0]
    st.dataframe(sc.style.format({"Revenue CAGR": "{:+.1%}", "Final-year EBIT margin": "{:.1%}", "Terminal growth": "{:.2%}",
                                  "DCF value": "{:,.2f}", "RI value": "{:,.2f}", "Blended value": "{:,.2f}",
                                  "vs. price": "{:+.1%}", "Probability": "{:.0%}"}, na_rep="–"))
    bear_v, base_v = res["Bear"]["dcf"]["per_share"], res["Base"]["dcf"]["per_share"]
    if base_v != bear_v:
        st.caption(f"The share price sits {(price - bear_v) / (base_v - bear_v):.0%} of the way from the Bear to the Base DCF value.")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for s in SCENARIOS:
            v = res[s]["dcf"]["per_share"]
            fig.add_scatter(x=["Share price", "DCF value"], y=[price, v], mode="lines+markers+text", name=s,
                            line=dict(color=SCEN_COLORS[s], width=3), text=["", f"{s} {v:,.2f}"], textposition="middle right")
        fig.add_scatter(x=["Share price", "DCF value"], y=[price, target], mode="lines+markers+text", name="Target",
                        line=dict(color="#F7A96B", width=2, dash="dash"), text=["", f"Target {target:,.2f}"],
                        textposition="bottom right")
        fig.update_layout(title="Share price vs. scenario values", height=380, showlegend=False,
                          xaxis=dict(range=[-0.3, 1.6]), margin=dict(t=50))
        st.plotly_chart(fig)
    with right:
        fig = go.Figure()
        for s in SCENARIOS:
            t = res[s]["dcf"]["table"]
            fig.add_scatter(x=t["Year"], y=t["EBIT margin"] * 100, name=s, mode="lines+markers", line=dict(color=SCEN_COLORS[s]))
        fig.update_layout(title="EBIT margin by scenario (%)", height=380, margin=dict(t=50))
        st.plotly_chart(fig)

# ---- DCF ------------------------------------------------------------------
with tab_dcf:
    explain("dcf")
    explain("wacc")
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Free cash flow to firm - {ss.scenario} case")
        st.dataframe(fmt_table(dcf["table"], ["Revenue growth", "EBIT margin"]))
    with right:
        st.subheader("WACC")
        st.table(pd.DataFrame({"Value": [f"{ke:.2%}", f"{ss.kd:.2f}% / {w['kd_after_tax']:.2%}",
                                         f"{w['we']:.1%} / {w['wd']:.1%}", f"{wacc:.2%}", f"{ss[f'tg_{ss.scenario}']:.2f}%"]},
                              index=["Cost of equity (CAPM)", "Cost of debt pre / after tax", "Equity / debt weight",
                                     "WACC", "Terminal growth"]))
        st.subheader("Equity bridge")
        st.table(pd.DataFrame({f"{currency} m": [dcf["pv_fcff"], dcf["pv_tv"], dcf["ev"], net_cash, dcf["equity_value"]],
                               "Per share": [dcf["pv_fcff"] / ss.shares, dcf["pv_tv"] / ss.shares, dcf["ev"] / ss.shares,
                                             net_cash / ss.shares, dcf_value]},
                              index=["PV of FCFF", "PV of terminal value", "Enterprise value", "+ Net cash / (- net debt)",
                                     "Equity value"]).style.format("{:,.2f}"))
        st.caption(f"Terminal value = {dcf['tv_share']:.0%} of enterprise value. Year-1 cash flow discounted one full year.")
    t = dcf["table"]
    fig = go.Figure()
    fig.add_bar(x=t["Year"], y=t["NOPAT"], name="NOPAT", marker_color=GREY)
    fig.add_bar(x=t["Year"], y=t["FCFF"], name="FCFF", marker_color=BLACK)
    fig.add_bar(x=t["Year"], y=t["PV of FCFF"], name="PV of FCFF", marker_color=ORANGE)
    fig.update_layout(title=f"NOPAT, FCFF and present value ({currency} m)", barmode="group", height=340)
    st.plotly_chart(fig)

# ---- Residual income ------------------------------------------------------
with tab_ri:
    explain("ri")
    left, right = st.columns([3, 2])
    with left:
        st.subheader(f"Residual income - {ss.scenario} case")
        st.dataframe(fmt_table(ri["table"], ["Revenue growth", "Net margin", "ROE"]))
    with right:
        st.subheader("Value bridge (per share)")
        sh = ss.shares
        fig = go.Figure(go.Waterfall(orientation="v", measure=["absolute", "relative", "relative", "total"],
                                     x=["Book value today", "PV of RI", "PV of terminal value", "Value per share"],
                                     y=[ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, 0],
                                     text=[f"{v:,.2f}" for v in [ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, ri_value]],
                                     textposition="outside", increasing=dict(marker_color=BLACK),
                                     decreasing=dict(marker_color=ORANGE), totals=dict(marker_color=GREY)))
        fig.add_hline(y=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Price {price:,.2f}")
        fig.update_layout(height=380, margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig)
        st.caption(f"Cost of equity {ke:.2%}; payout ratio {ss.payout:.1f}%.")

# ---- Sensitivity ----------------------------------------------------------
def heat(grid, xlab, ylab, xs, ys, title):
    fig = go.Figure(go.Heatmap(z=grid, x=[f"{xlab} {v:.2%}" for v in xs], y=[f"{ylab} {v:.2%}" for v in ys],
                               text=[[f"{v:,.2f}" if not pd.isna(v) else "–" for v in row] for row in grid],
                               texttemplate="%{text}", colorscale="RdYlGn", zmid=price, showscale=False))
    fig.update_layout(title=title, height=400, yaxis_autorange="reversed")
    return fig


with tab_sens:
    explain("sens")
    st.subheader(f"Tornado chart - DCF value drivers ({ss.scenario} case, base {dcf_value:,.2f})")
    tt = torn.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=tt["Driver"], x=tt["Downside value"] - dcf_value, base=dcf_value, orientation="h", name="Downside",
                marker_color=ORANGE, text=[f"{v:,.2f}" for v in tt["Downside value"]], textposition="outside")
    fig.add_bar(y=tt["Driver"], x=tt["Upside value"] - dcf_value, base=dcf_value, orientation="h", name="Upside",
                marker_color=BLACK, text=[f"{v:,.2f}" for v in tt["Upside value"]], textposition="outside")
    fig.add_vline(x=dcf_value, line_color=GREY)
    fig.update_layout(barmode="overlay", height=360, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)
    st.dataframe(torn.style.format({"Downside value": "{:,.2f}", "Upside value": "{:,.2f}", "Downside %": "{:+.1%}",
                                    "Upside %": "{:+.1%}", "Range": "{:,.2f}"}))
    tg_act = ss[f"tg_{ss.scenario}"] / 100
    g_vals = [round(tg_act + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
    w_vals = [round(wacc + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
    dgrid = [[_val(wacc=wv, g=gv) for gv in g_vals] for wv in w_vals]
    k_vals = [round(ke + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
    rgrid = sensitivity_grid(act["ri_kwargs"], k_vals, g_vals).values
    l, r = st.columns(2)
    l.plotly_chart(heat(dgrid, "g", "WACC", g_vals, w_vals, f"DCF value per share - {ss.scenario}"))
    r.plotly_chart(heat(rgrid, "g", "ke", g_vals, k_vals, f"Residual income value per share - {ss.scenario}"))
    st.caption(f"Green = above the share price ({price:,.2f}), red = below.")

# ---- Monte Carlo ----------------------------------------------------------
with tab_mc:
    explain("mc")
    st.subheader(f"Monte Carlo simulation - {ss.scenario} case DCF")
    mcc = st.columns(5)
    mcc[0].select_slider("Simulations", options=[1000, 2000, 3000, 5000, 10000], key="mc_n")
    mcc[1].number_input("SD revenue growth (pp)", 0.0, 10.0, step=0.25, key="mc_sg")
    mcc[2].number_input("SD EBIT margin (pp)", 0.0, 10.0, step=0.25, key="mc_sm")
    mcc[3].number_input("SD WACC (pp)", 0.0, 5.0, step=0.05, key="mc_sw")
    mcc[4].number_input("SD terminal growth (pp)", 0.0, 3.0, step=0.05, key="mc_st")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Median value", f"{pc[3]:,.2f}")
    m2.metric("10th - 90th percentile", f"{pc[1]:,.0f} - {pc[5]:,.0f}")
    m3.metric("P(value > share price)", f"{(mc > price).mean():.0%}")
    m4.metric("P(value > target price)", f"{(mc > target).mean():.0%}")
    fig = go.Figure(go.Histogram(x=np.clip(mc, np.percentile(mc, 0.5), np.percentile(mc, 99.5)), nbinsx=60,
                                 marker_color=BLACK, opacity=0.85))
    fig.add_vline(x=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Price {price:,.2f}")
    fig.add_vline(x=target, line_color=ORANGE, annotation_text=f"Target {target:,.2f}", annotation_position="top left")
    fig.add_vline(x=pc[3], line_color=GREY, annotation_text=f"Median {pc[3]:,.2f}", annotation_position="bottom right")
    fig.update_layout(height=380, xaxis_title="Value per share", yaxis_title="Number of simulations", bargap=0.03)
    st.plotly_chart(fig)
    st.dataframe(pd.DataFrame({"Percentile": ["5th", "10th", "25th", "50th", "75th", "90th", "95th"],
                               "Value per share": pc}).set_index("Percentile").T.style.format("{:,.2f}"))
    st.caption("Shifts are applied to every forecast year (fully correlated across years). Draws with WACC - g < 0.5pp are "
               "discarded. Fixed random seed, so results are reproducible.")

# ---- Reverse DCF ----------------------------------------------------------
with tab_rev:
    explain("reverse")
    st.subheader(f"What does a share price of {price:,.2f} imply? ({ss.scenario} case as starting point)")
    rows = [("EBIT margin (all years shifted)", f"final year {final_m:.1%}",
             "not solvable" if rev_margin is None else f"final year {final_m + rev_margin:.1%} ({rev_margin * 100:+.1f}pp)"),
            ("Revenue growth (all years shifted)", f"avg {np.mean(k_act['growth']):.1%}",
             "not solvable" if rev_growth is None else f"avg {np.mean(k_act['growth']) + rev_growth:.1%} ({rev_growth * 100:+.1f}pp)"),
            ("WACC", f"{wacc:.2%}", "not solvable" if rev_wacc is None else f"{rev_wacc:.2%}"),
            ("Terminal growth", f"{k_act['g']:.2%}", "not solvable" if rev_tg is None else f"{rev_tg:.2%}")]
    st.table(pd.DataFrame(rows, columns=["Driver (others unchanged)", "Our assumption", "Implied by share price"]).set_index(
        "Driver (others unchanged)"))
    if rev_margin is not None:
        verdict = "below" if rev_margin < 0 else "above"
        st.success(f"The market price is consistent with a final-year EBIT margin of {final_m + rev_margin:.1%}, "
                   f"{abs(rev_margin) * 100:.1f}pp {verdict} our {ss.scenario} case. "
                   f"{'Our case is more optimistic than the market.' if rev_margin < 0 else 'Our case is more conservative than the market.'}")
    margins = np.linspace(max(final_m - 0.08, -0.05), final_m + 0.08, 33)
    vals = [_val(ebit_margin=[m + (mm - final_m) for m in k_act["ebit_margin"]]) for mm in margins]
    fig = go.Figure(go.Scatter(x=margins * 100, y=vals, mode="lines", line=dict(color=BLACK, width=3)))
    fig.add_hline(y=price, line_dash="dash", line_color=ORANGE, annotation_text=f"Share price {price:,.2f}")
    fig.add_vline(x=final_m * 100, line_color=GREY, annotation_text=f"Our {final_m:.1%}")
    fig.update_layout(title="DCF value vs. final-year EBIT margin", xaxis_title="Final-year EBIT margin (%)",
                      yaxis_title="Value per share", height=360)
    st.plotly_chart(fig)

# ---- Trading comps --------------------------------------------------------
with tab_comps:
    explain("comps")
    st.subheader("Comparable companies")
    st.caption(f"Peers (edit in the sidebar): {', '.join(peer_list) if peer_list else 'none'}")
    if peers.empty:
        st.info("Enter peer tickers in the sidebar ('Trading comps peers') to run the trading comps.")
    else:
        stats = peers[["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]].agg(["median", "mean"])
        stats.index = ["Peer median", "Peer mean"]
        st.dataframe(pd.concat([peers.set_index("Ticker"), stats, pd.DataFrame([peer_row(ticker, info)]).set_index("Ticker")])
                     .style.format({"Market cap (bn)": "{:,.1f}", "EV (bn)": "{:,.1f}", "Revenue LTM (bn)": "{:,.1f}",
                                    "EBITDA LTM (bn)": "{:,.1f}", "EV/Revenue": "{:.2f}x", "EV/EBITDA": "{:.2f}x",
                                    "P/E": "{:.1f}x", "P/Book": "{:.2f}x"}, na_rep="–"))
        st.caption(f"Last row = {ticker} itself (reference only). EV = market cap + total debt - cash; LTM = last twelve months.")
        if not comps_prices.empty:
            st.subheader(f"Implied share price for {ticker}")
            st.dataframe(comps_prices.set_index("Multiple").style.format(
                {"Peer 25th pct": "{:.2f}x", "Peer median": "{:.2f}x", "Peer 75th pct": "{:.2f}x",
                 "Low": "{:,.2f}", "Mid": "{:,.2f}", "High": "{:,.2f}"}))
            st.caption(f"Comps value = average of the medians: {live_comps:,.2f}. Yahoo provides trailing figures, so values "
                       "differ slightly from the report (FY27E consensus EPS, tariff-adjusted EBITDA).")

# ---- Football field -------------------------------------------------------
with tab_ff:
    explain("ff")
    st.subheader("Valuation range by method")
    dv = [res[s]["dcf"]["per_share"] for s in SCENARIOS]
    rv = [res[s]["ri"]["per_share"] for s in SCENARIOS]
    bars = [("DCF: Bear - Base - Bull", min(dv), dv[1], max(dv), BLACK),
            ("Monte Carlo: 10th - 90th pct", pc[1], pc[3], pc[5], "#444444"),
            ("Residual income: Bear - Base - Bull", min(rv), rv[1], max(rv), GREY)]
    for _, r in comps_prices.iterrows():
        bars.append((f"Comps: {r['Multiple']}", r["Low"], r["Mid"], r["High"], ORANGE))
    if info.get("targetLowPrice") and info.get("targetHighPrice"):
        bars.append(("Analyst price targets", info["targetLowPrice"], info.get("targetMeanPrice") or info["targetLowPrice"],
                     info["targetHighPrice"], "#F7A96B"))
    lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
    if lo52 and hi52:
        bars.append(("52-week trading range", lo52, (lo52 + hi52) / 2, hi52, LGREY))
    ff = pd.DataFrame(bars, columns=["Method", "Low", "Mid", "High", "Color"])
    fig = go.Figure()
    fig.add_bar(y=ff["Method"], x=ff["High"] - ff["Low"], base=ff["Low"], orientation="h", marker_color=ff["Color"],
                text=[f"{l:,.2f} – {h:,.2f}" for l, h in zip(ff["Low"], ff["High"])], textposition="outside")
    fig.add_scatter(y=ff["Method"], x=ff["Mid"], mode="markers", marker=dict(color="white", symbol="line-ns-open", size=18, line_width=3))
    fig.add_vline(x=price, line_dash="dash", line_color=GREY, annotation_text=f"Price {price:,.2f}")
    fig.add_vline(x=target, line_color=ORANGE, line_width=2, annotation_text=f"Target {target:,.2f}", annotation_position="bottom right")
    fig.update_layout(height=140 + 62 * len(ff), showlegend=False, yaxis_autorange="reversed",
                      xaxis_title="Value per share", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig)
    st.dataframe(ff.drop(columns="Color").set_index("Method").style.format("{:,.2f}"))

# ---- Compare --------------------------------------------------------------
with tab_cmp:
    explain("compare")
    st.text_input("Compare with ticker", key="cmp_ticker")
    other = ss.cmp_ticker.strip().upper()
    if not other:
        st.info("Enter a second ticker to compare.")
    else:
        try:
            d2 = load_company(other)
            ok = d2["revenue"] is not None and d2["equity"] is not None and d2["net_income"] is not None
        except Exception as e:
            ok = False
            st.warning(f"Could not load {other}: {e}")
        if ok:
            H2 = hist_ratios(d2)
            i2 = d2["info"]
            v1, w1 = quick_dcf(H, ss.rf, ss.erp, ss.kd)
            v2, w2 = quick_dcf(H2, ss.rf, ss.erp, ss.kd)
            p1, p2 = peer_row(ticker, info), peer_row(other, i2)

            def col(h, i, p, v, wv):
                return [i.get("shortName", ""), i.get("currency", ""), h["price"], p["Market cap (bn)"], h["growth"],
                        h["ebit"], h["margin"], _div(h["margin"] * h["rev0"], h["bv0"]), _div(h["dps"], h["price"]),
                        p["EV/Revenue"], p["EV/EBITDA"], p["P/E"], p["P/Book"], h["beta"], wv, v, _div(v, h["price"]) - 1]
            idx = ["Name", "Currency", "Share price", "Market cap (bn)", "Revenue growth (last FY)", "EBIT margin (last FY)",
                   "Net margin (last FY)", "ROE (last FY)", "Dividend yield", "EV/Revenue", "EV/EBITDA", "P/E", "P/Book",
                   "Beta", "WACC (auto)", "Quick DCF value (auto Base)", "Quick DCF vs. price"]
            comp = pd.DataFrame({ticker: col(H, info, p1, v1, w1), other: col(H2, i2, p2, v2, w2)}, index=idx)
            fmt_rows = {"Revenue growth (last FY)": "{:+.1%}", "EBIT margin (last FY)": "{:.1%}", "Net margin (last FY)": "{:.1%}",
                        "ROE (last FY)": "{:.1%}", "Dividend yield": "{:.1%}", "WACC (auto)": "{:.2%}", "Quick DCF vs. price": "{:+.1%}",
                        "EV/Revenue": "{:.2f}x", "EV/EBITDA": "{:.1f}x", "P/E": "{:.1f}x", "P/Book": "{:.2f}x",
                        "Share price": "{:,.2f}", "Market cap (bn)": "{:,.1f}", "Beta": "{:.2f}", "Quick DCF value (auto Base)": "{:,.2f}"}
            show = comp.copy().astype(object)
            for r_, f_ in fmt_rows.items():
                show.loc[r_] = [f_.format(x) if isinstance(x, (int, float)) and not pd.isna(x) else "–" for x in comp.loc[r_]]
            st.table(show)
            st.caption("Quick DCF = automatic Base case for both companies: last fiscal year's growth and EBIT margin held "
                       "constant for 5 years, terminal growth 2.15%, same risk-free rate, equity risk premium and cost of debt.")
            if len(data["prices"]) and len(d2["prices"]):
                fig = go.Figure()
                for name_, s_, c_ in [(ticker, data["prices"], BLACK), (other, d2["prices"], ORANGE)]:
                    s_ = s_.dropna()
                    fig.add_scatter(x=s_.index, y=s_ / s_.iloc[0] * 100, name=name_, line=dict(color=c_, width=2))
                fig.update_layout(title="Share price performance, 5 years (rebased to 100, local currency)", height=360)
                st.plotly_chart(fig)

# ---- Company data ---------------------------------------------------------
with tab_data:
    st.subheader("Historical financials (Yahoo Finance)")
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
    if len(data["prices"]):
        fig = go.Figure(go.Scatter(x=data["prices"].index, y=data["prices"].values, mode="lines", line=dict(color=BLACK)))
        fig.add_hline(y=target, line_dash="dot", line_color=ORANGE, annotation_text=f"Target {target:,.2f}")
        fig.update_layout(title="Share price - last 5 years", height=360)
        st.plotly_chart(fig)

st.caption("Team 10 · WHU PT-MiM · Built with AI assistance (Claude) · Academic project, not investment advice.")
