"""
app.py - Team 10 (WHU PT-MiM) - Interactive Residual Income & Trading Comps valuation app.

Run with:   streamlit run app.py
Needs:      valuation.py in the same folder, internet access (Yahoo Finance).
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from valuation import (
    implied_prices_from_comps,
    residual_income_valuation,
    sensitivity_grid,
)

st.set_page_config(page_title="Team 10 | Equity Valuation", page_icon="📈", layout="wide")

BLACK, ORANGE, GREY = "#111111", "#F25C05", "#8A8A8A"     # Team 10 / Nike report colours

DEFAULT_PEERS = {"NKE": "ADS.DE, PUMA.DE, DECK, ONON, LULU, UAA"}

# Nike Phase 2 Excel model - used to prove the app reproduces the manual valuation
EXCEL_CHECK = {
    "growth": [-1.59, 4.70, 4.70, 3.26, 3.26],
    "margin": [7.321017182, 8.626545316, 8.778371510, 9.087953208, 9.390908032],
    "rev0": 46398.0, "bv0": 14865.0, "shares": 1479.8,
    "payout": 62.32, "rf": 4.44, "beta": 1.11, "erp": 4.18, "g": 2.15,
    "target_price": 36.97,
}


# --------------------------------------------------------------------------
# 1. DATA FROM YAHOO FINANCE
# --------------------------------------------------------------------------
def _row(df, names):
    """Return the first matching row of a yfinance statement, newest year first."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for name in names:
        if name in df.index:
            s = pd.to_numeric(df.loc[name], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index(ascending=False)
    return None


KNOWN = {"NKE": {"longName": "NIKE, Inc.", "sector": "Consumer Cyclical", "industry": "Footwear & Accessories"}}


def _safe(fn, default=None):
    try:
        v = fn()
        return default if v is None else v
    except Exception:
        return default


def fill_info(tk, t, info, inc, bs, prices):
    """Yahoo's profile endpoint is often blocked on cloud servers. Fill every missing key figure
    from sources that work reliably: fast_info, the financial statements and the price history."""
    info = dict(info or {})
    fi = _safe(lambda: t.fast_info)
    fget = lambda a: _safe(lambda: getattr(fi, a)) if fi is not None else None
    latest = lambda df, names: (lambda r: float(r.iloc[0]) if r is not None else None)(_row(df, names))

    def put(key, value):
        if not info.get(key) and value:
            info[key] = value
    put("currentPrice", fget("last_price") or (float(prices.iloc[-1]) if len(prices) else None))
    put("marketCap", fget("market_cap"))
    put("sharesOutstanding", fget("shares") or latest(bs, ["Ordinary Shares Number", "Share Issued"]))
    put("currency", fget("currency"))
    put("totalDebt", latest(bs, ["Total Debt"]))
    put("totalCash", latest(bs, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"]))
    put("totalRevenue", latest(inc, ["Total Revenue", "Operating Revenue"]))
    put("ebitda", latest(inc, ["EBITDA", "Normalized EBITDA"]))
    put("netIncomeToCommon", latest(inc, ["Net Income Common Stockholders", "Net Income"]))
    eq, sh = latest(bs, ["Stockholders Equity", "Common Stock Equity"]), info.get("sharesOutstanding")
    put("bookValue", eq / sh if (eq and sh) else None)
    if len(prices):
        last_year = prices[prices.index >= prices.index[-1] - pd.Timedelta(days=365)]
        put("fiftyTwoWeekLow", float(last_year.min()))
        put("fiftyTwoWeekHigh", float(last_year.max()))
    for k, v in KNOWN.get(tk, {}).items():
        put(k, v)
    return info


@st.cache_data(ttl=3600, show_spinner=False)
def load_company(ticker):
    t = yf.Ticker(ticker)
    info = _safe(lambda: t.info, {}) or {}
    inc, bs, cf = _safe(lambda: t.income_stmt), _safe(lambda: t.balance_sheet), _safe(lambda: t.cashflow)
    data = {
        "info": info,
        "revenue": _row(inc, ["Total Revenue", "Operating Revenue"]),
        "net_income": _row(inc, ["Net Income", "Net Income Common Stockholders"]),
        "ebitda": _row(inc, ["EBITDA", "Normalized EBITDA"]),
        "equity": _row(bs, ["Stockholders Equity", "Common Stock Equity"]),
        "dividends": _row(cf, ["Cash Dividends Paid", "Common Stock Dividend Paid"]),
    }
    hist = _safe(lambda: t.history(period="5y"), pd.DataFrame())
    data["prices"] = hist["Close"].dropna() if "Close" in hist else pd.Series(dtype=float)
    data["info"] = fill_info(ticker, t, info, inc, bs, data["prices"])
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_multiples(tickers):
    rows = []
    for tk in tickers:
        try:
            i = load_company(tk)["info"]          # same robust data as for the company itself
        except Exception:
            continue
        if not i.get("marketCap") or not i.get("totalRevenue"):
            continue
        rows.append(peer_row(tk, i))
    df = pd.DataFrame(rows)
    for c in ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[df[c] <= 0, c] = np.nan     # negative multiples are not meaningful
    return df


def _div(a, b):
    return a / b if (a is not None and b not in (None, 0) and not pd.isna(a) and not pd.isna(b)) else np.nan


def peer_row(tk, i):
    """Calculate the multiples from Yahoo raw data: EV = market cap + debt - cash."""
    mcap = i.get("marketCap")
    ev = (mcap + (i.get("totalDebt") or 0) - (i.get("totalCash") or 0)) if mcap else i.get("enterpriseValue")
    rev, ebitda, ni = i.get("totalRevenue"), i.get("ebitda"), i.get("netIncomeToCommon")
    px = i.get("currentPrice") or i.get("regularMarketPrice")
    same_ccy = (i.get("financialCurrency") or i.get("currency")) == i.get("currency")
    if same_ccy:   # market data and financials in the same currency -> compute ourselves
        ev_rev, ev_ebitda, pe = _div(ev, rev), _div(ev, ebitda), _div(mcap, ni)
    else:          # e.g. ADRs: fall back to Yahoo's own (currency-adjusted) ratios
        ev_rev, ev_ebitda, pe = i.get("enterpriseToRevenue"), i.get("enterpriseToEbitda"), i.get("trailingPE")
    return {
        "Ticker": tk, "Name": i.get("shortName", tk), "Currency": i.get("currency"),
        "Market cap (bn)": _div(mcap, 1e9), "EV (bn)": _div(ev, 1e9),
        "Revenue LTM (bn)": _div(rev, 1e9), "EBITDA LTM (bn)": _div(ebitda, 1e9),
        "EV/Revenue": ev_rev, "EV/EBITDA": ev_ebitda, "P/E": pe,
        "P/Book": _div(px, i.get("bookValue")) if same_ccy else i.get("priceToBook"),
    }


def fmt_pct(x):
    return "n/a" if x is None or pd.isna(x) else f"{x:.1%}"


# --------------------------------------------------------------------------
# 2. SIDEBAR - TICKER AND ASSUMPTIONS
# --------------------------------------------------------------------------
ss = st.session_state
st.sidebar.title("📈 Valuation inputs")
st.sidebar.caption("Team 10 · WHU PT-MiM · Final Project Phase III B")
ticker = st.sidebar.text_input("Stock ticker", value="NKE", help="Yahoo Finance symbol, e.g. NKE, AAPL, ADS.DE").strip().upper()

try:
    with st.spinner(f"Loading {ticker} from Yahoo Finance ..."):
        data = load_company(ticker)
except Exception as e:
    st.error(f"Could not load data for **{ticker}**: {e}")
    st.stop()

info = data["info"]
if data["revenue"] is None or data["equity"] is None or data["net_income"] is None:
    st.error(f"Yahoo Finance has no complete annual financials for **{ticker}**. Check the ticker.")
    st.stop()

rev_hist, ni_hist, eq_hist = data["revenue"] / 1e6, data["net_income"] / 1e6, data["equity"] / 1e6
div_hist = data["dividends"].abs() / 1e6 if data["dividends"] is not None else None
base_year = int(rev_hist.index[0].year)
price = info.get("currentPrice") or info.get("regularMarketPrice") or (
    float(data["prices"].iloc[-1]) if len(data["prices"]) else np.nan)
# Shares: Yahoo's "sharesOutstanding" can miss share classes (Nike: Class A + Class B),
# so take the larger of implied shares and market cap / price.
_share_candidates = [info.get("impliedSharesOutstanding"), info.get("sharesOutstanding")]
if info.get("marketCap") and price:
    _share_candidates.append(info["marketCap"] / price)
_share_candidates = [x for x in _share_candidates if x]
shares_yf = max(_share_candidates) / 1e6 if _share_candidates else np.nan
currency = info.get("financialCurrency") or info.get("currency") or "USD"

# Historical defaults: median of the last fiscal years instead of one single year,
# so one unusual year (or a data error) does not produce absurd assumptions
_growth_rates = (rev_hist / rev_hist.shift(-1) - 1).dropna()
hist_growth = float(_growth_rates.median()) if len(_growth_rates) else 0.03
hist_margin = float((ni_hist / rev_hist).dropna().median())
if div_hist is not None:
    _payouts = (div_hist / ni_hist)[ni_hist > 0].dropna()
    hist_payout = float(_payouts.median()) if len(_payouts) else 0.0
else:
    hist_payout = 0.0


def clip(x, lo, hi):
    return float(min(max(x, lo), hi))


# Keep values of widgets that are temporarily hidden (Streamlit would otherwise forget them)
WIDGET_KEYS = ["growth", "margin", "payout", "rf", "erp", "g", "ke_manual", "beta", "years",
               "ke_mode", "mode", "rev0", "bv0", "shares", "peers"]
for k in WIDGET_KEYS:
    if k in ss:
        ss[k] = ss[k]

def load_excel_check():
    """Button callback: load the Nike Phase 2 Excel assumptions."""
    c = EXCEL_CHECK
    ss.update({"rev0": c["rev0"], "bv0": c["bv0"], "shares": c["shares"], "payout": c["payout"],
               "rf": c["rf"], "beta": c["beta"], "erp": c["erp"], "g": c["g"], "years": 5,
               "ke_mode": "CAPM", "mode": "Year by year"})
    ss["yearly_preset"] = pd.DataFrame({"Revenue growth %": c["growth"], "Net margin %": c["margin"]})
    ss["preset_id"] = ss.get("preset_id", 0) + 1


# Reset inputs to the new company's data whenever the ticker changes
if ss.get("loaded_ticker") != ticker:
    ss.update({
        "growth": round(clip(hist_growth * 100, -10, 15), 1),
        "margin": round(clip(hist_margin * 100, -20, 40), 1),
        "payout": round(clip(hist_payout * 100, 0, 100), 1),
        "rf": 4.44, "erp": 4.18, "g": 2.15, "ke_manual": 9.0,
        "beta": round(clip(info.get("beta") or 1.0, 0, 3), 2),
        "years": 5, "ke_mode": "CAPM", "mode": "Simple (one value for all years)",
        "rev0": float(rev_hist.iloc[0]), "bv0": float(eq_hist.iloc[0]),
        "shares": float(shares_yf) if not np.isnan(shares_yf) else 1.0,
        "peers": DEFAULT_PEERS.get(ticker, ""),
        "yearly_preset": None,
    })
    if ticker == "NKE":          # Nike opens with our Phase II Excel model
        load_excel_check()
    ss["loaded_ticker"] = ticker


st.sidebar.subheader("Forecast")
years = st.sidebar.slider("Forecast period (years)", 3, 10, key="years")
mode = st.sidebar.radio("Assumption mode", ["Simple (one value for all years)", "Year by year"], key="mode")

if mode.startswith("Simple"):
    st.sidebar.slider("Revenue growth (% p.a.)", -30.0, 40.0, step=0.1, key="growth",
                      help=f"Median of the last fiscal years: {fmt_pct(hist_growth)}")
    st.sidebar.slider("Net income margin (%)", -20.0, 50.0, step=0.1, key="margin",
                      help=f"Median of the last fiscal years: {fmt_pct(hist_margin)}")
    growth = [ss.growth / 100] * years
    margin = [ss.margin / 100] * years
else:
    preset = ss.get("yearly_preset")
    if preset is not None and len(preset) == years:
        base_df = preset.copy()
    else:
        base_df = pd.DataFrame({"Revenue growth %": [ss.growth] * years, "Net margin %": [ss.margin] * years})
    base_df.index = [str(base_year + t) for t in range(1, years + 1)]
    st.sidebar.caption("Edit the table cells (values in %):")
    edited = st.sidebar.data_editor(base_df, key=f"yearly_{ticker}_{years}_{ss.get('preset_id', 0)}")
    if edited.isna().any().any():
        st.sidebar.error("Please fill in every cell of the table.")
        st.stop()
    growth = (edited["Revenue growth %"].astype(float) / 100).tolist()
    margin = (edited["Net margin %"].astype(float) / 100).tolist()

st.sidebar.slider("Dividend payout ratio (%)", 0.0, 100.0, step=0.1, key="payout",
                  help=f"Median of the last fiscal years: {fmt_pct(hist_payout)}")

st.sidebar.subheader("Discount rate")
ke_mode = st.sidebar.radio("Cost of equity", ["CAPM", "Manual"], horizontal=True, key="ke_mode")
if ke_mode == "CAPM":
    st.sidebar.number_input("Risk-free rate (%)", 0.0, 15.0, step=0.01, key="rf")
    st.sidebar.number_input("Beta", 0.0, 3.0, step=0.01, key="beta", help=f"Yahoo beta: {info.get('beta', 'n/a')}")
    st.sidebar.number_input("Equity risk premium (%)", 0.0, 15.0, step=0.01, key="erp")
    ke = (ss.rf + ss.beta * ss.erp) / 100
    st.sidebar.markdown(f"**Cost of equity = {ss.rf:.2f}% + {ss.beta:.2f} × {ss.erp:.2f}% = {ke:.2%}**")
else:
    st.sidebar.slider("Cost of equity (%)", 3.0, 20.0, step=0.05, key="ke_manual")
    ke = ss.ke_manual / 100
st.sidebar.slider("Terminal growth (%)", -2.0, 5.0, step=0.05, key="g")
g = ss.g / 100

with st.sidebar.expander("Starting values (from Yahoo, editable)"):
    st.number_input(f"Revenue FY{base_year} ({currency} m)", key="rev0", step=100.0)
    st.number_input(f"Book value of equity FY{base_year} ({currency} m)", key="bv0", step=100.0)
    st.number_input("Shares outstanding (m)", key="shares", step=1.0)

if ticker == "NKE":
    st.sidebar.button("↺ Reload Team 10 Excel model (Phase II)", on_click=load_excel_check,
                      help="Loads exactly the assumptions of our manual Excel model. Expected result: $36.97")

# --------------------------------------------------------------------------
# 3. CALCULATIONS
# --------------------------------------------------------------------------
base_kwargs = dict(rev0=ss.rev0, bv0=ss.bv0, shares=ss.shares, growth=growth, margin=margin,
                   payout=ss.payout / 100, ke=ke, g=g, base_year=base_year)
try:
    ri = residual_income_valuation(**base_kwargs)
except ValueError as e:
    st.error(str(e))
    st.stop()

ri_value = ri["per_share"]
if ss.bv0 <= 0:
    st.warning("Book value of equity is negative (e.g. after large buybacks). "
               "A residual income valuation is not meaningful here - rely on the trading comps.")
upside = ri_value / price - 1 if price else np.nan

# Sensitivity grid: cost of equity ±2pp, terminal growth ±1pp
ke_values = [round(ke + d, 4) for d in np.arange(-0.02, 0.0201, 0.005)]
g_values = [round(g + d, 4) for d in np.arange(-0.01, 0.0101, 0.005)]
grid = sensitivity_grid(base_kwargs, ke_values, g_values)
inner = grid.loc[[ke_values[2], ke_values[4], ke_values[6]], [g_values[1], g_values[2], g_values[3]]]  # ke ±1pp, g ±0.5pp

# --------------------------------------------------------------------------
# 4. PAGE HEADER
# --------------------------------------------------------------------------
st.title(f"{info.get('longName', ticker)} ({ticker})")
st.caption(f"{info.get('sector', '')} · {info.get('industry', '')} · Financials in {currency} millions · "
           f"Base year FY{base_year} · Data: Yahoo Finance")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current share price", f"{price:,.2f}")
c2.metric("Residual income value", f"{ri_value:,.2f}", f"{upside:+.1%} vs. market" if not pd.isna(upside) else None)
c3.metric("Cost of equity", f"{ke:.2%}")
c4.metric("Terminal growth", f"{g:.2%}")
verdict = "n/a" if pd.isna(upside) else ("Undervalued" if upside > 0.10 else "Overvalued" if upside < -0.10 else "Fairly valued")
c5.metric("Signal (±10% band)", verdict)

if ticker == "NKE" and mode == "Year by year" and abs(ss.rev0 - EXCEL_CHECK["rev0"]) < 1:
    diff = ri_value / EXCEL_CHECK["target_price"] - 1
    (st.success if abs(diff) < 0.001 else st.info)(
        f"{'✅ ' if abs(diff) < 0.001 else ''}**Excel check:** app value {ri_value:.2f} vs. Team 10 manual Excel model "
        f"{EXCEL_CHECK['target_price']:.2f} → difference {diff:+.2%}")

tab_ri, tab_sens, tab_comps, tab_ff, tab_data, tab_method = st.tabs(
    ["📊 Residual Income", "🎯 Sensitivity", "🏢 Trading Comps", "🏈 Football Field", "📁 Company data", "ℹ️ Method"])

# --------------------------------------------------------------------------
# 5. RESIDUAL INCOME TAB
# --------------------------------------------------------------------------
with tab_ri:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Forecast & residual income")
        tbl = ri["table"].set_index("Year").T
        pct_rows = ["Revenue growth", "Net margin", "ROE"]
        show = tbl.copy().astype(object)
        for r in show.index:
            show.loc[r] = [f"{v:.2%}" if r in pct_rows else (f"{v:.4f}" if r == "Discount factor" else f"{v:,.0f}")
                           for v in tbl.loc[r]]
        show.columns = [f"FY{c}" for c in show.columns]
        st.dataframe(show)
    with right:
        st.subheader("Value bridge (per share)")
        sh = ss.shares
        fig = go.Figure(go.Waterfall(
            orientation="v", measure=["absolute", "relative", "relative", "total"],
            x=["Book value today", "PV of RI (forecast)", "PV of terminal value", "Value per share"],
            y=[ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, 0],
            text=[f"{v:,.2f}" for v in [ri["bv0"] / sh, ri["pv_ri"] / sh, ri["pv_tv"] / sh, ri_value]],
            textposition="outside", increasing_marker_color=ORANGE, decreasing_marker_color=GREY,
            totals_marker_color=BLACK))
        fig.add_hline(y=price, line_dash="dash", line_color=GREY, annotation_text=f"Market price {price:,.2f}")
        fig.update_layout(height=380, margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig, key="chart_1")
        tv_share = ri["pv_tv"] / ri["equity_value"]
        st.caption(f"Terminal value = {tv_share:.0%} of total value. "
                   f"TV = RI(final year) × (1+g) / (ke − g) = {ri['terminal_value']:,.0f}")

    a_col, v_col = st.columns(2)
    with a_col:
        st.subheader("Assumptions used")
        g_txt = f"{growth[0]:.2%}" if len(set(growth)) == 1 else f"{min(growth):.2%} to {max(growth):.2%} (by year)"
        m_txt = f"{margin[0]:.2%}" if len(set(margin)) == 1 else f"{min(margin):.2%} to {max(margin):.2%} (by year)"
        st.table(pd.DataFrame({"Value": [
            f"{years} years (FY{base_year + 1}-FY{base_year + years})", g_txt, m_txt, f"{ss.payout / 100:.2%}",
            f"{ke:.2%}" + (f"  (CAPM: {ss.rf:.2f}% + {ss.beta:.2f} × {ss.erp:.2f}%)" if ke_mode == "CAPM" else "  (manual)"),
            f"{g:.2%}", f"{ss.rev0:,.0f}", f"{ss.bv0:,.0f}", f"{ss.shares:,.1f}"]},
            index=["Forecast period", "Revenue growth", "Net income margin", "Payout ratio", "Cost of equity",
                   "Terminal growth", f"Revenue FY{base_year} ({currency} m)",
                   f"Book value FY{base_year} ({currency} m)", "Shares outstanding (m)"]))
    with v_col:
        st.subheader("Valuation output")
        st.table(pd.DataFrame({
            f"Total ({currency} m)": [ri["bv0"], ri["pv_ri"], ri["pv_tv"], ri["equity_value"]],
            "Per share": [ri["bv0"] / ss.shares, ri["pv_ri"] / ss.shares, ri["pv_tv"] / ss.shares, ri_value]},
            index=["Book value of equity today", "PV of residual income (forecast)", "PV of terminal value",
                   "Equity value"]).style.format({f"Total ({currency} m)": "{:,.0f}", "Per share": "{:,.2f}"}))
        st.markdown(f"**Market price {price:,.2f} → implied upside/(downside) {upside:+.1%}**")

    t = ri["table"]
    fig2 = go.Figure()
    fig2.add_bar(x=t["Year"], y=t["Net income"], name="Net income", marker_color=BLACK)
    fig2.add_bar(x=t["Year"], y=t["Equity charge"], name="Equity charge (ke × BV)", marker_color=GREY)
    fig2.add_scatter(x=t["Year"], y=t["Residual income"], name="Residual income", mode="lines+markers",
                     line_color=ORANGE)
    fig2.update_layout(title=f"Net income vs. cost of equity ({currency} m)", barmode="group", height=360)
    st.plotly_chart(fig2, key="chart_2")

# --------------------------------------------------------------------------
# 6. SENSITIVITY TAB
# --------------------------------------------------------------------------
with tab_sens:
    st.subheader("Value per share: cost of equity × terminal growth")
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=[f"g {v:.2%}" for v in g_values], y=[f"ke {v:.2%}" for v in ke_values],
        text=[[f"{v:,.2f}" if not pd.isna(v) else "–" for v in row] for row in grid.values],
        texttemplate="%{text}", colorscale="RdYlGn", zmid=price, showscale=False))
    fig.update_layout(height=420, yaxis_autorange="reversed")
    st.plotly_chart(fig, key="chart_3")
    st.caption(f"Green = above the current market price ({price:,.2f}), red = below.")

# --------------------------------------------------------------------------
# 7. TRADING COMPS TAB
# --------------------------------------------------------------------------
with tab_comps:
    st.subheader("Comparable companies")
    st.text_input("Peer tickers (comma separated)", key="peers")
    peer_list = tuple(p.strip().upper() for p in ss.peers.split(",") if p.strip() and p.strip().upper() != ticker)
    comps_prices = pd.DataFrame()
    if not peer_list:
        st.info("Enter at least one peer ticker to run the trading comps.")
    else:
        with st.spinner("Loading peer multiples ..."):
            peers = load_peer_multiples(peer_list)
        if peers.empty:
            st.warning("No peer data found.")
        else:
            target_row = peer_row(ticker, info)
            stats = peers[["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]].agg(["median", "mean"])
            stats.index = ["Peer median", "Peer mean"]
            st.dataframe(
                pd.concat([peers.set_index("Ticker"), stats, pd.DataFrame([target_row]).set_index("Ticker")])
                .style.format({"Market cap (bn)": "{:,.1f}", "EV (bn)": "{:,.1f}", "Revenue LTM (bn)": "{:,.1f}",
                               "EBITDA LTM (bn)": "{:,.1f}", "EV/Revenue": "{:.2f}x", "EV/EBITDA": "{:.2f}x",
                               "P/E": "{:.1f}x", "P/Book": "{:.2f}x"}, na_rep="–"),
                )
            st.caption(f"Last row = {ticker} itself (for reference, not used in the peer statistics). "
                       "EV = market cap + total debt − cash; LTM = last twelve months; negative multiples excluded.")

            net_debt = ((info.get("totalDebt") or 0) - (info.get("totalCash") or 0)) / 1e6
            comps_prices = implied_prices_from_comps(
                peers, revenue=(info.get("totalRevenue") or 0) / 1e6, ebitda=(info.get("ebitda") or 0) / 1e6,
                eps=_div(info.get("netIncomeToCommon"), ss.shares * 1e6) if info.get("netIncomeToCommon") else info.get("trailingEps"),
                bvps=ss.bv0 / ss.shares, net_debt=net_debt, shares=ss.shares)
            st.subheader(f"Implied share price for {ticker}")
            st.caption(f"EV multiples: (multiple × metric − net debt {net_debt:,.0f}m) ÷ shares. "
                       "Range = peer 25th to 75th percentile.")
            if comps_prices.empty:
                st.warning("Not enough data to compute implied prices.")
            else:
                st.dataframe(comps_prices.set_index("Multiple").style.format(
                    {"Peer 25th pct": "{:.2f}x", "Peer median": "{:.2f}x", "Peer 75th pct": "{:.2f}x",
                     "Low": "{:,.2f}", "Mid": "{:,.2f}", "High": "{:,.2f}"}))
            if info.get("currency") != info.get("financialCurrency") and info.get("financialCurrency"):
                st.warning("Trading and reporting currency differ for this ticker - check the comps carefully.")

# --------------------------------------------------------------------------
# 8. FOOTBALL FIELD TAB
# --------------------------------------------------------------------------
with tab_ff:
    st.subheader("Valuation range by method")
    bars = [("Residual income (ke ±1pp, g ±0.5pp)", np.nanmin(inner.values), ri_value, np.nanmax(inner.values))]
    for _, r in comps_prices.iterrows():
        bars.append((f"Comps: {r['Multiple']}", r["Low"], r["Mid"], r["High"]))
    lo52, hi52 = info.get("fiftyTwoWeekLow"), info.get("fiftyTwoWeekHigh")
    if lo52 and hi52:
        bars.append(("52-week trading range", lo52, (lo52 + hi52) / 2, hi52))
    ff = pd.DataFrame(bars, columns=["Method", "Low", "Mid", "High"])

    fig = go.Figure()
    fig.add_bar(y=ff["Method"], x=ff["High"] - ff["Low"], base=ff["Low"], orientation="h",
                marker_color=[BLACK] + [ORANGE] * len(comps_prices) + ["#C9C9C9"] * (len(ff) - 1 - len(comps_prices)),
                text=[f"{l:,.2f} – {h:,.2f}" for l, h in zip(ff["Low"], ff["High"])], textposition="outside",
                hovertemplate="%{y}<br>Low %{base:,.2f}<extra></extra>")
    fig.add_scatter(y=ff["Method"], x=ff["Mid"], mode="markers", marker=dict(color=GREY, symbol="line-ns-open",
                    size=18, line_width=3), name="Mid / median")
    fig.add_vline(x=price, line_dash="dash", line_color=GREY, annotation_text=f"Market price {price:,.2f}")
    fig.update_layout(height=120 + 70 * len(ff), showlegend=False, xaxis_title=f"Value per share ({info.get('currency', '')})",
                      yaxis_autorange="reversed", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, key="chart_4")
    st.dataframe(ff.set_index("Method").style.format("{:,.2f}"))

# --------------------------------------------------------------------------
# 9. COMPANY DATA TAB
# --------------------------------------------------------------------------
with tab_data:
    st.subheader("Historical financials (Yahoo Finance)")
    hist = pd.DataFrame({"Revenue": rev_hist, "Net income": ni_hist, "Book value of equity": eq_hist})
    if div_hist is not None:
        hist["Dividends"] = div_hist
    hist = hist.sort_index()
    hist["Revenue growth"] = hist["Revenue"].pct_change()
    hist["Net margin"] = hist["Net income"] / hist["Revenue"]
    hist["ROE"] = hist["Net income"] / hist["Book value of equity"]
    if "Dividends" in hist:
        hist["Payout ratio"] = hist["Dividends"] / hist["Net income"]
    hist.index = [f"FY{d.year}" for d in hist.index]
    pct_cols = [c for c in ["Revenue growth", "Net margin", "ROE", "Payout ratio"] if c in hist]
    st.dataframe(hist.T.style.format("{:,.0f}", na_rep="–")
                 .format("{:.1%}", subset=pd.IndexSlice[pct_cols, :], na_rep="–"))

    if len(data["prices"]):
        fig = go.Figure(go.Scatter(x=data["prices"].index, y=data["prices"].values, mode="lines", line_color=BLACK))
        fig.add_hline(y=ri_value, line_dash="dot", line_color=ORANGE, annotation_text=f"RI value {ri_value:,.2f}")
        fig.update_layout(title="Share price - last 5 years", height=360)
        st.plotly_chart(fig, key="chart_5")

# --------------------------------------------------------------------------
# 10. METHOD & SOURCES TAB
# --------------------------------------------------------------------------
with tab_method:
    st.subheader("How the app values a company")
    st.markdown(
        "1. **Forecast:** revenue grows with the revenue growth rate; net income = revenue x net income margin; "
        "dividends = net income x payout ratio.\n"
        "2. **Book value (clean surplus):** ending book value = opening book value + net income - dividends.\n"
        "3. **Residual income (RI):** RI = net income - cost of equity x opening book value. "
        "A company only creates value if its return on equity is above its cost of equity.\n"
        "4. **Value:** book value today + present value of RI in the forecast years + present value of the "
        "terminal value, TV = RI(final year) x (1 + g) / (ke - g). Divided by the shares = value per share.\n"
        "5. **Cost of equity (CAPM):** ke = risk-free rate + beta x equity risk premium.\n"
        "6. **Trading comps:** peer median multiples (EV/Revenue, EV/EBITDA, P/E, P/Book) applied to the company; "
        "EV = market cap + debt - cash; range = peer 25th to 75th percentile.\n"
        "7. **Football field:** all value ranges next to the current share price.")
    st.subheader("Check against our Excel model")
    st.markdown(f"For Nike the app opens with exactly the assumptions of our manual Phase II Excel model "
                f"(cost of equity {EXCEL_CHECK['rf']:.2f}% + {EXCEL_CHECK['beta']:.2f} x {EXCEL_CHECK['erp']:.2f}%, "
                f"terminal growth {EXCEL_CHECK['g']:.2f}%) and reproduces its value of "
                f"**${EXCEL_CHECK['target_price']:.2f}** per share. Any other ticker starts with automatic assumptions "
                "(median of its last fiscal years) that can be changed in the sidebar.")
    st.subheader("Data")
    st.markdown("Yahoo Finance via the `yfinance` package: financial statements, share prices and peer data. "
                "If Yahoo's company profile is unavailable, key figures are taken from the statements and the "
                "price history instead.")
    st.caption("Academic project by Team 10 (WHU Otto Beisheim School of Management) - not investment advice.")
