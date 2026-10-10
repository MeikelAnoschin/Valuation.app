"""
valuation.py - the calculation engine of the app (no Streamlit, no internet).

Kept separate from app.py so the maths can be tested on its own and
compared with the manual Excel model.
All money amounts are in millions; rates are decimals (0.05 = 5 %).
"""
import numpy as np
import pandas as pd


def residual_income_valuation(rev0, bv0, shares, growth, margin, payout, ke, g, base_year):
    """Simplified residual income model.

    rev0, bv0  : revenue and book value of equity in the base year
    shares     : shares outstanding (millions)
    growth     : list of revenue growth rates, one per forecast year
    margin     : list of net income margins, one per forecast year
    payout     : dividend payout ratio
    ke, g      : cost of equity, terminal growth rate
    """
    if ke <= g:
        raise ValueError("Cost of equity must be higher than terminal growth.")
    if len(growth) != len(margin):
        raise ValueError("Growth and margin lists must have the same length.")

    rows = []
    revenue, bv = rev0, bv0
    pv_ri = 0.0
    for t, (gr, m) in enumerate(zip(growth, margin), start=1):
        revenue = revenue * (1 + gr)
        net_income = revenue * m
        dividends = net_income * payout
        bv_begin = bv
        equity_charge = ke * bv_begin                 # cost of equity in $
        ri = net_income - equity_charge               # residual income
        bv = bv_begin + net_income - dividends        # clean surplus roll-forward
        discount = 1 / (1 + ke) ** t
        pv_ri += ri * discount
        rows.append({
            "Year": base_year + t,
            "Revenue growth": gr,
            "Revenue": revenue,
            "Net margin": m,
            "Net income": net_income,
            "Dividends": dividends,
            "Beginning book value": bv_begin,
            "Ending book value": bv,
            "ROE": net_income / bv_begin if bv_begin else np.nan,
            "Equity charge": equity_charge,
            "Residual income": ri,
            "Discount factor": discount,
            "PV of RI": ri * discount,
        })

    terminal_value = ri * (1 + g) / (ke - g)          # at end of final year
    pv_tv = terminal_value * discount
    equity_value = bv0 + pv_ri + pv_tv
    return {
        "table": pd.DataFrame(rows),
        "bv0": bv0,
        "pv_ri": pv_ri,
        "terminal_value": terminal_value,
        "pv_tv": pv_tv,
        "equity_value": equity_value,
        "per_share": equity_value / shares,
    }


def sensitivity_grid(base_kwargs, ke_values, g_values):
    """Value per share for every combination of cost of equity and terminal growth."""
    grid = pd.DataFrame(index=ke_values, columns=g_values, dtype=float)
    for ke in ke_values:
        for g in g_values:
            try:
                grid.loc[ke, g] = residual_income_valuation(**{**base_kwargs, "ke": ke, "g": g})["per_share"]
            except ValueError:
                grid.loc[ke, g] = np.nan
    return grid


MULTIPLES = ["EV/Revenue", "EV/EBITDA", "P/E", "P/Book"]


def implied_prices_from_comps(peers, revenue, ebitda, eps, bvps, net_debt, shares):
    """Apply the peers' 25th percentile / median / 75th percentile multiples to the target.

    peers    : DataFrame with one column per multiple (NaN where not meaningful)
    revenue, ebitda, net_debt : target company, millions
    eps, bvps: target company, per share
    shares   : millions
    Returns a DataFrame: Multiple, Low, Mid, High (implied price per share), Peers used
    """
    out = []
    for mult in MULTIPLES:
        vals = pd.to_numeric(peers.get(mult), errors="coerce").dropna() if mult in peers else pd.Series(dtype=float)
        vals = vals[vals > 0]
        if vals.empty:
            continue
        q = vals.quantile([0.25, 0.5, 0.75]).values
        if mult == "EV/Revenue":
            metric_ok = revenue and revenue > 0
            prices = [(x * revenue - net_debt) / shares for x in q] if metric_ok else None
        elif mult == "EV/EBITDA":
            metric_ok = ebitda and ebitda > 0
            prices = [(x * ebitda - net_debt) / shares for x in q] if metric_ok else None
        elif mult == "P/E":
            metric_ok = eps and eps > 0
            prices = [x * eps for x in q] if metric_ok else None
        else:  # P/Book
            metric_ok = bvps and bvps > 0
            prices = [x * bvps for x in q] if metric_ok else None
        if prices is None:
            continue
        out.append({
            "Multiple": mult,
            "Peer 25th pct": q[0], "Peer median": q[1], "Peer 75th pct": q[2],
            "Low": prices[0], "Mid": prices[1], "High": prices[2],
            "Peers used": len(vals),
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# DCF (free cash flow to firm) - same logic as the 'DCF' sheet of Team 10 Final.xlsx
# ---------------------------------------------------------------------------
def wacc_calc(ke, kd_pre_tax, tax, equity_value, debt):
    """Weighted average cost of capital using market value of equity and book debt."""
    total = equity_value + debt
    we = equity_value / total if total else 1.0
    wd = debt / total if total else 0.0
    kd = kd_pre_tax * (1 - tax)
    return {"wacc": we * ke + wd * kd, "we": we, "wd": wd, "kd_after_tax": kd}


def dcf_valuation(rev0, growth, ebit_margin, tax, da_pct, capex_pct, nwc_pct, one_off,
                  wacc, g, net_cash, shares, base_year):
    """Five-to-ten-year FCFF DCF.

    growth, ebit_margin, tax, one_off : lists, one value per forecast year
    one_off   : cash costs such as restructuring (negative numbers), excluded from terminal value
    net_cash  : cash minus debt (negative if net debt)
    FCFF = EBIT x (1 - tax) + D&A - capex - increase in NWC + one-off cash items
    Terminal value = (final-year FCFF excl. one-offs) x (1 + g) / (WACC - g)
    """
    if wacc <= g:
        raise ValueError("WACC must be higher than terminal growth.")
    rows, rev, pv = [], rev0, 0.0
    for t in range(len(growth)):
        prev = rev
        rev = prev * (1 + growth[t])
        ebit = rev * ebit_margin[t]
        taxes = -ebit * tax[t]
        nopat = ebit + taxes
        da = rev * da_pct
        capex = -rev * capex_pct
        d_nwc = -(rev - prev) * nwc_pct
        fcff = nopat + da + capex + d_nwc + one_off[t]
        disc = 1 / (1 + wacc) ** (t + 1)
        pv += fcff * disc
        rows.append({"Year": base_year + t + 1, "Revenue": rev, "Revenue growth": growth[t], "EBIT margin": ebit_margin[t],
                     "EBIT": ebit, "Taxes": taxes, "NOPAT": nopat, "D&A": da, "Capex": capex, "Change in NWC": d_nwc,
                     "One-off cash costs": one_off[t], "FCFF": fcff, "Discount factor": disc, "PV of FCFF": fcff * disc})
    tv = (fcff - one_off[-1]) * (1 + g) / (wacc - g)
    pv_tv = tv * disc
    ev = pv + pv_tv
    equity = ev + net_cash
    return {"table": pd.DataFrame(rows), "pv_fcff": pv, "terminal_value": tv, "pv_tv": pv_tv, "ev": ev,
            "net_cash": net_cash, "equity_value": equity, "per_share": equity / shares,
            "tv_share": pv_tv / ev if ev else np.nan}


# ---------------------------------------------------------------------------
# Vectorised DCF (for Monte Carlo) and helper analyses
# ---------------------------------------------------------------------------
def dcf_vectorized(rev0, growth, ebit_margin, tax, da_pct, capex_pct, nwc_pct, one_off, wacc, g, net_cash, shares):
    """Same maths as dcf_valuation, for many simulations at once.

    growth, ebit_margin : arrays (n_sims x years);  tax, one_off : arrays (years,)
    wacc, g             : arrays (n_sims,)
    Returns value per share for every simulation.
    """
    growth, ebit_margin = np.atleast_2d(growth), np.atleast_2d(ebit_margin)
    tax, one_off = np.asarray(tax, float), np.asarray(one_off, float)
    wacc, g = np.asarray(wacc, float), np.asarray(g, float)
    rev = rev0 * np.cumprod(1 + growth, axis=1)
    prev = np.hstack([np.full((rev.shape[0], 1), rev0), rev[:, :-1]])
    fcff = rev * ebit_margin * (1 - tax) + rev * da_pct - rev * capex_pct - (rev - prev) * nwc_pct + one_off
    t = np.arange(1, rev.shape[1] + 1)
    disc = (1 + wacc)[:, None] ** (-t)
    tv = (fcff[:, -1] - one_off[-1]) * (1 + g) / (wacc - g)
    return ((fcff * disc).sum(axis=1) + tv * disc[:, -1] + net_cash) / shares


def monte_carlo(dcf_kwargs, n=3000, sd_growth=0.02, sd_margin=0.015, sd_wacc=0.0075, sd_g=0.005, seed=42):
    """Random shifts (normal) applied to every forecast year of a scenario; returns array of values per share.
    Draws where WACC - g < 0.5pp are discarded (terminal value not meaningful)."""
    rng = np.random.default_rng(seed)
    k = dcf_kwargs
    gr, mr = np.array(k["growth"]), np.array(k["ebit_margin"])
    dg, dm = rng.normal(0, sd_growth, n), rng.normal(0, sd_margin, n)
    w = k["wacc"] + rng.normal(0, sd_wacc, n)
    tg = k["g"] + rng.normal(0, sd_g, n)
    ok = (w - tg) > 0.005
    vals = dcf_vectorized(k["rev0"], gr[None, :] + dg[ok, None], mr[None, :] + dm[ok, None], k["tax"], k["da_pct"],
                          k["capex_pct"], k["nwc_pct"], k["one_off"], w[ok], tg[ok], k["net_cash"], k["shares"])
    return vals


def solve_for_price(value_fn, target, lo, hi, tol=1e-6, max_iter=200):
    """Bisection: find x in [lo, hi] with value_fn(x) = target. Returns None if not bracketed."""
    flo, fhi = value_fn(lo) - target, value_fn(hi) - target
    if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
        return None
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        fm = value_fn(mid) - target
        if abs(fm) < tol:
            return mid
        if flo * fm < 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return (lo + hi) / 2


def tornado(dcf_kwargs, base_value):
    """One-at-a-time DCF sensitivities. Returns DataFrame sorted by impact (largest first)."""
    k = dcf_kwargs
    shift = lambda lst, d: [x + d for x in lst]
    tests = [
        ("Revenue growth (+/-1pp every year)", {"growth": shift(k["growth"], -0.01)}, {"growth": shift(k["growth"], 0.01)}),
        ("EBIT margin (+/-1pp every year)", {"ebit_margin": shift(k["ebit_margin"], -0.01)}, {"ebit_margin": shift(k["ebit_margin"], 0.01)}),
        ("WACC (+/-0.5pp)", {"wacc": k["wacc"] + 0.005}, {"wacc": k["wacc"] - 0.005}),
        ("Terminal growth (+/-0.5pp)", {"g": k["g"] - 0.005}, {"g": k["g"] + 0.005}),
        ("Tax rate (+/-2pp)", {"tax": shift(k["tax"], 0.02)}, {"tax": shift(k["tax"], -0.02)}),
        ("Capex (+/-0.5pp of revenue)", {"capex_pct": k["capex_pct"] + 0.005}, {"capex_pct": k["capex_pct"] - 0.005}),
        ("Net working capital (+/-2pp of revenue)", {"nwc_pct": k["nwc_pct"] + 0.02}, {"nwc_pct": k["nwc_pct"] - 0.02}),
    ]
    rows = []
    for name, down, up in tests:
        try:
            lo = dcf_valuation(**{**k, **down})["per_share"]
            hi = dcf_valuation(**{**k, **up})["per_share"]
        except ValueError:
            continue
        rows.append({"Driver": name, "Downside value": lo, "Upside value": hi,
                     "Downside %": lo / base_value - 1, "Upside %": hi / base_value - 1,
                     "Range": abs(hi - lo)})
    df = pd.DataFrame(rows).sort_values("Range", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    return df
