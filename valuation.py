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
