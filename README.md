# Equity Valuation App - Team 10 (WHU PT-MiM)

Interactive valuation app built with **Streamlit** and **Yahoo Finance** for the WHU Part-Time Master in Management
final project (Phase III B). Enter any stock ticker and value the company with a **residual income model** and
**trading comparables**.

**Live app:** https://whu-team10-valuation.streamlit.app

> Default company: **NIKE, Inc. (NKE)**. The app opens with exactly the assumptions of our manual Phase II Excel model
> and reproduces its value of **$36.97 per share** (green check box at the top of the page).

## Features

| Tab | What it shows |
|---|---|
| Residual Income | Forecast table, value bridge (book value + PV of residual income + PV of terminal value), assumptions and output |
| Scenarios | Bear / Base / Bull values, probability-weighted value and residual income by scenario |
| Sensitivity | Value per share for cost of equity x terminal growth |
| Trading Comps | Peer multiples (EV/Revenue, EV/EBITDA, P/E, P/Book) and implied share price |
| Football Field | All value ranges next to the current share price |
| Company data | Historical financials and 5-year share price |
| Method | Formulas, Excel check and data sources |

The sidebar lets you change the forecast period, growth, margin, payout ratio, cost of equity (CAPM or manual),
terminal growth and the starting values - either one value for all years or year by year. The scenario switch
(Bear / Base / Bull) applies to every tab; spreads and probabilities are set under "Scenario settings".

## Method (short)

- **Residual income:** `RI = net income - ke x opening book value`; book value rolls forward with net income minus
  dividends (clean surplus).
- **Value:** book value today + PV of RI (forecast years) + PV of terminal value `RI x (1 + g) / (ke - g)`.
- **Cost of equity (CAPM):** `ke = rf + beta x ERP` (course inputs: rf 4.44%, beta 1.11, ERP 4.18%).
- **Trading comps:** peer median multiples, range = 25th to 75th percentile; EV = market cap + debt - cash.
- **Scenarios:** Base = sidebar assumptions; Bear / Bull = revenue growth -/+ 2 pp and net margin -/+ 1 pp (default),
  weighted 25 / 50 / 25 %.
- **Automatic assumptions** for other tickers: median of the last fiscal years, capped at realistic ranges.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface and Yahoo Finance data |
| `valuation.py` | Calculation engine (residual income, sensitivity, comps) - no Streamlit, testable |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Colour theme |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## How we used AI

The app was built for Phase III B with an LLM (Claude) as required by the assignment: the LLM wrote the code from our
requirements, we validated the result against our manual Excel model (Nike: $36.97), reported problems from the live
deployment back (e.g. unrealistic automatic assumptions and Yahoo data gaps on Streamlit Cloud) and iterated on the app.
We also tested a much larger version and deliberately went back to this focused version.

## Disclaimer

Academic project by Team 10 (WHU Otto Beisheim School of Management). Not investment advice. Data from Yahoo Finance
may be delayed, incomplete or inaccurate. NIKE is a trademark of NIKE, Inc., used for identification only.
