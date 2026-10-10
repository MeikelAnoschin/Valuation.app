# Equity Valuation App - Team 10 (WHU PT-MiM)

Interactive valuation app built with **Streamlit** and **Yahoo Finance** for the WHU Part-Time Master in Management final
project (Phase III B). Enter any stock ticker and value the company with a DCF, a residual income model and trading
comparables - with Bear / Base / Bull scenarios, a 12-month target price and a Buy / Hold / Sell rating.

**Live app:** `https://<your-app-name>.streamlit.app` *(replace with your link)*

> Default company: **NIKE, Inc. (NKE)**. Two built-in presets reproduce our Excel model exactly:
> the course Phase II residual income value (**$36.97**) and our sell-side report
> (DCF Bear **$25.42** / Base **$40.64** / Bull **$56.87**, target price **$38**, rating **HOLD**).

## Features

| Area | What it does |
|---|---|
| Data | Yahoo Finance via `yfinance` (prices, financial statements, profile where available). Robust fallbacks when Yahoo's profile endpoint is blocked; offline Nike data set (9 Oct 2026) as a safety net |
| Valuation models | FCFF **DCF** with WACC (CAPM), **residual income** model (clean surplus), **trading comps** (EV/Revenue, EV/EBITDA, P/E, P/Book) |
| Scenarios | Editable Bear / Base / Bull assumptions per year (growth, margins, tax, one-off costs), probability weighting |
| Target & rating | 12-month target price from DCF + comps, total return incl. dividends, Buy / Hold / Sell thresholds |
| Risk analysis | Tornado chart with driver ranking, WACC x growth sensitivity tables, **Monte Carlo** simulation, **reverse DCF** (what the share price implies) |
| Market | Football field, analyst consensus, peer suggestions by industry, side-by-side company comparison |
| Conventions | Optional mid-year discounting, valuation-date stub period, leases as debt, full equity bridge |
| Output | Excel export of all inputs and results, one-page **PDF tearsheet**, save / load assumptions (.json), shareable link |
| Help | Explain mode with short explanations, methodology tab with all formulas and sources |

## Methodology (short)

- **DCF:** `FCFF = EBIT x (1 - tax) + D&A - capex - increase in NWC + one-offs`; terminal value with the Gordon growth model
  (excluding one-offs); equity value = EV + cash - financial debt (- leases / minorities / pensions if selected).
- **WACC:** `ke = rf + beta x ERP` (course inputs: rf 4.44%, beta 1.11, ERP 4.18%); debt at book value, equity at market value.
- **Residual income:** `RI = NI - ke x opening book value`; value = book value + PV of RI + PV of terminal RI.
- **Target price:** `(w x DCF + (1 - w) x comps) x (1 + ke) - dividends over 12 months`.
- Default convention = the Excel model: end-of-year discounting, year 1 discounted one full year.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface |
| `valuation.py` | Calculation engine (DCF, RI, comps, Monte Carlo, reverse DCF, tornado) - no Streamlit, testable |
| `pdf_report.py` | One-page PDF tearsheet |
| `nke_offline.json` | Offline Nike data set with sources (used if Yahoo is unavailable) |
| `test_valuation.py` | Automated tests: the engine must reproduce the Excel values |
| `requirements.txt` | Python packages |
| `.streamlit/config.toml` | Colour theme |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
pip install pytest && pytest -q      # 12 tests, all must pass
```

## How we used AI

The app was built for Phase III B with an LLM (Claude) as required by the assignment: the LLM wrote the code from our
requirements, we validated every number against our manual Excel model (presets and automated tests), reported bugs from
the live deployment back (e.g. Yahoo's profile endpoint being blocked on Streamlit Cloud, which produced a wrong share
count) and iterated on design and functionality over several versions.

## Disclaimer

Academic project by Team 10 (WHU Otto Beisheim School of Management). Not investment advice. Data from Yahoo Finance
may be delayed, incomplete or inaccurate. NIKE is a trademark of NIKE, Inc., used for identification only.
