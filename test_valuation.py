"""
test_valuation.py - automated checks that the app's valuation engine reproduces the Team 10 Excel model.

Run locally:   pip install pytest   then   pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from valuation import (dcf_valuation, dcf_vectorized, implied_prices_from_comps, monte_carlo,
                       residual_income_valuation, solve_for_price, tornado, wacc_calc)

SHARES = 1484.699                      # diluted shares (m), Market Data sheet
KE = 0.0444 + 1.11 * 0.0418            # CAPM cost of equity, course inputs
WACC = wacc_calc(KE, 0.032, 0.23, 34.71 * SHARES, 7890)["wacc"]
BASE = dict(rev0=46398, growth=[-0.0819132290184922, 0.0136610412046372, 0.0465742243009708, 0.0488446334351913,
                                0.0405815365507638],
            ebit_margin=[0.06, 0.075, 0.095, 0.11, 0.12], tax=[0.25, 0.23, 0.23, 0.23, 0.23], da_pct=0.016,
            capex_pct=0.017, nwc_pct=5884 / 46398, one_off=[-300, -250, -200, -150, -100], wacc=WACC, g=0.0215,
            net_cash=480, shares=SHARES, base_year=2026)
BEAR = {**BASE, "growth": [-0.10, -0.02, 0.02, 0.03, 0.03], "ebit_margin": [0.05, 0.055, 0.07, 0.08, 0.09], "g": 0.015}
BULL = {**BASE, "growth": [-0.06, 0.05, 0.07, 0.06, 0.05], "ebit_margin": [0.065, 0.09, 0.115, 0.13, 0.14], "g": 0.0275}


def test_cost_of_capital_matches_excel():
    assert KE == pytest.approx(0.090798, abs=1e-9)
    assert WACC == pytest.approx(0.0820139, abs=1e-6)


def test_phase2_residual_income_value():
    """Course Phase II model: $36.97 per share."""
    rev = [45660.2718, 47806.3045746, 50053.200889606196, 51684.93523860736, 53369.864127385954]
    ni = [3342.7963438299985, 4124.03252791902, 4393.855926965158, 4697.1027298564695, 5011.914857077373]
    r = residual_income_valuation(46398, 14865, 1479.8, [-0.0159, 0.047, 0.047, 0.0326, 0.0326],
                                  [n / v for n, v in zip(ni, rev)], 0.6232, KE, 0.0215, 2026)
    assert r["per_share"] == pytest.approx(36.9704, abs=1e-4)
    assert r["pv_ri"] == pytest.approx(10293.86, abs=0.01)
    assert r["terminal_value"] == pytest.approx(45632.98, abs=0.01)


@pytest.mark.parametrize("kwargs,expected", [(BEAR, 25.4168), (BASE, 40.6366), (BULL, 56.8696)])
def test_report_dcf_scenarios(kwargs, expected):
    """Sell-side report: DCF Bear $25.42 / Base $40.64 / Bull $56.87."""
    assert dcf_valuation(**kwargs)["per_share"] == pytest.approx(expected, abs=1e-4)


def test_report_comps_and_target_price():
    peers = pd.DataFrame({"EV/Revenue": [1.17, 0.79, 1.84, 2.73, 1.01, 2.17],
                          "EV/EBITDA": [9.20, 25.81, 7.65, 17.94, 4.70, 13.29],
                          "P/E": [14.17, np.nan, 10.88, 19.03, 11.34, 18.77],
                          "P/Book": [4.03, 1.78, 4.92, 4.96, 2.20, 2.30]})
    imp = implied_prices_from_comps(peers, revenue=45888, ebitda=3904, eps=1.32, bvps=14865 / SHARES,
                                    net_debt=-480, shares=SHARES)
    comps = imp["Mid"].mean()
    assert comps == pytest.approx(31.7809, abs=1e-4)
    fair = 0.5 * dcf_valuation(**BASE)["per_share"] + 0.5 * comps
    target = round(fair * (1 + KE) - 1.64)
    assert target == 38


def test_vectorised_dcf_equals_scalar_dcf():
    for stub, mid in [(1.0, False), (1.0, True), (0.64, False), (0.64, True)]:
        scalar = dcf_valuation(**BASE, stub=stub, mid_year=mid)["per_share"]
        vec = dcf_vectorized(BASE["rev0"], np.array([BASE["growth"]]), np.array([BASE["ebit_margin"]]), BASE["tax"],
                             BASE["da_pct"], BASE["capex_pct"], BASE["nwc_pct"], BASE["one_off"], np.array([WACC]),
                             np.array([0.0215]), 480, SHARES, stub, mid)[0]
        assert vec == pytest.approx(scalar, rel=1e-12)


def test_conventions_move_value_in_expected_direction():
    base = dcf_valuation(**BASE)["per_share"]
    assert dcf_valuation(**BASE, mid_year=True)["per_share"] > base       # cash earlier -> higher value
    assert dcf_valuation(**BASE, stub=0.64)["per_share"] > base           # valuation date closer to cash flows


def test_monte_carlo_is_reproducible_and_centred():
    a, b = monte_carlo(BASE, n=2000), monte_carlo(BASE, n=2000)
    assert np.array_equal(a, b)
    assert np.median(a) == pytest.approx(40.6, abs=1.5)


def test_reverse_dcf_recovers_price():
    f = lambda x: dcf_valuation(**{**BASE, "ebit_margin": [m + x for m in BASE["ebit_margin"]]})["per_share"]
    shift = solve_for_price(f, 34.71, -0.3, 0.4)
    assert f(shift) == pytest.approx(34.71, abs=1e-4)
    assert 0.12 + shift == pytest.approx(0.1046, abs=5e-4)


def test_tornado_ranks_margin_first():
    t = tornado(BASE, dcf_valuation(**BASE)["per_share"])
    assert t.iloc[0]["Driver"].startswith("EBIT margin")


def test_wacc_must_exceed_growth():
    with pytest.raises(ValueError):
        dcf_valuation(**{**BASE, "g": 0.09})
