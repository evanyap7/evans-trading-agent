from __future__ import annotations

import numpy as np
import pandas as pd

from trading_agent import ideas


def fixture():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2025-08-01", periods=300)
    mkt = rng.normal(0.0004, 0.01, 300)
    cols = {"SPY": mkt, "TECH": 1.5 * mkt + rng.normal(0, 0.005, 300),            # held, tech
            "HEDGE": rng.normal(0.0002, 0.008, 300),                              # uncorrelated, calm
            "CYCL": 1.2 * mkt + rng.normal(0, 0.006, 300),                        # market-like
            "WILD": rng.normal(0, 0.04, 300),                                     # far too volatile
            "HELD2": rng.normal(0, 0.008, 300)}
    px = pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in cols.items()}, index=idx)
    table = pd.DataFrame({"Security": ["Tech Co", "Hedge Co", "Cyc Co", "Wild Co", "Held Two"],
                          "GICS Sector": ["Information Technology", "Utilities", "Utilities", "Utilities", "Utilities"],
                          "GICS Sub-Industry": ["x"] * 5}, index=["TECH", "HEDGE", "CYCL", "WILD", "HELD2"])
    return table, px


def test_screen_finds_gaps_and_prefers_low_correlation():
    table, px = fixture()
    sc = ideas.screen(table, px, {"SPYM": 0.7, "TECH": 0.3, "HELD2": 0.02}, core="SPYM")
    assert "Utilities" in sc["gaps"] and "Information Technology" not in sc["gaps"]  # HELD2 is only 2%: still a gap
    assert sc["sector_pct"]["Broad index"] == 70.0
    syms = [d["symbol"] for d in sc["shortlist"]]
    assert "WILD" not in syms and "TECH" not in syms and "HELD2" not in syms        # too volatile / already held
    assert syms.index("HEDGE") < syms.index("CYCL")                                 # lower correlation ranks first


def test_check_ideas_only_keeps_screened_symbols_and_caps_count():
    short = [{"symbol": s} for s in "ABCD"]
    mk = lambda s: ideas.Idea(symbol=s, size_pct=2, why="w", wrong_if="x")  # noqa: E731
    kept, dropped = ideas.check_ideas([mk("A"), mk("ZZZ"), mk("B"), mk("C"), mk("D")], short)
    assert [i.symbol for i in kept] == ["A", "B", "C"] and {i.symbol for i in dropped} == {"ZZZ", "D"}
    assert "Nothing on the screened list" in ideas.render_ideas({"gaps": ["Utilities"], "shortlist": short}, [])
