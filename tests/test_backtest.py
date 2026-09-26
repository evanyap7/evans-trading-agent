"""Replay engine: fill/exit mechanics, no lookahead, and the live pipeline wired end to end."""

from datetime import date, datetime, time, timedelta

import pytest

from conftest import ScriptedAgent, good_proposal
from trading_agent.agents import BaselineMomentumAgent
from trading_agent.backtest import AgentOutputCache, Backtester, OpenPosition, manage_bar
from trading_agent.config import load_risk_limits, load_universe
from trading_agent.market_calendar import ET
from trading_agent.schemas import AgentOutput, Bar

HISTORY = 260


def weekdays(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def path(px: float, returns: list[float], start=date(2024, 1, 1), volume=5e6) -> list[Bar]:
    """Daily bars: each opens at the prior close and moves by the given percent, with a 1% range."""
    bars = []
    for d, r in zip(weekdays(start, len(returns)), returns):
        nxt = px * (1 + r / 100)
        bars.append(Bar(ts=datetime.combine(d, time(16), ET), open=px, high=max(px, nxt) * 1.01,
                        low=min(px, nxt) * 0.99, close=nxt, volume=volume))
        px = nxt
    return bars


def run(bars, agent, cash=100_000, limits=None, **kw):
    start = bars["SPY"][HISTORY - 1].ts.date()
    bt = Backtester(bars=bars, agent=agent, limits=limits or load_risk_limits(), universe=load_universe(),
                    start=start, end=bars["SPY"][-1].ts.date(), starting_cash=cash, **kw)
    return bt.run()


def first_day_only(ctx):
    first = getattr(first_day_only, "day", None) or ctx.as_of
    first_day_only.day = first
    return AgentOutput(market_view="x", proposals=[good_proposal(ctx)] if ctx.as_of == first else [])


# -- exit mechanics --------------------------------------------------------------------


def pos(**kw):
    base = dict(decision_id="d", symbol="XLK", qty=10, entry=100.0, initial_stop=96.0, stop=96.0,
                take_profit=108.0, entry_idx=0, time_stop_idx=10)
    return OpenPosition(**base | kw)


def bar(o, h, l, c):
    return Bar(ts=datetime(2024, 6, 3, 16, tzinfo=ET), open=o, high=h, low=l, close=c, volume=1e6)


def test_stop_gap_target_and_ambiguous_bar():
    assert manage_bar(pos(), bar(101, 102, 99, 101), 1, 0) is None
    assert manage_bar(pos(), bar(101, 102, 95, 97), 1, 0) == (96.0, "stop")
    assert manage_bar(pos(), bar(90, 91, 88, 89), 1, 0) == (90, "stop_gap")        # gap fills at the open, not the stop
    assert manage_bar(pos(), bar(101, 109, 100, 108), 1, 0) == (108.0, "take_profit")
    assert manage_bar(pos(), bar(110, 111, 109, 110), 1, 0) == (110, "take_profit")  # gap up: sold at the open
    assert manage_bar(pos(), bar(101, 109, 95, 100), 1, 0) == (96.0, "stop")       # both touched: assume stop first


def test_time_stop_and_entry_day():
    assert manage_bar(pos(), bar(101, 102, 100, 101), 10, 0) == (101, "time_stop")
    # A position bought this morning ignores the opening gap checks but not its intraday range.
    assert manage_bar(pos(entry_idx=3), bar(95, 97, 94, 96), 3, 0) == (96.0, "stop")
    px, _ = manage_bar(pos(), bar(101, 102, 95, 97), 1, 10)  # 10 bps of slippage on the sell
    assert px == pytest.approx(96 * 0.999)


# -- replay ------------------------------------------------------------------------------


def trend_universe(extra: list[float] | None = None):
    tail = extra or [0.1] * 40
    return {"SPY": path(400, [0.05] * HISTORY + [0.05] * len(tail)),
            "XLK": path(70, [0.15] * HISTORY + tail)}


def test_agent_never_sees_the_future():
    seen = []

    def spy(ctx):
        seen.append((ctx.as_of, max(f["bar_date"] for f in ctx.features.values())))
        return AgentOutput(market_view="x")

    res = run(trend_universe(), ScriptedAgent(spy))
    assert seen and all(date.fromisoformat(latest) <= as_of for as_of, latest in seen)
    assert len(res.equity) == len(res.benchmark) == 41


def test_baseline_end_to_end_trades_through_live_pipeline():
    res = run(trend_universe(), BaselineMomentumAgent())
    s = res.summary()
    assert s["trades"] >= 1 and s["proposals"] >= 1
    assert all(t.symbol in ("SPY", "XLK") for t in res.trades)
    assert s["ending_equity"] == pytest.approx(res.equity[-1][1])


def test_trailing_stop_turns_a_round_trip_into_breakeven():
    # Five up days (~+1.4R at the close), then a day that trades back through the original stop.
    scenario = [1.2] * 5 + [-12.0] + [0.0] * 5
    trailing = run(trend_universe(scenario), ScriptedAgent(first_day_only))
    first_day_only.day = None
    limits = load_risk_limits()
    fixed = limits.model_copy(update={"execution": limits.execution.model_copy(update={"trailing_stops": False})})
    no_trail = run(trend_universe(scenario), ScriptedAgent(first_day_only), limits=fixed)
    first_day_only.day = None

    (t1,), (t2,) = trailing.trades, no_trail.trades
    assert t1.reason == t2.reason == "stop"
    assert t1.r_multiple == pytest.approx(0, abs=0.03)  # breakeven less slippage both ways
    assert t2.r_multiple == pytest.approx(-1, abs=0.05)


def test_drawdown_kill_switch_stops_new_entries():
    def every_day(ctx):
        held = {p["symbol"] for p in ctx.positions}
        return AgentOutput(market_view="x", proposals=[] if "XLK" in held else [good_proposal(ctx, requested_risk_pct=2.0)])

    def crashing():
        bars = trend_universe([0.2] * 3 + [-60.0] + [0.2] * 20)
        b = bars["XLK"][HISTORY + 3]  # open the crash day at its close: a gap far below the stop
        bars["XLK"][HISTORY + 3] = b.model_copy(update={"open": b.close, "high": b.close * 1.01})
        return bars

    res = run(crashing(), ScriptedAgent(every_day), cash=1500)
    assert res.halted_on is not None
    assert all(t.entry_date <= res.halted_on for t in res.trades)

    kept_going = run(crashing(), ScriptedAgent(every_day), cash=1500, halt_on_kill_switch=False)
    assert kept_going.halted_on == res.halted_on and kept_going.kill_switch_trips[0] == res.halted_on
    assert any(t.entry_date > res.halted_on for t in kept_going.trades)


def test_output_cache_replays_without_calling_the_agent(tmp_path):
    calls = []

    def counted(ctx):
        calls.append(ctx.as_of)
        return AgentOutput(market_view="x")

    run(trend_universe(), ScriptedAgent(counted), output_cache=AgentOutputCache(tmp_path))
    n = len(calls)
    run(trend_universe(), ScriptedAgent(counted), output_cache=AgentOutputCache(tmp_path))
    assert n > 0 and len(calls) == n

    budget = AgentOutputCache(tmp_path / "fresh", max_new_calls=2)
    res = run(trend_universe(), ScriptedAgent(counted), output_cache=budget)
    assert budget.misses == 2 and res.rejections["agent error: RuntimeError"] > 0


def test_calibration_buckets_confidence_against_outcomes():
    from trading_agent.backtest import calibration

    table = calibration([(0.62, 2.0), (0.65, -1.0), (0.68, -1.0), (0.81, 1.5), (1.0, 2.0)])
    assert table["0.6-0.7"] == {"trades": 3, "avg_confidence": 0.65, "win_rate": 0.333, "avg_r": 0.0}
    assert table["0.9-1.0"]["trades"] == 1  # confidence 1.0 lands in the top bucket, not a new one
    assert set(table) == {"0.6-0.7", "0.8-0.9", "0.9-1.0"}
