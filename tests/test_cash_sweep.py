"""Cash sweep: idle cash parked in an ETF, sold to fund entries, invisible to limits and breakers."""

import pytest

from conftest import AFTER_CLOSE, IN_SESSION, ScriptedAgent, good_proposal, make_orchestrator, seeded_broker
from test_backtest import HISTORY, path, run
from trading_agent.agents import BaselineMomentumAgent
from trading_agent.config import CashSweep, Events, TradingMode, load_risk_limits, load_universe
from trading_agent.schemas import AgentOutput, OrderState, Position
from trading_agent.sweep import PURPOSE

SYM = "SPYM"


def sweep_limits(**kw):
    base = load_risk_limits()
    return base.model_copy(update={"cash_sweep": CashSweep(enabled=True, symbol=SYM, **kw)})


def broker_with_sweep_quote(cash):
    b = seeded_broker(IN_SESSION, cash=cash)
    b.set_trend_bars(SYM, start=60, daily_pct=0.05, end=IN_SESSION)
    b.set_quote(SYM, bid=89.99, ask=90.01, last=90.0, fetched_at=IN_SESSION)
    return b


def no_ideas(ctx):
    return AgentOutput(market_view="flat", no_trade_reason="nothing")


def one_idea(ctx):
    return AgentOutput(market_view="risk-on", proposals=[good_proposal(ctx)])


def sweep_orders(orch):
    return list(orch.ledger.iter_rows("SELECT * FROM orders WHERE purpose=?", (PURPOSE,)))


def test_config_default_off_and_symbol_checked():
    assert load_risk_limits().cash_sweep.enabled is False
    with pytest.raises(ValueError):
        CashSweep(enabled=True, symbol="spy m")


def test_execute_parks_spare_cash_above_reserve(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    rep = orch.execute()
    assert any("cash sweep: BUY" in n for n in rep.notes), rep.notes
    held = broker.positions[SYM].quantity
    # 5% reserve of $2,000 stays in cash; whole shares at the collared ask of ~$90.46
    assert held == 21 and broker.cash >= 100  # floor(1,900 / 90.46)
    assert orch.ledger.open_trades() == []  # the sweep is not a trade


def test_no_sweep_buys_while_kill_switch_engaged(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    orch.kill.engage("test", by="test")
    orch.execute()
    assert SYM not in broker.positions and broker.place_calls == 0


def test_disabled_sweep_places_nothing(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION)
    orch.execute()
    assert SYM not in broker.positions


def test_entry_is_funded_by_selling_sweep_shares(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    limits = sweep_limits()
    make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=limits).execute()
    assert broker.cash < 200  # nearly everything is swept

    research = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), AFTER_CLOSE, limits=limits)
    rep = research.research()
    # Sized from spendable cash (sweep included), and the sweep's ETF is not counted as exposure or risk.
    assert any("XLK: queued" in n for n in rep.notes), rep.notes

    execute = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), IN_SESSION, limits=limits)
    rep = execute.execute()
    assert any("cash sweep: SELL" in n for n in rep.notes), rep.notes
    assert any("XLK: BUY" in n and "FILLED" in n for n in rep.notes), rep.notes
    assert broker.positions["XLK"].quantity == execute.ledger.open_trades()[0]["quantity"]
    assert broker.cash >= 0


def test_agent_never_sees_or_closes_sweep_shares(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    limits = sweep_limits()
    make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=limits).execute()
    seen = {}

    def record(ctx):
        seen["positions"] = [p["symbol"] for p in ctx.positions]
        seen["cash"] = ctx.account_summary["cash_usd"]
        return no_ideas(ctx)

    make_orchestrator(tmp_path, broker, ScriptedAgent(record), AFTER_CLOSE, limits=limits).research()
    assert SYM not in seen["positions"]
    assert seen["cash"] > 1_800  # sweep value shows up as cash


def test_manual_holding_in_sweep_symbol_is_not_swept(tmp_path):
    broker = broker_with_sweep_quote(cash=100)
    broker.positions[SYM] = Position(symbol=SYM, quantity=10, avg_cost=80, last_price=90)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    orch._account()
    assert orch._sweep.qty == 0  # bought by hand, so the sweep does not own it and will not sell it
    assert orch._view(broker.get_account()).position(SYM).quantity == 10


def test_market_drop_in_sweep_does_not_trip_drawdown_breaker(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    orch.execute()
    broker.set_quote(SYM, bid=80.99, ask=81.01, last=81.0, fetched_at=IN_SESSION)  # -10%: ~-9% of the account
    rep = orch.monitor()
    assert not orch.kill.engaged(), rep.notes


def test_same_drop_in_a_manual_holding_still_trips_it(tmp_path):
    broker = broker_with_sweep_quote(cash=200)
    broker.positions[SYM] = Position(symbol=SYM, quantity=20, avg_cost=90, last_price=90)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    orch.monitor()
    broker.set_quote(SYM, bid=80.99, ask=81.01, last=81.0, fetched_at=IN_SESSION)
    orch.monitor()
    assert orch.kill.engaged()


def test_missing_sweep_shares_fail_reconciliation(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, limits=sweep_limits())
    orch.execute()
    broker.positions.pop(SYM)  # sold outside the agent
    rep = orch.monitor()
    assert orch.kill.engaged() and any("cash sweep expects" in n for n in rep.notes), rep.notes


def test_risk_engine_rejects_agent_entries_in_sweep_symbol(tmp_path):
    from trading_agent.risk import RiskContext, evaluate
    from trading_agent.schemas import AccountState, SizedOrder

    limits = sweep_limits()
    order = SizedOrder(decision_id="d", symbol=SYM, side="BUY", quantity=1, limit_price=90, stop_loss=85,
                       take_profit=100, notional=90, risk_usd=5)
    acct = AccountState(account_id="a", equity=1000, cash=1000, buying_power=1000, positions=[], open_orders=[],
                        as_of=IN_SESSION)
    ctx = RiskContext(now=IN_SESSION, trading_date=IN_SESSION.date(), account=acct, quote=None,
                      kill_switch_engaged=False, sends_real_orders_to_prod=False, require_session=False,
                      reconciled=True, entries_today=0, start_of_day_equity=None, peak_equity=None,
                      decision_already_executed=False)
    d = evaluate(order, "ETF", 5, ctx, limits, load_universe(), Events())
    assert "not_cash_sweep_symbol" in d.failed_checks


def test_shadow_mode_records_sweep_orders_without_sending(tmp_path):
    broker = broker_with_sweep_quote(cash=2_000)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(no_ideas), IN_SESSION, mode=TradingMode.SHADOW,
                             limits=sweep_limits())
    orch.execute()
    rows = sweep_orders(orch)
    assert broker.place_calls == 0 and rows and rows[0]["state"] == OrderState.SHADOW.value
    assert orch._sweep.qty == 0


# -- backtest ----------------------------------------------------------------------------


def _bars():
    n = HISTORY + 60
    return {"SPY": path(400, [0.1] * n), "XLK": path(70, [0.2] * n), SYM: path(60, [0.1] * n)}


def test_backtest_sweep_invests_idle_cash_and_reconciles():
    bars = _bars()
    plain = run(bars, BaselineMomentumAgent(), cash=900)
    swept = run(bars, BaselineMomentumAgent(), cash=900, limits=sweep_limits())
    s = swept.summary()
    assert s["sweep"]["symbol"] == SYM and s["sweep"]["orders"] >= 1 and s["sweep"]["avg_swept_pct"] > 50
    assert s["total_return_pct"] > plain.summary()["total_return_pct"]  # a rising sweep ETF earns on idle cash
    assert plain.summary().get("sweep") is None


def test_backtest_reports_trend_benchmark():
    s = run(_bars(), BaselineMomentumAgent(), cash=900).summary()
    # SPY rises every day, so the 200-day filter stays invested and matches buy-and-hold closely.
    assert s["trend_benchmark_return_pct"] == pytest.approx(s["benchmark_return_pct"], abs=0.5)
    assert "excess_vs_benchmark_pct" in s and "avg_invested_pct" in s
