"""End-to-end cycles against the simulated broker."""

from unittest.mock import patch

from conftest import AFTER_CLOSE, IN_SESSION, ScriptedAgent, good_proposal, good_short_proposal, make_orchestrator, seeded_broker
from trading_agent.config import TradingMode
from trading_agent.schemas import AgentOutput, OrderState


def one_idea(ctx):
    return AgentOutput(market_view="risk-on", proposals=[good_proposal(ctx)])


def _run_research_then_execute(tmp_path, mode=TradingMode.BROKER, env="uat"):
    broker = seeded_broker(IN_SESSION)
    research = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), AFTER_CLOSE, mode, env)
    rep = research.research()
    assert any("queued" in n for n in rep.notes), rep.notes
    execute = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), IN_SESSION, mode, env)
    return broker, execute, execute.execute()


def test_full_cycle_entry_fill_stop_and_take_profit(tmp_path):
    broker, orch, rep = _run_research_then_execute(tmp_path)
    assert any("BUY" in n for n in rep.notes), rep.notes
    trades = orch.ledger.open_trades()
    assert len(trades) == 1 and broker.positions["XLK"].quantity == trades[0]["quantity"]
    stops = [o for o in broker.orders.values() if o.order_type == "STOP_LOSS"]
    assert len(stops) == 1 and stops[0].quantity == trades[0]["quantity"]

    tp = trades[0]["take_profit"]
    broker.set_quote("XLK", bid=tp + 0.5, ask=tp + 0.6, last=tp + 0.55, fetched_at=IN_SESSION)
    rep = orch.monitor()
    assert any("take_profit" in n for n in rep.notes), rep.notes
    assert "XLK" not in broker.positions
    assert orch.ledger.open_trades() == []
    assert all(o.status.value != "WORKING" for o in broker.orders.values())  # stop was cancelled, not orphaned


def test_shadow_mode_records_without_sending(tmp_path):
    broker, orch, rep = _run_research_then_execute(tmp_path, mode=TradingMode.SHADOW)
    assert broker.place_calls == 0
    assert any(OrderState.SHADOW.value in n for n in rep.notes)


def test_prod_blocked_while_live_trading_disabled(tmp_path):
    from trading_agent.config import load_risk_limits

    broker = seeded_broker(IN_SESSION)
    disabled_limits = load_risk_limits()
    disabled_limits = disabled_limits.model_copy(update={"live_trading": disabled_limits.live_trading.model_copy(update={"enabled": False})})
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), AFTER_CLOSE, env="prod", limits=disabled_limits)
    rep = orch.research()
    assert any("live_trading_enabled_for_prod" in n for n in rep.notes), rep.notes
    assert broker.place_calls == 0


def test_agent_crash_is_a_no_trade(tmp_path):
    def boom(ctx):
        raise RuntimeError("model unavailable")

    orch = make_orchestrator(tmp_path, seeded_broker(IN_SESSION), ScriptedAgent(boom), AFTER_CLOSE)
    rep = orch.research()
    assert any("agent error" in n for n in rep.notes)


def test_hallucinated_symbol_never_reaches_broker(tmp_path):
    def fake(ctx):
        p = good_proposal(ctx).model_copy(update={"symbol": "ZZZZ"})
        return AgentOutput(market_view="", proposals=[p])

    broker = seeded_broker(IN_SESSION)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(fake), AFTER_CLOSE)
    rep = orch.research()
    assert any("not in universe" in n for n in rep.notes)
    assert orch.ledger.pending() == []


def test_position_mismatch_engages_kill_switch_and_blocks_entries(tmp_path):
    broker, orch, _ = _run_research_then_execute(tmp_path)
    broker.positions.pop("XLK")  # position vanished outside the system
    broker.orders.clear()
    rep = orch.monitor()
    assert orch.kill.engaged() and any("KILL SWITCH" in n for n in rep.notes)
    rep = orch.execute()
    assert any("kill switch engaged" in n for n in rep.notes)


def test_operator_kill_cancels_working_entries(tmp_path):
    broker = seeded_broker(IN_SESSION)

    def resting(ctx):  # limit well below market so it rests
        f = ctx.features["XLK"]
        lim = round(f["close"] - 1.5 * f["atr14"], 2)
        return AgentOutput(market_view="", proposals=[good_proposal(
            ctx, entry={"order_type": "LIMIT", "limit_price": lim},
            exit={"take_profit": round(lim + 4 * f["atr14"], 2), "stop_loss": round(lim - 2 * f["atr14"], 2),
                  "time_stop_days": 10},
            invalidation_price=round(lim - 2 * f["atr14"], 2), expected_return_pct=2.0)])

    make_orchestrator(tmp_path, broker, ScriptedAgent(resting), AFTER_CLOSE).research()
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(resting), IN_SESSION)
    orch.execute()
    working = [o for o in broker.orders.values() if o.status.value == "WORKING"]
    assert len(working) == 1
    orch.kill.engage("operator test", by="test")
    orch.execute()
    assert broker.orders[working[0].client_order_id].status.value == "CANCELLED"


def test_duplicate_research_run_enters_once(tmp_path):
    broker = seeded_broker(IN_SESSION)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), AFTER_CLOSE)
    orch.research()
    orch.research()  # duplicate research run: same symbol queued twice
    ex = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), IN_SESSION)
    rep = ex.execute()
    buys = [n for n in rep.notes if "BUY" in n]
    assert len(buys) == 1, rep.notes  # second is blocked by no_existing_exposure_in_symbol


def test_tick_runs_each_cycle_once_per_day(tmp_path):
    from datetime import datetime

    from trading_agent.market_calendar import ET

    broker = seeded_broker(IN_SESSION)
    kinds = lambda reps: [r.kind for r in reps]  # noqa: E731
    at = lambda h, m: make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), datetime(2026, 9, 22, h, m, tzinfo=ET))  # noqa: E731
    assert kinds(at(9, 35).tick()) == ["monitor"]            # before the execute window
    assert kinds(at(9, 50).tick()) == ["execute", "monitor"]
    assert kinds(at(9, 55).tick()) == ["monitor"]            # execute already ran today
    assert kinds(at(16, 30).tick()) == ["research", "monitor"]
    assert kinds(at(16, 35).tick()) == []
    sat = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), datetime(2026, 9, 26, 12, 0, tzinfo=ET))
    assert sat.tick() == []


def test_pre_existing_position_exit_for_capital_rotation(tmp_path):
    from trading_agent.schemas import ExitProposal, Position

    broker = seeded_broker(IN_SESSION)
    # Simulate a pre-existing holding in the broker not opened by the ledger
    broker.positions["GOOG"] = Position(symbol="GOOG", quantity=1.0, avg_cost=150.0, last_price=170.0)
    broker.set_quote("GOOG", bid=169.90, ask=170.10, last=170.0, fetched_at=IN_SESSION)
    broker.set_trend_bars("GOOG", start=140, daily_pct=0.05, end=IN_SESSION)

    def exit_goog(ctx):
        px_id = next(e.evidence_id for e in ctx.evidence if e.kind == "price_features" and e.symbol == "GOOG")
        return AgentOutput(
            market_view="rotate into higher momentum",
            exits=[ExitProposal(symbol="GOOG", reason="Rotate capital into higher velocity breakout", evidence_ids=[px_id])]
        )

    from trading_agent.config import load_risk_limits

    base = load_risk_limits()

    def with_manual_closes(allowed):
        return base.model_copy(update={"live_trading": base.live_trading.model_copy(
            update={"agent_may_close_manual_positions": allowed})})

    opted_in = with_manual_closes(True)

    # Switched off: the agent may not sell a holding it did not open.
    blocked = make_orchestrator(tmp_path / "blocked", broker, ScriptedAgent(exit_goog), AFTER_CLOSE,
                                limits=with_manual_closes(False)).research()
    assert any("exit for GOOG ignored: not a system trade" in n for n in blocked.notes), blocked.notes

    orch_res = make_orchestrator(tmp_path, broker, ScriptedAgent(exit_goog), AFTER_CLOSE, limits=opted_in)
    rep_res = orch_res.research()
    assert any("exit queued for GOOG" in n for n in rep_res.notes), rep_res.notes

    orch_exec = make_orchestrator(tmp_path, broker, ScriptedAgent(exit_goog), IN_SESSION, limits=opted_in)
    rep_exec = orch_exec.execute()
    assert any("GOOG: agent exit SELL 1" in n for n in rep_exec.notes), rep_exec.notes


@patch("trading_agent.alerts.send_telegram", return_value=True)
def test_morning_briefing_cycle(mock_send, tmp_path):
    broker = seeded_broker(IN_SESSION)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), IN_SESSION)
    rep = orch.morning_briefing()
    assert any("morning briefing dispatched: True" in n for n in rep.notes), rep.notes
    assert mock_send.called


def test_small_gain_no_longer_forces_an_exit(tmp_path):
    """The old +$10 'daily target' sale capped winners far below 1R; open trades now run to stop/target/time."""
    broker, orch, rep = _run_research_then_execute(tmp_path)
    t = orch.ledger.open_trades()[0]
    entry, r = t["entry_price"], t["entry_price"] - t["stop_loss"]
    last = round(entry + 0.5 * r, 2)
    broker.set_quote(t["symbol"], bid=last, ask=last + 0.02, last=last, fetched_at=IN_SESSION)
    rep = orch.monitor()
    assert not any("SELL" in n for n in rep.notes), rep.notes
    assert t["symbol"] in broker.positions and len(orch.ledger.open_trades()) == 1


def test_stop_trails_to_breakeven_at_one_r(tmp_path):
    broker, orch, rep = _run_research_then_execute(tmp_path)
    t = orch.ledger.open_trades()[0]
    entry, r = t["entry_price"], t["entry_price"] - t["stop_loss"]
    last = round(entry + 1.2 * r, 2)  # take-profit sits at +2R
    broker.set_quote(t["symbol"], bid=last, ask=last + 0.02, last=last, fetched_at=IN_SESSION)
    rep = orch.monitor()
    assert any("trail stop" in n for n in rep.notes), rep.notes

    trade = orch.ledger.open_trades()[0]
    assert trade["stop_loss"] == entry and trade["initial_stop"] == t["stop_loss"]
    working = [o for o in broker.orders.values() if o.order_type == "STOP_LOSS" and o.status.value == "WORKING"]
    assert len(working) == 1 and working[0].stop_price == entry  # old stop cancelled, one replacement working

    rep = orch.monitor()  # same price: nothing more to do
    assert not any("trail stop" in n for n in rep.notes), rep.notes
    assert len([o for o in broker.orders.values() if o.order_type == "STOP_LOSS"]) == 2



def test_continuous_intraday_trading_tick(tmp_path):
    from datetime import datetime
    from trading_agent.market_calendar import ET

    broker = seeded_broker(IN_SESSION)
    kinds = lambda reps: [r.kind for r in reps]  # noqa: E731
    at = lambda h, m: make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea),
                                        datetime(2026, 9, 22, h, m, tzinfo=ET), continuous_trading=True)
    reps = at(9, 50).tick()
    # Continuous trading active: runs execute, research (intraday slot), execute (pending entries), monitor
    assert "research" in kinds(reps)
    assert "monitor" in kinds(reps)


# -- Short trading end-to-end tests ---------------------------------------------

def one_short_idea(ctx):
    return AgentOutput(market_view="risk-off", proposals=[good_short_proposal(ctx)])


def _run_short_research_then_execute(tmp_path, mode=TradingMode.BROKER, env="uat"):
    broker = seeded_broker(IN_SESSION)
    from trading_agent.config import load_risk_limits
    short_limits = load_risk_limits()
    short_limits = short_limits.model_copy(update={
        "live_trading": short_limits.live_trading.model_copy(update={"permit_shorting": True})
    })
    research = make_orchestrator(tmp_path, broker, ScriptedAgent(one_short_idea), AFTER_CLOSE, mode, env, limits=short_limits)
    rep = research.research()
    assert any("queued" in n for n in rep.notes), rep.notes
    execute = make_orchestrator(tmp_path, broker, ScriptedAgent(one_short_idea), IN_SESSION, mode, env, limits=short_limits)
    return broker, execute, execute.execute(), short_limits


def test_full_cycle_short_entry_fill_stop_and_take_profit(tmp_path):
    broker, orch, rep, short_limits = _run_short_research_then_execute(tmp_path)
    assert any("SELL" in n for n in rep.notes), rep.notes
    trades = orch.ledger.open_trades()
    assert len(trades) == 1
    t = trades[0]
    assert t["side"] == "SELL_SHORT"
    assert broker.positions["XLK"].quantity == -t["quantity"]

    stops = [o for o in broker.orders.values() if o.order_type == "STOP_LOSS"]
    assert len(stops) == 1
    assert stops[0].side == "BUY"
    assert stops[0].stop_price == t["stop_loss"]
    assert stops[0].quantity == t["quantity"]

    # Price drops to take-profit (profitable for short)
    tp = t["take_profit"]
    broker.set_quote("XLK", bid=tp - 0.5, ask=tp - 0.4, last=tp - 0.45, fetched_at=IN_SESSION)
    rep = orch.monitor()
    assert any("take_profit" in n for n in rep.notes), rep.notes
    assert "XLK" not in broker.positions
    assert orch.ledger.open_trades() == []
    assert all(o.status.value != "WORKING" for o in broker.orders.values())  # protective stop was cancelled


def test_short_stop_trails_to_breakeven_at_one_r(tmp_path):
    broker, orch, rep, short_limits = _run_short_research_then_execute(tmp_path)
    t = orch.ledger.open_trades()[0]
    entry, r = t["entry_price"], t["stop_loss"] - t["entry_price"]  # stop is ABOVE entry for short
    last = round(entry - 1.2 * r, 2)  # price drops by 1.2R
    broker.set_quote(t["symbol"], bid=last - 0.02, ask=last, last=last, fetched_at=IN_SESSION)
    rep = orch.monitor()
    assert any("trail stop" in n for n in rep.notes), rep.notes

    trade = orch.ledger.open_trades()[0]
    assert trade["stop_loss"] == entry and trade["initial_stop"] == t["stop_loss"]
    working = [o for o in broker.orders.values() if o.order_type == "STOP_LOSS" and o.status.value == "WORKING"]
    assert len(working) == 1 and working[0].stop_price == entry
    assert working[0].side == "BUY"


