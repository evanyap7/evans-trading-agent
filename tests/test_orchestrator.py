"""End-to-end cycles against the simulated broker."""

from conftest import AFTER_CLOSE, IN_SESSION, ScriptedAgent, good_proposal, make_orchestrator, seeded_broker
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
    broker = seeded_broker(IN_SESSION)
    orch = make_orchestrator(tmp_path, broker, ScriptedAgent(one_idea), AFTER_CLOSE, env="prod")
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
