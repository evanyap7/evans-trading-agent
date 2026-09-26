import sqlite3
from datetime import timedelta

import pytest

from trading_agent.broker.base import OrderRequest
from trading_agent.broker.simulated import SimulatedBroker
from trading_agent.execution import ExecutionEngine, client_order_id
from trading_agent.ledger import Ledger
from trading_agent.schemas import OrderState, utcnow


def test_events_are_append_only(ledger):
    ledger.append("x", {"a": 1})
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute("UPDATE events SET payload='{}'")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.db.execute("DELETE FROM events")


def test_hash_chain_detects_tampering(tmp_path):
    led = Ledger(tmp_path / "l.db")
    for i in range(3):
        led.append("x", {"i": i})
    assert led.verify_chain()
    led.db.execute("DROP TRIGGER events_no_update")
    led.db.execute("UPDATE events SET payload='{\"i\":99}' WHERE seq=2")
    assert not led.verify_chain()


def _req(decision="d1", qty=5, limit=101.0):
    return OrderRequest(client_order_id=client_order_id(decision, "ENTRY"), symbol="XLK", side="BUY",
                        order_type="LIMIT", quantity=qty, time_in_force="DAY", limit_price=limit)


@pytest.fixture
def sim():
    b = SimulatedBroker()
    b.set_quote("XLK", bid=100.99, ask=101.01, last=101.0)
    return b


def test_shadow_mode_sends_nothing(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=False)
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.SHADOW
    assert sim.place_calls == 0


def test_duplicate_scheduler_run_places_once(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    eng.submit(_req(), "d1", "ENTRY")
    eng.submit(_req(), "d1", "ENTRY")
    assert sim.place_calls == 1


def test_timeout_after_broker_received_is_reconciled_not_retried(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "timeout_received"
    state = eng.submit(_req(limit=100.0), "d1", "ENTRY")  # rests (not marketable)
    assert state == OrderState.ACKNOWLEDGED
    assert sim.place_calls == 1


def test_timeout_where_order_was_lost_stays_unknown_then_resolves(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "timeout_lost"
    coid = _req().client_order_id
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.UNKNOWN_RECONCILE
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.UNKNOWN_RECONCILE  # no blind retry
    assert sim.place_calls == 1
    old = (utcnow() - timedelta(minutes=30)).isoformat()
    ledger.db.execute("UPDATE orders SET updated_at=? WHERE client_order_id=?", (old, coid))
    assert eng.sync(coid) == OrderState.REJECTED


def test_explicit_rejection_is_terminal(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    sim.fail_next_place = "reject"
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.REJECTED


def test_preview_mismatch_blocks_submission(sim, ledger, limits):
    from trading_agent.broker.base import PreviewResult

    sim.preview = lambda o: PreviewResult(ok=True, estimated_cost=99_999.0, estimated_fees=0)
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    assert eng.submit(_req(), "d1", "ENTRY") == OrderState.REJECTED
    assert sim.place_calls == 0


def test_partial_fill_then_restart_is_recovered(sim, ledger, limits, tmp_path):
    """A new engine (after restart) picks up broker state from the ledger + broker."""
    from trading_agent.schemas import BrokerOrderStatus

    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    coid = _req(limit=100.0).client_order_id
    eng.submit(_req(limit=100.0), "d1", "ENTRY")
    o = sim.orders[coid]
    sim.orders[coid] = o.model_copy(update={"status": BrokerOrderStatus.PARTIALLY_FILLED, "filled_quantity": 2,
                                            "filled_price": 100.0})
    restarted = ExecutionEngine(sim, ledger, limits, send_orders=True)
    assert restarted.sync(coid) == OrderState.PARTIALLY_FILLED
    assert ledger.get_order(coid)["filled_quantity"] == 2


# -- Short execution and ledger tests -------------------------------------------

def test_short_protective_stop_is_buy_order(sim, ledger, limits):
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    state = eng.place_protective_stop("d_short_1", "XLK", 10, 105.0, is_short=True)
    assert state == OrderState.ACKNOWLEDGED
    stop_order = next(o for o in sim.orders.values() if o.order_type == "STOP_LOSS")
    assert stop_order.side == "BUY"
    assert stop_order.stop_price == 105.0


def test_short_exit_is_buy_cover(sim, ledger, limits):
    from datetime import date
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    state = eng.exit_trade("d_short_2", "XLK", 5, 95.0, "take_profit", date(2026, 9, 22), is_short=True)
    assert state == OrderState.ACKNOWLEDGED
    coid = client_order_id("d_short_2", "EXIT:2026-09-22")
    exit_order = sim.orders[coid]
    assert exit_order.side == "BUY"
    # Buy to cover limit collar is above market
    assert exit_order.limit_price >= 95.0


def test_ledger_stores_side_and_raise_stop_direction(ledger):
    from datetime import date
    # Long trade: stop only moves UP
    ledger.open_trade("d_long", "AAPL", 10, 100.0, 95.0, 110.0, date(2026, 10, 1), side="BUY")
    t_long = next(t for t in ledger.open_trades() if t["decision_id"] == "d_long")
    assert t_long["side"] == "BUY"
    ledger.raise_stop("d_long", 97.0)  # moves up: allowed
    assert next(t for t in ledger.open_trades() if t["decision_id"] == "d_long")["stop_loss"] == 97.0
    ledger.raise_stop("d_long", 94.0)  # moves down: rejected for long
    assert next(t for t in ledger.open_trades() if t["decision_id"] == "d_long")["stop_loss"] == 97.0

    # Short trade: stop only moves DOWN
    ledger.open_trade("d_short", "TSLA", 5, 200.0, 210.0, 180.0, date(2026, 10, 1), side="SELL_SHORT")
    t_short = next(t for t in ledger.open_trades() if t["decision_id"] == "d_short")
    assert t_short["side"] == "SELL_SHORT"
    ledger.raise_stop("d_short", 205.0)  # moves down (toward profit): allowed
    assert next(t for t in ledger.open_trades() if t["decision_id"] == "d_short")["stop_loss"] == 205.0
    ledger.raise_stop("d_short", 212.0)  # moves up (away from profit): rejected for short
    assert next(t for t in ledger.open_trades() if t["decision_id"] == "d_short")["stop_loss"] == 205.0


def test_exit_waits_for_async_stop_cancel(sim, ledger, limits, monkeypatch):
    """Webull confirms cancels a few seconds later. The exit must wait for that rather than defer and let
    the next monitor pass re-arm the stop (which left an LLM-requested SMCI exit unsent in production)."""
    from datetime import date

    from trading_agent.schemas import BrokerOrderStatus

    monkeypatch.setattr("time.sleep", lambda s: None)
    eng = ExecutionEngine(sim, ledger, limits, send_orders=True)
    eng.place_protective_stop("d_async", "XLK", 5, 95.0)
    stop_coid = client_order_id("d_async", "STOP")
    real_cancel, real_get_order, lag = sim.cancel, sim.get_order, {"polls": 0}

    def async_cancel(coid):  # accepted, but reported WORKING for two more polls
        lag["polls"] = 2

    def lagging_get_order(coid):
        if coid == stop_coid and lag["polls"]:
            lag["polls"] -= 1
            if not lag["polls"]:
                real_cancel(coid)
            return real_get_order(coid).model_copy(update={"status": BrokerOrderStatus.WORKING})
        return real_get_order(coid)

    monkeypatch.setattr(sim, "cancel", async_cancel)
    monkeypatch.setattr(sim, "get_order", lagging_get_order)

    state = eng.exit_trade("d_async", "XLK", 5, 101.0, "agent_thesis_exit", date(2026, 9, 22))
    assert ledger.get_order(stop_coid)["state"] == OrderState.CANCELLED.value
    assert state == OrderState.FILLED  # the marketable exit went out instead of deferring
    assert not ledger.has_event("exit_deferred", "d_async")

