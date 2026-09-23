from datetime import date, timedelta

import pytest

from conftest import IN_SESSION, good_proposal, seeded_broker
from trading_agent.agents import AgentContext
from trading_agent.config import Events
from trading_agent.features import build_evidence
from trading_agent.portfolio import size_order
from trading_agent.risk import OpenRisk, RiskContext, evaluate
from trading_agent.verifier import verify


@pytest.fixture
def setup():
    b = seeded_broker(IN_SESSION)
    ev, feats = build_evidence(b.bars, b.quotes, "SPY")
    ctx = AgentContext(as_of=IN_SESSION.date(), evidence=ev, universe={}, account_summary={}, positions=[],
                       known_earnings={}, features=feats)
    return b, ctx, {e.evidence_id: e.symbol for e in ev}, feats


def run_verify(p, cyc, feats, b, limits, universe, now=IN_SESSION, fresh=True):
    return verify(p, cycle_evidence=cyc, features=feats.get(p.symbol), quote=b.quotes.get(p.symbol), limits=limits,
                  universe=universe, now=now, require_fresh_quote=fresh)


def test_good_proposal_verifies(setup, limits, universe):
    b, ctx, cyc, feats = setup
    v = run_verify(good_proposal(ctx), cyc, feats, b, limits, universe)
    assert v.passed, v.failures


def test_hallucinated_evidence_rejected(setup, limits, universe):
    b, ctx, cyc, feats = setup
    v = run_verify(good_proposal(ctx, evidence_ids=["px_XLK_20260922_madeup"]), cyc, feats, b, limits, universe)
    assert not v.passed and any("unknown evidence" in f for f in v.failures)


def test_hallucinated_price_rejected(setup, limits, universe):
    b, ctx, cyc, feats = setup
    close = feats["XLK"]["close"]
    p = good_proposal(ctx, entry={"order_type": "LIMIT", "limit_price": round(close * 1.10, 2)},
                      exit={"take_profit": close * 1.3, "stop_loss": close * 1.05, "time_stop_days": 10},
                      invalidation_price=close * 1.05)
    assert any("collar" in f for f in run_verify(p, cyc, feats, b, limits, universe).failures)


def test_stale_data_rejected(setup, limits, universe):
    b, ctx, cyc, feats = setup
    later = IN_SESSION + timedelta(days=10)
    v = run_verify(good_proposal(ctx), cyc, feats, b, limits, universe, now=later)
    assert any("stale" in f for f in v.failures) and any("fresh quote" in f for f in v.failures)


def test_no_edge_after_costs_rejected(setup, limits, universe):
    b, ctx, cyc, feats = setup
    v = run_verify(good_proposal(ctx, expected_return_pct=0.3), cyc, feats, b, limits, universe)
    assert any("net edge" in f for f in v.failures)


def test_overclaimed_return_rejected(setup, limits, universe):
    b, ctx, cyc, feats = setup
    v = run_verify(good_proposal(ctx, expected_return_pct=40), cyc, feats, b, limits, universe)
    assert any("exceeds the take-profit" in f for f in v.failures)


def test_sizing_uses_risk_not_llm(setup, limits):
    b, ctx, _, feats = setup
    acct = b.get_account()
    s = size_order("d1", good_proposal(ctx, requested_risk_pct=1.0), acct, feats["XLK"]["avg_volume_20d"], limits)
    assert not isinstance(s, str)
    assert s.notional <= limits.account.max_order_value_usd
    assert s.risk_usd / acct.equity * 100 <= limits.account.max_risk_per_trade_pct + 1e-9


def _ctx(b, **kw):
    base = dict(now=IN_SESSION, trading_date=IN_SESSION.date(), account=b.get_account(), quote=b.quotes["XLK"],
                kill_switch_engaged=False, sends_real_orders_to_prod=False, require_session=True, reconciled=True,
                entries_today=0, start_of_day_equity=None, peak_equity=None, decision_already_executed=False)
    base.update(kw)
    return RiskContext(**base)


def _sized(setup, limits):
    b, ctx, _, feats = setup
    p = good_proposal(ctx)
    return b, p, size_order("d1", p, b.get_account(), feats["XLK"]["avg_volume_20d"], limits)


def test_risk_approves_clean_order(setup, limits, universe):
    b, p, s = _sized(setup, limits)
    d = evaluate(s, "ETF", 10, _ctx(b), limits, universe, Events())
    assert d.approved, d.failed_checks


@pytest.mark.parametrize("kw,check", [
    ({"kill_switch_engaged": True}, "kill_switch_off"),
    ({"sends_real_orders_to_prod": True}, "live_trading_enabled_for_prod"),
    ({"reconciled": False}, "broker_reconciled"),
    ({"decision_already_executed": True}, "not_duplicate_decision"),
    ({"entries_today": 3}, "new_trades_per_day_ok"),
    ({"now": IN_SESSION.replace(hour=18)}, "market_session_open"),
    ({"start_of_day_equity": 1_100_000}, "daily_loss_ok"),
    ({"peak_equity": 1_200_000}, "drawdown_ok"),
    ({"open_risks": [OpenRisk("XLK", 10, 100, 95)]}, "no_existing_exposure_in_symbol"),
    ({"open_risks": [OpenRisk("AAPL", 2000, 100, None)]}, "portfolio_risk_ok"),
])
def test_risk_limits_block(setup, limits, universe, kw, check):
    b, p, s = _sized(setup, limits)
    d = evaluate(s, "ETF", 10, _ctx(b, **kw), limits, universe, Events())
    assert not d.approved and check in d.failed_checks


def test_wide_spread_blocks(setup, limits, universe):
    b, p, s = _sized(setup, limits)
    last = b.quotes["XLK"].last
    b.set_quote("XLK", bid=last * 0.99, ask=last * 1.01, last=last, fetched_at=IN_SESSION)
    d = evaluate(s, "ETF", 10, _ctx(b, quote=b.quotes["XLK"]), limits, universe, Events())
    assert "spread_ok" in d.failed_checks


def test_stock_without_earnings_date_blocked_and_earnings_in_window_blocked(setup, limits, universe):
    b, p, s = _sized(setup, limits)
    s = s.model_copy(update={"symbol": "AAPL"})
    no_data = evaluate(s, "EQUITY", 10, _ctx(b), limits, universe, Events())
    assert "earnings_date_known" in no_data.failed_checks
    soon = Events(earnings={"AAPL": IN_SESSION.date() + timedelta(days=7)})
    assert "no_earnings_in_holding_window" in evaluate(s, "EQUITY", 10, _ctx(b), limits, universe, soon).failed_checks
    later = Events(earnings={"AAPL": date(2026, 12, 1)})
    assert "no_earnings_in_holding_window" not in evaluate(s, "EQUITY", 10, _ctx(b), limits, universe, later).failed_checks
