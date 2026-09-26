from __future__ import annotations

from datetime import datetime

from conftest import IN_SESSION, good_short_proposal, seeded_broker
from trading_agent.agents import AgentContext, BaselineMomentumAgent
from trading_agent.config import Events, load_risk_limits, load_universe
from trading_agent.features import build_evidence
from trading_agent.portfolio import size_order
from trading_agent.risk import RiskContext, evaluate
from trading_agent.schemas import Evidence


def test_default_risk_limits_have_shorting_enabled():
    limits = load_risk_limits()
    assert limits.live_trading.permit_shorting is True
    assert limits.account.max_short_exposure_pct == 50
    assert limits.account.max_gross_exposure_pct == 150
    assert limits.account.max_portfolio_risk_pct == 10.0
    assert limits.account.max_new_trades_per_day == 8
    assert limits.account.max_order_value_usd == 1500


def test_expanded_universe_contents():
    u = load_universe()
    assert len(u.symbols) >= 55

    # Check ETFs
    for sym in ("SPY", "QQQ", "IWM", "SMH", "XBI", "ARKK", "KRE", "XLK", "XLF", "XLE"):
        assert sym in u.symbols
        assert u.symbols[sym].type == "ETF"

    # Check Key Individual Equities for momentum & shorting
    for sym in ("AAPL", "NVDA", "INTC", "SNOW", "NKE", "PYPL", "COIN", "BA", "MRNA", "LLY"):
        assert sym in u.symbols
        assert u.symbols[sym].type == "EQUITY"


def test_short_order_passes_live_risk_checks(universe):
    limits = load_risk_limits()
    broker = seeded_broker(IN_SESSION)
    account = broker.get_account()
    symbols = ["SPY", "XLK"]
    evidence, features = build_evidence(broker.bars, broker.quotes, "SPY")

    ctx = AgentContext(
        as_of=IN_SESSION.date(),
        evidence=evidence,
        universe={s: {"type": universe.symbols[s].type, "sector": universe.symbols[s].sector} for s in symbols},
        account_summary={"equity_usd": account.equity, "cash_usd": account.cash},
        positions=[],
        known_earnings={},
        features=features,
    )

    prop = good_short_proposal(ctx, symbol="XLK")
    sized = size_order("d_live_short", prop, account, features["XLK"]["avg_volume_20d"], limits)
    assert not isinstance(sized, str)
    assert sized.side == "SELL_SHORT"

    risk_ctx = RiskContext(
        now=IN_SESSION,
        trading_date=IN_SESSION.date(),
        account=account,
        quote=broker.quotes["XLK"],
        kill_switch_engaged=False,
        sends_real_orders_to_prod=False,
        require_session=True,
        reconciled=True,
        entries_today=0,
        start_of_day_equity=account.equity,
        peak_equity=account.equity,
        decision_already_executed=False,
        open_risks=[],
    )

    decision = evaluate(sized, "ETF", 10, risk_ctx, limits, universe, Events())
    assert decision.approved, f"Failed checks: {decision.failed_checks}"
    assert "long_only" not in decision.failed_checks


def test_baseline_agent_proposes_shorts_in_bear_regime():
    """In a bear regime (benchmark below 200 SMA), the baseline agent should actively propose shorts."""
    now = IN_SESSION
    broker = seeded_broker(now)

    # Set SPY in a downtrend (below 200 SMA)
    broker.set_trend_bars("SPY", start=500, daily_pct=-0.15, end=now)
    # Set XLK in a downtrend (weakest mover, below both SMAs)
    broker.set_trend_bars("XLK", start=120, daily_pct=-0.25, end=now)

    for sym in ("SPY", "XLK"):
        last = broker.bars[sym][-1].close
        broker.set_quote(sym, bid=round(last - 0.01, 2), ask=round(last + 0.01, 2), last=last, fetched_at=now)

    evidence, features = build_evidence(broker.bars, broker.quotes, "SPY")

    ctx = AgentContext(
        as_of=now.date(),
        evidence=evidence,
        universe={"SPY": {"type": "ETF", "sector": "Broad Market"}, "XLK": {"type": "ETF", "sector": "Technology"}},
        account_summary={"equity_usd": 100_000, "cash_usd": 100_000},
        positions=[],
        known_earnings={},
        features=features,
    )

    agent = BaselineMomentumAgent(max_ideas=4)
    out = agent.propose(ctx)

    assert "risk-off" in out.market_view
    assert len(out.proposals) >= 1
    short_proposals = [p for p in out.proposals if p.side == "SELL_SHORT"]
    assert len(short_proposals) >= 1
    assert short_proposals[0].symbol == "XLK"
    assert short_proposals[0].exit.stop_loss > short_proposals[0].entry.limit_price
    assert short_proposals[0].exit.take_profit < short_proposals[0].entry.limit_price
