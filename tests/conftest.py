from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from trading_agent.agents import AgentContext
from trading_agent.broker.simulated import SimulatedBroker
from trading_agent.config import Events, Settings, TradingMode, load_risk_limits, load_universe
from trading_agent.ledger import Ledger
from trading_agent.market_calendar import ET
from trading_agent.orchestrator import Orchestrator
from trading_agent.schemas import AgentOutput, TradeProposal

# Tuesday 2026-09-22, 10:30 ET: a regular session.
IN_SESSION = datetime(2026, 9, 22, 10, 30, tzinfo=ET)
AFTER_CLOSE = datetime(2026, 9, 21, 17, 0, tzinfo=ET)


@pytest.fixture
def limits():
    return load_risk_limits()


@pytest.fixture
def universe():
    return load_universe()


@pytest.fixture
def ledger():
    return Ledger(":memory:")


def make_settings(tmp_path: Path, mode: TradingMode = TradingMode.BROKER, env: str = "uat") -> Settings:
    return Settings(trading_mode=mode, webull_environment=env, webull_region="sg", webull_account_id="SIM",
                    state_dir=tmp_path, llm_model="test")


def seeded_broker(now: datetime, cash: float = 1_000_000) -> SimulatedBroker:
    """SPY in an uptrend (risk-on regime) plus XLK trending at ~$100."""
    b = SimulatedBroker(cash=cash)
    b.set_trend_bars("SPY", start=400, daily_pct=0.05, end=now)
    b.set_trend_bars("XLK", start=70, daily_pct=0.15, end=now)
    for sym in ("SPY", "XLK"):
        last = b.bars[sym][-1].close
        b.set_quote(sym, bid=round(last - 0.01, 2), ask=round(last + 0.01, 2), last=last, fetched_at=now)
    return b


class ScriptedAgent:
    """Returns whatever proposal-builder it is given, so tests control the LLM's output."""

    name = "llm"
    model = "scripted"

    def __init__(self, build):
        self.build = build

    def propose(self, ctx: AgentContext) -> AgentOutput:
        return self.build(ctx)


def good_proposal(ctx: AgentContext, symbol: str = "XLK", **overrides) -> TradeProposal:
    f = ctx.features[symbol]
    px_id = next(e.evidence_id for e in ctx.evidence if e.kind == "price_features" and e.symbol == symbol)
    close, a = f["close"], f["atr14"]
    limit = round(close * 1.001, 2)
    data = dict(
        symbol=symbol, instrument_type="ETF", side="BUY", strategy="trend", thesis="Strong uptrend above averages.",
        holding_period_days=10, confidence=0.6, expected_return_pct=2.0, invalidation_price=round(limit - 2 * a, 2),
        entry={"order_type": "LIMIT", "limit_price": limit},
        exit={"take_profit": round(limit + 4 * a, 2), "stop_loss": round(limit - 2 * a, 2), "time_stop_days": 10},
        requested_risk_pct=0.25, evidence_ids=[px_id],
    )
    data.update(overrides)
    return TradeProposal(**data)


def make_orchestrator(tmp_path, broker, agent, now, mode=TradingMode.BROKER, env="uat", limits=None, events=None, enable_news=False, continuous_trading=None):
    return Orchestrator(settings=make_settings(tmp_path, mode, env), limits=limits or load_risk_limits(),
                        universe=load_universe(), events=events or Events(), broker=broker,
                        ledger=Ledger(tmp_path / "ledger.db"), agent=agent, now=lambda: now,
                        enable_news=enable_news, continuous_trading=continuous_trading)
