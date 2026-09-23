"""Deterministic pre-trade risk engine. Pure functions of config and observed state.

Every check is named and evaluated (no short-circuit) so each rejection
records exactly which limits failed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime

from .config import Events, RiskLimits, Universe
from .market_calendar import is_regular_session
from .schemas import AccountState, Quote, RiskDecision, SizedOrder


@dataclass
class OpenRisk:
    """A position or working entry order and the stop that bounds its loss."""

    symbol: str
    quantity: float
    price: float
    stop: float | None


@dataclass
class RiskContext:
    now: datetime
    trading_date: date
    account: AccountState
    quote: Quote | None
    kill_switch_engaged: bool
    sends_real_orders_to_prod: bool
    require_session: bool
    reconciled: bool
    entries_today: int
    start_of_day_equity: float | None
    peak_equity: float | None
    decision_already_executed: bool
    open_risks: list[OpenRisk] = field(default_factory=list)


def evaluate(order: SizedOrder, instrument_type: str, holding_period_days: int, ctx: RiskContext,
             limits: RiskLimits, universe: Universe, events: Events) -> RiskDecision:
    a, ex, lt = limits.account, limits.execution, limits.live_trading
    eq = ctx.account.equity
    sec = universe.get(order.symbol)
    checks: dict[str, bool] = {}

    checks["kill_switch_off"] = not ctx.kill_switch_engaged
    checks["live_trading_enabled_for_prod"] = lt.enabled or not ctx.sends_real_orders_to_prod
    checks["instrument_allowed"] = instrument_type in lt.allowed_instruments
    checks["symbol_allowlisted"] = sec is not None
    checks["long_only"] = order.side == "BUY" or lt.permit_shorting
    checks["account_fresh"] = (ctx.now - ctx.account.as_of).total_seconds() <= ex.max_account_age_seconds
    checks["broker_reconciled"] = ctx.reconciled
    checks["not_duplicate_decision"] = not ctx.decision_already_executed
    checks["no_existing_exposure_in_symbol"] = not any(r.symbol == order.symbol for r in ctx.open_risks)
    checks["positive_equity"] = eq > 0

    if ctx.require_session:
        q = ctx.quote
        checks["market_session_open"] = is_regular_session(ctx.now)
        checks["quote_fresh"] = q is not None and q.age_seconds(ctx.now) <= ex.max_quote_age_seconds
        checks["spread_ok"] = q is not None and math.isfinite(q.spread_pct) and q.spread_pct <= ex.max_spread_pct
        checks["price_collar"] = q is not None and q.ask > 0 and order.limit_price <= q.ask * (1 + ex.price_collar_pct / 100)

    committed = sum(r.quantity * r.price for r in ctx.open_risks)
    sector_committed = sum(r.quantity * r.price for r in ctx.open_risks
                           if sec and (s := universe.get(r.symbol)) and s.sector == sec.sector)
    open_risk_usd = sum(
        r.quantity * max(r.price - r.stop, 0) if r.stop is not None else r.quantity * r.price * a.unknown_stop_risk_pct / 100
        for r in ctx.open_risks
    )
    pct = (lambda x: x / eq * 100) if eq > 0 else (lambda x: math.inf)

    checks["order_value_ok"] = order.notional <= a.max_order_value_usd
    checks["position_pct_ok"] = pct(order.notional) <= a.max_position_pct
    checks["sector_pct_ok"] = pct(sector_committed + order.notional) <= a.max_sector_pct
    checks["total_exposure_ok"] = pct(committed + order.notional) <= a.max_total_exposure_pct
    checks["risk_per_trade_ok"] = pct(order.risk_usd) <= a.max_risk_per_trade_pct + 1e-9
    checks["portfolio_risk_ok"] = pct(open_risk_usd + order.risk_usd) <= a.max_portfolio_risk_pct
    checks["new_trades_per_day_ok"] = ctx.entries_today < a.max_new_trades_per_day
    checks["cash_covers_order"] = order.notional + ex.fee_per_order_usd <= min(ctx.account.cash, ctx.account.buying_power)
    checks["daily_loss_ok"] = (ctx.start_of_day_equity is None
                               or pct(eq - ctx.start_of_day_equity) > -a.max_daily_loss_pct)
    checks["drawdown_ok"] = ctx.peak_equity is None or (eq / ctx.peak_equity - 1) * 100 > -a.max_drawdown_pct

    earnings = events.earnings.get(order.symbol)
    if instrument_type == "EQUITY":
        if earnings is None:
            checks["earnings_date_known"] = not limits.events.require_earnings_data_for_stocks
        else:
            # Block if earnings lands anywhere in the holding window (trading days -> calendar days) plus buffer.
            days = (earnings - ctx.trading_date).days
            window = math.ceil(holding_period_days * 7 / 5) + limits.events.earnings_blackout_days
            checks["no_earnings_in_holding_window"] = not (-1 <= days <= window)

    return RiskDecision(
        approved=all(checks.values()),
        failed_checks=[k for k, v in checks.items() if not v],
        passed_checks=[k for k, v in checks.items() if v],
        limits_sha256=limits.source_sha256,
    )
