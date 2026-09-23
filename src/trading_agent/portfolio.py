"""Turns an approved idea into a share quantity. The LLM never picks size."""

from __future__ import annotations

import math

from .config import ExecutionLimits, RiskLimits
from .schemas import AccountState, SizedOrder, TradeProposal


def size_order(decision_id: str, p: TradeProposal, account: AccountState, avg_volume_20d: float,
               limits: RiskLimits) -> SizedOrder | str:
    """Return a SizedOrder, or a string explaining why no size is possible."""
    a, ex = limits.account, limits.execution
    limit, stop = p.entry.limit_price, p.exit.stop_loss
    per_share_risk = (limit - stop) + limit * ex.slippage_bps / 10_000 * 2
    if per_share_risk <= 0 or account.equity <= 0:
        return "non-positive risk per share or equity"

    risk_pct = min(p.requested_risk_pct, a.max_risk_per_trade_pct)
    caps = {
        "risk budget": (account.equity * risk_pct / 100) / per_share_risk,
        "max position": account.equity * a.max_position_pct / 100 / limit,
        "max order value": a.max_order_value_usd / limit,
        "ADV participation": avg_volume_20d * a.max_adv_participation_pct / 100,
        "cash": max(account.cash - ex.fee_per_order_usd, 0) / limit,
    }
    binding = min(caps, key=caps.get)
    qty = math.floor(caps[binding])
    if qty < 1:
        # Whole-share floor: on small accounts, if 1 share's risk is within the hard
        # max_risk_per_trade_pct and all hard limits allow 1 share, grant 1 share.
        hard_risk_cap = (account.equity * a.max_risk_per_trade_pct / 100) / per_share_risk
        if (
            hard_risk_cap >= 1.0
            and caps["cash"] >= 1.0
            and caps["max position"] >= 1.0
            and caps["max order value"] >= 1.0
            and caps["ADV participation"] >= 1.0
        ):
            qty = 1
            binding = "whole-share floor (within max_risk_per_trade_pct)"
        else:
            return f"size rounds to 0 shares (binding cap: {binding})"
    return SizedOrder(
        decision_id=decision_id, symbol=p.symbol, side="BUY", quantity=qty, limit_price=limit, stop_loss=stop,
        take_profit=p.exit.take_profit, notional=round(qty * limit, 2), risk_usd=round(qty * per_share_risk, 2),
        sizing_notes=[f"binding cap: {binding}", f"risk_pct used: {risk_pct}"],
    )


def trailed_stop(entry: float, initial_stop: float, current_stop: float, last: float,
                 ex: ExecutionLimits) -> float | None:
    """The new, higher stop for an open long, or None if it should stay where it is.

    Rules are in multiples of the initial risk R: breakeven at +trail_breakeven_r, then trail
    trail_distance_r behind the last price from +trail_start_r. Moves smaller than
    trail_min_step_r are skipped, and the stop never moves down."""
    r = entry - initial_stop
    if not ex.trailing_stops or r <= 0 or last <= entry:
        return None
    gain_r = (last - entry) / r
    candidate = current_stop
    if gain_r >= ex.trail_breakeven_r:
        candidate = max(candidate, entry)
    if gain_r >= ex.trail_start_r:
        candidate = max(candidate, last - ex.trail_distance_r * r)
    candidate = math.floor(candidate * 100) / 100  # whole cents, rounded away from the market
    if candidate >= last or candidate - current_stop < ex.trail_min_step_r * r:
        return None
    return candidate
