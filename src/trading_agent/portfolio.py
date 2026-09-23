"""Turns an approved idea into a share quantity. The LLM never picks size."""

from __future__ import annotations

import math

from .config import RiskLimits
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
        return f"size rounds to 0 shares (binding cap: {binding})"
    return SizedOrder(
        decision_id=decision_id, symbol=p.symbol, side="BUY", quantity=qty, limit_price=limit, stop_loss=stop,
        take_profit=p.exit.take_profit, notional=round(qty * limit, 2), risk_usd=round(qty * per_share_risk, 2),
        sizing_notes=[f"binding cap: {binding}", f"risk_pct used: {risk_pct}"],
    )
