"""Turns an approved idea into a share quantity. The LLM never picks size."""

from __future__ import annotations

import math

from .config import ExecutionLimits, RiskLimits
from .schemas import AccountState, SizedOrder, TradeProposal


def kelly_risk_fraction(p: float, upside: float, downside: float) -> float:
    """Full-Kelly fraction of equity to put at risk on a win-`upside` / lose-`downside` bet won with probability p.

    Losing the stop costs the risked amount; hitting the target pays reward_risk times it, so this is the
    classic binary Kelly f* = p - (1 - p) / b. Zero or negative means no edge."""
    if upside <= 0 or downside <= 0:
        return 0.0
    return p - (1 - p) / (upside / downside)


def size_order(decision_id: str, p: TradeProposal, account: AccountState, avg_volume_20d: float,
               limits: RiskLimits) -> SizedOrder | str:
    """Return a SizedOrder, or a string explaining why no size is possible."""
    a, ex = limits.account, limits.execution
    limit, stop = p.entry.limit_price, p.exit.stop_loss
    is_short = p.is_short

    # Risk per share: distance from entry to stop, plus slippage on both legs.
    if is_short:
        per_share_risk = (stop - limit) + limit * ex.slippage_bps / 10_000 * 2
        upside = limit - p.exit.take_profit   # profit when price falls
        downside = stop - limit               # loss when price rises
    else:
        per_share_risk = (limit - stop) + limit * ex.slippage_bps / 10_000 * 2
        upside = p.exit.take_profit - limit
        downside = limit - stop

    if per_share_risk <= 0 or account.equity <= 0:
        return "non-positive risk per share or equity"

    kelly_pct = limits.signal.kelly_fraction * kelly_risk_fraction(
        p.confidence, upside, downside) * 100
    if kelly_pct <= 0:
        return "no Kelly edge: confidence does not beat the reward/risk break-even"
    risk_pct = min(p.requested_risk_pct, kelly_pct, a.max_risk_per_trade_pct)
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
        decision_id=decision_id, symbol=p.symbol, side=p.side, quantity=qty, limit_price=limit, stop_loss=stop,
        take_profit=p.exit.take_profit, notional=round(qty * limit, 2), risk_usd=round(qty * per_share_risk, 2),
        sizing_notes=[f"binding cap: {binding}", f"risk_pct used: {risk_pct:.3f}", f"kelly risk_pct: {kelly_pct:.3f}"],
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


def trailed_stop_short(entry: float, initial_stop: float, current_stop: float, last: float,
                       ex: ExecutionLimits) -> float | None:
    """The new, lower stop for an open short, or None if it should stay where it is.

    Mirror of `trailed_stop` for short positions: the stop ratchets *down* as the price falls
    (i.e. as the short becomes more profitable). R = initial_stop - entry (the distance above
    entry where the stop sits). The stop never moves up (toward the market)."""
    r = initial_stop - entry
    if not ex.trailing_stops or r <= 0 or last >= entry:
        return None
    gain_r = (entry - last) / r  # profit in R-multiples
    candidate = current_stop
    if gain_r >= ex.trail_breakeven_r:
        candidate = min(candidate, entry)  # lock in breakeven
    if gain_r >= ex.trail_start_r:
        candidate = min(candidate, last + ex.trail_distance_r * r)  # trail down
    candidate = math.ceil(candidate * 100) / 100  # whole cents, rounded toward the market (up for shorts)
    if candidate <= last or current_stop - candidate < ex.trail_min_step_r * r:
        return None
    return candidate
