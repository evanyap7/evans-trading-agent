"""Independent quant verification of a proposal.

Recomputes what the proposal implies from our own data and rejects anything
ungrounded, stale, internally inconsistent, or without edge after costs.
"""

from __future__ import annotations

import math
from datetime import date, datetime

from .config import RiskLimits, Universe
from .schemas import Quote, TradeProposal, Verification


def verify(
    p: TradeProposal,
    *,
    cycle_evidence: dict[str, str | None],
    features: dict[str, float] | None,
    quote: Quote | None,
    limits: RiskLimits,
    universe: Universe,
    now: datetime,
    require_fresh_quote: bool,
) -> Verification:
    fails: list[str] = []
    m: dict[str, float] = {}
    sig, ex = limits.signal, limits.execution
    is_short = p.is_short

    sec = universe.get(p.symbol)
    if sec is None:
        return Verification(passed=False, failures=[f"{p.symbol} not in universe"], metrics={})
    if sec.type != p.instrument_type:
        fails.append(f"instrument_type {p.instrument_type} != universe type {sec.type}")

    # 1. Grounding: every cited ID exists in this cycle, and the symbol's own price evidence is cited.
    missing = [e for e in p.evidence_ids if e not in cycle_evidence]
    if missing:
        fails.append(f"unknown evidence ids: {missing[:5]}")
    if not any(cycle_evidence.get(e) == p.symbol and e.startswith("px_") for e in p.evidence_ids):
        fails.append("no price evidence for the traded symbol cited")

    # 2. Freshness.
    if not features:
        return Verification(passed=False, failures=fails + ["no features for symbol"], metrics=m)
    bar_age = (now.date() - date.fromisoformat(str(features["bar_date"]))).days
    m["bar_age_days"] = bar_age
    if bar_age > ex.max_bar_age_days:
        fails.append(f"daily bars stale ({bar_age}d)")
    close = float(features["close"])
    if close < ex.min_price_usd:
        fails.append(f"price {close:.2f} below minimum {ex.min_price_usd}")
    dollar_volume = float(features.get("avg_dollar_volume_20d") or 0)
    if dollar_volume < ex.min_avg_dollar_volume_usd:
        fails.append(f"20d avg dollar volume {dollar_volume:,.0f} below minimum {ex.min_avg_dollar_volume_usd:,.0f}")
    quote_ok = quote is not None and quote.age_seconds(now) <= ex.max_quote_age_seconds and math.isfinite(quote.spread_pct)
    if require_fresh_quote and not quote_ok:
        fails.append("no fresh quote")

    # 3. Price sanity against our reference price (direction-aware).
    limit, stop, tp = p.entry.limit_price, p.exit.stop_loss, p.exit.take_profit
    a = float(features["atr14"])
    if is_short:
        # Short entry: selling, so reference is the bid; collar prevents selling too far below it.
        ref = quote.bid if quote_ok and quote and quote.bid > 0 else float(features["close"])
        m.update(reference_price=ref, atr14=a)
        if limit < ref * (1 - ex.price_collar_pct / 100):
            fails.append(f"short limit {limit} below collar of reference {ref:.2f}")
        if limit > ref + 2 * a:
            fails.append(f"short limit {limit} more than 2 ATR above reference {ref:.2f}")
        stop_atr = (stop - limit) / a if a > 0 else math.nan
        up = (limit - tp) / limit * 100     # profit on price decline
        down = (stop - limit) / limit * 100  # loss on price rise
    else:
        # Long entry: buying, so reference is the ask.
        ref = quote.ask if quote_ok and quote else float(features["close"])
        m.update(reference_price=ref, atr14=a)
        if limit > ref * (1 + ex.price_collar_pct / 100):
            fails.append(f"limit {limit} above collar of reference {ref:.2f}")
        if limit < ref - 2 * a:
            fails.append(f"limit {limit} more than 2 ATR below reference {ref:.2f}")
        stop_atr = (limit - stop) / a if a > 0 else math.nan
        up = (tp - limit) / limit * 100
        down = (limit - stop) / limit * 100

    # 4. Stop placement relative to volatility.
    m["stop_atr"] = round(stop_atr, 3)
    if not (sig.min_stop_atr <= stop_atr <= sig.max_stop_atr):
        fails.append(f"stop distance {stop_atr:.2f} ATR outside [{sig.min_stop_atr}, {sig.max_stop_atr}]")

    # 5. Reward/risk and economic edge.
    rr = up / down
    breakeven_p = down / (up + down)  # win rate at which this stop/target pair has zero expectancy
    prob_edge = p.confidence - breakeven_p
    ev_from_confidence = p.confidence * up - (1 - p.confidence) * down
    edge = min(p.expected_return_pct, ev_from_confidence)
    spread = quote.spread_pct if quote_ok and quote else 0.05
    fee_pct = 2 * ex.fee_per_order_usd / limits.account.max_order_value_usd * 100
    costs = spread + 2 * ex.slippage_bps / 100 + fee_pct
    net = edge - costs - sig.safety_buffer_pct
    m.update(upside_pct=round(up, 3), downside_pct=round(down, 3), reward_risk=round(rr, 3),
             breakeven_probability=round(breakeven_p, 3), probability_edge=round(prob_edge, 3),
             ev_from_confidence_pct=round(ev_from_confidence, 3), cost_pct=round(costs, 3), net_edge_pct=round(net, 3))
    if p.expected_return_pct > up + 1e-9:
        fails.append("expected return exceeds the take-profit move")
    if rr < sig.min_reward_risk:
        fails.append(f"reward/risk {rr:.2f} < {sig.min_reward_risk}")
    if p.confidence < sig.min_confidence:
        fails.append(f"confidence {p.confidence} < {sig.min_confidence}")
    if prob_edge < sig.min_probability_edge:
        fails.append(f"probability edge {prob_edge:.3f} < {sig.min_probability_edge} "
                     f"(confidence {p.confidence} vs break-even {breakeven_p:.3f})")
    if net < sig.min_net_edge_pct:
        fails.append(f"net edge {net:.2f}% < {sig.min_net_edge_pct}%")

    return Verification(passed=not fails, failures=fails, metrics=m)
