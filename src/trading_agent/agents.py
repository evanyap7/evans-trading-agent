"""Idea generators. They return `AgentOutput` and cannot touch the broker.

- `BaselineMomentumAgent`: a transparent rule-based agent. Its confidence
  numbers are placeholders, not calibrated probabilities. The LLM agents were
  removed after a 2024 backtest showed no edge over it (docs/backtest-llm-vs-baseline-2024.md).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from .schemas import AgentOutput, Evidence, TradeProposal


@dataclass
class AgentContext:
    as_of: date
    evidence: list[Evidence]
    universe: dict[str, dict]
    account_summary: dict
    positions: list[dict]
    known_earnings: dict[str, str]
    features: dict[str, dict]


class ResearchAgent(Protocol):
    name: str
    model: str

    def propose(self, ctx: AgentContext) -> AgentOutput: ...


def render_context(ctx: AgentContext) -> str:
    trusted = [e for e in ctx.evidence if not e.untrusted_text]
    untrusted = [e for e in ctx.evidence if e.untrusted_text]
    parts = [
        f"Decision date (US/Eastern): {ctx.as_of.isoformat()}. Orders will be placed at the next regular session.",
        "## Universe\n" + json.dumps(ctx.universe, sort_keys=True),
        "## Account\n" + json.dumps(ctx.account_summary, sort_keys=True),
        "## Open positions\n" + json.dumps(ctx.positions, sort_keys=True, default=str),
        "## Known earnings dates\n" + json.dumps(ctx.known_earnings, sort_keys=True),
        "## Evidence\n" + "\n".join(
            json.dumps({"evidence_id": e.evidence_id, "kind": e.kind, "symbol": e.symbol,
                        "as_of": e.as_of.date().isoformat(), **e.payload}, sort_keys=True, default=str)
            for e in trusted
        ),
    ]
    from .news import strip_untrusted_tags

    for e in untrusted:
        text = strip_untrusted_tags(str(e.payload.get("text", "")))
        source = strip_untrusted_tags(e.source).replace('"', "'")
        parts.append(f'<untrusted_document evidence_id="{e.evidence_id}" source="{source}">\n{text}\n</untrusted_document>')
    parts.append("Return your decisions in the required JSON schema.")
    return "\n\n".join(parts)



class BaselineMomentumAgent:
    """Trend-following control: strongest 60-day movers above their 50/200-day averages (long),
    plus weakest movers below both averages (short)."""

    name = "baseline"
    model = "baseline-momentum-v1"

    def __init__(self, max_ideas: int = 4):
        self.max_ideas = max_ideas

    def propose(self, ctx: AgentContext) -> AgentOutput:
        regime = next((e for e in ctx.evidence if e.kind == "regime"), None)
        is_bull_regime = regime is not None and bool(regime.payload.get("benchmark_above_sma200"))
        regime_ev_id = [regime.evidence_id] if regime else []
        held = {p["symbol"] for p in ctx.positions}
        px_ev = {e.symbol: e for e in ctx.evidence if e.kind == "price_features"}

        # Long candidates: above both SMAs, sorted by 60d momentum descending
        long_candidates = []
        # Short candidates: below both SMAs, sorted by 60d momentum ascending (most negative first)
        short_candidates = []
        for sym, f in ctx.features.items():
            if sym in held or sym not in ctx.universe or sym not in px_ev:
                continue
            dist200 = f.get("dist_sma200_pct") or -1
            dist50 = f.get("dist_sma50_pct") or -1
            ret60 = f.get("ret_60d_pct")
            if ret60 is None:
                continue
            if dist200 > 0 and dist50 > 0:
                long_candidates.append((ret60, sym))
            elif dist200 < 0 and dist50 < 0:
                short_candidates.append((ret60, sym))

        proposals = []
        # Long proposals (propose in bull/healthy regime)
        if is_bull_regime:
            long_quota = max(1, self.max_ideas // 2)
            for _, sym in sorted(long_candidates, reverse=True)[: long_quota]:
                f = ctx.features[sym]
                close, a = f["close"], f["atr14"]
                limit = round(close * 1.002, 2)
                stop, tp = round(limit - 2 * a, 2), round(limit + 4 * a, 2)
                up, down = (tp - limit) / limit * 100, (limit - stop) / limit * 100
                conf = 0.55
                proposals.append(TradeProposal(
                    symbol=sym, instrument_type=ctx.universe[sym]["type"], side="BUY", strategy="momentum_trend",
                    thesis=f"{sym} is among the strongest 60-day performers and trades above its 50 and 200-day averages.",
                    holding_period_days=15, confidence=conf, expected_return_pct=round(conf * up - (1 - conf) * down, 2),
                    invalidation_price=stop, entry={"order_type": "LIMIT", "limit_price": limit},
                    exit={"take_profit": tp, "stop_loss": stop, "time_stop_days": 15}, requested_risk_pct=0.25,
                    evidence_ids=[px_ev[sym].evidence_id] + regime_ev_id,
                ))

        # Short proposals (propose in both regimes, prioritize in bear regimes)
        short_quota = self.max_ideas if not is_bull_regime else max(1, self.max_ideas // 2)
        for _, sym in sorted(short_candidates)[: short_quota]:
            f = ctx.features[sym]
            close, a = f["close"], f["atr14"]
            limit = round(close * 0.998, 2)
            stop = round(limit + 2 * a, 2)   # stop above entry
            tp = round(limit - 4 * a, 2)      # target below entry
            if tp <= 0:
                continue
            up = (limit - tp) / limit * 100     # profit on decline
            down = (stop - limit) / limit * 100  # loss on rise
            conf = 0.55
            proposals.append(TradeProposal(
                symbol=sym, instrument_type=ctx.universe[sym]["type"], side="SELL_SHORT", strategy="momentum_breakdown",
                thesis=f"{sym} is among the weakest 60-day performers and trades below its 50 and 200-day averages.",
                holding_period_days=15, confidence=conf, expected_return_pct=round(conf * up - (1 - conf) * down, 2),
                invalidation_price=stop, entry={"order_type": "LIMIT", "limit_price": limit},
                exit={"take_profit": tp, "stop_loss": stop, "time_stop_days": 15}, requested_risk_pct=0.25,
                evidence_ids=[px_ev[sym].evidence_id] + regime_ev_id,
            ))

        market_view = "risk-on: benchmark above 200-day average" if is_bull_regime else "risk-off: benchmark below 200-day average (shorting focus)"
        no_trade = None if proposals else ("no qualifying trends" if is_bull_regime else "no qualifying breakdown trends to short")
        return AgentOutput(market_view=market_view, proposals=proposals, no_trade_reason=no_trade)
