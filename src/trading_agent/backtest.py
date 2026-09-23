"""Offline replay of the live decision pipeline over historical daily bars.

The agent, verifier, sizing, risk engine and trailing-stop rule are the live ones; only the
market is replayed. Each trading day D:

  open of D   fill entries queued at D-1's close (DAY limit: filled only if the bar trades through
              the limit), then sell any CLOSE the agent asked for.
  during D    time stop at the open; otherwise stop / take-profit from the day's range. Gaps fill at
              the open, and when one bar touches both the stop is assumed to have come first.
  close of D  ratchet trailing stops on the close, mark to market, drawdown kill switch, then
              research: the agent sees bars up to D only and its ideas go through
              verify -> size_order -> risk.evaluate exactly as in `Orchestrator.research`.

Known optimism that no replay removes: the universe is today's list (survivorship), there is no
historical news, and an LLM has read about these dates in training (lookahead). Treat an LLM
backtest as an upper bound, and the baseline as the honest control.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path
from statistics import fmean, pstdev

from .agents import AgentContext, ResearchAgent, render_context
from .config import Events, RiskLimits, Universe
from .features import build_evidence
from .market_calendar import ET
from .portfolio import size_order, trailed_stop
from .risk import OpenRisk, RiskContext, evaluate
from .schemas import AccountState, AgentOutput, Bar, Evidence, Position, SizedOrder, TradeProposal
from .verifier import verify

FEATURE_WINDOW = 300  # bars handed to feature code each day; covers the 252-day high and 200-day SMA
RESEARCH_TIME = time(16, 20)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class OpenPosition:
    decision_id: str
    symbol: str
    qty: int
    entry: float
    initial_stop: float
    stop: float
    take_profit: float
    entry_idx: int
    time_stop_idx: int


@dataclass
class ClosedTrade:
    symbol: str
    entry_date: date
    exit_date: date
    qty: int
    entry: float
    exit: float
    initial_stop: float
    reason: str
    pnl: float
    r_multiple: float
    bars_held: int


@dataclass
class PendingEntry:
    order: SizedOrder
    trade: TradeProposal


@dataclass
class BacktestResult:
    start: date
    end: date
    agent: str
    starting_equity: float
    trades: list[ClosedTrade]
    equity: list[tuple[date, float]]
    benchmark: list[tuple[date, float]]
    proposals: int = 0
    unfilled_entries: int = 0
    rejections: Counter = field(default_factory=Counter)
    halted_on: date | None = None
    halt_reason: str = ""
    kill_switch_trips: list[date] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        eq = [v for _, v in self.equity]
        rs = [t.r_multiple for t in self.trades]
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]
        gross_win, gross_loss = sum(t.pnl for t in wins), -sum(t.pnl for t in losses)
        bench = [v for _, v in self.benchmark]
        return {
            "period": f"{self.start} .. {self.end}",
            "trading_days": len(eq),
            "agent": self.agent,
            "starting_equity": round(self.starting_equity, 2),
            "ending_equity": round(eq[-1], 2) if eq else self.starting_equity,
            "total_return_pct": round(_ret(eq), 2),
            "cagr_pct": round(_cagr(eq), 2),
            "max_drawdown_pct": round(_max_dd(eq), 2),
            "sharpe": round(_sharpe(eq), 2),
            "trades": len(self.trades),
            "win_rate_pct": round(len(wins) / len(self.trades) * 100, 1) if self.trades else None,
            "expectancy_r": round(fmean(rs), 3) if rs else None,
            "avg_win_r": round(fmean([t.r_multiple for t in wins]), 3) if wins else None,
            "avg_loss_r": round(fmean([t.r_multiple for t in losses]), 3) if losses else None,
            "best_r": round(max(rs), 2) if rs else None,
            "worst_r": round(min(rs), 2) if rs else None,
            "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else None,
            "avg_bars_held": round(fmean([t.bars_held for t in self.trades]), 1) if self.trades else None,
            "exit_reasons": dict(Counter(t.reason for t in self.trades).most_common()),
            "proposals": self.proposals,
            "unfilled_entries": self.unfilled_entries,
            "top_rejections": dict(self.rejections.most_common(8)),
            "benchmark_return_pct": round(_ret(bench), 2),
            "benchmark_max_drawdown_pct": round(_max_dd(bench), 2),
            "kill_switch_tripped": f"{self.halted_on} ({self.halt_reason})" if self.halted_on else None,
            "kill_switch_trips": len(self.kill_switch_trips),
        }

    def report(self) -> str:
        s = self.summary()
        width = max(len(k) for k in s)
        lines = [f"{k:<{width}}  {v}" for k, v in s.items()]
        return "\n".join(lines + [f"WARNING: {w}" for w in self.warnings])

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=str))
        with open(out_dir / "trades.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(ClosedTrade.__dataclass_fields__))
            w.writeheader()
            w.writerows(asdict(t) for t in self.trades)
        bench = dict(self.benchmark)
        with open(out_dir / "equity.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["date", "equity", "benchmark"])
            w.writerows((d, round(v, 2), round(bench.get(d, math.nan), 2)) for d, v in self.equity)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Backtester:
    def __init__(self, *, bars: dict[str, list[Bar]], agent: ResearchAgent, limits: RiskLimits, universe: Universe,
                 start: date, end: date, starting_cash: float,
                 earnings_history: dict[str, list[date]] | None = None, halt_on_kill_switch: bool = True,
                 output_cache: "AgentOutputCache | None" = None):
        self.limits, self.universe, self.agent = limits, universe, agent
        self.halt_on_kill_switch = halt_on_kill_switch
        self.cache = output_cache
        self.earnings = {s: sorted(ds) for s, ds in (earnings_history or {}).items()}
        self.bars = {s: sorted(b, key=lambda x: x.ts) for s, b in bars.items() if s in universe.symbols and b}
        self.dates = {s: [b.ts.date() for b in bs] for s, bs in self.bars.items()}
        bench = universe.regime_benchmark
        if bench not in self.bars:
            raise ValueError(f"no bars for regime benchmark {bench}")
        self.calendar = [d for d in self.dates[bench] if start <= d <= end]
        if not self.calendar:
            raise ValueError(f"no {bench} bars between {start} and {end}")
        if bisect.bisect_left(self.dates[bench], self.calendar[0]) < 200:
            raise ValueError("need ~200 bars of history before the start date for the long moving averages")

        self.cash = starting_cash
        self.positions: dict[str, OpenPosition] = {}
        self.pending_entries: list[PendingEntry] = []
        self.pending_closes: set[str] = set()
        self.peak = starting_cash
        self.prev_close_equity = starting_cash
        self.halted = False
        self.result = BacktestResult(start=self.calendar[0], end=self.calendar[-1], agent=f"{agent.name}:{agent.model}",
                                     starting_equity=starting_cash, trades=[], equity=[], benchmark=[])

    # -- data access -----------------------------------------------------------------

    def _bar(self, sym: str, d: date) -> Bar | None:
        ds = self.dates[sym]
        i = bisect.bisect_left(ds, d)
        return self.bars[sym][i] if i < len(ds) and ds[i] == d else None

    def _history(self, sym: str, d: date) -> list[Bar]:
        """Bars up to and including d: the only data research may see on day d."""
        i = bisect.bisect_right(self.dates[sym], d)
        return self.bars[sym][max(0, i - FEATURE_WINDOW):i]

    def _last_close(self, sym: str, d: date) -> float:
        h = self._history(sym, d)
        return h[-1].close if h else 0.0

    def _next_earnings(self, sym: str, d: date) -> date | None:
        ds = self.earnings.get(sym, [])
        i = bisect.bisect_left(ds, d)
        return ds[i] if i < len(ds) else None

    # -- main loop --------------------------------------------------------------------

    def run(self) -> BacktestResult:
        ex = self.limits.execution
        bench = self.universe.regime_benchmark
        bench_start = self._bar(bench, self.calendar[0]).close
        for idx, d in enumerate(self.calendar):
            self._open(idx, d)
            for sym in list(self.positions):
                p = self.positions[sym]
                bar = self._bar(sym, d)
                if bar is None:
                    continue
                hit = manage_bar(p, bar, idx, ex.slippage_bps)
                if hit:
                    self._close(sym, hit[0], d, idx, hit[1])
                    continue
                new_stop = trailed_stop(p.entry, p.initial_stop, p.stop, bar.close, ex)
                if new_stop is not None:
                    p.stop = new_stop
            equity = self._equity(d)
            self.result.equity.append((d, equity))
            self.result.benchmark.append((d, self.result.starting_equity * self._bar(bench, d).close / bench_start))
            self.peak = max(self.peak, equity)
            if not self.halted and (equity / self.peak - 1) * 100 <= -self.limits.account.max_drawdown_pct:
                self.result.kill_switch_trips.append(d)
                if self.result.halted_on is None:
                    self.result.halted_on = d
                    self.result.halt_reason = f"drawdown {(equity / self.peak - 1) * 100:.1f}% from peak {self.peak:.2f}"
                if self.halt_on_kill_switch:
                    self.halted = True
                    self.pending_entries.clear()
                else:
                    # As if the operator released the kill switch and reset the high-water mark; without the
                    # reset the risk engine's drawdown_ok check would keep blocking entries anyway.
                    self.peak = equity
            if idx < len(self.calendar) - 1:
                self._research(idx, d, equity)
            self.prev_close_equity = equity
        # Mark anything still open at the last close so the trade list is complete.
        last = self.calendar[-1]
        for sym in list(self.positions):
            px = self._last_close(sym, last) * (1 - ex.slippage_bps / 10_000)
            self._close(sym, px, last, len(self.calendar) - 1, "open_at_end")
        if self.cache is not None and self.cache.misses:
            self.result.warnings.append(f"{self.cache.misses} uncached agent calls were made (now cached)")
        return self.result

    def _equity(self, d: date) -> float:
        return self.cash + sum(p.qty * self._last_close(p.symbol, d) for p in self.positions.values())

    # -- open of day ------------------------------------------------------------------

    def _open(self, idx: int, d: date) -> None:
        ex = self.limits.execution
        slip = ex.slippage_bps / 10_000
        for sym in sorted(self.pending_closes):
            p, bar = self.positions.get(sym), self._bar(sym, d)
            if p and bar:
                self._close(sym, bar.open * (1 - slip), d, idx, "agent_close")
        self.pending_closes.clear()

        for pe in self.pending_entries:
            o = pe.order
            bar = self._bar(o.symbol, d)
            if bar is None or o.symbol in self.positions or bar.low > o.limit_price:
                self.result.unfilled_entries += 1
                continue
            px = min(bar.open, o.limit_price) * (1 + slip)
            if px <= o.stop_loss:  # live execute re-verifies on a fresh quote and would refuse this
                self.result.rejections["entry: gapped below stop"] += 1
                continue
            cost = o.quantity * px + ex.fee_per_order_usd
            if cost > self.cash:
                self.result.rejections["entry: cash at fill"] += 1
                continue
            self.cash -= cost
            self.positions[o.symbol] = OpenPosition(
                decision_id=o.decision_id, symbol=o.symbol, qty=o.quantity, entry=px, initial_stop=o.stop_loss,
                stop=o.stop_loss, take_profit=o.take_profit, entry_idx=idx,
                time_stop_idx=idx + pe.trade.exit.time_stop_days)
        self.pending_entries.clear()

    def _close(self, sym: str, px: float, d: date, idx: int, reason: str) -> None:
        p = self.positions.pop(sym)
        fee = self.limits.execution.fee_per_order_usd
        self.cash += p.qty * px - fee
        pnl = (px - p.entry) * p.qty - 2 * fee
        risk = p.entry - p.initial_stop
        self.result.trades.append(ClosedTrade(
            symbol=sym, entry_date=self.calendar[p.entry_idx], exit_date=d, qty=p.qty, entry=round(p.entry, 4),
            exit=round(px, 4), initial_stop=p.initial_stop, reason=reason, pnl=round(pnl, 2),
            r_multiple=round((px - p.entry) / risk, 3) if risk > 0 else 0.0, bars_held=idx - p.entry_idx))

    # -- close of day: research ------------------------------------------------------------

    def _research(self, idx: int, d: date, equity: float) -> None:
        now = datetime.combine(d, RESEARCH_TIME, ET)
        history = {s: h for s in self.bars if len(h := self._history(s, d)) > 0 and h[-1].ts.date() >= d - timedelta(days=10)}
        evidence, features = build_evidence(history, {}, self.universe.regime_benchmark)
        positions = [Position(symbol=p.symbol, quantity=p.qty, avg_cost=p.entry, last_price=self._last_close(p.symbol, d))
                     for p in self.positions.values()]
        for p in self.positions.values():
            payload = {"quantity": p.qty, "avg_cost": p.entry, "last_price": self._last_close(p.symbol, d),
                       "system_stop": p.stop, "system_target": p.take_profit,
                       "time_stop_date": self.calendar[min(p.time_stop_idx, len(self.calendar) - 1)].isoformat()}
            evidence.append(Evidence(evidence_id=f"pos_{p.symbol}_{d:%Y%m%d}", kind="position", symbol=p.symbol,
                                     as_of=now, source="backtest.positions", payload=payload))
        account = AccountState(account_id="backtest", equity=equity, cash=self.cash, buying_power=self.cash,
                               positions=positions, open_orders=[], as_of=now)
        earnings = {s: nd for s, sec in self.universe.symbols.items()
                    if sec.type == "EQUITY" and (nd := self._next_earnings(s, d)) is not None}
        exposure = sum(p.quantity * p.last_price for p in positions)
        ctx = AgentContext(
            as_of=d, evidence=evidence,
            universe={s: {"type": sec.type, "sector": sec.sector} for s, sec in self.universe.symbols.items()},
            account_summary={"equity_usd": round(equity, 2), "cash_usd": round(self.cash, 2),
                             "gross_exposure_pct": round(exposure / equity * 100, 2) if equity else None},
            positions=[{"symbol": p.symbol, "quantity": p.quantity, "avg_cost": p.avg_cost, "last": p.last_price}
                       for p in positions],
            known_earnings={s: v.isoformat() for s, v in earnings.items()},
            features=features,
        )
        try:
            output = self.cache.propose(self.agent, ctx) if self.cache else self.agent.propose(ctx)
        except Exception as e:  # same as live: a failed model call is a NO_TRADE
            self.result.rejections[f"agent error: {type(e).__name__}"] += 1
            return

        cycle_ev = {e.evidence_id: e.symbol for e in evidence}
        freed = 0.0
        for c in output.exits:
            grounded = all(e in cycle_ev for e in c.evidence_ids) and any(
                cycle_ev.get(e) == c.symbol and e.startswith(("px_", "pos_")) for e in c.evidence_ids)
            if c.symbol in self.positions and grounded:
                self.pending_closes.add(c.symbol)
                freed += self.positions[c.symbol].qty * self._last_close(c.symbol, d)
            else:
                self.result.rejections["exit: ungrounded or not held"] += 1
        if self.halted:
            return

        cash = self.cash + freed
        events = Events(earnings=earnings)
        for n, trade in enumerate(output.proposals):
            self.result.proposals += 1
            decision_id = f"{d:%Y%m%d}-{n}"
            v = verify(trade, cycle_evidence=cycle_ev, features=features.get(trade.symbol), quote=None,
                       limits=self.limits, universe=self.universe, now=now, require_fresh_quote=False)
            if not v.passed:
                for f in v.failures:
                    self.result.rejections["verifier: " + _normalize(f)] += 1
                continue
            acct = account.model_copy(update={"cash": cash, "buying_power": cash})
            sized = size_order(decision_id, trade, acct, features[trade.symbol]["avg_volume_20d"], self.limits)
            if isinstance(sized, str):
                self.result.rejections["sizing: " + _normalize(sized)] += 1
                continue
            open_risks = [OpenRisk(p.symbol, p.qty, self._last_close(p.symbol, d), p.stop)
                          for p in self.positions.values() if p.symbol not in self.pending_closes]
            open_risks += [OpenRisk(pe.order.symbol, pe.order.quantity, pe.order.limit_price, pe.order.stop_loss)
                           for pe in self.pending_entries]
            rctx = RiskContext(
                now=now, trading_date=d, account=acct, quote=None, kill_switch_engaged=self.halted,
                sends_real_orders_to_prod=False, require_session=False, reconciled=True,
                entries_today=len(self.pending_entries), start_of_day_equity=self.prev_close_equity,
                peak_equity=self.peak, decision_already_executed=False, open_risks=open_risks)
            decision = evaluate(sized, trade.instrument_type, trade.holding_period_days, rctx, self.limits,
                                self.universe, events)
            if not decision.approved:
                for c in decision.failed_checks:
                    self.result.rejections["risk: " + c] += 1
                continue
            self.pending_entries.append(PendingEntry(order=sized, trade=trade))
            cash -= sized.notional + self.limits.execution.fee_per_order_usd


def manage_bar(p: OpenPosition, bar: Bar, idx: int, slippage_bps: float) -> tuple[float, str] | None:
    """Exit price and reason for an open long over one daily bar, or None if it survives the day.

    Positions opened this morning were bought at (or after) the open, so only their intraday range counts."""
    slip = slippage_bps / 10_000
    opened_today = idx == p.entry_idx
    if not opened_today:
        if bar.open <= p.stop:
            return bar.open * (1 - slip), "stop_gap"
        if idx >= p.time_stop_idx:
            return bar.open * (1 - slip), "time_stop"
        if bar.open >= p.take_profit:
            return bar.open * (1 - slip), "take_profit"
    if bar.low <= p.stop:  # checked before the target: the conservative reading of an ambiguous bar
        return p.stop * (1 - slip), "stop"
    if bar.high >= p.take_profit:
        return p.take_profit * (1 - slip), "take_profit"
    return None


# ---------------------------------------------------------------------------
# Agent output cache (LLM runs are slow and cost money; re-runs should be free)
# ---------------------------------------------------------------------------


class AgentOutputCache:
    def __init__(self, directory: Path, max_new_calls: int | None = None):
        self.dir = directory
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_new_calls = max_new_calls
        self.misses = 0

    def propose(self, agent: ResearchAgent, ctx: AgentContext) -> AgentOutput:
        key = hashlib.sha256(f"{agent.name}|{agent.model}|{render_context(ctx)}".encode()).hexdigest()[:24]
        path = self.dir / f"{ctx.as_of.isoformat()}_{key}.json"
        if path.exists():
            return AgentOutput.model_validate_json(path.read_text())
        if self.max_new_calls is not None and self.misses >= self.max_new_calls:
            raise RuntimeError(f"agent call budget of {self.max_new_calls} used up")
        self.misses += 1
        out = agent.propose(ctx)
        path.write_text(out.model_dump_json())
        return out


# ---------------------------------------------------------------------------
# Historical data (yfinance, cached on disk)
# ---------------------------------------------------------------------------


def load_bars(symbols: list[str], start: date, end: date, cache_dir: Path) -> dict[str, list[Bar]]:
    """Split- and dividend-adjusted daily bars from ~420 calendar days before `start` through `end`."""
    first = start - timedelta(days=420)
    out: dict[str, list[Bar]] = {}
    missing = []
    for sym in symbols:
        path = cache_dir / f"bars_{sym}_{first}_{end}.json"
        if path.exists():
            out[sym] = [Bar.model_validate(b) for b in json.loads(path.read_text())]
        else:
            missing.append(sym)
    if missing:
        import yfinance as yf

        df = yf.download(missing, start=first.isoformat(), end=(end + timedelta(days=1)).isoformat(),
                         interval="1d", auto_adjust=True, progress=False, group_by="ticker")
        cache_dir.mkdir(parents=True, exist_ok=True)
        for sym in missing:
            sub = df[sym] if len(missing) > 1 else df
            if len(missing) == 1 and hasattr(sub.columns, "nlevels") and sub.columns.nlevels > 1:
                sub = sub[sym]
            bars = [Bar(ts=datetime.combine(ix.date(), time(16, 0), ET), open=float(r["Open"]), high=float(r["High"]),
                        low=float(r["Low"]), close=float(r["Close"]), volume=float(r["Volume"]))
                    for ix, r in sub.dropna().iterrows()]
            if bars:
                out[sym] = bars
                (cache_dir / f"bars_{sym}_{first}_{end}.json").write_text(
                    json.dumps([b.model_dump(mode="json") for b in bars]))
    return out


def load_earnings_history(symbols: list[str], cache_dir: Path) -> dict[str, list[date]]:
    """Past and scheduled earnings dates per stock. A stock with none stays blocked, as in live."""
    path = cache_dir / "earnings.json"
    cached: dict[str, list[str]] = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in symbols if s not in cached]
    if todo:
        import yfinance as yf

        for sym in todo:
            try:
                df = yf.Ticker(sym).get_earnings_dates(limit=60)
                cached[sym] = sorted({ix.date().isoformat() for ix in df.index}) if df is not None else []
            except Exception:
                cached[sym] = []
        cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=1))
    return {s: [date.fromisoformat(x) for x in cached.get(s, [])] for s in symbols}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _normalize(msg: str) -> str:
    """Group rejection messages by kind: drop the numbers and symbols that vary per case."""
    msg = re.sub(r"\b[A-Z]{1,5}\b(?= )", "SYM", msg)
    return re.sub(r"-?\d[\d,]*\.?\d*", "#", msg)[:80]


def _ret(eq: list[float]) -> float:
    return (eq[-1] / eq[0] - 1) * 100 if len(eq) > 1 and eq[0] else 0.0


def _cagr(eq: list[float]) -> float:
    if len(eq) < 2 or eq[0] <= 0 or eq[-1] <= 0:
        return 0.0
    return ((eq[-1] / eq[0]) ** (252 / (len(eq) - 1)) - 1) * 100


def _max_dd(eq: list[float]) -> float:
    peak, worst = -math.inf, 0.0
    for v in eq:
        peak = max(peak, v)
        worst = min(worst, (v / peak - 1) * 100 if peak > 0 else 0.0)
    return worst


def _sharpe(eq: list[float]) -> float:
    rets = [b / a - 1 for a, b in zip(eq, eq[1:]) if a > 0]
    if len(rets) < 2 or pstdev(rets) == 0:
        return 0.0
    return fmean(rets) / pstdev(rets) * math.sqrt(252)
