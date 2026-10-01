#!/usr/bin/env python3
"""Backtest harness and research workbench for Evan's Trading Agent.

Week 1: Critical path backtest on 2+ years of historical data (Top 100 liquid US stocks + SPY/SPYM).
Week 2: Isolate LLM contribution (Test A: Claude vs Test B: Quantitative Baseline).
Week 3: Decision checkpoint (Absolute & Relative hurdles: Sharpe > 1.0, Max DD < 30%, 2x SPY Sharpe).

Usage:
    # 1. Test harness with 6 months of 1 stock (sanity check)
    uv run python scripts/backtest_harness.py smoke --symbol NVDA

    # 2. Download/cache 2+ years of historical data for Top 100 universe
    uv run python scripts/backtest_harness.py download --start 2024-01-01 --end 2026-09-30

    # 3. Run full 2-year backtest
    uv run python scripts/backtest_harness.py run --agent baseline --start 2024-01-01 --end 2026-09-30

    # 4. Compare Test A (Claude LLM) vs Test B (Baseline)
    uv run python scripts/backtest_harness.py compare --start 2024-01-01 --end 2026-09-30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

# Anchor project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from trading_agent.agents import BaselineMomentumAgent, TieredResearchAgent
from trading_agent.backtest import Backtester, load_bars
from trading_agent.config import (
    CONFIG_DIR,
    CashSweep,
    Events,
    RiskLimits,
    Security,
    Universe,
    load_events,
    load_risk_limits,
    load_universe,
)
from trading_agent.schemas import Bar

TOP100_CONFIG = CONFIG_DIR / "universe_top100.yaml"
CACHE_DIR = PROJECT_ROOT / "state" / "backtest_cache"


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def cmd_smoke(args: argparse.Namespace) -> None:
    """Run a fast 6-month single-stock test to verify the replay loop, slippage, and Kelly sizing."""
    sym = args.symbol.upper()
    start_d = _parse_date(args.start)
    end_d = _parse_date(args.end)
    print(f"=== Smoke Test: 6 Months of {sym} ({start_d} to {end_d}) ===")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = [sym, "SPY"]
    bars = load_bars(symbols, start_d, end_d, CACHE_DIR)
    if sym not in bars or not bars[sym]:
        print(f"ERROR: Could not load bars for {sym}")
        return

    universe = Universe(
        symbols={sym: Security(symbol=sym, type="EQUITY", sector="Technology"),
                 "SPY": Security(symbol="SPY", type="ETF", sector="Broad Market")},
        regime_benchmark="SPY",
    )
    limits = load_risk_limits()
    agent = BaselineMomentumAgent()

    bt = Backtester(
        bars=bars,
        agent=agent,
        limits=limits,
        universe=universe,
        start=start_d,
        end=end_d,
        starting_cash=args.cash,
        halt_on_kill_switch=False,
    )
    res = bt.run()
    s = res.summary()

    print("\n--- Smoke Test Results ---")
    print(f"Trades Taken:    {s['trades']}")
    print(f"Win Rate:        {s['win_rate_pct']}%")
    print(f"Total Return:    {s['total_return_pct']}%")
    print(f"Max Drawdown:    {s['max_drawdown_pct']}%")
    print(f"Sharpe Ratio:    {s['sharpe']}")
    print(f"SPY Benchmark:   {s['benchmark_return_pct']}% (Sharpe: {s.get('benchmark_sharpe')})")
    print(f"Profit Factor:   {s.get('profit_factor')}")
    print("\n✓ Smoke test loop completed successfully!")


def cmd_download(args: argparse.Namespace) -> None:
    """Download and cache 2+ years of daily OHLCV bars for the Top 100 universe."""
    universe_path = TOP100_CONFIG if TOP100_CONFIG.exists() else None
    univ = load_universe(universe_path)
    symbols = sorted(univ.symbols.keys())
    start_d = _parse_date(args.start)
    end_d = _parse_date(args.end)

    print(f"Downloading historical data for {len(symbols)} symbols from {start_d} to {end_d}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    bars = load_bars(symbols, start_d, end_d, CACHE_DIR)
    loaded = sum(1 for s in symbols if s in bars and len(bars[s]) > 0)
    print(f"Successfully loaded and cached {loaded}/{len(symbols)} symbols in {CACHE_DIR}.")


def _run_backtest_instance(agent_name: str, start_d: date, end_d: date, cash: float,
                           sweep_sym: str | None = None) -> dict:
    universe_path = TOP100_CONFIG if TOP100_CONFIG.exists() else None
    univ = load_universe(universe_path)
    symbols = sorted(univ.symbols.keys())
    if sweep_sym and sweep_sym not in symbols:
        symbols.append(sweep_sym)

    bars = load_bars(symbols, start_d, end_d, CACHE_DIR)
    limits = load_risk_limits()
    if sweep_sym:
        limits = limits.model_copy(update={"cash_sweep": CashSweep(enabled=True, symbol=sweep_sym, reserve_pct=5.0)})

    from trading_agent.backtest import load_earnings_history
    earnings_hist = load_earnings_history(symbols, CACHE_DIR)

    agent = TieredResearchAgent() if agent_name == "llm" else BaselineMomentumAgent()

    bt = Backtester(
        bars=bars,
        agent=agent,
        limits=limits,
        universe=univ,
        start=start_d,
        end=end_d,
        starting_cash=cash,
        earnings_history=earnings_hist,
        halt_on_kill_switch=False,
    )
    res = bt.run()
    return res.summary()


def cmd_run(args: argparse.Namespace) -> None:
    """Run full 2-year backtest for a specific agent."""
    start_d = _parse_date(args.start)
    end_d = _parse_date(args.end)
    print(f"Running 2-Year Backtest ({args.agent.upper()}) from {start_d} to {end_d}...")
    s = _run_backtest_instance(args.agent, start_d, end_d, args.cash, args.sweep)

    print("\n" + "=" * 50)
    print(f"  2-YEAR BACKTEST SUMMARY: {args.agent.upper()}")
    print("=" * 50)
    for k, v in s.items():
        if k not in ("calibration", "top_rejections", "exit_reasons"):
            print(f"  {k:<32} {v}")

    # Week 1 Evaluation Checkpoint
    sharpe = s.get("sharpe", 0.0) or 0.0
    max_dd = abs(s.get("max_drawdown_pct", 0.0) or 0.0)
    ret = s.get("total_return_pct", 0.0) or 0.0
    bench_sharpe = s.get("benchmark_sharpe", 0.0) or 0.0

    print("\n" + "-" * 50)
    print("  WEEK 1 EVALUATION CHECKPOINT")
    print("-" * 50)
    c1 = sharpe > 1.0
    c2 = max_dd < 30.0
    c3 = ret > 0.0
    c4 = bench_sharpe > 0 and (sharpe >= 2.0 * bench_sharpe)

    print(f"  [1] Absolute Sharpe > 1.0:        {'✓ PASS' if c1 else '✗ FAIL'} ({sharpe})")
    print(f"  [2] Absolute Max DD < 30%:        {'✓ PASS' if c2 else '✗ FAIL'} ({max_dd:.1f}%)")
    print(f"  [3] Absolute Positive Return:     {'✓ PASS' if c3 else '✗ FAIL'} ({ret:.1f}%)")
    print(f"  [4] Relative: 2x SPY Sharpe:      {'✓ PASS' if c4 else '✗ FAIL'} ({sharpe} vs 2x {bench_sharpe})")

    if c1 and c2 and c3 and c4:
        print("\n🏆 STRATEGY PASSES ALL WEEK 1 HURDLES.")
    else:
        print("\n⚠️ STRATEGY DOES NOT MEET ALL HURDLES YET.")


def cmd_compare(args: argparse.Namespace) -> None:
    """Week 2: Run parallel backtests to isolate Claude's contribution vs Quantitative Baseline."""
    start_d = _parse_date(args.start)
    end_d = _parse_date(args.end)
    print(f"Running Parallel Comparison (Week 2): Claude LLM vs Quantitative Baseline...")
    print(f"Window: {start_d} to {end_d} | Starting Cash: ${args.cash:,.2f}")

    print("\n[1/2] Replaying Test B: Quantitative Baseline...")
    base_res = _run_backtest_instance("baseline", start_d, end_d, args.cash, args.sweep)

    print("\n[2/2] Replaying Test A: Claude LLM System...")
    llm_res = _run_backtest_instance("llm", start_d, end_d, args.cash, args.sweep)

    # Scorecard
    llm_wr = llm_res.get("win_rate_pct") or 0.0
    base_wr = base_res.get("win_rate_pct") or 0.0
    llm_sharpe = llm_res.get("sharpe") or 0.0
    base_sharpe = base_res.get("sharpe") or 0.0
    llm_dd = abs(llm_res.get("max_drawdown_pct") or 0.0)
    base_dd = abs(base_res.get("max_drawdown_pct") or 0.0)
    llm_ret = llm_res.get("total_return_pct") or 0.0
    base_ret = base_res.get("total_return_pct") or 0.0

    wr_edge = "✓ Claude wins" if llm_wr > base_wr else ("✗ Baseline wins" if base_wr > llm_wr else "Tie")
    sharpe_edge = "✓ Claude wins" if llm_sharpe > base_sharpe else ("✗ Baseline wins" if base_sharpe > llm_sharpe else "Tie")
    dd_edge = "✓ Claude wins" if llm_dd < base_dd else ("✗ Baseline wins" if base_dd < llm_dd else "Tie")
    ret_edge = "✓ Claude wins" if llm_ret > base_ret else ("✗ Baseline wins" if base_ret > llm_ret else "Tie")

    print("\n" + "=" * 68)
    print("  WEEK 2 SCORECARD: ISOLATING THE LLM'S CONTRIBUTION")
    print("=" * 68)
    header = f"{'Metric':<18} | {'Test A (Claude)':<16} | {'Test B (Baseline)':<18} | {'Edge?'}"
    print(header)
    print("-" * len(header))
    print(f"{'Win Rate':<18} | {llm_wr:>14.1f}% | {base_wr:>16.1f}% | {wr_edge}")
    print(f"{'Sharpe Ratio':<18} | {llm_sharpe:>15.2f} | {base_sharpe:>17.2f} | {sharpe_edge}")
    print(f"{'Max Drawdown':<18} | {llm_dd:>14.1f}% | {base_dd:>16.1f}% | {dd_edge}")
    print(f"{'Net Return':<18} | {llm_ret:>14.1f}% | {base_ret:>16.1f}% | {ret_edge}")
    print("=" * 68)

    print("\nDECISION RULE:")
    if llm_sharpe > base_sharpe and llm_wr > base_wr:
        print("✓ Edge Confirmed: Claude beats the baseline on both Sharpe and Win Rate.")
        print("  -> Proceed to Week 3 Path A (Resume live trading with 50% capital).")
    else:
        print("✗ No LLM Edge: Claude does not beat the baseline on both Sharpe and Win Rate.")
        print("  -> LLM is overhead. Follow Path B (Pivot to systematic rules or specialized quant model).")


def main() -> None:
    p = argparse.ArgumentParser(description="Evan's Trading Agent Backtest Harness & Evaluation Workbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Smoke
    sm = sub.add_parser("smoke", help="run fast 6-month single-stock smoke test")
    sm.add_argument("--symbol", default="NVDA", help="single stock ticker")
    sm.add_argument("--start", default="2024-01-01", help="start date YYYY-MM-DD")
    sm.add_argument("--end", default="2024-07-01", help="end date YYYY-MM-DD")
    sm.add_argument("--cash", type=float, default=1500.0, help="starting cash USD")

    # Download
    dl = sub.add_parser("download", help="download and cache 2+ years of historical bars")
    dl.add_argument("--start", default="2024-01-01", help="start date YYYY-MM-DD")
    dl.add_argument("--end", default="2026-09-30", help="end date YYYY-MM-DD")

    # Run
    rn = sub.add_parser("run", help="run full 2-year backtest for an agent")
    rn.add_argument("--agent", choices=["baseline", "llm"], default="baseline")
    rn.add_argument("--start", default="2024-01-01", help="start date YYYY-MM-DD")
    rn.add_argument("--end", default="2026-09-30", help="end date YYYY-MM-DD")
    rn.add_argument("--cash", type=float, default=1500.0, help="starting cash USD")
    rn.add_argument("--sweep", default=None, help="cash sweep ETF symbol, e.g. SPYM")

    # Compare
    cmp = sub.add_parser("compare", help="run parallel Test A vs Test B scorecard")
    cmp.add_argument("--start", default="2024-01-01", help="start date YYYY-MM-DD")
    cmp.add_argument("--end", default="2026-09-30", help="end date YYYY-MM-DD")
    cmp.add_argument("--cash", type=float, default=1500.0, help="starting cash USD")
    cmp.add_argument("--sweep", default=None, help="cash sweep ETF symbol, e.g. SPYM")

    args = p.parse_args()
    if args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "download":
        cmd_download(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
