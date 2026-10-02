# Backtest report: Claude LLM agent vs momentum baseline (2024)

Run on 2024-10-01/02 (SGT). Branch `worktree-harness-fixes`.

## Verdict

Claude does **not** add a measurable edge over the free momentum baseline. By the decision rule
agreed earlier (Claude must beat the baseline on **both** Sharpe and win rate), it fails: its Sharpe
lead is 0.03 (noise) and its win rate is 3.5 points lower. Its one real advantage is roughly half the
drawdown. It also costs about $27 per year in API calls on a $1,500 account, against $0 for the
baseline. Neither strategy beats plain SPY buy-and-hold in 2024.

**Recommendation:** remove the LLM layer (branch `refactor/remove-llm-layer`), keep the baseline,
the SPYM cash sweep and the harness.

## Setup

| Item | Value |
|---|---|
| Window | 2024-01-01 to 2024-12-31 (about 252 trading days) |
| Universe | top-100 liquid US stocks plus SPYM (112 symbols cached) |
| Starting cash | $1,500 |
| Cash sweep | SPYM, 5% reserve |
| Claude agent | Haiku 4.5 screens, Opus 5.5 proposes (daily close-bar research cycle) |
| Baseline agent | `baseline-momentum-v1` (deterministic rules) |
| Both agents | same verifier gate, sizing, stops, slippage, kill switch |

## Results

| Metric | Claude | Baseline | SPY buy-and-hold |
|---|---|---|---|
| Ending equity | $1,806.09 | $1,871.96 | n/a |
| Net return | +20.32% | **+24.71%** | +25.59% |
| Sharpe | **1.39** | 1.36 | 1.89 |
| Max drawdown | **-6.36%** | -12.36% | -8.41% |
| Win rate | 43.9% | **47.4%** | n/a |
| Expectancy (R per trade) | 0.221 | **0.33** | n/a |
| Profit factor | 1.42 | **1.64** | n/a |
| Trades | 107 | 133 | n/a |
| Avg invested | 63.2% | 45.4% | n/a |
| Kill switch tripped | 2024-08-05 (-5.9%) | 2024-03-15 (-5.1%) | n/a |

Claude only: 493 proposals, 34 unfilled entries, avg win 1.458 R, avg loss -0.748 R, avg hold 10.4 bars,
159 SPYM sweep orders.

Plan hurdles: both agents pass absolute Sharpe > 1.0, max drawdown < 30% and positive return. Both
fail the relative hurdle (Sharpe at least 2x SPY: 1.39 and 1.36 against a required 3.78).

## Cost

| Item | Amount |
|---|---|
| Measured cost per research cycle | about $0.11 (Haiku $0.0085 + Opus $0.104) |
| Spend logged by the new ledger | $27.70 |
| Spend reported before the ledger existed | $13.00 (seed entry) |
| Total ledger | **$40.70** of the $42 cap |

A full year of daily Claude cycles costs about $27 on a $1,500 account, roughly 1.8% of capital, before
any benefit. That is a large drag next to a +20% return.

## What went wrong along the way

1. **First compare was invalid.** The harness never loaded `.env`, so every LLM call failed with an
   authentication error. The backtester swallowed that as NO_TRADE and printed a fake "Claude 0%
   win rate". Fixed: the harness loads `.env`, records the first error of each kind, and aborts if
   the LLM agent only errors.
2. **Two runs were stopped wrongly.** I monitored the main checkout's cache while the runs wrote to the
   worktree's own cache, so a healthy run looked hung (an LLM call waiting on the network shows 0% CPU).
   I killed it after 40 minutes. Roughly $7.50 of work was stranded.
3. **Cache is fragile.** The cache key covers the whole prompt, including account state that depends on
   earlier Claude decisions, and the window's end date. Reruns only reuse the exact same chain, and
   changing the end date forfeits everything.
4. **No spend visibility.** Nothing logged billed calls, so about $13 looked like "nothing". Fixed with the
   spend ledger, a hard cap, a 3-failures-in-a-row abort and a 240 s client timeout.

## Caveats

- One year of data, and 2024 was a strong bull market for the SPY benchmark.
- One run of a nondeterministic model. A rerun would give different decisions and different numbers.
- The Sharpe difference (1.39 vs 1.36) is well inside run-to-run noise.
- Small account ($1,500): fixed per-order fees and rounding to whole shares affect both agents.
- The earlier baseline figures in the previous session (Sharpe 1.42, +61%) used a different window and
  are not comparable.
- A 49-call partial run earlier showed a 65% win rate for Claude on only a handful of trades. That is not
  statistically meaningful and is superseded by this full-year run.

## Files and where things live

| What | Where |
|---|---|
| Harness | `scripts/backtest_harness.py` (baseline only; the spend guard and `compare` were removed with the LLM layer, see git history at commit `7e2f386`) |
| Backtester error surfacing | `src/trading_agent/backtest.py` |
| Spend ledger (historical) | `state/backtest_cache/llm_spend.jsonl` |
| Run log | `$CLAUDE_JOB_DIR/tmp/llm_2024.log` |
| Saved partial LLM-removal edit | `$CLAUDE_JOB_DIR/tmp/partial-llm-removal.patch` |

## Next steps

1. Finish removing the LLM layer on `refactor/remove-llm-layer`: drop the Claude agents, the model and
   scan-interval settings, and the harness's `llm` path; keep the baseline and the sweep. Run the tests
   and commit.
2. Resume live trading only at reduced size, per the earlier plan (about half capital, 30 trades,
   compare live results with the backtest).
3. If an edge is still wanted, test other signals on the baseline's framework rather than adding an LLM.
