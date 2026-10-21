# Evan's Trading Agent

[![CI](https://github.com/evanyap7/evans-trading-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/evanyap7/evans-trading-agent/actions/workflows/ci.yml)

**Author**: Evan Yap ([@evanyap7](https://github.com/evanyap7)), SUTD Engineering Systems & Design

I built a trading system for a small **Webull Singapore** account to find an edge. It found none.

The project started as one question: is there an edge here, and how would I know? So I built three things. A rule-based
momentum agent, a deterministic risk and execution layer, and a walk-forward research harness that tests every idea
against luck-adjusted statistics before it gets near money.

**Short answer so far: no edge found.** I ran four independent studies on liquid US equities and ETFs. They all say the same
thing, and each one is written up with its method and caveats. I would rather say that plainly than tune until a chart looks good.

**Status:** the system placed its first real orders on 2026-09-23, one share at a time. I paused live trading with the
kill switch on 2026-10-01 until the longer backtests below were done.

## Research findings

| Study | Question | Result | Report |
|---|---|---|---|
| Classic quant signals | Do momentum, reversal, low-vol (51 US stocks) or trend following (18 and 25 ETFs) beat buy-and-hold out of sample? | **No.** Deflated Sharpe 0.05 to 0.65, all below the ~0.95 bar. Trend following cut risk (about a third of the volatility, half the drawdown), not return. | [quant-research-report](docs/quant-research-report.md) |
| Insider buying | Do SEC Form 4 open-market purchases predict returns? | **No usable edge.** A small raw effect, mostly survivorship and noise, that vanishes inside sizing, stops and risk limits. | [insider-signal-study](docs/insider-signal-study.md) |
| LLM agent | Does a Claude agent (Haiku screen, Opus proposal) beat a free momentum baseline? | **No.** Sharpe lead of 0.03 over the baseline (noise), lower win rate, and both trailed SPY (+20.3% and +24.7% vs +25.6% in 2024). About $27 a year in API cost. Removed. | [backtest-llm-vs-baseline-2024](docs/backtest-llm-vs-baseline-2024.md) |
| Momentum baseline | Does the live rule set beat SPY? | **Matches it, does not beat it** (+24.7% vs +25.6% in 2024; +66.8% vs +70.1% over 2022 to 2026 at roughly half the drawdown). | same reports |

The walk-forward cross-sectional long/short on 51 stocks scored an out-of-sample Sharpe of **-0.22** (95% interval -1.33 to 0.74)
against 1.46 for equal-weight on the same stocks. Beta is near zero, so the shortfall is not market exposure.

### Method

I wanted a negative result I could trust, so the research layer does this:

- **No lookahead, tested.** A test rewrites the future of a price series and asserts that past signals do not move.
- **Walk-forward selection.** I pick parameters on 2 years, score them only on the next 6 months, then roll. Only
  out-of-sample stretches are joined into the reported series.
- **Luck correction.** Deflated Sharpe ratio (Bailey & Lopez de Prado), which accounts for skew, kurtosis, sample length
  and the number of variants tried, plus a block-bootstrap interval on the Sharpe.
- **Trial counting.** Every grid and every earlier study raises the hurdle for the next one (30 trials, then 36).
- **Pre-registration.** I wrote down the extra diversifier ETFs before the wider run. This is self-attested, with no external timestamp.
- **Stated costs and biases.** 10 bps on the stock book, 5 bps on ETFs, 50 bps a year borrow. The universe is today's
  survivors, so absolute returns are inflated. Both are listed under each report's caveats.

```bash
uv run trading-agent quant-research                                   # cross-sectional, 6 years
uv run trading-agent quant-research --strategy trend --diversified    # 25-ETF trend following
```

The maths behind the signals and sizing (descriptive stats, Bayes, Monte Carlo, regression) is in
[algorithmic_trading_maths.md](docs/algorithmic_trading_maths.md).

## System design

```
research (after US close)  evidence ─► agent ─► verifier ─► sizing ─► risk ─► queue
execute  (09:45 ET)        fresh quotes ─► re-verify ─► re-size ─► full risk ─► preview ─► place
monitor  (every 5 min)     reconcile ─► circuit breakers ─► stop / target / time-stop exits
```

The agent only proposes `OPEN` / `CLOSE` / `NO_TRADE`. Deterministic code verifies, sizes and approves or rejects, and the
agent never picks a quantity or goes around the risk envelope. The goal is controlled risk and contained
failures, not a promise of profit.

| Concern | Approach | Where |
|---|---|---|
| Trade verification | Grounding, freshness, price collar, ATR stop, R:R, probability edge over break-even, net edge after costs | `verifier.py` |
| Sizing | Fractional Kelly, capped by a per-trade risk budget and 6% total portfolio risk | `portfolio.py` |
| Risk limits | Every limit in `config/risk_limits.yaml`; absurd values (>100% exposure, shorts, margin, >5% per trade) refuse to load | `risk.py`, `config.py` |
| Execution | Deterministic client order IDs, preview check, no blind retries, 5xx treated as ambiguous and reconciled | `execution.py` |
| Stops | Broker-side GTC stop after every fill, re-armed if missing; ratchets to breakeven at +1R and trails from +2R | `execution.py` |
| Audit | Append-only, hash-chained SQLite ledger with a tamper check (`verify-ledger`) | `ledger.py` |
| Fail-safe | Reconciliation, auto and manual kill switch, drawdown breaker, one-cycle-at-a-time lock | `orchestrator.py`, `killswitch.py` |
| Data | Webull quotes, Databento historical and live, yfinance fallback; no invented prices, stale or one-sided quotes block entries | `broker/webull.py`, `databento_data.py` |
| Backtesting | Replays daily bars through the live agent, verifier, sizing and risk engine; stop assumed first when a bar touches both | `backtest.py` |

It is about 6,500 lines of Python and **191 tests**. CI runs them offline on every push. They cover look-ahead, idempotency,
stop re-arming, kill-switch behaviour and the quant statistics.

## Backtest: baseline agent and the cash sweep

Early runs showed the agent holding only 17 to 32% of the account, so most of the money sat in cash. I added an optional sweep
that parks idle cash in an index ETF (`SPYM`, about $90 a share so a small account can hold whole shares) and sells it to fund entries.
Sweep shares are invisible to the agent and the breakers, so a market dip does not trip the kill switch.

$900 start, 2024-01-02 to 2026-09-25, kill switch on as live:

| | Return | Max drawdown | Last 12 months |
|---|---|---|---|
| No sweep | +8.5% | -5.5% | +5.8% |
| SGOV sweep | +18.3% | -5.3% | +8.2% |
| SPYM sweep | +60.8% | -14.5% | +47.4% |
| SPY buy-and-hold | +68.5% | -18.8% | +18.5% |

The sweep is mostly beta. Its contribution is about the market's return on the roughly 70% it holds. The agent's own trades
are lumpy at one share each, so do not extrapolate the last-12-months figure.

**Known biases:** the universe is today's list, no historical news is replayed, and the sample covers few regimes (2024 was a
strong bull year). Backtests are optimistic on all three.

```bash
uv run trading-agent backtest --start 2024-01-02                              # baseline, live limits
uv run trading-agent backtest --start 2024-01-02 --cash 900 --sweep SPYM      # A/B the cash sweep
```

Output includes expectancy in R, win rate, profit factor, drawdown, Sharpe, rejection reasons, and two benchmarks: SPY
buy-and-hold and SPY held only above its 200-day average.

## Run it

```bash
uv sync --extra webull
cp .env.example .env                           # fill in yourself; never paste keys into a chat
uv run pytest                                  # 191 passing, no credentials needed
uv run trading-agent research --broker sim     # offline dry run on synthetic data
```

Going live is gated one step at a time: UAT read-only, then shadow mode (everything runs except sending orders), then UAT
orders, then production. Real orders need `WEBULL_ENVIRONMENT=prod`, `TRADING_MODE=broker` **and**
`live_trading.enabled: true`. With any one off, entries are risk-rejected.

```bash
uv run trading-agent status          # kill switch, trades, orders, recent events
uv run trading-agent kill --reason "..."   # stop new orders, cancel working entries
uv run trading-agent verify-ledger   # detect tampering with the audit log
```

Telegram alerts cover nightly research summaries, fills, exits and a 9:00 SGT morning briefing.

## Safety guarantees

- **No invented market data.** One-sided, stale or timestamp-less quotes block entries instead of being filled in.
- **No position without a stop.** Each monitor pass re-arms a missing or rejected broker-side stop. Exits are never priced off a zero bid or a quote older than 15 minutes.
- **Stops only ratchet up.** The broker-side stop is replaced only after the old one's cancel is confirmed.
- **Agent exits need price or position evidence**, never news alone.
- **A crash in execute or research never skips monitor**, and it sends an alert.
- **Limits are sanity-checked on load.** Bad config refuses to start rather than trade.
- **Webull order status is matched exactly** by client order ID.

## Not built

Postgres/Timescale backend, web dashboard, WebSocket order events (polling is used), options (Webull SG does not support them).

**Next research step:** a universe where small capital can compete (liquid small caps) with a point-in-time constituent
list to remove survivorship bias.

> Not investment advice. The default mode is shadow, and the evidence so far says to hold the index.
