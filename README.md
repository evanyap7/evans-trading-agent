# Evan's Trading Agent

**Author**: Evan Yap ([@evanyap7](https://github.com/evanyap7))

An autonomous swing-trading agent for a **Webull Singapore** account, built from
`trading-agent.md`. The LLM invents trades. Deterministic code verifies
them, sizes them, and approves or rejects them. The official Webull SDK
executes them. No human approval step, and no way for the model to go around
the envelope.

```
research (after US close)  evidence ─► LLM ─► verifier ─► sizing ─► risk ─► queue
execute  (09:45 ET)        fresh quotes ─► re-verify ─► re-size ─► full risk ─► preview ─► place
monitor  (every 5 min)     reconcile ─► circuit breakers ─► stop / target / time-stop exits
```

> No model can promise profits. This system aims for controlled risk and reliable
> execution, and for failures to stay contained. Whether it has an edge is
> something you measure in shadow mode before you trust it with real money.

## What exists today (Phases 0, 1, 4 and most of 5)

| Blueprint component | Where |
|---|---|
| Structured LLM trade contract (`OPEN` / `CLOSE` / NO_TRADE) | `schemas.py` |
| Tiered LLM Intelligence (Fast quantitative screener + Deep reasoning strategist) | `agents.py` |
| Real-time multi-source financial news & macro evidence (Bloomberg, WSJ, Economist, Reuters, NYSE) | `news.py` |
| Telegram personal assistant bot alerts & 9:00 AM daily executive morning briefing | `alerts.py` |
| Rule-based momentum baseline, the benchmark the LLM has to beat | `agents.py` |
| Point-in-time price features as citable evidence | `features.py` |
| Quant verifier: grounding, freshness, price collar, ATR stop, R:R, probability edge vs break-even, net edge after costs | `verifier.py` |
| Position sizing from fractional Kelly, risk budget and caps (the LLM never picks quantity) | `portfolio.py` |
| Deterministic risk engine: every limit in `config/risk_limits.yaml` | `risk.py` |
| Capital rotation & pre-existing portfolio position exit engine | `orchestrator.py`, `execution.py` |
| Idempotent execution: deterministic client order IDs, preview check, no blind retries | `execution.py` |
| Broker-side GTC stop-loss placed after each fill (re-armed if missing) | `execution.py` |
| Append-only, hash-chained ledger (SQLite) | `ledger.py` |
| Reconciliation, kill switch (auto + operator), drawdown breaker | `orchestrator.py`, `killswitch.py` |
| Webull SG adapter (official SDK v3.0.2 + resilient yfinance fallback, UAT + prod) | `broker/webull.py` |
| Simulated broker for tests and offline runs | `broker/simulated.py` |
| 83 unit and integration tests covering the blueprint's failure list | `tests/` |

**Not built yet:** offline backtester replay engine (Phase 3), Postgres/Timescale database backend,
web dashboard, real-time WebSocket order-event stream (gRPC/polling used instead), options (Phase 7;
Webull's own MCP config marks SG as `supports_options=False`).

## Edge, Kelly and scan cadence

The rules below come from a Polymarket agent (scan often, only bet when the price is off by more
than 8%, size with Kelly, never have more than 6% at risk). Here they are applied to stocks:

- **Mispricing gate.** A stop/target pair has a break-even win rate of `down / (up + down)`, which is
  33% for a 2:1 setup. The verifier rejects any idea whose `confidence` doesn't beat that by
  `signal.min_probability_edge` (0.08).
- **Kelly sizing.** Risk per trade is `kelly_fraction x (p - (1 - p) / reward_risk)` of equity (quarter
  Kelly by default). It is still capped by `requested_risk_pct` and `max_risk_per_trade_pct`. An idea with no
  Kelly edge gets no size.
- **10% at risk.** `max_portfolio_risk_pct: 10.0` caps the sum of `(price - stop) x qty` across open positions.
- **10-minute scans.** With `CONTINUOUS_TRADING=true`, the research pass runs every
  `SCAN_INTERVAL_MINUTES` (default 10) during regular hours. Each scan is an LLM call, so costs scale with it.

## Cash sweep (idle cash in an index ETF)

Backtests showed the agent holding only 17-32% of the account on average, so most of the money sat in
cash while the market rose. With `cash_sweep.enabled: true` in `config/risk_limits.yaml`:

- **Execute** (09:45 ET) buys whole shares of `cash_sweep.symbol` with cash above `reserve_pct` of equity
  and above what working entry orders need. Default `SPYM`, an S&P 500 ETF at about $90 a share, so a
  small account can hold whole shares (SPY is about $770). Use `SGOV` (T-bills) for low risk instead.
- **Entries are funded by selling sweep shares** at execution when cash is short. If the sale does not fill
  within a few seconds, the entry is dropped rather than sent without cash.
- **The sweep is not a trade.** Its shares are removed from what the agent, sizing and risk engine see, and
  their value counts as spendable cash. So they never use up exposure, sector or portfolio-risk limits,
  and the agent cannot close them.
- **Breakers ignore the sweep's P&L.** The drawdown kill switch and daily-loss check watch trading equity
  (equity minus the sweep's gain or loss), so an ordinary market dip does not halt the agent. The account
  as a whole now moves with the market: expect index-sized drawdowns.
- Only shares the sweep bought itself count (tracked from its own fills in the ledger). SPYM you bought by
  hand stays a manual position. If sweep shares disappear at the broker, reconciliation fails and the kill
  switch engages.
- The kill switch stops sweep buys. In shadow mode sweep orders are recorded, never sent.

Backtest (baseline agent, $900, 2024-01-02 to 2026-09-25, kill switch on as live):

| | Return | Max drawdown | Last 12 months |
|---|---|---|---|
| No sweep | +8.5% | -5.5% | +5.8% |
| SGOV sweep | +18.3% | -5.3% | +8.2% |
| SPYM sweep | +60.8% | -14.5% | +47.4% |
| SPY buy-and-hold | +68.5% | -18.8% | +18.5% |

The SPYM sweep's own contribution is about the market's return on the ~70% it holds. The rest of the
last-12-months figure is agent trades, which are lumpy at one share each (a few names made most of it),
so do not count on that part repeating. Check cash settlement rules for your Webull SG account before
enabling: buying with unsettled sale proceeds and then selling again before settlement can break
cash-account rules.

## Setup

```bash
cd ~/trading-agent
uv sync --extra webull
cp .env.example .env         # then fill it in yourself; never paste keys into a chat
uv run pytest                # 149 passing
uv run trading-agent research --broker sim   # offline dry run on synthetic data
```

## Going live, one gate at a time

1. **UAT read-only:** set `WEBULL_ENVIRONMENT=uat`, then run `trading-agent accounts` → put the id in
   `WEBULL_ACCOUNT_ID` → run `trading-agent probe`. Check that balance, positions and snapshot fields
   match what `broker/webull.py` expects.
2. **Shadow mode** (`TRADING_MODE=shadow`, the default): schedule `tick --agent llm`
   (`scripts/com.trading-agent.tick.plist`). Everything runs except sending orders. Run it for weeks
   and compare LLM ideas against the baseline.
3. **UAT orders:** `TRADING_MODE=broker`, `WEBULL_ENVIRONMENT=uat`. Exercise preview, fills, stops,
   cancels and the kill switch against Webull's test environment.
4. **Live execution:** `WEBULL_ENVIRONMENT=prod`, `TRADING_MODE=broker`, **and** set
   `live_trading.enabled: true` in `config/risk_limits.yaml`.

Real orders need all three switches. With any one of them off, entries are risk-rejected.

## Telegram Personal Assistant Integration

The agent dispatches real-time events to your Telegram bot:
- **Nightly Research Summaries**: Dispatched after market close with ingested macro evidence and queued setups.
- **Order Executions & Fills**: Real-time notifications for live submissions and broker-side GTC stops.
- **Automated Exits & Capital Rotation**: Instant alert when take-profit, stop-loss, or rotation exits execute.
- **Daily 9:00 AM Morning Briefing**: Complete portfolio snapshot, 24h P&L, open positions with unrealized gains, and analyst stance delivered at 09:00 SGT (`trading-agent morning-report`).

## Backtesting

```bash
uv run trading-agent backtest --start 2024-01-02                  # baseline, $1,500, live limits
uv run trading-agent backtest --start 2024-01-02 --no-halt        # keep going after the kill switch trips
uv run trading-agent backtest --start 2024-01-02 --no-trailing    # A/B the trailing stop
uv run trading-agent backtest --start 2024-01-02 --cash 900 --sweep SPYM   # A/B the cash sweep (--no-sweep)
uv run trading-agent backtest --start 2025-06-02 --agent llm --max-llm-calls 40   # costs money; cached
```

`backtest.py` replays daily bars through the live agent, verifier, sizing, risk engine and trailing
stop. Research at each close sees only bars up to that day. Entries fill at the next open as DAY limits.
Stops, targets and gaps come from the daily range, and when a bar touches both, the stop is assumed first.
The run prints expectancy in R, win rate, profit factor, drawdown, Sharpe, exit reasons, the most common
rejection reasons, the average share of equity invested, and two benchmarks: SPY buy-and-hold (with the
excess return over it) and SPY held only while above its 200-day average. If the agent cannot beat both
after LLM costs, its trades add nothing. The run writes `summary.json`, `trades.csv` and `equity.csv` under
`state/backtests/`. Bars, earnings dates and LLM outputs are cached in `state/backtest_cache/`.

Results are optimistic in three known ways: the universe is today's list, no historical news is
replayed, and an LLM has seen these dates in training. Treat the baseline as the honest control and an
LLM run as an upper bound.

## Operating

```bash
uv run trading-agent status                     # kill switch, trades, orders, recent events
uv run trading-agent morning-report --broker webull # send fresh 9:00 AM briefing to Telegram
uv run trading-agent kill --reason "..."        # stop new orders; cancel working entries
uv run trading-agent unkill
uv run trading-agent verify-ledger              # detect tampering with the audit log
```

Earnings dates come from Webull's earnings calendar at every research and execute cycle (if the next
report is not listed yet, it is estimated as the last one plus a quarter). `config/events.yaml` overrides
the feed. Individual stocks with no date are blocked from new entries. ETFs are not affected.
`market_calendar.py` holds NYSE holidays through 2027 and refuses to run after that until extended.

## Safety guarantees

- **No invented market data.** Quotes come from Webull (Nasdaq Basic). The yfinance fallback uses only real
  bid/ask and the exchange timestamp; a one-sided or stale quote blocks new entries instead of being
  filled in. Bars without a timestamp are dropped.
- **The kill switch always hits the same file.** Relative `STATE_DIR` / `WEBULL_TOKEN_DIR` resolve against
  the project root, so `trading-agent kill` works from any directory.
- **Positions are not left without a stop.** Each monitor pass re-arms a missing or rejected broker-side
  stop (bounded, and never while an exit sell is working). A rejected exit is retried up to
  `max_exit_attempts_per_day`; after that the stop is left in place.
- **Winners run; stops only ratchet up.** Exits come from the stop, the take-profit or the time stop. From +1R
  the stop moves to breakeven, and from +2R it trails 1.5R behind price (`trail_*` in `risk_limits.yaml`).
  The broker-side stop is replaced only after the old one's cancel is confirmed. `daily_profit_target_usd` is
  reported but never triggers a sale.
- **Exits need a real price.** No sell is priced off a zero bid or a quote older than 15 minutes.
- **Agent exits need our own evidence.** Selling (including manual holdings, when
  `agent_may_close_manual_positions` is on) must cite price/position data, never news alone.
- **Liquidity floor for entries:** `min_price_usd` and `min_avg_dollar_volume_usd`.
- **One cycle at a time.** A lock (`state/agent.lock`) stops a manual run overlapping the scheduled tick, and
  network calls time out so a hung feed cannot stall monitoring.
- **A crash in execute or research never skips monitor**, and it sends a Telegram alert.
- **Alerts are not silently dropped.** Telegram messages fall back to plain text if Markdown is rejected.
- **Limits are sanity-checked on load.** Values such as >100% exposure, margin/short/options switches or
  >5% risk per trade refuse to load instead of trading.
- **Webull order status is matched exactly** by client order id, and 5xx submission errors are treated as
  ambiguous (reconciled, not assumed rejected).
