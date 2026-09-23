# Evan's Trading Agent

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
| LLM agent (structured outputs, reasoning and refusal fallback) | `agents.py` |
| Rule-based momentum baseline, the benchmark the LLM has to beat | `agents.py` |
| Point-in-time price features as citable evidence | `features.py` |
| Quant verifier: grounding, freshness, price collar, ATR stop, R:R, net edge after costs | `verifier.py` |
| Position sizing from risk budget and caps (the LLM never picks quantity) | `portfolio.py` |
| Deterministic risk engine: every limit in `config/risk_limits.yaml` | `risk.py` |
| Idempotent execution: deterministic client order IDs, preview check, no blind retries | `execution.py` |
| Broker-side GTC stop-loss placed after each fill | `execution.py` |
| Append-only, hash-chained ledger (SQLite) | `ledger.py` |
| Reconciliation, kill switch (auto + operator), drawdown breaker | `orchestrator.py`, `killswitch.py` |
| Webull SG adapter (official SDK v3.0.2, UAT + prod) | `broker/webull.py` |
| Simulated broker for tests and offline runs | `broker/simulated.py` |
| 79 tests covering the blueprint's failure list | `tests/` |

**Not built yet:** backtester (Phase 3), news/filings/earnings ingestion, Postgres/Timescale,
dashboard, real-time order-event stream (gRPC; polling is used instead), options (Phase 7;
Webull's own MCP config marks SG as `supports_options=False`).

## Setup

```bash
cd ~/trading-agent
uv sync --extra webull
cp .env.example .env         # then fill it in yourself; never paste keys into a chat
uv run --group dev pytest    # 79 passing
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
4. **Live canary:** `WEBULL_ENVIRONMENT=prod`, `TRADING_MODE=broker`, **and** set
   `live_trading.enabled: true` in `config/risk_limits.yaml` in a git commit. Keep the default caps
   ($500 per order, 20% total exposure, 1% daily loss, 5% drawdown).

Real orders need all three switches. With any one of them off, entries are risk-rejected.

## Operating

```bash
uv run trading-agent status                     # kill switch, trades, orders, recent events
uv run trading-agent kill --reason "..."        # stop new orders; cancel working entries
uv run trading-agent unkill
uv run trading-agent verify-ledger              # detect tampering with the audit log
```

Maintain `config/events.yaml` (earnings dates) by hand until an earnings feed is wired in.
Individual stocks with no date on file are blocked from new entries. ETFs are not affected.
`market_calendar.py` holds NYSE holidays through 2027 and refuses to run after that until extended.

## Safety guarantees

- **No invented market data.** Without a Webull quote subscription, the yfinance fallback uses only real
  bid/ask and the exchange timestamp. A one-sided or stale quote blocks new entries instead of being
  filled in. Bars without a timestamp are dropped.
- **The kill switch always hits the same file.** Relative `STATE_DIR` / `WEBULL_TOKEN_DIR` resolve against
  the project root, so `trading-agent kill` works from any directory.
- **Positions are not left without a stop.** Each monitor pass re-arms a missing or rejected broker-side
  stop (bounded, and never while an exit sell is working). A rejected exit is retried up to
  `max_exit_attempts_per_day`; after that the stop is left in place.
- **Exits need a real price.** No sell is priced off a zero bid or a quote older than 15 minutes.
- **Your own holdings are off-limits by default.** With `agent_may_close_manual_positions: false` the agent
  sells only what it bought. Agent exits must cite our own price/position evidence, never news alone.
- **Liquidity floor for entries:** `min_price_usd` and `min_avg_dollar_volume_usd`.
- **One cycle at a time.** A lock (`state/agent.lock`) stops a manual run overlapping the scheduled tick, and
  network calls time out so a hung feed cannot stall monitoring.
- **A crash in execute or research never skips monitor**, and it sends a Telegram alert.
- **Alerts are not silently dropped.** Telegram messages fall back to plain text if Markdown is rejected.
- **Limits are sanity-checked on load.** Values such as >100% exposure, margin/short/options switches or
  >5% risk per trade refuse to load instead of trading.
- **Webull order status is matched exactly** by client order id, and 5xx submission errors are treated as
  ambiguous (reconciled, not assumed rejected).
