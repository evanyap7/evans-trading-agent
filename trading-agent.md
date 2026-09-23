# Autonomous Webull Trading Agent Blueprint

## Project goal

Build a highly autonomous trading agent for a **Webull Singapore account** that can:

- Research markets and generate original trade ideas.
- Trade stocks, ETFs and eventually options.
- Decide whether to buy, sell, hold or exit.
- Choose entries, exits, stop losses and holding periods.
- Execute approved trades automatically without human confirmation.
- Monitor positions and close trades when the thesis or risk conditions change.

The target style is **daily/swing trading**, with positions generally held for days to weeks.

> **Important:** No trading model can guarantee accuracy or profits. The engineering objective is positive expected return after costs, controlled drawdowns, reliable execution and rapid containment of failures.

## Core design principle

The LLM may **invent and propose trades**, but it must not have unrestricted control over the brokerage account.

Use this permission model:

```text
LLM proposes creatively
        ↓
Software verifies objectively
        ↓
Risk engine authorizes automatically
        ↓
Webull previews and executes
        ↓
System monitors and reconciles
```

There is no required human approval in this flow. Approval is automatic when all deterministic checks pass.

### Responsibility split

| Component | Responsibility |
|---|---|
| LLM trading agent | Research securities, combine evidence, create theses and propose trades |
| Quant verifier | Recalculate the LLM's numbers and test whether estimated edge exceeds costs |
| Portfolio constructor | Convert approved ideas into target positions and quantities |
| Risk engine | Enforce hard limits and approve or reject trades |
| Execution engine | Preview, submit, modify and cancel orders |
| Reconciliation service | Compare internal records with Webull balances, positions, orders and fills |
| Monitoring service | Detect stale data, abnormal losses, broker disconnects and software failures |

The LLM must never be able to modify risk limits, disable safeguards, access raw credentials or bypass the risk engine.

## High-level architecture

```text
Market data / fundamentals / news / events
                    ↓
          Ingestion and validation
                    ↓
        Point-in-time data storage
                    ↓
        Feature and evidence layer
                    ↓
            LLM trading agent
                    ↓
       Structured trade proposal
                    ↓
       Independent quant verifier
                    ↓
          Portfolio constructor
                    ↓
       Deterministic risk engine
                    ↓
          Webull order preview
                    ↓
        Automatic order execution
                    ↓
      Order monitoring and ledger
                    ↓
     Reconciliation, alerts and audit
```

## Recommended technology stack

- **Python:** Data ingestion, quantitative analysis, backtesting, models and Webull SDK integration.
- **FastAPI:** Internal APIs and service orchestration.
- **Pydantic:** Strict validation of model outputs and internal messages.
- **PostgreSQL with TimescaleDB:** Orders, positions, time series, trading state and audit records.
- **Object storage:** Immutable source files, raw market data, prompts, model outputs and model artifacts.
- **Redis:** Short-lived locks, queues and idempotency controls where appropriate.
- **Next.js and TypeScript:** Monitoring dashboard and authenticated control plane.
- **Docker on a persistent VM/container host:** Live trading services, MQTT market-data streams and gRPC order events.
- **Secrets manager:** Webull App Key, App Secret and production credentials.

Do not run the core live trader solely in short-lived Vercel serverless functions. Vercel can host the dashboard, but persistent market-data and broker-event connections need a long-running process.

## Main trading cycle

For a daily/swing system:

1. Retrieve cash, buying power, positions, open orders and recent fills.
2. Retrieve validated market, fundamental, event and news data.
3. Check whether all required data is current and complete.
4. Ask the LLM to identify and rank opportunities.
5. Require the LLM to return structured trade proposals or `NO_TRADE`.
6. Recalculate prices, returns, volatility, costs and risk outside the LLM.
7. Reject unsupported or contradictory proposals.
8. Convert acceptable ideas into target positions.
9. Apply account-level and trade-level risk limits.
10. Preview the order through Webull.
11. Confirm that the broker preview matches the intended order.
12. Submit it automatically.
13. Monitor acknowledgements, partial fills, rejections and cancellations.
14. Reconcile internal state against Webull.
15. Re-evaluate exits, thesis invalidation and portfolio risk on schedule.

Run the main research cycle after the US session closes. Run smaller monitoring cycles during market hours for exits, fills, risk events and material new information.

## LLM trade contract

Never execute free-form text such as “buy some Apple calls.” Require schema-validated output similar to:

```json
{
  "decision_id": "uuid",
  "timestamp": "2026-09-23T20:05:00Z",
  "action": "OPEN",
  "instrument_type": "EQUITY",
  "symbol": "AAPL",
  "side": "BUY",
  "strategy": "momentum_breakout",
  "thesis": "Price and earnings-revision momentum remain positive.",
  "holding_period_days": 10,
  "confidence": 0.72,
  "expected_return_pct": 3.1,
  "invalidation_price": 224.50,
  "entry": {
    "order_type": "LIMIT",
    "limit_price": 230.00
  },
  "exit": {
    "take_profit": 239.00,
    "stop_loss": 224.50,
    "time_stop_days": 10
  },
  "requested_risk_pct": 0.4,
  "evidence_ids": [
    "price_snapshot_...",
    "earnings_revision_...",
    "news_document_..."
  ]
}
```

The LLM should request risk rather than select an unrestricted quantity. The portfolio service calculates the final quantity from account equity, stop distance, liquidity, portfolio exposure and correlation.

## Suggested system prompt

```text
You are an autonomous swing-trading portfolio manager.

Identify US stock and ETF opportunities with expected holding periods of 3–20 trading days. You may create original theses by combining price trends, volume, volatility, fundamentals, earnings revisions, news, events and market regime.

Return NO_TRADE unless the supplied, timestamped evidence supports positive expected return after spread, slippage and fees. Every factual claim must reference an evidence ID supplied in the context.

You may choose the instrument, direction, strategy, entry, exit, stop, target and thesis-invalidation conditions. You may not fabricate unavailable facts, alter risk limits, submit unstructured orders or proceed when required data is missing or stale.

Output only the required JSON schema.
```

## Required data

### Stocks and ETFs

| Dataset | Required fields | Purpose |
|---|---|---|
| Price bars | Unadjusted and adjusted OHLCV, VWAP, trade count, timestamps and session | Signals, labels and backtests |
| Quotes | Bid, ask, sizes, midpoint, spread and quote timestamp | Costs and execution decisions |
| Security master | Permanent ID, ticker history, listing/delisting dates, type, exchange, sector and currency | Avoid symbol and survivorship errors |
| Corporate actions | Splits, dividends, mergers, spin-offs and symbol changes | Correct historical prices and positions |
| Fundamentals | Value, period, filing/release timestamp and later revisions | Point-in-time fundamental features |
| Events | Earnings times, ex-dividend dates, economic releases, holidays and halts | Event and gap-risk controls |
| Broker data | Balances, buying power, positions, orders, fills, fees and margin state | Sizing and reconciliation |
| Operational metadata | Source, arrival time, checksum, missing/stale flags and dataset version | Audit and incident diagnosis |

For daily/swing strategies, begin with daily and 5–30 minute bars for a small universe of highly liquid US stocks and ETFs. Add finer quote data when researching execution quality.

### Options

Options require contract-level, point-in-time data:

- Underlying price and symbol.
- Contract identifier, call/put, strike and expiration.
- Multiplier, deliverable and exercise style.
- Bid, ask, sizes, midpoint and quote timestamp.
- OHLC, volume and open interest.
- Implied volatility, delta, gamma, theta, vega and rho.
- Volatility skew, surface and term structure.
- Risk-free rate and dividend assumptions.
- Earnings and ex-dividend timing.
- Exercise and assignment records.
- Commissions, fees, spread and expected slippage.

Options should be a later phase. An options trade requires correct forecasts of direction, volatility and timing while accounting for time decay, spread, liquidity and assignment risk.

### Text and alternative data

Possible sources include:

- Company filings and earnings releases.
- Earnings-call transcripts.
- Timestamped news.
- Analyst estimate revisions.
- Economic releases and interest rates.
- Volatility indices and market-regime indicators.

Store the original document, its first-known timestamp, source, extraction prompt, model version and structured result. This prevents edited webpages or future model knowledge from leaking into historical tests.

## Measuring performance

Do not optimize only for directional accuracy.

| Layer | Useful metrics |
|---|---|
| Forecast | Rank information coefficient, precision among selected trades, Brier score, log loss and calibration |
| Strategy | Net return, Sharpe, Sortino, maximum drawdown, Calmar ratio, hit rate and payoff ratio |
| Portfolio | Turnover, beta, sector concentration, gross/net exposure and liquidity usage |
| Execution | Fill rate, spread paid, slippage, implementation shortfall and rejection rate |
| Reliability | Duplicate orders, stale-data incidents, reconciliation breaks, uptime and recovery time |
| Drift | Feature drift, prediction drift, calibration changes and live-versus-backtest slippage |

A system can have high prediction accuracy and still lose money if losing trades are larger than winners or if spreads, fees and slippage consume the edge.

## Strategy development

Start with one transparent strategy rather than a universal agent.

A suitable initial research candidate is a liquid stock/ETF trend or cross-sectional momentum strategy with volatility, spread, market-regime and event-risk filters. Treat this as a hypothesis to test, not an assumption that it will work.

Development sequence:

1. Define a point-in-time universe.
2. Define exactly when decisions occur.
3. Create forward-return labels for the intended holding period.
4. Subtract estimated spread, slippage and fees from labels and results.
5. Build cash, benchmark, equal-weight and simple rule-based baselines.
6. Test linear or logistic models.
7. Test a tabular tree model such as LightGBM or XGBoost.
8. Let the LLM add evidence-based qualitative reasoning and idea generation.
9. Keep prediction, portfolio sizing and execution as separate stages.
10. Trade only when expected edge exceeds expected costs plus a safety buffer.

## Backtesting and validation

Use the same core strategy and order logic in backtest, shadow, paper and live modes. Only the data and broker adapters should change.

Required safeguards:

- Chronological train, validation and untouched test periods.
- Walk-forward retraining using only previously available information.
- Purging or embargo where forward-return labels overlap.
- Point-in-time joins for constituents, fundamentals, news and events.
- Delisted instruments and historical ticker changes.
- Correct splits, dividends and corporate actions.
- Bid/ask-aware execution assumptions.
- Conservative slippage and latency.
- Partial fills, rejected orders, halts and market sessions.
- Parameter, cost, universe and start-date sensitivity tests.
- A complete record of successful and failed experiments.
- Comparison against simple benchmarks.
- Shadow and paper tests using the production services.

Backtests can overfit when many variations are tried and only the winner is reported. Keep an experiment registry and assess whether performance is broad and stable rather than concentrated in one parameter or time period.

## Quant verification

Every LLM proposal should pass independent checks:

1. **Grounding:** Every factual claim maps to a stored evidence ID.
2. **Freshness:** Prices, quotes and account data are recent enough.
3. **Numerical correctness:** Returns, volatility, valuation, Greeks and position size are recalculated outside the model.
4. **Economic edge:** Expected return exceeds spreads, slippage, fees and a safety margin.
5. **Historical support:** Comparable situations show sufficiently stable results out of sample.
6. **Contradiction check:** A critic model identifies missing or conflicting evidence.
7. **Portfolio fit:** The trade does not create excessive concentration or correlated risk.

A second LLM can act as a critic, but it must not be the final safety control because models can share similar failure modes.

## Deterministic risk engine

All risk controls must be code-based and inaccessible to prompts.

### Pre-trade controls

- Approved product and symbol allowlist.
- Valid instrument and open market session.
- Maximum order value and quantity.
- Maximum position and sector exposure.
- Maximum gross and net portfolio exposure.
- Maximum risk per trade and total portfolio risk.
- Maximum number of new trades per day.
- Minimum expected edge and confidence.
- Maximum spread and minimum liquidity.
- Maximum quote and data age.
- Price collars around current bid/ask or reference prices.
- Earnings and corporate-action restrictions.
- Duplicate decision and client-order-ID rejection.
- Position and cash reconciliation before submission.
- Daily-loss and portfolio-drawdown circuit breakers.

### Example approval logic

```python
def approve_trade(proposal, account, market):
    checks = [
        proposal.symbol in ALLOWED_SYMBOLS,
        market.data_age_seconds <= MAX_DATA_AGE,
        market.spread_pct <= MAX_SPREAD,
        proposal.expected_return_pct >= MIN_EXPECTED_EDGE,
        proposal.confidence >= MIN_CONFIDENCE,
        account.daily_pnl_pct > -MAX_DAILY_LOSS_PCT,
        account.drawdown_pct > -MAX_DRAWDOWN_PCT,
        projected_position_pct(proposal) <= MAX_POSITION_PCT,
        projected_sector_pct(proposal) <= MAX_SECTOR_PCT,
        projected_portfolio_risk(proposal) <= MAX_PORTFOLIO_RISK,
        not has_duplicate_intent(proposal.decision_id),
        not is_near_restricted_event(proposal),
        broker_and_internal_positions_match()
    ]

    return "APPROVED" if all(checks) else "REJECTED"
```

### Example initial limits

These values are illustrative starting controls, not recommendations:

```yaml
live_trading:
  enabled: false
  allowed_instruments:
    - EQUITY
    - ETF
  max_position_pct: 2
  max_total_exposure_pct: 20
  max_risk_per_trade_pct: 0.25
  max_daily_loss_pct: 1
  max_drawdown_pct: 5
  max_new_trades_per_day: 3
  max_order_value_usd: 500
  permit_shorting: false
  permit_margin: false
  permit_options: false
  emergency_liquidation: manual
```

Initially keep `enabled: false`. Turn it on only after shadow and paper testing. Limits must be changed through authenticated, version-controlled configuration—not model prompts.

## Order execution

Use the official Webull SDK and documented OpenAPI rather than private or reverse-engineered endpoints.

Suggested order state machine:

```text
PROPOSED
  → RISK_APPROVED
  → SUBMITTING
  → ACKNOWLEDGED
  → PARTIALLY_FILLED
  → FILLED

Alternative states:
  → CANCEL_PENDING → CANCELLED
  → REJECTED
  → EXPIRED
  → UNKNOWN_RECONCILE
```

Execution requirements:

- Generate and persist a stable client order ID before submission.
- Make every command idempotent.
- Preview orders before submitting them.
- Confirm preview details against the intended order.
- On a timeout, query Webull before retrying.
- Never assume an HTTP error means the broker did not receive the order.
- Handle partial fills and cancel/replace operations explicitly.
- Consume real-time order events where available.
- Reconcile after startup, reconnects, submissions, fills and each session.

## Reconciliation and ledger

Maintain an append-only internal ledger containing:

- Trade proposals and evidence.
- Quant-verification results.
- Risk decisions and failed checks.
- Order intents and previews.
- Broker order IDs and client order IDs.
- Status changes, modifications and cancellations.
- Partial and complete fills.
- Fees and realized/unrealized P&L.
- Position and cash snapshots.
- Model, prompt, data and configuration versions.

At startup and throughout the session, compare internal records against Webull balances, positions, open orders, history and fills. Block new orders when discrepancies cannot be resolved safely.

## Kill switches

Implement three independent stops:

1. **Automatic software stop:** Triggered by loss limits, stale data, reconciliation failure, abnormal order volume or broker-event gaps.
2. **Operator stop:** Authenticated “cancel open orders and disable new orders” control outside the model process.
3. **Infrastructure stop:** Revoke production credentials or stop execution without disabling monitoring.

Send alerts through at least two channels. Include sanitized order and account context but never credentials.

## Options-specific agent output

Require additional fields for options:

```json
{
  "instrument_type": "OPTION",
  "underlying": "AAPL",
  "option_type": "CALL",
  "strike": 230,
  "expiration": "2026-11-20",
  "side": "BUY",
  "quantity_requested": 1,
  "max_loss": 350,
  "underlying_thesis": {},
  "volatility_thesis": {},
  "entry_iv": 0.31,
  "delta": 0.55,
  "gamma": 0.03,
  "theta_daily": -0.08,
  "vega": 0.17,
  "exit_conditions": {}
}
```

The risk engine must replace or verify every numeric value using the authoritative options feed. Reject stale, illiquid or unusually wide contracts and enforce limits on maximum loss, expiry concentration and portfolio Greeks.

Start with defined-risk positions only. Do not enable naked options, unrestricted shorting or margin during early production.

## Webull Singapore considerations

Webull Singapore documents an official OpenAPI with authentication, official SDKs, account access, market data, trading endpoints and real-time order events. API access requires an application plus an App Key and App Secret.

There is a capability inconsistency that must be resolved before automating options:

- Webull Singapore's Trading API overview has documented single-leg US options functionality.
- Some official Webull agent tooling has shown different regional product support.

Before implementing live options execution:

1. Apply for Webull Singapore OpenAPI and UAT access.
2. Confirm the account is approved for the relevant products.
3. Test option-instrument lookup in UAT.
4. Test an option-order preview without placing a live order.
5. Ask Webull Singapore whether SG production credentials support single-leg US option orders.
6. Confirm the required OPRA/non-display market-data entitlement.
7. Save Webull's response and the tested SDK/API versions.

If automated options are unavailable for the SG account, keep the options component in analysis/alert mode or evaluate another properly supported broker after legal and operational review.

## Security rules

- Store credentials only in a secrets manager or protected server environment.
- Never expose credentials to the browser, prompts, model context or logs.
- Do not commit `.env` files or keys to Git.
- Give research and model services no brokerage credentials.
- Restrict production credentials to the execution service.
- Use separate UAT and production accounts/configurations.
- Encrypt sensitive data in transit and at rest.
- Validate and sanitize all external documents before including them in model context.
- Treat news and webpages as untrusted content that may contain prompt injection.
- Maintain an allowlist of callable internal tools and strict JSON schemas.
- Use authenticated service-to-service communication.
- Log all sensitive actions with credential redaction.

## Suggested repository structure

```text
trading-agent/
  apps/
    dashboard/                 # Next.js monitoring and controls
  services/
    orchestrator/              # Trading-cycle scheduler
    data_ingestion/
    feature_service/
    research_agent/
    signal_service/
    quant_verifier/
    portfolio_service/
    risk_service/
    execution_service/
    reconciliation_service/
    alert_service/
  packages/
    schemas/                   # Pydantic contracts
    webull_adapter/
    backtest_engine/
    data_quality/
    observability/
  research/
    experiments/               # Cannot import production broker client
  migrations/
  tests/
    unit/
    integration/
    replay/
    failure_injection/
  infra/
```

Enforce a dependency rule: research and signal services can create trade candidates but cannot import the production Webull client. Only the execution service can access production credentials, and it accepts only risk-approved internal commands.

## Development roadmap

### Phase 0 — Specification

- Define universe, holding period and supported strategies.
- Decide whether shorting, leverage and overnight positions are permitted.
- Set initial capital and risk limits.
- Define benchmarks and promotion criteria.
- Create threat models and failure-mode tables.

### Phase 1 — Read-only Webull integration

- Apply for SG OpenAPI and UAT access.
- Install the official Python SDK.
- Retrieve accounts, balances, positions, open orders and order history.
- Retrieve stock/ETF snapshots and historical bars.
- Subscribe to order-status events.
- Implement normalized schemas and audit storage.
- Keep order submission disabled.

### Phase 2 — Data platform

- Ingest bars, quotes, security master and corporate actions.
- Add earnings, events, fundamentals and timestamped text sources.
- Create immutable dated raw-data partitions.
- Add data-quality checks and point-in-time feature generation.
- Version every dataset and feature set.

### Phase 3 — Backtesting

- Implement simple benchmarks.
- Build one transparent strategy.
- Add realistic costs, walk-forward testing and sensitivity tests.
- Create full trade and attribution reports.
- Preserve all experiments, including failed ones.

### Phase 4 — Autonomous shadow mode

- Allow the LLM to generate original trade ideas.
- Run verification, sizing and risk checks.
- Record intended orders without sending them.
- Compare expected and realized outcomes.
- Test prompt injection and malformed-output handling.

### Phase 5 — UAT and paper trading

- Route the production pipeline to UAT/paper execution.
- Test order previews, submissions, replacements and cancellations.
- Inject stale feeds, duplicated events, timeouts, partial fills, token expiry, restarts and database failures.
- Confirm every failure moves the system into a safe state.

### Phase 6 — Small live canary

- Use a small allowlist of liquid stocks/ETFs.
- Disable options, margin and shorting.
- Cap order value, total exposure and daily losses tightly.
- Monitor every order and reconcile continuously.
- Increase permissions only after stable live evidence.

### Phase 7 — Options

- Confirm SG API functionality and data entitlements.
- Acquire point-in-time historical options data.
- Implement volatility forecasts, contract selection and Greek limits.
- Start with simulated defined-risk trades.
- Move to tightly capped live options only after separate validation.

## Failure tests

Before live trading, verify safe behavior for:

- Missing or stale market data.
- Contradictory data sources.
- LLM hallucinated symbols or prices.
- Invalid or incomplete JSON.
- Prompt injection inside news or filings.
- Duplicate scheduler runs.
- Timeout after order submission.
- Partial fill followed by restart.
- Webull disconnect or expired token.
- Repeated or out-of-order broker events.
- Internal and broker position mismatch.
- Database outage during submission.
- Rapid market gap through a stop.
- Daily loss or drawdown-limit breach.
- Manual kill-switch activation.

## Immediate next steps

1. Obtain Webull Singapore OpenAPI UAT access.
2. Create a Python project using the official SDK.
3. Build read-only account, position, order and market-data adapters.
4. Design Pydantic schemas for evidence, proposals, risk decisions and broker orders.
5. Build the append-only ledger and reconciliation service.
6. Implement deterministic risk checks and kill switches.
7. Select a small liquid US stock/ETF universe.
8. Acquire point-in-time price, quote, corporate-action and event data.
9. Build one rule-based baseline and walk-forward backtester.
10. Add the LLM research/proposal layer.
11. Run shadow mode, then UAT/paper mode.
12. Launch a tightly capped live stock/ETF canary.
13. Confirm Webull SG options support before building live options execution.

## Official resources

- Webull Singapore OpenAPI: <https://developer.webull.com.sg/apis/docs/getting-started/>
- Webull Trading API overview: <https://developer.webull.com.sg/apis/docs/trade-api/overview>
- Webull Market Data API overview: <https://developer.webull.com.sg/apis/docs/market-data-api/overview>
- Webull OpenAPI Python SDK: <https://github.com/webull-inc/openapi-python-sdk>
- Webull OpenAPI MCP server: <https://github.com/webull-inc/webull-openapi-mcp>
- Webull agent tooling: <https://github.com/webull-inc/webull-openapi-skills>
- Webull Singapore Agentic Trading: <https://www.webull.com.sg/agentic>
- MAS digital-advisory guidelines: <https://www.mas.gov.sg/regulation/guidelines/guidelines-on-provision-of-digital-advisory-services>

## Final design statement

The desired system is an **autonomous LLM portfolio agent operating inside a deterministic trading envelope**:

- The LLM has freedom to research and invent trades.
- Independent code verifies facts and calculations.
- The portfolio service determines safe quantities.
- The risk engine automatically approves or rejects orders.
- The execution service manages Webull orders reliably.
- Reconciliation, monitoring and kill switches constrain failures.

This preserves autonomous decision-making without allowing a hallucination, malicious document, malformed tool call or software retry to create an unrestricted financial action.
