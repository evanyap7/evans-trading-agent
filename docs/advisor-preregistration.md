# Advisor prediction log: pre-registration

Written 2026-10-31, before any call was logged. Self-attested; the git commit timestamp is the only external stamp.
Author: Evan Yap.

## Question

Does an LLM that reads news and SEC filings on my holdings predict which of them will beat SPY, better than the same
model without the news?

## What is logged

Each advisor run records, per call: date, arm, symbol, direction (`up` = beats SPY, `down` = trails it),
horizon (5 or 20 trading days), stated probability, the stock price and SPY price at call time, and the model.
Two arms, same model and same holdings:

- `news`: sees yfinance headlines, EDGAR 8-K/10-Q/10-K filings, earnings dates and my one-line theses.
- `blind`: sees only the holdings and prices. This control shows whether the news adds anything.

The log is the hash-chained `state/advisor/ledger.db`; the same tamper check as the trading ledger applies.

## Scoring (fixed now)

- Outcome: the call is correct if the stock's return over the horizon, from the logged price, beats SPY's return from
  the logged SPY price (`up`) or trails it (`down`).
- Metrics: hit rate with a 95% binomial interval, and Brier score `(p - correct)^2`. A coin flip at p = 0.5 scores 0.25.
- Independence: calls on the same symbol inside one horizon overlap, so only the first call per symbol per horizon
  window counts toward n.

## Pass bar (one test, no tuning)

The `news` arm passes only if all hold after at least 300 independent scored calls:

1. Hit rate above 50% with the lower end of the 95% interval above 50% (at n = 300 that needs roughly 55.5% or better).
2. Brier score under 0.25.
3. A paired comparison (bootstrap over calls) shows `news` beats `blind` on Brier, with the 95% interval excluding zero.

With about 5 to 10 independent calls a week that is several months of data. No result is reported before n = 300.

## Rules

- No prompt, model, horizon or threshold changes after the first logged call. A change restarts the log under a new
  version of this file.
- One test, so no trial-count correction is needed. If I add a second hypothesis, it counts against this one.
- The advisor never trades. Calls do not become advice until the bar is passed, and even then through the existing
  verifier, sizing and risk limits.
- A fail is published the same way as a pass.
- The new-ideas section (S&P 500 gap screen, separate model call, logged as `advisor_idea`) is also outside this test.
- Scoring is `uv run trading-agent advise-score`, which implements the rules above and prints no verdict below n = 300.

## Known weaknesses

- The model's knowledge may overlap the sample window. Forward-only logging reduces this but does not remove it for
  events the model already knew about.
- Holdings are not a random draw of stocks, so a pass would not generalise to the whole market.
- yfinance prices at call time are the last close or a delayed quote, not a fill.

## Design freeze

The prompt, the data sources (Yahoo RSS, EDGAR filings, EDGAR XBRL fundamentals incl. annual revenue growth, gross
margin and free cash flow, the text of 8-K item 2.02 earnings releases, the last reported EPS surprise, earnings dates,
theses) and the `max_tokens` budget were frozen on 2026-10-31 before the first logged call. The `news` arm sees all of
these; `blind` sees none. Portfolio risk statistics and 12-1 momentum are given only to the separate advice call, never
to either logged arm, so they cannot confound the news-vs-blind comparison. Any later change to them is a new version
of this experiment and restarts the count.

## Trade advice is outside this test

On 2026-10-31 I added buy/sell advice at my request. It is a separate third model call, so the prompts of the two
logged arms are unchanged. Advice is logged (`advisor_advice`) but is not scored by this pre-registration and does not
count toward the pass bar. It is not validated, and each digest says so.
