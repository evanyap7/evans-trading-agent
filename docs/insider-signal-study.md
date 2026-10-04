# Insider buying signal (SEC Form 4): study and backtest

Data: SEC quarterly Form 3/4/5 data sets (free). Open-market purchases only (code `P`), dated by FILING date,
so nothing is used before the public could see it. Code: `src/trading_agent/insider.py`,
`scripts/insider_study.py`, `backtest_harness.py run --insider-min-usd`.

## Event study (top-100 US stocks, 2022-01 to 2026-09)

Event: trailing-90-day filed buys reach $100k. Entry at the next open. Excess return is stock minus SPY.
The control is the same stocks and spacing with no insider condition.

| Hold | Insider mean excess | Control mean excess | Insider edge |
|---|---|---|---|
| 5d | +0.08% | +0.18% | none |
| 20d | +1.39% (n=518) | +0.61% | +0.8% |
| 60d | +3.63% (n=518) | +1.84% | +1.8% |

The control is positive because the universe is today's survivors, so the raw numbers overstate any edge.
Hit rate is about 50% for both, so the gain comes from a few big winners. At 60 days the insider group lost to
SPY in 2024 (-1.9%) and led in the other four years. Events cluster around market dips, so the t-stats
(2.5 at 20d, 3.4 at 60d) are optimistic.

## Backtest: baseline momentum vs baseline plus insider longs (2022-01 to 2026-09, $1,500, no sweep)

| Variant | Return | Sharpe | Max DD | Win rate | Trades |
|---|---|---|---|---|---|
| Baseline | +66.8% | 0.98 | -11.9% | 42.7% | 536 |
| + insider longs, $100k | +64.2% | 1.00 | -12.2% | 42.1% | 535 |
| + insider longs, $250k | +59.6% | 0.94 | -12.4% | 42.0% | 517 |
| SPY buy-and-hold | +70.1% | 0.74 | -24.5% | | |

## Verdict

No usable edge on a large-cap universe. The event study shows a small raw effect that is mostly survivorship
and noise, and it vanishes inside the agent's sizing, stops and risk limits. Insider buys are rare and
concentrated in small caps, which the liquidity floor (`min_avg_dollar_volume_usd`) excludes. The flag stays
off by default. Testing small caps needs a different universe and liquidity rules, which is a separate project.

Not tested: sells and cluster-of-sellers signals, 13F data, congressional trades, officer-role weighting.
Trading on non-public information is illegal. This uses only public filings.
