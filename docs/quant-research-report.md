# Quant research report: do classic signals beat buy-and-hold here?

Run on 2026-10-21. Code: `src/trading_agent/quant/`, command: `trading-agent quant-research`.

## Verdict

No. Neither of the two classic strategy families tested shows an edge that survives out-of-sample testing
and a correction for the number of variants tried.

| Strategy | Out-of-sample Sharpe | 95% interval | Deflated Sharpe | Buy-and-hold Sharpe |
|---|---|---|---|---|
| Cross-sectional long/short, 51 US stocks | **-0.22** | -1.33 to 0.74 | 0.05 | 1.46 |
| Trend following, 18 ETFs | **0.42** | -0.22 to 1.24 | 0.50 | 0.82 |
| Trend following, 25 ETFs (adds bonds, commodities, dollar, ex-US) | **0.57** | -0.08 to 1.38 | 0.65 | 0.83 |

A deflated Sharpe of about 0.95 or more means the result is unlikely to be luck after counting every variant
tried. None gets close. The one useful finding is that trend following **cuts risk, not return**: about a third of
the volatility and half the worst drawdown of buy-and-hold, at roughly a quarter of the return.

This agrees with two earlier studies in this repo: [Claude did not beat the momentum baseline](backtest-llm-vs-baseline-2024.md)
(and cost about $27 a year), and [insider buying showed no usable edge on large caps](insider-signal-study.md).
Three different approaches reach the same answer on the same kind of universe: liquid US large caps are about
as efficient as free daily data can reveal.

## What was tested, and why these

| Strategy | Source | Idea |
|---|---|---|
| Momentum | Jegadeesh & Titman (1993) | Winners over 3-12 months keep winning; the latest month is skipped |
| Short-term reversal | Jegadeesh (1990) | Last week's losers bounce |
| Low volatility | Ang et al. (2006), Frazzini & Pedersen (2014) | Calm stocks earn more per unit of risk |
| Trend following | Moskowitz, Ooi & Pedersen (2012) | An asset's own past return predicts its next month; size each asset to a common volatility |

These are the best-documented anomalies. They are also public, so the expectation going in was a weak or decayed
edge. The aim was to measure that honestly, not to find a winner.

## Method

- **No lookahead.** Every signal uses data up to day t and earns the return from t to t+1. A test changes the
  future of a price series and asserts the past signals do not move.
- **Walk-forward selection.** Settings are chosen on 2 years of data and scored only on the next 6 months, then the
  window rolls. Only the out-of-sample stretches are joined into the reported series.
- **Costs.** 10 bps per unit of turnover on the stock book, 5 bps on the ETF book, and 50 bps a year borrow on shorts.
  These are assumptions, not measurements.
- **Luck correction.** The deflated Sharpe ratio (Bailey & Lopez de Prado) asks whether the best result is
  distinguishable from the best of N skill-free strategies, accounting for skew, kurtosis and sample length.
  A block bootstrap gives a 95% interval for the Sharpe.
- **Trial counting.** The cross-sectional grid has 24 settings. The ETF trend grid has 6 (three lookbacks, long-short
  or long-only), run on two asset sets. The deflated figures for trend count all earlier work, so the hurdle
  rises with every look: 30 trials for the first ETF set, 36 for the second.
- **Pre-registration.** The seven added ETFs (IEF, LQD, DBC, UUP, EFA, EEM, VNQ) were written down before the wider run.
  Nothing was tuned after seeing results. This is self-attested, since there is no external timestamp.

## Results

### Cross-sectional long/short, 51 US stocks, 2020-10 to 2026-10

Dollar-neutral: long the top 20-30% by score, short the bottom, rebalanced every 5-21 days. 7 out-of-sample folds, 882 days.

| | Out-of-sample | Equal-weight same stocks |
|---|---|---|
| Sharpe | -0.22 | 1.46 |
| CAGR | -3.9% | +31.6% |
| Max drawdown | -33.7% | -24.8% |
| Beta to the stock universe | -0.04 | |

Full-sample (not out-of-sample) Sharpe at default settings: momentum 0.34, reversal -0.46, low volatility -0.64,
composite -0.22. Walk-forward picked momentum in 5 of 7 folds, and the chosen setting lost money in 5 of 7 test windows.
The shortfall is not market exposure: beta is near zero.

### Trend following, ETFs, 2016-10 to 2026-10 (15 out-of-sample folds, 1,890 days)

| | 18 ETFs | 25 ETFs | Buy-and-hold (25 ETFs) |
|---|---|---|---|
| Sharpe | 0.42 | 0.57 | 0.83 |
| CAGR | +2.4% | +2.9% | +12.8% |
| Annual volatility | 5.9% | 5.3% | 16.1% |
| Max drawdown | -16.9% | -14.9% | -29.1% |
| Beta to the ETF mix | 0.21 | 0.21 | 1.0 |
| Deflated Sharpe | 0.50 | 0.65 | |

In-sample, long-only variants beat long-short ones (Sharpe 0.55-0.88 against 0.17-0.43 on the wider set), which says the
value is mostly in sitting out downturns and not in shorting them. The strategy lost money in the 2022-2023 test windows,
the rate-shock period in which trend following has historically done well.

## Why there is no edge here

- **The universe is the most efficient corner of the market.** Large caps and broad ETFs are the most-watched
  instruments, traded by firms with faster data and better execution.
- **The signals are published.** Returns to published anomalies are known to fall sharply after publication.
- **Buy-and-hold already collects a risk premium.** Over this decade it paid a Sharpe above 0.8. An active strategy must beat that after costs.
- **Daily bars cannot see what pays.** Anything dependent on speed, order flow or queue position is invisible at this resolution.

## Caveats

- The stock universe is today's list, so absolute returns are inflated by survivorship. The dollar-neutral book cancels
  most of it but not all, and buy-and-hold is flattered the most.
- 51 of the 52 listed stocks had full history; the recent listing was dropped to avoid biasing ranks.
- About 7.5 years of out-of-sample data and a mostly rising market limit what any Sharpe can prove.
- The ETF set is mostly equity funds with a few diversifiers, so it is less diversified than a managed-futures universe.
- Costs and borrow are assumed. No intraday slippage, no market impact, no taxes.
- The deflated Sharpe only counts the grids shown here, not every idea considered before they were written down.

## Reproduce

```bash
uv run trading-agent quant-research                          # cross-sectional, 6 years
uv run trading-agent quant-research --strategy trend         # trend, the repo's 18 ETFs, 10 years
uv run trading-agent quant-research --strategy trend --diversified   # adds the pre-registered 7 ETFs
uv run pytest tests/test_quant.py                            # 11 tests: statistics, no-lookahead, costs, noise rejection
```

Reports are written to `state/quant/` as JSON, including every fold's chosen setting and its in-sample and out-of-sample Sharpe.

## Next steps

1. Wire trend following in as a small risk-reducing sleeve, only if the goal is lower drawdown and not higher return.
2. Test a universe where retail-sized capital can still compete: small caps with real liquidity rules, which is a separate project.
3. Add a point-in-time constituent list, which removes the survivorship caveat and needs a paid or curated data source.
4. Keep counting trials. Any new variant must be added to the trial total before it is judged.
