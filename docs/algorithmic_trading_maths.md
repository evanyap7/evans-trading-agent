# Algorithmic Trading Mathematics Guide

This document establishes the quantitative mathematical framework powering **Evan's Trading Agent**, adapted from the [QuantInsti Algorithmic Trading Maths Curriculum](https://www.quantinsti.com/articles/algorithmic-trading-maths/).

---

## Architecture Overview

Mathematical concepts in this trading system are organized into five primary pillars:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   ALGORITHMIC TRADING MATHEMATICS                     │
├─────────────────────┬────────────────────┬─────────────────────────────┤
│ 1. Descriptive      │ 2. Probability &   │ 3. Linear Algebra           │
│    Statistics       │    Stochastics     │                             │
│ • Mean & Crossovers │ • Bernoulli Trials │ • Return & Weight Vectors   │
│ • Median & Outliers │ • Random Walk      │ • Covariance Matrix (Σ)     │
│ • Bucketed Mode     │ • Bayes' Updating  │ • Portfolio Variance (wᵀΣw) │
│ • Range, QD, MAD, σ │ • Monte Carlo Sim  │ • Matrix Rank, Trace, Det   │
├─────────────────────┴────────────────────┴─────────────────────────────┤
│ 4. Linear Regression & ML                │ 5. Calculus                 │
│ • OLS Slope (Trend Velocity)             │ • 1st Derivative (dp/dt)    │
│ • R² (Trend Conviction vs Noise)         │ • 2nd Derivative (d²p/dt²)  │
│ • Intercept & Regression Channels        │ • Discrete Integral (VWAP)  │
└──────────────────────────────────────────┴─────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM IMPLEMENTATION                           │
├────────────────────────────────┬───────────────────────────────────────┤
│ math_quant.py                  │ Pure mathematical functions & tests   │
│ features.py                    │ Deterministic evidence generation     │
│ agents.py                      │ Quantitative reasoning prompts        │
│ verifier.py & portfolio.py     │ Asymmetry, Kelly sizing, risk bounds  │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 1. Descriptive Statistics

### 1.1 Central Tendency
- **Mean & Moving Average Crossovers**:
  - Fast SMA (20-day) vs Slow SMA (50-day / 200-day).
  - Meeting point & crossover direction:
    - **Bullish Crossover**: Fast crosses above slow $\rightarrow$ upward trend momentum.
    - **Bearish Crossover**: Fast drops below slow $\rightarrow$ breakdown / downward momentum.
- **Median**:
  - Resilient to outlier wicks and single-day gap anomalies.
  - Used via `dist_median_20d_pct` to verify true central price support.
- **Mode & Frequency Clustering**:
  - Bucketed histogram modes identify high-volume price consolidation nodes (support/resistance accumulation).

### 1.2 Dispersion & Volatility
- **Range ($X_{\max} - X_{\min}$)**: Identifies range-bound boundaries and channel breakout thresholds (`range_20d_pct`).
- **Quartile Deviation (QD)**: $\frac{1}{2}(Q_3 - Q_1)$ measures spread of central 50% of prices, ignoring fat tails (`quartile_dev_20d_pct`).
- **Mean Absolute Deviation (MAD)**: $\frac{1}{n} \sum |x_i - \mu|$, linear penalty on deviation (`mad_20d_pct`).
- **Variance & Standard Deviation ($\sigma$)**:
  - Standard deviation measures price dispersion and sets baseline volatility collars.
  - Used in ATR stops (1–3 ATR), realized volatility tracking, and Bollinger Bands.

---

## 2. Probability Theory & Stochastic Processes

### 2.1 Bernoulli Trials & Expectancy
Trade setups are modeled as discrete Bernoulli trials:
- Binary outcomes: Win ($+b \times \text{risk}$) with probability $p$; Loss ($-1 \times \text{risk}$) with probability $1 - p$.
- Expectancy: $\mathbb{E}[R] = p \cdot \text{upside} - (1 - p) \cdot \text{downside}$.
- Break-even win rate: $p_{\text{be}} = \frac{\text{down}}{\text{up} + \text{down}}$.
- Sizing via Fractional Kelly: $f^* = p - \frac{1 - p}{b}$.

### 2.2 Random Walk Hypothesis
Price series exhibit significant stochastic noise ($P_t = P_{t-1} + \epsilon_t$). Consequently, the agent does not attempt to predict every tick; it extracts statistical edge by enforcing minimum reward-to-risk ($\ge 1.5:1$, ideally $2:1$ to $3:1+$) and positive probability edge ($p - p_{\text{be}} \ge 0.08$).

### 2.3 Bayesian Evidence Updating
- Prior probability $P(\text{Win})$: Base historical win rate ($\approx 45\%$).
- Posterior probability:
  $$P(\text{Win} \mid \text{Evidence}) = \frac{P(\text{Evidence} \mid \text{Win}) P(\text{Win})}{P(\text{Evidence})}$$
- Updates confidence based on multi-factor evidence confluence (trend alignment, regression $R^2$, positive acceleration, and volume).

### 2.4 Monte Carlo Simulations
- Simulates 2,000+ stochastic equity trajectories over a forward horizon of trades.
- Calculates empirical **Maximum Drawdown Distribution**, **95% Value-at-Risk (VaR)**, and **Probability of Ruin**.

---

## 3. Linear Algebra & Portfolio Matrix Math

- **Multi-Asset Vectors**: Return vector $\mathbf{r}$ and portfolio weight vector $\mathbf{w}$.
- **Covariance Matrix ($\mathbf{\Sigma}$)**: Measures pairwise asset covariances.
- **Portfolio Variance**:
  $$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$
- **Diversification**: Selecting trade ideas across distinct, uncorrelated sectors reduces off-diagonal covariance terms, keeping portfolio variance within safe limits.
- **Matrix Properties**: Verification of matrix rank, trace, and condition number ensures non-singular portfolio optimization.

---

## 4. Linear Regression & Predictive Modeling

- **Ordinary Least Squares (OLS)**: $Y = mX + b$.
- **Slope ($m$)**: Annualized rate of price change (`linreg_slope_20d_pct`).
- **Goodness-of-Fit ($R^2$)**:
  - $R^2 > 0.70$: Clean, strong trend with high directional conviction.
  - $R^2 < 0.30$: Choppy, random-walk market conditions where trend-following setups are de-prioritized.
- **Dynamic Regression Channels**: Deviation from the regression line provides mean-reversion signals.

---

## 5. Continuous Mathematics & Calculus

- **1st Derivative (Velocity / Instantaneous Momentum)**:
  $$\text{Velocity} = \frac{dp}{dt}$$
  Measures instantaneous rate of price change per day (`price_velocity_20d`).
- **2nd Derivative (Acceleration / Curvature)**:
  $$\text{Acceleration} = \frac{d^2p}{dt^2}$$
  - Positive acceleration ($\frac{d^2p}{dt^2} > 0$): Momentum expansion.
  - Negative acceleration ($\frac{d^2p}{dt^2} < 0$): Momentum deceleration / exhaustion (signals potential top or bottom before price reverses).
- **Integral Calculus**:
  - **Volume-Weighted Average Price (VWAP)**:
    $$\text{VWAP} = \frac{\int P(t) V(t) \, dt}{\int V(t) \, dt}$$

---

## Usage in Code

All mathematical algorithms are implemented in [`src/trading_agent/math_quant.py`](file:///Users/evanyap7/trading-agent/src/trading_agent/math_quant.py):

```python
from trading_agent.math_quant import (
    moving_average_crossover,
    dispersion_metrics,
    bayes_posterior_probability,
    monte_carlo_trade_simulation,
    linear_regression,
    momentum_calculus,
    portfolio_variance,
)

# 1. Moving average crossover
crossover = moving_average_crossover(closes, fast_period=20, slow_period=50)

# 2. Linear regression & R^2
reg = linear_regression(closes[-20:])
print(f"Slope: {reg.slope_annualized_pct}%, R^2: {reg.r_squared}")

# 3. Momentum Calculus (1st and 2nd derivatives)
mom = momentum_calculus(closes, window=20)
print(f"Velocity: {mom.velocity}, Acceleration: {mom.acceleration}")

# 4. Bayesian Probability Update
posterior_win = bayes_posterior_probability(prior_prob=0.45, likelihood_true=0.70, likelihood_false=0.30)

# 5. Monte Carlo Drawdown Simulation
sim = monte_carlo_trade_simulation(win_rate=0.55, reward_risk=2.0, horizon_trades=50)
print(f"Expected Max Drawdown: {sim.expected_max_drawdown_pct}%, 95% VaR: {sim.var_95_pct}%")
```
