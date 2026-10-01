---
name: algorithmic-trading-maths
description: Master quantitative mathematics for algorithmic trading, including descriptive statistics, probability theory, linear algebra, linear regression, continuous calculus derivatives, and Monte Carlo risk modeling.
---

# Algorithmic Trading Mathematics: Core Quantitative Foundations

This skill equips the trading agent with rigorous mathematical principles and operational formulas for algorithmic trading, portfolio construction, risk management, and signal verification, modeled after the QuantInsti curriculum.

---

## 1. Descriptive Statistics

Descriptive statistics summarize and characterize market data distributions across two primary dimensions: **Central Tendency** and **Dispersion**.

### 1.1 Measures of Central Tendency

#### Arithmetic Mean ($\mu$) & Moving Averages
$$\mu = \frac{1}{n} \sum_{i=1}^{n} x_i$$
- **Algorithmic Application**: Simple Moving Average (SMA).
  - **Faster SMA** (e.g., 20-day): Captures short-term price momentum.
  - **Slower SMA** (e.g., 50-day or 200-day): Defines underlying regime/structural trend.
  - **Crossover Meeting Points**:
    - **Bullish Golden Cross**: Faster SMA crosses above Slower SMA from below $\rightarrow$ initiates/confirms long trend.
    - **Bearish Death Cross**: Faster SMA drops below Slower SMA from above $\rightarrow$ initiates/confirms short breakdown.
  - **Limitation**: Highly sensitive to extreme single-bar price outliers (flash spikes, earnings gaps).

#### Median
- Middle value of sorted series: $x_{(n+1)/2}$ (or mean of middle two values if $n$ is even).
- **Algorithmic Application**: Outlier-resistant price baseline. When price gaps violently due to an idiosyncratic event, the median remains stable while the mean shifts. If `dist_median_20d_pct` diverges heavily from `dist_sma20_pct`, an outlier spike is present.

#### Mode & Price Clustering
- The most frequently occurring value in a dataset.
- Continuous stock prices rarely repeat exact floating-point decimals. In quantitative trading, **bucketed mode** or **volume profile point-of-control (POC)** represents the price level where trading activity clusters most heavily.
- **Bimodal Distributions**: Signal transition or balance between two distinct valuation regimes (e.g., accumulation channel vs distribution zone).

---

### 1.2 Measures of Dispersion & Volatility

Dispersion quantifies how widely observations deviate from the central tendency—the quantitative definition of market risk and volatility.

#### Range
$$\text{Range} = X_{\max} - X_{\min}$$
- **Algorithmic Application**: Range-bound trading strategies, Donchian channels, support/resistance bands, and breakout volatility expansion filters.

#### Quartile Deviation (QD) / Semi-Interquartile Range
$$\text{QD} = \frac{1}{2} (Q_3 - Q_1)$$
- Where $Q_1$ is the 25th percentile and $Q_3$ is the 75th percentile (middle 50% spread).
- **Algorithmic Application**: Semi-interquartile range measures core volatility while ignoring the top and bottom 25% extremes. Superior to standard deviation when dealing with fat-tailed or jump-diffusion distributions.

#### Mean Absolute Deviation (MAD)
$$\text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \mu|$$
- **Algorithmic Application**: Linear penalty on price distance. Does not square deviations, making it less prone to distortion from rare flash crashes than variance.

#### Variance ($\sigma^2$) and Standard Deviation ($\sigma$)
$$\sigma^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \mu)^2, \quad \sigma = \sqrt{\sigma^2}$$
- **Algorithmic Application**:
  - **Bollinger Bands**: $\mu \pm k\sigma$ (typically $k=2$).
  - **Dynamic Volatility Stops**: Setting stop distances proportional to $\sigma$ or ATR (1–3 ATR / $\sigma$).
  - **Parametric Value-at-Risk (VaR)**: $\text{VaR}_\alpha = -(\mu - z_\alpha \sigma)$.

---

### 1.3 Data Visualisation in Algorithmic Trading
- **Histograms**: Essential for evaluating return distributions, assessing fat tails (kurtosis), and identifying skewness.
- **Line Charts**: Tracking multi-timeframe moving average channels, price vs regression trends, and stop-loss trailing trajectories.
- **Bar Charts**: Comparing categorical metrics (e.g., sector returns, volume bars, monthly performance).
- **Pie Charts**: Visualizing portfolio capital allocation across asset classes and sector risk weights.

---

## 2. Probability Theory & Stochastic Processes

### 2.1 Discrete vs. Continuous Probability & Bernoulli Trials
- Discrete probability mass function (PMF) vs continuous probability density function (PDF).
- A binary trade outcome (hit take-profit vs hit stop-loss) can be formalized as a **Bernoulli trial**:
$$P(X = x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}$$
- Expected return: $\mathbb{E}[R] = p \cdot \text{upside} - (1 - p) \cdot \text{downside}$.
- Zero-expectancy break-even win rate: $p_{\text{be}} = \frac{\text{downside}}{\text{upside} + \text{downside}} = \frac{1}{1 + b}$ where $b = \text{reward}/\text{risk}$.

### 2.2 Random Walk Hypothesis
$$P_t = P_{t-1} + \epsilon_t, \quad \epsilon_t \sim \text{i.i.d. } \mathcal{N}(0, \sigma^2)$$
- Stock prices exhibit high degrees of random walk behavior. Individual price steps have high noise.
- **Quantitative Implication**: A trading edge cannot rely on predicting every bar; edge comes from structural market inefficiencies, positive asymmetry ($R:R \ge 2:1$), and disciplined risk management.

### 2.3 Bayes' Theorem for Evidence Updating
$$P(\text{Win} \mid \text{Evidence}) = \frac{P(\text{Evidence} \mid \text{Win}) \cdot P(\text{Win})}{P(\text{Evidence})}$$
- **Application in LLM & Quantitative Sizing**:
  - Prior probability $P(\text{Win})$: The baseline historical win rate (e.g., 45%).
  - New evidence arrives: Bullish moving average crossover, high $R^2$ linear regression slope, positive 2nd derivative (acceleration), and macro tailwinds.
  - Calculate posterior probability to update trade `confidence` objectively before submitting to the verifier and Kelly sizer.

### 2.4 Monte Carlo Simulations
- Simulates thousands of stochastic paths using random sampling of historical returns or Bernoulli trade sequences.
- **Application**:
  - Stress-tests strategy robustness without overfitting to a single realized history.
  - Generates empirical distributions of **Maximum Drawdown (MDD)**, **95% Value-at-Risk (VaR)**, and **Probability of Ruin**.
  - Assesses whether a drawdown is within normal stochastic variance or indicates strategy failure.

---

## 3. Linear Algebra & Portfolio Matrix Operations

Multi-asset quantitative trading relies on vector and matrix representations:

### 3.1 Vectors and Matrices
- Asset return vector: $\mathbf{r} = [r_1, r_2, \dots, r_N]^T$.
- Portfolio weight vector: $\mathbf{w} = [w_1, w_2, \dots, w_N]^T$, where $\sum w_i \le 1$.
- Covariance matrix: $\mathbf{\Sigma} \in \mathbb{R}^{N \times N}$, where $\Sigma_{ij} = \text{Cov}(r_i, r_j)$.

### 3.2 Portfolio Expected Return and Variance
$$\mu_p = \mathbf{w}^T \mathbf{r}$$
$$\sigma_p^2 = \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$$
$$\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$$
- **Algorithmic Application**: Sizing and screening candidates across uncorrelated sectors reduces portfolio variance $\sigma_p^2$ through negative or zero covariance off-diagonal terms ($\Sigma_{ij} \approx 0$).

### 3.3 Core Matrix Operations
- **Rank**: Determines the number of independent factors driving portfolio returns.
- **Trace**: $\text{Tr}(\mathbf{\Sigma}) = \sum \sigma_i^2$, total system variance.
- **Determinant**: Measures matrix non-singularity. $\det(\mathbf{\Sigma}) = 0$ implies perfect multicollinearity.
- **Inverse ($\mathbf{\Sigma}^{-1}$)**: Required for Markowitz mean-variance portfolio optimization $\mathbf{w}^* \propto \mathbf{\Sigma}^{-1} \boldsymbol{\mu}$.
- **Eigenvalues & Eigenvectors**: Used in Principal Component Analysis (PCA) to extract statistical arbitrage factors and market modes.

---

## 4. Linear Regression & Predictive Modeling

### 4.1 Ordinary Least Squares (OLS)
$$Y = mX + b$$
- Slope $m$:
$$m = \frac{\sum (X_i - \bar{X})(Y_i - \bar{Y})}{\sum (X_i - \bar{X})^2} = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}$$
- Intercept $b$:
$$b = \bar{Y} - m\bar{X}$$
- Coefficient of Determination ($R^2$):
$$R^2 = \frac{[\text{Cov}(X, Y)]^2}{\text{Var}(X) \cdot \text{Var}(Y)}$$

### 4.2 Trading Applications
- **Slope as Trend Velocity**: Normalized annualized slope (`linreg_slope_20d_pct`) measures the speed and direction of price movement.
- **$R^2$ as Trend Conviction**:
  - $R^2 \ge 0.70$: High-conviction, low-noise directional trend (ideal for trend following).
  - $R^2 \le 0.30$: Noisy chop or sideways mean-reverting regime.
- **Dynamic Regression Channels**: Plotting $Y \pm 2 \cdot \text{RMSE}$ creates linear regression channels for mean-reversion trading when price touches channel boundaries.

---

## 5. Continuous Mathematics & Calculus

Calculus analyzes continuous rates of change and cumulative quantities in financial time series.

### 5.1 Differential Calculus: Rates of Change & Derivatives
- **1st Derivative (Velocity / Momentum)**:
$$\text{Velocity} = \frac{dp}{dt} \approx \lim_{\Delta t \to 0} \frac{P(t + \Delta t) - P(t)}{\Delta t}$$
Measures the instantaneous directional speed of price change.
- **2nd Derivative (Acceleration / Curvature)**:
$$\text{Acceleration} = \frac{d^2p}{dt^2} = \frac{d}{dt}\left(\frac{dp}{dt}\right)$$
  - $\frac{d^2p}{dt^2} > 0$ with positive velocity: Momentum is expanding and accelerating (high probability trend continuation).
  - $\frac{d^2p}{dt^2} < 0$ with positive velocity: Upward momentum is decelerating—signals trend exhaustion or impending reversal before price officially peaks.
- **Options Greeks**:
  - Delta: $\Delta = \frac{\partial V}{\partial S}$ (1st derivative of option price with respect to underlying stock price).
  - Gamma: $\Gamma = \frac{\partial^2 V}{\partial S^2}$ (2nd derivative, rate of change of Delta).
  - Theta: $\Theta = \frac{\partial V}{\partial t}$ (decay rate with respect to time).

### 5.2 Integral Calculus: Cumulative Quantities
- **Volume-Weighted Average Price (VWAP)**:
$$\text{VWAP} = \frac{\int_0^T P(t) \cdot V(t) \, dt}{\int_0^T V(t) \, dt} \approx \frac{\sum P_i V_i}{\sum V_i}$$
Provides the true volume-weighted benchmark for institutional execution quality.
- **Value at Risk (VaR)** as Area Under Density:
$$\text{VaR}_\alpha \text{ satisfies } \int_{-\infty}^{-\text{VaR}_\alpha} f(r) \, dr = \alpha$$

---

## 6. How the System Applies This Mathematics

In `Evan's Trading Agent`:
1. **Feature Engine (`src/trading_agent/features.py`)**: Computes `sma20_above_sma50`, `median_20d`, `dist_median_20d_pct`, `range_20d_pct`, `quartile_dev_20d_pct`, `mad_20d_pct`, `linreg_slope_20d_pct`, `linreg_r2_20d`, `price_velocity_20d`, and `price_acceleration_20d`.
2. **Deterministic Quant Module (`src/trading_agent/math_quant.py`)**: Provides pure-math functions for all the above formulas.
3. **LLM Strategist (`src/trading_agent/agents.py`)**: Leverages Bayesian updating, trend $R^2$, and momentum calculus derivatives to shortlist high-conviction trades with minimal covariance.
4. **Quant Verifier (`src/trading_agent/verifier.py`)**: Deterministically enforces break-even win rate $p_{\text{be}} = \frac{\text{down}}{\text{up} + \text{down}}$, minimum probability edge, and net edge after slippage/fees.
5. **Portfolio Sizer (`src/trading_agent/portfolio.py`)**: Uses fractional Kelly sizing $f^* = p - \frac{1-p}{b}$ derived from Bernoulli trial expectancy.
