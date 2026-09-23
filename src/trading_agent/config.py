"""Loads the human-controlled configuration.

Risk limits are frozen Pydantic models built from a YAML file under version
control. Nothing the LLM produces flows into these objects.
"""

from __future__ import annotations

import hashlib
import os
from datetime import date
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LiveTradingLimits(Frozen):
    enabled: bool = False
    allowed_instruments: tuple[str, ...] = ("EQUITY", "ETF")
    permit_shorting: bool = False
    permit_margin: bool = False
    permit_options: bool = False
    # Lets the agent sell positions it did not open (e.g. bought by hand). Off = it may only close its own trades.
    agent_may_close_manual_positions: bool = False

    @model_validator(mode="after")
    def _unsupported_permissions_off(self) -> "LiveTradingLimits":
        # The code has no short, margin or options handling; turning these on would silently do nothing safe.
        if self.permit_shorting or self.permit_margin or self.permit_options:
            raise ValueError("permit_shorting/permit_margin/permit_options are not supported and must stay false")
        if not set(self.allowed_instruments) <= {"EQUITY", "ETF"}:
            raise ValueError("allowed_instruments may only contain EQUITY and ETF")
        return self


class AccountLimits(Frozen):
    # Hard ceilings below are code-level sanity bounds: a typo in the YAML fails loudly instead of trading.
    max_position_pct: float = Field(gt=0, le=100)
    max_sector_pct: float = Field(gt=0, le=100)
    max_total_exposure_pct: float = Field(gt=0, le=100)  # cash account, no margin
    max_risk_per_trade_pct: float = Field(gt=0, le=5)
    max_portfolio_risk_pct: float = Field(gt=0, le=25)
    unknown_stop_risk_pct: float = Field(gt=0, le=100)
    max_daily_loss_pct: float = Field(gt=0, le=10)
    max_drawdown_pct: float = Field(gt=0, le=25)
    max_new_trades_per_day: int = Field(ge=0, le=10)
    max_order_value_usd: float = Field(gt=0)
    max_adv_participation_pct: float = Field(gt=0, le=5)
    daily_profit_target_usd: float = Field(default=10.0, ge=0)  # reporting only; never drives exits


class SignalLimits(Frozen):
    min_confidence: float = Field(ge=0, le=1)
    min_net_edge_pct: float = Field(ge=0)
    safety_buffer_pct: float = Field(ge=0)
    min_reward_risk: float = Field(ge=1)
    min_stop_atr: float = Field(gt=0)
    max_stop_atr: float = Field(gt=0)


class ExecutionLimits(Frozen):
    max_spread_pct: float = Field(gt=0, le=2)
    max_quote_age_seconds: float = Field(gt=0, le=900)
    max_bar_age_days: int = Field(ge=1, le=10)
    max_account_age_seconds: float = Field(gt=0, le=900)
    price_collar_pct: float = Field(gt=0, le=3)
    slippage_bps: float = Field(ge=0)
    fee_per_order_usd: float = Field(ge=0)
    broker_side_stops: bool = True
    min_price_usd: float = Field(default=5.0, ge=0)                  # no new entries in penny stocks
    min_avg_dollar_volume_usd: float = Field(default=20_000_000, ge=0)  # 20-day average, for new entries
    max_exit_attempts_per_day: int = Field(default=3, ge=1, le=10)
    # Ratcheting stop, in multiples of the trade's initial risk R = entry - initial stop.
    trailing_stops: bool = True
    trail_breakeven_r: float = Field(default=1.0, gt=0)   # at +this R, stop moves to entry
    trail_start_r: float = Field(default=2.0, gt=0)       # from +this R, stop trails price...
    trail_distance_r: float = Field(default=1.5, gt=0)    # ...by this many R
    trail_min_step_r: float = Field(default=0.25, gt=0)   # smaller moves are not worth a broker order

    @model_validator(mode="after")
    def _trail_locks_in_profit(self) -> "ExecutionLimits":
        # A trail wider than its start point would sit below entry, i.e. a stop that loosens the breakeven.
        if self.trail_distance_r > self.trail_start_r:
            raise ValueError("trail_distance_r must not exceed trail_start_r")
        return self


class EventLimits(Frozen):
    earnings_blackout_days: int
    require_earnings_data_for_stocks: bool = True


class RiskLimits(Frozen):
    live_trading: LiveTradingLimits
    account: AccountLimits
    signal: SignalLimits
    execution: ExecutionLimits
    events: EventLimits
    source_sha256: str = Field(default="", description="Hash of the YAML this was loaded from")


class Security(Frozen):
    symbol: str
    type: str
    sector: str


class Universe(Frozen):
    symbols: dict[str, Security]
    regime_benchmark: str

    def get(self, symbol: str) -> Security | None:
        return self.symbols.get(symbol)


class Events(Frozen):
    earnings: dict[str, date] = Field(default_factory=dict)


class TradingMode(str, Enum):
    SHADOW = "shadow"
    BROKER = "broker"


class Settings(Frozen):
    """Process settings from the environment. Credentials are read only by the broker adapter."""

    trading_mode: TradingMode
    webull_environment: str
    webull_region: str
    webull_account_id: str
    state_dir: Path
    llm_model: str = "claude-opus-5-5"
    llm_model_fast: str = "claude-haiku-4-5"
    continuous_trading: bool = False

    @property
    def is_production(self) -> bool:
        return self.webull_environment == "prod"


def _read_yaml(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return yaml.safe_load(raw) or {}, hashlib.sha256(raw).hexdigest()


def load_risk_limits(path: Path | None = None) -> RiskLimits:
    data, digest = _read_yaml(path or CONFIG_DIR / "risk_limits.yaml")
    return RiskLimits(**data, source_sha256=digest)


def load_universe(path: Path | None = None) -> Universe:
    data, _ = _read_yaml(path or CONFIG_DIR / "universe.yaml")
    symbols = {s: Security(symbol=s, **v) for s, v in data["symbols"].items()}
    return Universe(symbols=symbols, regime_benchmark=data["regime_benchmark"])


def load_events(path: Path | None = None) -> Events:
    data, _ = _read_yaml(path or CONFIG_DIR / "events.yaml")
    return Events(earnings=data.get("earnings") or {})


def _project_path(value: str | os.PathLike) -> Path:
    """Relative paths are anchored to the project, not the caller's cwd, so every process sees the same state/KILL."""
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_settings() -> Settings:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if token_dir := os.environ.get("WEBULL_TOKEN_DIR"):
        os.environ["WEBULL_TOKEN_DIR"] = str(_project_path(token_dir))
    env = os.environ.get("WEBULL_ENVIRONMENT", "uat").lower()
    if env not in {"uat", "prod"}:
        raise ValueError("WEBULL_ENVIRONMENT must be 'uat' or 'prod'")
    return Settings(
        trading_mode=TradingMode(os.environ.get("TRADING_MODE", "shadow").lower()),
        webull_environment=env,
        webull_region=os.environ.get("WEBULL_REGION", "sg").lower(),
        webull_account_id=os.environ.get("WEBULL_ACCOUNT_ID", ""),
        state_dir=_project_path(os.environ.get("STATE_DIR", "state")),
        llm_model=os.environ.get("LLM_MODEL", "claude-opus-5-5"),
        llm_model_fast=os.environ.get("LLM_MODEL_FAST", "claude-haiku-4-5"),
        continuous_trading=os.environ.get("CONTINUOUS_TRADING", "true").lower() in ("1", "true", "yes"),
    )
