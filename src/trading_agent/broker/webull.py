"""Webull OpenAPI adapter (official `webull-openapi-python-sdk`, order_v3 + account_v2).

Only this module touches Webull credentials. Field names come from the SDK
samples and Webull's own MCP server; `trading-agent probe` dumps raw responses
so the mappings can be checked against a real SG account.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from ..schemas import AccountState, Bar, BrokerOrder, BrokerOrderStatus, Position, Quote, utcnow
from .base import BrokerError, OrderRequest, PreviewResult

# Test-environment hosts (the SDK's endpoints.json only ships production hosts).
UAT_ENDPOINTS = {
    "sg": {"api": "sg-api.uat.webullbroker.com", "quotes-api": "data-api.uat.webullbroker.com",
           "events-api": "sg-events-api.uat.webullbroker.com"},
}

STATUS_MAP = {
    "SUBMITTED": BrokerOrderStatus.WORKING,
    "PENDING": BrokerOrderStatus.WORKING,
    "WORKING": BrokerOrderStatus.WORKING,
    "PARTIAL_FILLED": BrokerOrderStatus.PARTIALLY_FILLED,
    "PARTIALLY_FILLED": BrokerOrderStatus.PARTIALLY_FILLED,
    "FILLED": BrokerOrderStatus.FILLED,
    "CANCELLED": BrokerOrderStatus.CANCELLED,
    "CANCELED": BrokerOrderStatus.CANCELLED,
    "FAILED": BrokerOrderStatus.REJECTED,
    "REJECTED": BrokerOrderStatus.REJECTED,
    "EXPIRED": BrokerOrderStatus.CANCELLED,
}


def _f(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _ts(v: Any) -> datetime | None:
    if v in (None, ""):
        return None
    if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
        n = int(v)
        return datetime.fromtimestamp(n / 1000 if n > 1e11 else n, tz=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _json(resp: Any) -> Any:
    if getattr(resp, "status_code", 200) != 200:
        raise BrokerError(f"HTTP {resp.status_code}: {getattr(resp, 'text', '')[:300]}")
    return resp.json() if hasattr(resp, "json") else resp


def _unwrap_list(data: Any, *keys: str) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in keys:
            if isinstance(data.get(k), list):
                return data[k]
    return []


class WebullBroker:
    name = "webull"
    _cached_account: AccountState | None = None
    _cached_account_time: float = 0.0

    def __init__(self, account_id: str, region: str = "sg", environment: str = "uat"):
        from webull.core.client import ApiClient
        from webull.core.common.api_type import DEFAULT, EVENTS, QUOTES
        from webull.data.data_client import DataClient
        from webull.trade.trade_client import TradeClient

        key, secret = os.environ.get("WEBULL_APP_KEY"), os.environ.get("WEBULL_APP_SECRET")
        if not key or not secret:
            raise BrokerError("WEBULL_APP_KEY / WEBULL_APP_SECRET are not set")
        self.account_id = account_id
        self.environment = environment
        client = ApiClient(key, secret, region, connect_timeout=5, timeout=15)
        if token_dir := os.environ.get("WEBULL_TOKEN_DIR"):
            client.set_token_dir(token_dir)
        if environment == "uat":
            hosts = UAT_ENDPOINTS.get(region)
            if not hosts:
                raise BrokerError(f"no UAT endpoints known for region {region}")
            for kind, api_type in (("api", DEFAULT), ("quotes-api", QUOTES), ("events-api", EVENTS)):
                client.add_endpoint(region, hosts[kind], api_type)
        self._trade = TradeClient(client)
        self._data = DataClient(client)
        self._cached_account: AccountState | None = None
        self._cached_account_time: float = 0.0

    def _call_with_retry(self, fn, max_retries: int = 3, delay: float = 2.0):
        import time
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except Exception as e:
                err_str = (str(e) + " " + getattr(e, "error_code", "")).upper()
                if ("TOO_MANY_REQUESTS" in err_str or "TOOMANYREQUESTS" in err_str or "429" in err_str) and attempt < max_retries:
                    time.sleep(delay * (attempt + 1))
                    continue
                raise

    # -- account --------------------------------------------------------------------

    def list_accounts(self) -> list[dict]:
        return _unwrap_list(_json(self._call_with_retry(lambda: self._trade.account_v2.get_account_list())), "data", "accounts")

    def get_account(self, force: bool = False, ttl_seconds: float = 10.0) -> AccountState:
        import time
        now = time.time()
        if not force and self._cached_account and (now - self._cached_account_time) < ttl_seconds:
            return self._cached_account

        bal = _json(self._call_with_retry(lambda: self._trade.account_v2.get_account_balance(self.account_id)))
        positions = self._positions()
        usd = next((a for a in bal.get("account_currency_assets", []) if a.get("currency") == "USD"), None)
        if bal.get("total_asset_currency") == "USD":
            cash = _f(bal.get("total_cash_balance"))
            equity = _f(bal.get("total_net_liquidation_value")) or cash + _f(bal.get("total_market_value"))
        elif usd is not None:
            # Non-USD base currency: count only USD cash plus US positions, so sizing never
            # relies on an FX conversion we did not perform.
            cash = _f(usd.get("cash_balance"))
            equity = cash + sum(p.market_value for p in positions)
        else:
            raise BrokerError("balance response has no USD figures; run `trading-agent probe` and check mapping")
        buying_power = _f((usd or {}).get("buying_power"), cash)
        acct = AccountState(account_id=self.account_id, equity=equity, cash=cash, buying_power=buying_power,
                            positions=positions, open_orders=self._open_orders(), as_of=utcnow())
        self._cached_account = acct
        self._cached_account_time = time.time()
        return acct

    def _positions(self) -> list[Position]:
        rows = _unwrap_list(_json(self._call_with_retry(lambda: self._trade.account_v2.get_account_position(self.account_id))), "data", "holdings")
        out = []
        for r in rows:
            qty = _f(r.get("quantity"))
            if qty and r.get("symbol"):
                last = _f(r.get("last_price")) or _f(r.get("market_value")) / qty
                out.append(Position(symbol=r["symbol"], quantity=qty, avg_cost=_f(r.get("cost_price")), last_price=last))
        return out

    def _open_orders(self) -> list[BrokerOrder]:
        data = _json(self._call_with_retry(lambda: self._trade.order_v3.get_order_open(account_id=self.account_id)))
        return [self._to_order(o) for o in self._flatten_orders(_unwrap_list(data, "data", "orders"))]

    # -- market data ----------------------------------------------------------------

    def _yfinance_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Fallback when the Webull quote subscription is missing.

        Uses only real bid/ask and the exchange timestamp. Nothing is estimated: a missing side
        is left at 0 so the spread reads as infinite and the risk engine refuses new entries.
        """
        import yfinance as yf
        now, out = utcnow(), {}
        for sym in symbols:
            try:
                info = yf.Ticker(sym).info or {}
                last = _f(info.get("regularMarketPrice")) or _f(info.get("currentPrice"))
                if last <= 0:
                    continue
                out[sym] = Quote(
                    symbol=sym, bid=_f(info.get("bid")), ask=_f(info.get("ask")), last=last,
                    bid_size=_f(info.get("bidSize")), ask_size=_f(info.get("askSize")),
                    volume=_f(info.get("regularMarketVolume")),
                    last_trade_time=_ts(info.get("regularMarketTime")), fetched_at=now, source="yfinance",
                )
            except Exception:
                continue
        return out

    def _yfinance_bars(self, symbols: list[str], count: int) -> dict[str, list[Bar]]:
        import yfinance as yf
        period = "1y" if count <= 260 else "2y"
        out: dict[str, list[Bar]] = {}
        try:
            df = yf.download(symbols, period=period, interval="1d", progress=False, group_by="ticker")
            for sym in symbols:
                sub = df[sym] if len(symbols) > 1 and sym in df else df
                bars = []
                for idx, r in sub.dropna().iterrows():
                    ts = idx.to_pydatetime()
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    bars.append(Bar(
                        ts=ts, open=_f(r["Open"]), high=_f(r["High"]),
                        low=_f(r["Low"]), close=_f(r["Close"]), volume=_f(r["Volume"]),
                    ))
                if bars:
                    out[sym] = sorted(bars[-count:], key=lambda b: b.ts)
        except Exception:
            pass
        return out

    def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        from webull.data.common.category import Category

        now, out = utcnow(), {}
        try:
            for i in range(0, len(symbols), 100):
                chunk = ",".join(symbols[i:i + 100])
                for s in _unwrap_list(_json(self._data.market_data.get_snapshot(chunk, Category.US_STOCK.name)), "data"):
                    out[s["symbol"]] = Quote(
                        symbol=s["symbol"], bid=_f(s.get("bid")), ask=_f(s.get("ask")), last=_f(s.get("price")),
                        bid_size=_f(s.get("bid_size")), ask_size=_f(s.get("ask_size")), volume=_f(s.get("volume")),
                        last_trade_time=_ts(s.get("last_trade_time")), fetched_at=now, source="webull",
                    )
            if out:
                return out
        except Exception:
            pass
        return self._yfinance_quotes(symbols)

    def get_daily_bars(self, symbols: list[str], count: int) -> dict[str, list[Bar]]:
        from webull.data.common.category import Category
        from webull.data.common.timespan import Timespan

        out: dict[str, list[Bar]] = {}
        try:
            for i in range(0, len(symbols), 20):
                resp = self._data.market_data.get_batch_history_bar(symbols[i:i + 20], Category.US_STOCK.name,
                                                                    Timespan.D.name, str(count))
                for group in _unwrap_list(_json(resp), "result", "data"):  # SG returns {"result": [...]}
                    # A bar with no timestamp is dropped, never stamped "now": that would make stale data look fresh.
                    bars = [Bar(ts=ts, open=_f(b["open"]), high=_f(b["high"]),
                                low=_f(b["low"]), close=_f(b["close"]), volume=_f(b.get("volume")))
                            for b in group.get("result", []) if (ts := _ts(b.get("time"))) is not None]
                    out[group["symbol"]] = sorted(bars, key=lambda b: b.ts)
            if out:
                return out
        except Exception:
            pass
        return self._yfinance_bars(symbols, count)

    # -- events ---------------------------------------------------------------------

    QUARTER_DAYS = 91

    def get_earnings_dates(self, symbols: list[str], today: date) -> dict[str, date]:
        """Next earnings date per stock from Webull's earnings calendar.

        Uses the earliest unpublished report on or after yesterday. If Webull has not listed the next
        report yet, estimates it as the last published report plus one quarter (never earlier than today,
        so an overdue report blocks entries). Symbols with no calendar data are omitted: unknown stays unknown."""
        out: dict[str, date] = {}
        for sym in symbols:
            try:
                rows = _unwrap_list(_json(self._data.fundamentals.get_earnings_calendar(sym)), "data")
            except Exception:
                continue
            upcoming, published = [], []
            for r in rows:
                try:
                    d = date.fromisoformat(str(r.get("expected_publish_date"))[:10])
                except ValueError:
                    continue
                (published if r.get("eps_actual") not in (None, "") else upcoming).append(d)
            upcoming = [d for d in upcoming if d >= today - timedelta(days=1)]
            if upcoming:
                out[sym] = min(upcoming)
            elif published:
                out[sym] = max(max(published) + timedelta(days=self.QUARTER_DAYS), today)
        return out

    # -- orders ---------------------------------------------------------------------

    @staticmethod
    def _payload(o: OrderRequest) -> dict:
        p = {
            "combo_type": "NORMAL",
            "client_order_id": o.client_order_id,
            "symbol": o.symbol,
            "instrument_type": "EQUITY",
            "market": "US",
            "order_type": o.order_type,
            "quantity": str(int(o.quantity)),
            "support_trading_session": "CORE",
            "side": o.side,
            "time_in_force": o.time_in_force,
            "entrust_type": "QTY",
        }
        if o.limit_price is not None:
            p["limit_price"] = f"{o.limit_price:.2f}"
        if o.stop_price is not None:
            p["stop_price"] = f"{o.stop_price:.2f}"
        return p

    def preview(self, order: OrderRequest) -> PreviewResult:
        try:
            data = _json(self._trade.order_v3.preview_order(self.account_id, [self._payload(order)]))
        except Exception as e:  # preview never places anything; any failure is a clean "no"
            return PreviewResult(ok=False, estimated_cost=None, estimated_fees=None, error=str(e)[:300])
        data = data if isinstance(data, dict) else {}
        cost = data.get("estimated_cost", data.get("estimated_amount"))
        fees = data.get("estimated_transaction_fee", data.get("estimated_commission"))
        return PreviewResult(ok=True, estimated_cost=_f(cost, None), estimated_fees=_f(fees, None), raw=data)

    def place(self, order: OrderRequest) -> str | None:
        from webull.core.exception.exceptions import ServerException

        try:
            data = _json(self._trade.order_v3.place_order(account_id=self.account_id, new_orders=[self._payload(order)]))
        except ServerException as e:
            # A 4xx business error means the order was not accepted. A 5xx (or missing status) may
            # have reached the matching engine, so it is ambiguous and must be reconciled, not assumed dead.
            status = getattr(e, "http_status", None)
            ambiguous = not (isinstance(status, int) and 400 <= status < 500)
            raise BrokerError(f"{'ambiguous' if ambiguous else 'rejected'}: {getattr(e, 'error_code', '')} {e}",
                              ambiguous=ambiguous) from e
        except BrokerError as e:
            raise BrokerError(str(e), ambiguous=True) from e
        except Exception as e:
            # Timeouts, connection resets, unknown SDK errors: the order may exist. Caller must reconcile.
            raise BrokerError(f"ambiguous submission failure: {e}", ambiguous=True) from e
        self._cached_account = None
        if isinstance(data, dict):
            return data.get("order_id") or next((x.get("order_id") for x in data.get("orders", []) if x), None)
        return None

    def cancel(self, client_order_id: str) -> None:
        self._cached_account = None
        _json(self._call_with_retry(lambda: self._trade.order_v3.cancel_order(self.account_id, client_order_id)))

    def get_order(self, client_order_id: str) -> BrokerOrder:
        from webull.core.exception.exceptions import ServerException

        cached = getattr(self, "_cached_account", None)
        if cached and cached.open_orders:
            for o in cached.open_orders:
                if o.client_order_id == client_order_id:
                    return o

        try:
            data = _json(self._call_with_retry(lambda: self._trade.order_v3.get_order_detail(self.account_id, client_order_id)))
        except ServerException as e:
            if "NOT_FOUND" in str(getattr(e, "error_code", "")).upper() or "not exist" in str(e).lower():
                return BrokerOrder(client_order_id=client_order_id, symbol="", side="BUY", order_type="",
                                   quantity=0, status=BrokerOrderStatus.NOT_FOUND)
            raise
        items = self._flatten_orders([data] if isinstance(data, dict) else _unwrap_list(data, "data"))
        # Exact match only: returning some other order's status could fake a fill we never got.
        match = next((o for o in items if o.get("client_order_id") == client_order_id), None)
        if match is None:
            return BrokerOrder(client_order_id=client_order_id, symbol="", side="BUY", order_type="",
                               quantity=0, status=BrokerOrderStatus.NOT_FOUND)
        return self._to_order(match)

    @staticmethod
    def _flatten_orders(rows: list[dict]) -> list[dict]:
        out = []
        for r in rows:
            out.extend(r["orders"] if isinstance(r.get("orders"), list) else [r])
        return out

    @staticmethod
    def _to_order(o: dict) -> BrokerOrder:
        raw = str(o.get("status", "")).upper().replace(" ", "_")
        return BrokerOrder(
            client_order_id=o.get("client_order_id", ""), broker_order_id=o.get("order_id"),
            symbol=o.get("symbol", ""), side="SELL" if str(o.get("side", "")).upper() == "SELL" else "BUY",
            order_type=o.get("order_type", ""), quantity=_f(o.get("total_quantity", o.get("quantity"))),
            filled_quantity=_f(o.get("filled_quantity")), filled_price=_f(o.get("filled_price"), None),
            limit_price=_f(o.get("limit_price"), None), stop_price=_f(o.get("stop_price"), None),
            status=STATUS_MAP.get(raw, BrokerOrderStatus.WORKING), raw_status=raw,
        )

    # -- diagnostics ----------------------------------------------------------------

    def probe(self, symbol: str = "SPY") -> dict:
        """Raw responses for checking field mappings against a real account."""
        from webull.data.common.category import Category

        def _safe(fn):
            try:
                return _json(fn())
            except Exception as e:
                return {"error": str(e)}

        q = self.get_quotes([symbol]).get(symbol)
        return {
            "accounts": _safe(lambda: self._trade.account_v2.get_account_list()),
            "balance": _safe(lambda: self._trade.account_v2.get_account_balance(self.account_id)),
            "positions": _safe(lambda: self._trade.account_v2.get_account_position(self.account_id)),
            "open_orders": _safe(lambda: self._trade.order_v3.get_order_open(account_id=self.account_id)),
            "webull_snapshot": _safe(lambda: self._data.market_data.get_snapshot(symbol, Category.US_STOCK.name)),
            "real_time_quote": q.model_dump(mode="json") if q else None,
        }
