"""Real-time financial news ingestion from premier global sources.

Sources included:
- Bloomberg Markets
- The Wall Street Journal (Markets & US Business)
- The Economist (Finance & Economics)
- Reuters Markets
- New York Stock Exchange (NYSE)
- Yahoo Finance (Ticker-specific syndicated stories & analyst actions)

All stories are sanitized and packaged into typed `Evidence` objects with
`untrusted_text=True` to guard against prompt injection.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests

from .schemas import Evidence, utcnow

FEED_TIMEOUT_SECONDS = 10

MACRO_FEEDS = {
    "Wall Street Journal Markets": "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "Wall Street Journal Business": "https://feeds.a.dj.com/rss/WSJcomUSBusiness.xml",
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
    "Bloomberg": "https://news.google.com/rss/search?q=site:bloomberg.com+markets+when:2d&hl=en-US&gl=US&ceid=US:en",
    "Reuters": "https://news.google.com/rss/search?q=site:reuters.com+markets+when:2d&hl=en-US&gl=US&ceid=US:en",
    "NYSE": 'https://news.google.com/rss/search?q="New+York+Stock+Exchange"+OR+NYSE+when:2d&hl=en-US&gl=US&ceid=US:en',
}

_TAG_RE = re.compile(r"<[^>]+>")


def _clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = _TAG_RE.sub(" ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return strip_untrusted_tags(text)


_UNTRUSTED_TAG_RE = re.compile(r"<\s*/?\s*untrusted_document[^>]*>?", re.IGNORECASE)


def strip_untrusted_tags(text: str) -> str:
    """Remove any spelling of the wrapper tag so a story cannot 'close' its sandbox and speak as the system."""
    return _UNTRUSTED_TAG_RE.sub("", text)


def _eid(source: str, symbol: str | None, title: str) -> str:
    h = hashlib.sha256(f"{source}_{symbol}_{title}".encode()).hexdigest()[:10]
    sym_tag = symbol.lower() if symbol else "mkt"
    src_tag = "".join(c for c in source.lower() if c.isalnum())[:8]
    return f"doc_{src_tag}_{sym_tag}_{h}"


def _parse_entry_date(entry: Any) -> datetime:
    parsed = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return utcnow()


def fetch_macro_news(max_per_feed: int = 4) -> list[Evidence]:
    """Ingests top market and macroeconomic stories from Bloomberg, WSJ, The Economist, and Reuters."""
    evidence: list[Evidence] = []
    for source_name, feed_url in MACRO_FEEDS.items():
        try:
            # feedparser's own fetch has no timeout; a hung feed would stall the whole tick (and monitoring).
            resp = requests.get(feed_url, timeout=FEED_TIMEOUT_SECONDS, headers={"User-Agent": "trading-agent/0.1"})
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
            for entry in feed.entries[:max_per_feed]:
                title = _clean_text(getattr(entry, "title", ""))
                summary = _clean_text(getattr(entry, "summary", "")) or title
                link = getattr(entry, "link", "")
                if not title:
                    continue
                pub_dt = _parse_entry_date(entry)
                evidence.append(Evidence(
                    evidence_id=_eid(source_name, None, title),
                    kind="document",
                    symbol=None,
                    as_of=pub_dt,
                    source=source_name,
                    payload={"title": title, "text": summary, "url": link},
                    untrusted_text=True,
                ))
        except Exception:
            continue
    return evidence


def fetch_ticker_news(symbols: list[str], max_per_symbol: int = 3) -> list[Evidence]:
    """Fetches real-time ticker-specific news and analyst updates via Yahoo Finance aggregator."""
    import yfinance as yf

    evidence: list[Evidence] = []
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            raw_news = getattr(ticker, "news", []) or []
            for item in raw_news[:max_per_symbol]:
                c = item.get("content", item)
                title = _clean_text(c.get("title", ""))
                summary = _clean_text(c.get("summary", "")) or title
                link = c.get("canonicalUrl", {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")
                provider = "Yahoo Finance"
                if isinstance(c.get("provider"), dict):
                    provider = c["provider"].get("displayName", provider)
                if not title:
                    continue
                pub_time = c.get("pubDate")
                pub_dt = utcnow()
                if pub_time:
                    try:
                        pub_dt = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                    except Exception:
                        pass
                evidence.append(Evidence(
                    evidence_id=_eid(provider, sym, title),
                    kind="document",
                    symbol=sym,
                    as_of=pub_dt,
                    source=provider,
                    payload={"title": title, "text": summary, "url": link, "symbol": sym},
                    untrusted_text=True,
                ))
        except Exception:
            continue
    return evidence


def fetch_all_market_news(universe_symbols: list[str] | None = None) -> list[Evidence]:
    """Combines macro market feeds (Bloomberg, WSJ, Economist, Reuters) with ticker-level news."""
    macro = fetch_macro_news(max_per_feed=3)
    ticker_docs: list[Evidence] = []
    if universe_symbols:
        # Sample or prioritize universe symbols (top 10 to keep network light and fast)
        sample = universe_symbols[:10]
        ticker_docs = fetch_ticker_news(sample, max_per_symbol=2)
    return macro + ticker_docs


def _pct_change(now: Any, then: Any) -> float | None:
    try:
        now, then = float(now), float(then)
    except (TypeError, ValueError):
        return None
    return round((now / then - 1) * 100, 2) if then else None


def estimate_revision_payload(revisions: Any, trend: Any, actions: Any, today: datetime) -> dict:
    """Sell-side estimate momentum for one stock: the catalyst data a desk analyst checks first.

    `revisions` / `trend` are yfinance eps_revisions / eps_trend frames (rows 0y = this fiscal year,
    +1y = next); `actions` is upgrades_downgrades (dated index, Action in up/down/main/init/reit)."""
    out: dict = {}
    for period, label in (("0y", "fy0"), ("+1y", "fy1")):
        if revisions is not None and period in getattr(revisions, "index", []):
            r = revisions.loc[period]
            out[f"{label}_eps_up_30d"] = int(r.get("upLast30days") or 0)
            out[f"{label}_eps_down_30d"] = int(r.get("downLast30days") or 0)
        if trend is not None and period in getattr(trend, "index", []):
            t = trend.loc[period]
            out[f"{label}_eps_est_chg_30d_pct"] = _pct_change(t.get("current"), t.get("30daysAgo"))
            out[f"{label}_eps_est_chg_90d_pct"] = _pct_change(t.get("current"), t.get("90daysAgo"))
    if actions is not None and len(actions) and "Action" in actions:
        idx = actions.index.tz_localize(None) if getattr(actions.index, "tz", None) else actions.index
        recent = actions[idx >= today.replace(tzinfo=None) - timedelta(days=30)]
        acts = recent["Action"].str.lower()
        out["analyst_upgrades_30d"] = int((acts == "up").sum())
        out["analyst_downgrades_30d"] = int((acts == "down").sum())
        if {"currentPriceTarget", "priorPriceTarget"} <= set(recent.columns):
            moves = [m for c, p in zip(recent["currentPriceTarget"], recent["priorPriceTarget"])
                     if (m := _pct_change(c, p)) is not None]
            out["avg_price_target_chg_30d_pct"] = round(sum(moves) / len(moves), 2) if moves else None
    return out


def fetch_estimate_revisions(symbols: list[str]) -> list[Evidence]:
    """One numeric evidence item per stock. Missing data is skipped, never guessed."""
    import yfinance as yf

    now, out = utcnow(), []
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            payload = estimate_revision_payload(t.eps_revisions, t.eps_trend, t.upgrades_downgrades, now)
        except Exception:
            continue
        if payload:
            h = hashlib.sha256(repr(sorted(payload.items())).encode()).hexdigest()[:10]
            out.append(Evidence(evidence_id=f"est_{sym}_{now:%Y%m%d}_{h}", kind="event", symbol=sym, as_of=now,
                                source="yfinance.estimates", payload=payload))
    return out
