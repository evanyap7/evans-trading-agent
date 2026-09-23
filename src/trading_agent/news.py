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
from datetime import datetime, timezone
from typing import Any

import feedparser

from .schemas import Evidence, utcnow

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
    return text.replace("</untrusted_document>", "")


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
            feed = feedparser.parse(feed_url)
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
