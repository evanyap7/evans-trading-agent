"""Open-market insider purchases from SEC's free quarterly Form 3/4/5 data sets.

Point-in-time: every purchase is dated by its FILING date (when the public could first see it), never the
trade date. Only Form 4 code `P` (open-market buy) counts. Awards, option exercises and sales are noise.
Data: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"
# SEC requires a contact in the User-Agent.
DEFAULT_UA = "trading-agent evanyap7@gmail.com"


def _quarters(start: date, end: date):
    y, q = start.year, (start.month - 1) // 3 + 1
    while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)


def _download(y: int, q: int, cache: Path, ua: str) -> Path | None:
    path = cache / f"{y}q{q}_form345.zip"
    if path.exists():
        return path
    req = urllib.request.Request(URL.format(y=y, q=q), headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            path.write_bytes(r.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:  # quarter not published yet
            return None
        raise
    return path


def _parse(zpath: Path) -> list[dict]:
    """Return open-market buys in one quarterly zip as {symbol, filed, usd, owner_cik, accession}."""
    with zipfile.ZipFile(zpath) as z:
        def rows(name):
            return csv.DictReader(io.TextIOWrapper(z.open(name), encoding="utf-8", errors="replace"), delimiter="\t")

        subs = {r["ACCESSION_NUMBER"]: r for r in rows("SUBMISSION.tsv") if r["DOCUMENT_TYPE"] in ("4", "4/A")}
        owners: dict = {}
        for r in rows("REPORTINGOWNER.tsv"):
            owners.setdefault(r["ACCESSION_NUMBER"], r["RPTOWNERCIK"])
        out = []
        for r in rows("NONDERIV_TRANS.tsv"):
            s = subs.get(r["ACCESSION_NUMBER"])
            if not s or r["TRANS_CODE"] != "P" or r["TRANS_ACQUIRED_DISP_CD"] != "A" or not s["ISSUERTRADINGSYMBOL"]:
                continue
            try:
                usd = float(r["TRANS_SHARES"]) * float(r["TRANS_PRICEPERSHARE"])
                filed = datetime.strptime(s["FILING_DATE"], "%d-%b-%Y").date()
            except ValueError:
                continue
            if usd > 0:
                out.append({"symbol": s["ISSUERTRADINGSYMBOL"].upper().strip(), "filed": filed.isoformat(),
                            "usd": round(usd), "owner_cik": owners.get(r["ACCESSION_NUMBER"], ""),
                            "accession": r["ACCESSION_NUMBER"]})
        return out


def load_insider_buys(start: date, end: date, cache_dir: Path, user_agent: str = DEFAULT_UA) -> list[dict]:
    """Open-market buys filed in [start, end]. Parsed quarters are cached as JSON next to the zips."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    buys: list[dict] = []
    for y, q in _quarters(start, end):
        parsed = cache_dir / f"buys_{y}q{q}.json"
        if not parsed.exists():
            zpath = _download(y, q, cache_dir, user_agent)
            if zpath is None:
                continue
            parsed.write_text(json.dumps(_parse(zpath)))
        buys += json.loads(parsed.read_text())
    return [b for b in buys if start.isoformat() <= b["filed"] <= end.isoformat()]


def buy_clusters(buys: list[dict], symbol: str, as_of: date, days: int = 90) -> dict:
    """What the public could see on `as_of`: buys by `symbol` filed in the trailing `days` days."""
    lo = (as_of - timedelta(days=days)).isoformat()
    hit = [b for b in buys if b["symbol"] == symbol and lo < b["filed"] <= as_of.isoformat()]
    return {"usd": sum(b["usd"] for b in hit), "insiders": len({b["owner_cik"] for b in hit})}


def demo() -> None:
    buys = [{"symbol": "X", "filed": "2024-02-01", "usd": 100, "owner_cik": "1", "accession": "a"},
            {"symbol": "X", "filed": "2024-02-10", "usd": 50, "owner_cik": "2", "accession": "b"},
            {"symbol": "X", "filed": "2024-06-01", "usd": 999, "owner_cik": "3", "accession": "c"}]
    assert buy_clusters(buys, "X", date(2024, 2, 15)) == {"usd": 150, "insiders": 2}
    assert buy_clusters(buys, "X", date(2024, 5, 20))["usd"] == 0   # aged out of the window
    assert buy_clusters(buys, "X", date(2024, 2, 5))["usd"] == 100   # later filings not yet visible
    assert list(_quarters(date(2023, 11, 1), date(2024, 4, 1))) == [(2023, 4), (2024, 1), (2024, 2)]


if __name__ == "__main__":
    demo()
    print("ok")
