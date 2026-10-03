"""Literature search for the Embodied Fly Lab (Europe PMC + OpenAlex, no API keys).

Public API (see docs/CONTRACTS.md):
    search_europepmc(query, n=5) -> list[dict]
    search_openalex(query, n=5)  -> list[dict]
    get_by_doi(doi)              -> dict | None
Each item: {"title","year","doi","pmid","authors","abstract","url","source"}.
All HTTP responses are cached as JSON in data/cache/literature/ so repeated calls are offline and fast.

CLI:
    python -m flylab.literature "moonwalker descending neuron" [-n 5] [--source europepmc|openalex|both]
    python -m flylab.literature --doi 10.1126/science.1249964
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache" / "literature"
EPMC_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
OPENALEX_URL = "https://api.openalex.org/works"
TIMEOUT_S = 20
HEADERS = {"User-Agent": "EmbodiedFlyLab/0.1 (hackathon research prototype)"}

__all__ = ["search_europepmc", "search_openalex", "get_by_doi", "search"]


# ----------------------------------------------------------------------------- helpers
def _cache_path(kind: str, key: str) -> Path:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{kind}_{h}.json"


def _cached_get(kind: str, url: str, params: dict[str, Any]) -> Any | None:
    """GET JSON with on-disk cache. Returns None on network/HTTP failure (never raises)."""
    key = url + "?" + json.dumps(params, sort_keys=True)
    path = _cache_path(kind, key)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["response"]
        except Exception:
            pass
    data = None
    for attempt in range(3):  # transient failures (rate limits, timeouts) seen in practice -> 3 tries, 1 s / 2 s backoff
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT_S)
            if r.status_code == 404:
                data = {"_not_found": True}
                break
            r.raise_for_status()
            data = r.json()
            break
        except Exception:
            if attempt < 2:
                time.sleep(1.0 + attempt)
    if data is None:
        return None
    # atomic write (several agents may share the cache): unique temp file, then replace
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps({"url": url, "params": params, "fetched": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "response": data}, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
    return data


# HTML/JATS tags such as <i>, </sup>, <jats:p ...>. A bare "<" followed by a space (e.g. "p < 0.05") is not a tag.
_TAG_RE = re.compile(r"</?[A-Za-z][\w:.-]*(?:\s[^<>]*)?/?>")


def _clean(text: str | None) -> str:
    if not text:
        return ""
    # unescape first: Europe PMC sometimes returns escaped markup ("&lt;i&gt;Drosophila&lt;/i&gt;")
    text = _TAG_RE.sub(" ", html.unescape(text))
    return re.sub(r"\s+", " ", text).strip()


def _clean_title(text: str | None) -> str:
    """Like _clean, plus no stray space before punctuation left by removed inline tags ('<i>Drosophila</i>.')."""
    return re.sub(r"\s+([.,;:?!)])", r"\1", _clean(text))


def _norm_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip()
    doi = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.lower()


def _epmc_item(r: dict) -> dict:
    doi = _norm_doi(r.get("doi"))
    pmid = r.get("pmid") or ""
    if doi:
        url = f"https://doi.org/{doi}"
    elif pmid:
        url = f"https://europepmc.org/article/MED/{pmid}"
    else:
        url = f"https://europepmc.org/article/{r.get('source', 'MED')}/{r.get('id', '')}"
    year = r.get("pubYear")
    return {
        "title": _clean_title(r.get("title")),
        "year": int(year) if year and str(year).isdigit() else None,
        "doi": doi,
        "pmid": str(pmid),
        "authors": _clean(r.get("authorString")),
        "abstract": _clean(r.get("abstractText")),
        "url": url,
        "source": "europepmc",
        "journal": _clean((r.get("journalInfo") or {}).get("journal", {}).get("title") or r.get("journalTitle")),
    }


def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


def _openalex_item(w: dict) -> dict:
    doi = _norm_doi(w.get("doi"))
    ids = w.get("ids") or {}
    pmid = str(ids.get("pmid") or "").rsplit("/", 1)[-1]
    authors = [((a.get("author") or {}).get("display_name") or "") for a in (w.get("authorships") or [])]
    authors = [a for a in authors if a]
    author_str = ", ".join(authors[:6]) + (", et al." if len(authors) > 6 else "")
    src = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "title": _clean_title(w.get("display_name") or w.get("title")),
        "year": w.get("publication_year"),
        "doi": doi,
        "pmid": pmid,
        "authors": author_str,
        "abstract": _clean(_openalex_abstract(w.get("abstract_inverted_index"))),
        "url": f"https://doi.org/{doi}" if doi else (w.get("id") or ""),
        "source": "openalex",
        "journal": _clean(src.get("display_name")),
    }


# ----------------------------------------------------------------------------- public API
def _epmc_raw(query: str, n: int) -> list[dict]:
    params = {"query": query, "format": "json", "resultType": "core", "pageSize": max(1, min(int(n), 100))}
    data = _cached_get("epmc", EPMC_URL, params)
    if not data or data.get("_not_found"):
        return []
    results = ((data.get("resultList") or {}).get("result")) or []
    return [_epmc_item(r) for r in results[:n]]


def search_europepmc(query: str, n: int = 5, title_abstract_only: bool = True) -> list[dict]:
    """Search Europe PMC (resultType=core => abstracts included). Returns [] on failure.

    Plain free-text queries are restricted to title+abstract (TITLE_ABS:(...)) for relevance; if that finds
    nothing, or the query already uses Europe PMC field syntax (contains ':'), the raw query is used.
    """
    if title_abstract_only and ":" not in query:
        res = _epmc_raw(f"TITLE_ABS:({query})", n)
        if res:
            return res
    return _epmc_raw(query, n)


def search_openalex(query: str, n: int = 5) -> list[dict]:
    """Search OpenAlex works (abstract reconstructed from the inverted index). Returns [] on failure."""
    params = {"search": query, "per-page": max(1, min(int(n), 50))}
    data = _cached_get("openalex", OPENALEX_URL, params)
    if not data or data.get("_not_found"):
        return []
    return [_openalex_item(w) for w in (data.get("results") or [])[:n]]


def search(query: str, n: int = 5) -> list[dict]:
    """Convenience: Europe PMC first, then OpenAlex results with DOIs not already seen."""
    out = search_europepmc(query, n)
    seen = {r["doi"] for r in out if r["doi"]}
    for r in search_openalex(query, n):
        if r["doi"] and r["doi"] in seen:
            continue
        out.append(r)
    return out


def get_by_doi(doi: str) -> dict | None:
    """Resolve one DOI. Europe PMC (exact DOI field match) first, OpenAlex as fallback / abstract filler.

    Returns None if neither source knows the DOI. The returned "doi" is always the normalised query DOI,
    and "verified_by" lists the sources that returned a record with exactly this DOI.
    """
    d = _norm_doi(doi)
    if not d:
        return None
    best: dict | None = None
    verified: list[str] = []
    for r in search_europepmc(f'DOI:"{d}"', n=5):
        if r["doi"] == d:
            # prefer the record with an abstract (MED over preprint duplicates etc.)
            if best is None or (not best["abstract"] and r["abstract"]):
                best = r
    if best:
        verified.append("europepmc")
    data = _cached_get("openalex_doi", f"{OPENALEX_URL}/https://doi.org/{d}", {})
    oa = _openalex_item(data) if data and not data.get("_not_found") and data.get("id") else None
    if oa and oa["doi"] == d:
        verified.append("openalex")
        if best is None:
            best = oa
        else:
            if not best["abstract"] and oa["abstract"]:
                best = {**best, "abstract": oa["abstract"]}
            if not best.get("year") and oa.get("year"):
                best["year"] = oa["year"]
    if best is None:
        return None
    best = dict(best)
    best["doi"] = d
    best["verified_by"] = verified
    return best


# ----------------------------------------------------------------------------- CLI
def _main() -> None:
    ap = argparse.ArgumentParser(description="Europe PMC / OpenAlex literature search (cached).")
    ap.add_argument("query", nargs="?", help="free-text query")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--source", choices=["europepmc", "openalex", "both"], default="both")
    ap.add_argument("--doi", help="resolve a single DOI")
    ap.add_argument("--abstract", action="store_true", help="print abstracts")
    a = ap.parse_args()
    if a.doi:
        print(json.dumps(get_by_doi(a.doi), indent=2, ensure_ascii=False))
        return
    if not a.query:
        ap.error("query or --doi required")
    if a.source == "europepmc":
        res = search_europepmc(a.query, a.n)
    elif a.source == "openalex":
        res = search_openalex(a.query, a.n)
    else:
        res = search(a.query, a.n)
    for r in res:
        print(f"[{r['source']}] {r['year']} {r['title']}  doi:{r['doi'] or '-'}  pmid:{r['pmid'] or '-'}")
        print(f"    {r['authors'][:120]}")
        if a.abstract and r["abstract"]:
            print("    " + r["abstract"][:800])


if __name__ == "__main__":
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    _main()
