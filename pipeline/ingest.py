"""
pipeline/ingest.py
──────────────────
Stage 1 — Data Ingestion

Fetches bibliographic records from the OpenAlex REST API.
Falls back to a synthetic dataset if the API is unreachable.

Public contract
───────────────
    ingest() -> pd.DataFrame
        Returns a raw DataFrame of research paper records.
"""

from __future__ import annotations

import time
import random
import requests
import pandas as pd
from datetime import datetime, date, timezone

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))
from config import OPENALEX_BASE, SEARCH_TERMS, RECORDS_PER_TERM, REQUEST_TIMEOUT
from logger import get_logger

log = get_logger("ingest")


# ── OpenAlex helpers ───────────────────────────────────────────────────────────

def _fetch_works(term: str, per_page: int) -> list[dict]:
    """Fetch raw work records from OpenAlex for a single search term."""
    url = f"{OPENALEX_BASE}/works"
    params = {
        "filter":   f"title.search:{term}",
        "per-page": per_page,
        "select":   (
            "id,title,publication_year,cited_by_count,"
            "authorships,primary_location,open_access,concepts"
        ),
    }
    log.debug("GET %s  params=%s", url, params)
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.json().get("results", [])


def _parse_work(raw: dict, search_term: str) -> dict:
    """Flatten one OpenAlex work object into a table row."""
    # Authors
    authors = [
        a["author"]["display_name"]
        for a in raw.get("authorships", [])
        if a.get("author") and a["author"].get("display_name")
    ]

    # Venue
    loc = raw.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = source.get("display_name", "Unknown")

    # Top concept
    concepts = sorted(
        raw.get("concepts", []),
        key=lambda c: c.get("score", 0),
        reverse=True,
    )
    top_concept = concepts[0]["display_name"] if concepts else "Unknown"

    return {
        "openalex_id":   raw.get("id", ""),
        "title":         raw.get("title", ""),
        "year":          raw.get("publication_year"),
        "citations":     raw.get("cited_by_count", 0),
        "author_count":  len(authors),
        "authors":       "; ".join(authors[:5]),   # store first 5
        "venue":         venue,
        "open_access":   raw.get("open_access", {}).get("is_oa", False),
        "top_concept":   top_concept,
        "search_term":   search_term,
        "ingested_at":   datetime.now(timezone.utc).isoformat(),
    }


def _fetch_from_api() -> pd.DataFrame:
    """Attempt to pull data from OpenAlex for all configured search terms."""
    rows: list[dict] = []
    for term in SEARCH_TERMS:
        log.info("Fetching '%s' from OpenAlex …", term)
        try:
            works = _fetch_works(term, RECORDS_PER_TERM)
            parsed = [_parse_work(w, term) for w in works]
            rows.extend(parsed)
            log.info("  → %d records retrieved", len(parsed))
            time.sleep(0.5)   # be polite to the API
        except requests.RequestException as exc:
            log.warning("API call failed for term '%s': %s", term, exc)
    return pd.DataFrame(rows)


# ── Fallback synthetic dataset ─────────────────────────────────────────────────

_VENUES = [
    "Nature Machine Intelligence", "NeurIPS", "ICML", "ICLR",
    "IEEE Transactions on Neural Networks", "Journal of Machine Learning Research",
    "ACM Computing Surveys", "Artificial Intelligence",
]
_CONCEPTS = [
    "Neural Networks", "Gradient Descent", "Attention Mechanism",
    "Convolutional Networks", "Reinforcement Learning", "Transfer Learning",
    "Generative Models", "Graph Neural Networks",
]
_AUTHOR_POOL = [
    "Y. LeCun", "G. Hinton", "Y. Bengio", "A. Vaswani", "I. Goodfellow",
    "K. He", "J. Devlin", "T. Brown", "S. Hochreiter", "A. Krizhevsky",
    "D. Silver", "O. Vinyals", "R. Sutton", "P. Abbeel", "C. Szegedy",
]


def _generate_synthetic(n: int = 150) -> pd.DataFrame:
    """Generate a realistic-looking synthetic research dataset."""
    log.warning("Generating synthetic fallback dataset (%d records).", n)
    random.seed(42)
    rows = []
    for i in range(n):
        term  = random.choice(SEARCH_TERMS)
        year  = random.randint(2015, 2024)
        n_authors = random.randint(1, 6)
        authors = random.sample(_AUTHOR_POOL, min(n_authors, len(_AUTHOR_POOL)))
        # Citations follow a power-law-ish distribution
        cites = max(0, int(random.expovariate(1 / 120)))
        rows.append({
            "openalex_id":  f"synthetic_{i:04d}",
            "title":        f"Advances in {term.title()} — Study {i+1}",
            "year":         year,
            "citations":    cites,
            "author_count": n_authors,
            "authors":      "; ".join(authors),
            "venue":        random.choice(_VENUES),
            "open_access":  random.random() < 0.45,
            "top_concept":  random.choice(_CONCEPTS),
            "search_term":  term,
            "ingested_at":  datetime.now(timezone.utc).isoformat(),
        })
    return pd.DataFrame(rows)


# ── Public entry point ─────────────────────────────────────────────────────────

def ingest() -> pd.DataFrame:
    """
    Fetch research paper records.

    Tries the OpenAlex API first; falls back to synthetic data on failure.
    Returns a raw (uncleaned) DataFrame.
    """
    log.info("=== INGEST stage started ===")
    try:
        df = _fetch_from_api()
        if df.empty:
            raise ValueError("API returned no rows.")
        log.info("API ingestion complete — %d raw records", len(df))
    except Exception as exc:
        log.warning("API ingestion failed (%s). Using synthetic data.", exc)
        df = _generate_synthetic()

    log.info("=== INGEST stage finished — %d records ===", len(df))
    return df


if __name__ == "__main__":
    df = ingest()
    print(df.head())
    print(df.dtypes)
