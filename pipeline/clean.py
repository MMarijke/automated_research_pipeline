"""
pipeline/clean.py
─────────────────
Stage 2 — Data Cleaning & Transformation

Receives raw DataFrame from ingest(), applies validation, deduplication,
type coercion, derived features, and returns a clean, analysis-ready DataFrame.

Public contract
───────────────
    clean(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from logger import get_logger

log = get_logger("clean")

DATA_DIR.mkdir(parents=True, exist_ok=True)


# ── Schema ─────────────────────────────────────────────────────────────────────

EXPECTED_COLS = {
    "openalex_id":  str,
    "title":        str,
    "year":         "Int64",
    "citations":    "Int64",
    "author_count": "Int64",
    "authors":      str,
    "venue":        str,
    "open_access":  bool,
    "top_concept":  str,
    "search_term":  str,
    "ingested_at":  str,
}

CURRENT_YEAR = 2024


# ── Step functions ─────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected columns exist; add missing ones as NA."""
    missing = set(EXPECTED_COLS) - set(df.columns)
    if missing:
        log.warning("Missing columns added as NA: %s", missing)
        for col in missing:
            df[col] = pd.NA
    extra = set(df.columns) - set(EXPECTED_COLS)
    if extra:
        log.debug("Dropping unexpected columns: %s", extra)
        df = df.drop(columns=list(extra))
    return df[list(EXPECTED_COLS)]          # canonical column order


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to declared dtypes; coerce errors to NA."""
    df = df.copy()

    # Numeric nullable integers
    for col in ("year", "citations", "author_count"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Boolean — handle strings like 'True'/'False'
    if df["open_access"].dtype == object:
        df["open_access"] = df["open_access"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
            if pd.notna(v) else False
        )
    df["open_access"] = df["open_access"].astype(bool)

    # String columns — strip whitespace, replace empty strings
    str_cols = ["openalex_id", "title", "authors", "venue",
                "top_concept", "search_term", "ingested_at"]
    for col in str_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )

    return df


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate by openalex_id; keep first occurrence."""
    before = len(df)
    df = df.drop_duplicates(subset=["openalex_id"], keep="first")
    removed = before - len(df)
    if removed:
        log.info("Deduplication removed %d rows (by openalex_id)", removed)

    # Secondary dedup: near-identical titles in the same year
    before = len(df)
    df["_title_norm"] = df["title"].str.lower().str.replace(r"\W+", " ", regex=True).str.strip()
    df = df.drop_duplicates(subset=["_title_norm", "year"], keep="first")
    df = df.drop(columns=["_title_norm"])
    removed2 = before - len(df)
    if removed2:
        log.info("Title-level deduplication removed %d additional rows", removed2)

    return df.reset_index(drop=True)


def _filter_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that cannot be salvaged."""
    before = len(df)

    # Title must exist
    df = df[df["title"].notna() & (df["title"].str.len() > 3)]

    # Year must be plausible
    df = df[df["year"].notna() & df["year"].between(1950, CURRENT_YEAR)]

    # Citations must be non-negative
    df = df[df["citations"].notna() & (df["citations"] >= 0)]

    removed = before - len(df)
    if removed:
        log.info("Filtered %d rows with invalid/missing critical fields", removed)

    return df.reset_index(drop=True)


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill remaining NA values with sensible defaults."""
    df = df.copy()
    df["venue"]       = df["venue"].fillna("Unknown")
    df["top_concept"] = df["top_concept"].fillna("Unknown")
    df["authors"]     = df["authors"].fillna("Unknown")
    df["author_count"] = df["author_count"].fillna(1)
    return df


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived columns useful for analysis."""
    if df.empty:
        # Add all derived columns as empty so schema stays consistent
        for col in ["paper_age","citations_per_year","impact_tier","decade","is_collaborative","title_word_count"]:
            df[col] = pd.Series(dtype=object)
        return df
    df = df.copy()

    # Age of paper
    df["paper_age"] = CURRENT_YEAR - df["year"].astype(int)

    # Citations per year (avoid div-by-zero for current-year papers)
    age_safe = df["paper_age"].clip(lower=1)
    df["citations_per_year"] = (
        df["citations"].astype(float) / age_safe
    ).round(2)

    # Impact tier based on citations
    bins   = [-1, 0, 10, 50, 200, float("inf")]
    labels = ["Uncited", "Low", "Medium", "High", "Highly Cited"]
    df["impact_tier"] = pd.cut(
        df["citations"].astype(float),
        bins=bins,
        labels=labels,
    ).astype(str)

    # Decade bucket
    df["decade"] = (df["year"].astype(int) // 10 * 10).astype(str) + "s"

    # Collaboration flag
    df["is_collaborative"] = df["author_count"] > 1

    # Normalised title length (word count)
    df["title_word_count"] = df["title"].str.split().str.len().fillna(0).astype(int)

    return df


def _audit(df: pd.DataFrame) -> None:
    """Log a data quality summary."""
    if df.empty:
        log.warning("Data quality check: DataFrame is empty — no rows survived cleaning filters.")
        return

    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        log.info("Data quality check PASSED — no null values remaining")
    else:
        log.warning("Null values remain:\n%s", null_cols.to_string())

    cite_mean = df["citations"].mean()
    cite_std  = df["citations"].std()
    log.info(
        "Clean dataset: %d rows × %d cols | years %s–%s | citations μ=%.1f σ=%s",
        len(df), len(df.columns),
        int(df["year"].min()), int(df["year"].max()),
        float(cite_mean) if cite_mean is not None and str(cite_mean) != "<NA>" else 0.0,
        f"{float(cite_std):.1f}" if cite_std is not None and str(cite_std) != "<NA>" else "n/a",
    )


# ── Public entry point ─────────────────────────────────────────────────────────

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full cleaning & transformation pipeline.

    Steps (in order):
        1. Schema validation
        2. Type coercion
        3. Deduplication
        4. Invalid-row removal
        5. Missing-value imputation
        6. Feature engineering
        7. Audit logging

    Returns a clean DataFrame ready for storage.
    """
    log.info("=== CLEAN stage started — %d input rows ===", len(df))

    df = _validate_schema(df)
    df = _coerce_types(df)
    df = _remove_duplicates(df)
    df = _filter_invalid(df)
    df = _impute_missing(df)
    df = _add_features(df)

    _audit(df)

    # Persist a snapshot for debugging / reproducibility
    snapshot_path = DATA_DIR / "cleaned_snapshot.csv"
    df.to_csv(snapshot_path, index=False)
    log.debug("Clean snapshot saved → %s", snapshot_path)

    log.info("=== CLEAN stage finished — %d rows ===", len(df))
    return df


if __name__ == "__main__":
    from ingest import ingest
    raw = ingest()
    clean_df = clean(raw)
    print(clean_df[["title", "year", "citations", "impact_tier", "citations_per_year"]].head(10))
