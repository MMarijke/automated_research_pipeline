"""
pipeline/store.py
─────────────────
Stage 3 — Storage

Manages a SQLite database with three tables:
    • papers          — core cleaned research records
    • pipeline_runs   — audit log of every pipeline execution
    • yearly_summary  — pre-aggregated yearly stats (materialised view pattern)

Public contract
───────────────
    store(df: pd.DataFrame, run_meta: dict) -> None
"""

from __future__ import annotations

import sqlite3
import hashlib
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Generator

import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from logger import get_logger

log = get_logger("store")


# ── DDL ────────────────────────────────────────────────────────────────────────

DDL_PAPERS = """
CREATE TABLE IF NOT EXISTS papers (
    row_hash         TEXT PRIMARY KEY,
    openalex_id      TEXT NOT NULL,
    title            TEXT NOT NULL,
    year             INTEGER,
    citations        INTEGER DEFAULT 0,
    author_count     INTEGER DEFAULT 1,
    authors          TEXT,
    venue            TEXT,
    open_access      INTEGER DEFAULT 0,
    top_concept      TEXT,
    search_term      TEXT,
    paper_age        INTEGER,
    citations_per_year  REAL,
    impact_tier      TEXT,
    decade           TEXT,
    is_collaborative INTEGER DEFAULT 0,
    title_word_count INTEGER,
    ingested_at      TEXT,
    stored_at        TEXT DEFAULT (datetime('now'))
);
"""

DDL_RUNS = """
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    finished_at  TEXT,
    status       TEXT DEFAULT 'running',
    rows_ingested INTEGER DEFAULT 0,
    rows_stored   INTEGER DEFAULT 0,
    rows_skipped  INTEGER DEFAULT 0,
    error_message TEXT,
    metadata      TEXT
);
"""

DDL_YEARLY = """
CREATE TABLE IF NOT EXISTS yearly_summary (
    year            INTEGER PRIMARY KEY,
    paper_count     INTEGER,
    avg_citations   REAL,
    median_citations REAL,
    avg_authors     REAL,
    open_access_pct REAL,
    top_concept     TEXT,
    top_venue       TEXT,
    refreshed_at    TEXT
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_papers_year       ON papers(year);",
    "CREATE INDEX IF NOT EXISTS idx_papers_citations  ON papers(citations);",
    "CREATE INDEX IF NOT EXISTS idx_papers_venue      ON papers(venue);",
    "CREATE INDEX IF NOT EXISTS idx_papers_concept    ON papers(top_concept);",
    "CREATE INDEX IF NOT EXISTS idx_papers_term       ON papers(search_term);",
]


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with WAL mode and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _init_schema() -> None:
    """Create tables and indexes if they don't exist."""
    with _get_conn() as conn:
        conn.execute(DDL_PAPERS)
        conn.execute(DDL_RUNS)
        conn.execute(DDL_YEARLY)
        for idx in DDL_INDEXES:
            conn.execute(idx)
    log.debug("Schema initialised at %s", DB_PATH)


# ── Row hashing (idempotent upsert key) ────────────────────────────────────────

def _row_hash(row: pd.Series) -> str:
    """Stable SHA-1 hash over business-key fields for idempotent upserts."""
    key = f"{row['openalex_id']}|{row['title']}|{row['year']}"
    return hashlib.sha1(key.encode()).hexdigest()


# ── Core write helpers ─────────────────────────────────────────────────────────

_PAPER_COLS = [
    "row_hash", "openalex_id", "title", "year", "citations",
    "author_count", "authors", "venue", "open_access", "top_concept",
    "search_term", "paper_age", "citations_per_year", "impact_tier",
    "decade", "is_collaborative", "title_word_count", "ingested_at",
]

def _upsert_papers(conn: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    """
    INSERT OR IGNORE pattern — skip rows whose hash already exists.
    Returns (inserted, skipped).
    """
    if df.empty:
        log.debug("_upsert_papers: empty DataFrame, nothing to insert.")
        return 0, 0

    df = df.copy()
    df["row_hash"]         = df.apply(_row_hash, axis=1)
    df["open_access"]      = df["open_access"].astype(int)
    df["is_collaborative"] = df["is_collaborative"].astype(int)

    # Convert pandas nullable Int64 → standard Python int so SQLite
    # stores INTEGER not raw 8-byte little-endian blobs.
    for col in ("year", "citations", "author_count", "paper_age",
                "title_word_count"):
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), other=None)
            df[col] = df[col].apply(
                lambda v: int(v) if v is not None else None
            )

    # Fetch existing hashes
    existing = {
        r[0] for r in conn.execute("SELECT row_hash FROM papers").fetchall()
    }

    new_rows = df[~df["row_hash"].isin(existing)][_PAPER_COLS]
    skipped  = len(df) - len(new_rows)

    if not new_rows.empty:
        placeholders = ", ".join(["?"] * len(_PAPER_COLS))
        sql = f"INSERT INTO papers ({', '.join(_PAPER_COLS)}) VALUES ({placeholders})"
        conn.executemany(sql, new_rows.itertuples(index=False, name=None))

    log.info("Papers: %d inserted, %d skipped (already stored)", len(new_rows), skipped)
    return len(new_rows), skipped


def _refresh_yearly_summary(conn: sqlite3.Connection) -> None:
    """Recompute yearly_summary from the papers table."""
    conn.execute("DELETE FROM yearly_summary;")

    rows = conn.execute("""
        SELECT
            year,
            COUNT(*)                                              AS paper_count,
            ROUND(AVG(CAST(citations AS REAL)), 2)               AS avg_citations,
            ROUND(AVG(CAST(author_count AS REAL)), 2)            AS avg_authors,
            ROUND(100.0 * SUM(open_access) / COUNT(*), 1)        AS open_access_pct
        FROM papers
        GROUP BY year
        ORDER BY year
    """).fetchall()

    import struct

    def _to_int(v):
        if isinstance(v, (bytes, bytearray)):
            return struct.unpack('<q', v)[0]
        return int(v.item()) if hasattr(v, 'item') else int(v)

    def _to_float(v):
        if isinstance(v, (bytes, bytearray)):
            return float(struct.unpack('<q', v)[0])
        return float(v.item()) if hasattr(v, 'item') else float(v)

    for r in rows:
        yr = _to_int(r[0])
        top_concept = conn.execute(
            "SELECT top_concept FROM papers WHERE year=? AND top_concept!='Unknown' "
            "GROUP BY top_concept ORDER BY COUNT(*) DESC LIMIT 1", (yr,)
        ).fetchone()
        top_venue = conn.execute(
            "SELECT venue FROM papers WHERE year=? AND venue!='Unknown' "
            "GROUP BY venue ORDER BY COUNT(*) DESC LIMIT 1", (yr,)
        ).fetchone()

        conn.execute(
            """INSERT OR REPLACE INTO yearly_summary
               (year, paper_count, avg_citations, median_citations, avg_authors,
                open_access_pct, top_concept, top_venue, refreshed_at)
               VALUES (?,?,?,?,?,?,?,?,datetime('now'))""",
            (
                yr,
                _to_int(r[1]), _to_float(r[2]), _to_float(r[2]),
                _to_float(r[3]), _to_float(r[4]),
                top_concept[0] if top_concept else "Unknown",
                top_venue[0] if top_venue else "Unknown",
            )
        )
    log.debug("yearly_summary refreshed (%d years).", len(rows))


# ── Run tracking ───────────────────────────────────────────────────────────────

def start_run(meta: dict | None = None) -> int:
    """Insert a pipeline_run row and return its run_id."""
    _init_schema()
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO pipeline_runs (started_at, status, metadata) VALUES (?, 'running', ?)",
            (datetime.now(timezone.utc).isoformat(), json.dumps(meta or {})),
        )
        return cur.lastrowid


def finish_run(
    run_id: int,
    status: str,
    rows_ingested: int = 0,
    rows_stored: int = 0,
    rows_skipped: int = 0,
    error_message: str | None = None,
) -> None:
    """Update an existing pipeline_run row with final stats."""
    with _get_conn() as conn:
        conn.execute(
            """
            UPDATE pipeline_runs
            SET finished_at=?, status=?, rows_ingested=?,
                rows_stored=?, rows_skipped=?, error_message=?
            WHERE run_id=?
            """,
            (
                datetime.now(timezone.utc).isoformat(), status,
                rows_ingested, rows_stored, rows_skipped,
                error_message, run_id,
            ),
        )


# ── Public entry point ─────────────────────────────────────────────────────────

def store(df: pd.DataFrame, run_id: int) -> tuple[int, int]:
    """
    Persist a clean DataFrame to SQLite.

    Performs:
        • Idempotent upsert of paper records
        • Refresh of yearly_summary aggregate table
        • Update of pipeline_run status

    Returns (rows_stored, rows_skipped).
    """
    log.info("=== STORE stage started — %d rows to process ===", len(df))
    _init_schema()

    if df.empty:
        log.warning("STORE: received empty DataFrame — nothing to store.")
        return 0, 0

    try:
        with _get_conn() as conn:
            inserted, skipped = _upsert_papers(conn, df)
            _refresh_yearly_summary(conn)
    except Exception as exc:
        log.error("STORE stage failed: %s", exc, exc_info=True)
        finish_run(run_id, "failed", len(df), 0, 0, str(exc))
        raise

    log.info("=== STORE stage finished — %d new, %d skipped ===", inserted, skipped)
    return inserted, skipped


def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Convenience wrapper — execute a SELECT and return a DataFrame."""
    with _get_conn() as conn:
        return pd.read_sql_query(sql, conn, params=params)


if __name__ == "__main__":
    from ingest import ingest
    from clean import clean
    raw   = ingest()
    clean_df = clean(raw)
    run_id = start_run({"test": True})
    ins, skip = store(clean_df, run_id)
    finish_run(run_id, "success", len(raw), ins, skip)
    print(query("SELECT year, paper_count, avg_citations FROM yearly_summary ORDER BY year DESC LIMIT 10;"))
