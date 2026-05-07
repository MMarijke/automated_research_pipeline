"""
tests/test_store.py
────────────────────
Unit and integration tests for pipeline/store.py
"""

import sys
import json
import sqlite3
import pytest
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from pipeline.store import (
    store,
    start_run,
    finish_run,
    query,
    _init_schema,
    _upsert_papers,
    _refresh_yearly_summary,
    _get_conn,
    _row_hash,
)


# ── Row hashing ────────────────────────────────────────────────────────────────

class TestRowHash:
    def test_same_inputs_same_hash(self):
        row = pd.Series({"openalex_id": "W001", "title": "Test", "year": 2020})
        assert _row_hash(row) == _row_hash(row)

    def test_different_ids_different_hash(self):
        r1 = pd.Series({"openalex_id": "W001", "title": "Test", "year": 2020})
        r2 = pd.Series({"openalex_id": "W002", "title": "Test", "year": 2020})
        assert _row_hash(r1) != _row_hash(r2)

    def test_hash_is_40_char_hex(self):
        row = pd.Series({"openalex_id": "W001", "title": "Test", "year": 2020})
        h = _row_hash(row)
        assert len(h) == 40
        assert all(c in "0123456789abcdef" for c in h)


# ── Schema initialisation ──────────────────────────────────────────────────────

class TestInitSchema:
    def test_creates_all_tables(self, tmp_db):
        _init_schema()
        with sqlite3.connect(tmp_db) as conn:
            tables = {
                r[0] for r in
                conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
        assert {"papers", "pipeline_runs", "yearly_summary"}.issubset(tables)

    def test_idempotent_schema_init(self, tmp_db):
        """Calling _init_schema multiple times should not raise."""
        _init_schema()
        _init_schema()
        _init_schema()

    def test_indexes_created(self, tmp_db):
        _init_schema()
        with sqlite3.connect(tmp_db) as conn:
            indexes = {
                r[0] for r in
                conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            }
        assert "idx_papers_year" in indexes
        assert "idx_papers_citations" in indexes


# ── Run tracking ───────────────────────────────────────────────────────────────

class TestRunTracking:
    def test_start_run_returns_integer_id(self, tmp_db):
        run_id = start_run()
        assert isinstance(run_id, int)
        assert run_id >= 1

    def test_start_run_with_metadata(self, tmp_db):
        run_id = start_run({"env": "test", "version": "1.0"})
        df = query("SELECT metadata FROM pipeline_runs WHERE run_id = ?", (run_id,))
        meta = json.loads(df.iloc[0]["metadata"])
        assert meta["env"] == "test"

    def test_initial_status_is_running(self, tmp_db):
        run_id = start_run()
        df = query("SELECT status FROM pipeline_runs WHERE run_id = ?", (run_id,))
        assert df.iloc[0]["status"] == "running"

    def test_finish_run_updates_status(self, tmp_db):
        run_id = start_run()
        finish_run(run_id, "success", 100, 95, 5)
        df = query("SELECT status, rows_ingested, rows_stored, rows_skipped "
                   "FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = df.iloc[0]
        assert row["status"] == "success"
        assert row["rows_ingested"] == 100
        assert row["rows_stored"] == 95
        assert row["rows_skipped"] == 5

    def test_finish_run_with_error_message(self, tmp_db):
        run_id = start_run()
        finish_run(run_id, "failed", error_message="Something went wrong")
        df = query("SELECT error_message FROM pipeline_runs WHERE run_id = ?", (run_id,))
        assert df.iloc[0]["error_message"] == "Something went wrong"

    def test_multiple_runs_increment_ids(self, tmp_db):
        ids = [start_run() for _ in range(3)]
        assert ids == sorted(ids)
        assert len(set(ids)) == 3


# ── Upsert logic ───────────────────────────────────────────────────────────────

class TestUpsertPapers:
    def test_inserts_new_rows(self, tmp_db, clean_df):
        _init_schema()
        with _get_conn() as conn:
            inserted, skipped = _upsert_papers(conn, clean_df)
        assert inserted == len(clean_df)
        assert skipped == 0

    def test_skips_existing_rows(self, tmp_db, clean_df):
        _init_schema()
        with _get_conn() as conn:
            _upsert_papers(conn, clean_df)
            inserted2, skipped2 = _upsert_papers(conn, clean_df)
        assert inserted2 == 0
        assert skipped2 == len(clean_df)

    def test_partial_overlap(self, tmp_db, clean_df):
        _init_schema()
        half = clean_df.head(2)
        with _get_conn() as conn:
            _upsert_papers(conn, half)
            ins, skip = _upsert_papers(conn, clean_df)
        assert ins == len(clean_df) - 2
        assert skip == 2

    def test_empty_dataframe_returns_zero(self, tmp_db):
        _init_schema()
        with _get_conn() as conn:
            ins, skip = _upsert_papers(conn, pd.DataFrame())
        assert ins == 0
        assert skip == 0

    def test_year_stored_as_integer_not_bytes(self, tmp_db, clean_df):
        """Regression test: pandas Int64 must not be stored as raw bytes."""
        _init_schema()
        with _get_conn() as conn:
            _upsert_papers(conn, clean_df)
            rows = conn.execute("SELECT year FROM papers LIMIT 3").fetchall()
        for (year,) in rows:
            assert isinstance(year, int), f"year stored as {type(year)}, expected int"
            assert 1950 <= year <= 2024


# ── Yearly summary ─────────────────────────────────────────────────────────────

class TestYearlySummary:
    def test_summary_populated_after_store(self, populated_db):
        df = query("SELECT * FROM yearly_summary ORDER BY year")
        assert len(df) > 0

    def test_summary_years_match_papers(self, populated_db):
        paper_years = set(query("SELECT DISTINCT year FROM papers")["year"].astype(int))
        summary_years = set(query("SELECT year FROM yearly_summary")["year"].astype(int))
        assert paper_years == summary_years

    def test_paper_count_correct(self, populated_db):
        for _, row in query("SELECT year, paper_count FROM yearly_summary").iterrows():
            actual = query(
                "SELECT COUNT(*) as n FROM papers WHERE year = ?", (int(row["year"]),)
            ).iloc[0]["n"]
            assert int(row["paper_count"]) == int(actual)

    def test_avg_citations_non_negative(self, populated_db):
        df = query("SELECT avg_citations FROM yearly_summary")
        assert (df["avg_citations"] >= 0).all()

    def test_open_access_pct_between_0_and_100(self, populated_db):
        df = query("SELECT open_access_pct FROM yearly_summary")
        assert (df["open_access_pct"] >= 0).all()
        assert (df["open_access_pct"] <= 100).all()

    def test_summary_refreshes_on_rerun(self, tmp_db, clean_df):
        """Second store() call should not duplicate yearly_summary rows."""
        run1 = start_run()
        store(clean_df, run1)
        finish_run(run1, "success")

        run2 = start_run()
        store(clean_df, run2)
        finish_run(run2, "success")

        count = query("SELECT COUNT(*) as n FROM yearly_summary").iloc[0]["n"]
        # Should equal number of distinct years, not 2x
        distinct_years = query("SELECT COUNT(DISTINCT year) as n FROM papers").iloc[0]["n"]
        assert count == distinct_years


# ── store() integration ────────────────────────────────────────────────────────

class TestStoreIntegration:
    def test_store_returns_tuple(self, tmp_db, clean_df):
        run_id = start_run()
        result = store(clean_df, run_id)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_store_empty_df_returns_zeros(self, tmp_db):
        run_id = start_run()
        ins, skip = store(pd.DataFrame(), run_id)
        assert ins == 0
        assert skip == 0

    def test_total_papers_in_db_matches_inserted(self, tmp_db, clean_df):
        run_id = start_run()
        ins, _ = store(clean_df, run_id)
        count = query("SELECT COUNT(*) as n FROM papers").iloc[0]["n"]
        assert count == ins

    def test_idempotent_across_three_runs(self, tmp_db, clean_df):
        for i in range(3):
            run_id = start_run()
            ins, skip = store(clean_df, run_id)
            finish_run(run_id, "success")
            if i == 0:
                assert ins == len(clean_df)
            else:
                assert ins == 0
                assert skip == len(clean_df)
        total = query("SELECT COUNT(*) as n FROM papers").iloc[0]["n"]
        assert total == len(clean_df)
