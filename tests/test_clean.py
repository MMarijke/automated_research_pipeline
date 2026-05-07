"""
tests/test_clean.py
────────────────────
Unit tests for pipeline/clean.py
"""

import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from pipeline.clean import (
    clean,
    _validate_schema,
    _coerce_types,
    _remove_duplicates,
    _filter_invalid,
    _impute_missing,
    _add_features,
    EXPECTED_COLS,
)


# ── Schema validation ──────────────────────────────────────────────────────────

class TestValidateSchema:
    def test_passes_complete_schema(self, raw_df):
        result = _validate_schema(raw_df)
        for col in EXPECTED_COLS:
            assert col in result.columns

    def test_adds_missing_columns_as_na(self):
        df = pd.DataFrame([{"title": "Test", "year": 2020}])
        result = _validate_schema(df)
        assert "openalex_id" in result.columns
        assert result["openalex_id"].isna().all()

    def test_drops_extra_columns(self, raw_df):
        df = raw_df.copy()
        df["unexpected_col"] = "noise"
        result = _validate_schema(df)
        assert "unexpected_col" not in result.columns

    def test_returns_canonical_column_order(self, raw_df):
        result = _validate_schema(raw_df)
        assert list(result.columns) == list(EXPECTED_COLS.keys())


# ── Type coercion ──────────────────────────────────────────────────────────────

class TestCoerceTypes:
    def test_year_becomes_int64(self, raw_df):
        df = _validate_schema(raw_df)
        result = _coerce_types(df)
        assert str(result["year"].dtype) == "Int64"

    def test_citations_becomes_int64(self, raw_df):
        df = _validate_schema(raw_df)
        result = _coerce_types(df)
        assert str(result["citations"].dtype) == "Int64"

    def test_non_numeric_year_becomes_na(self):
        df = pd.DataFrame([{
            "openalex_id": "X", "title": "T", "year": "not_a_year",
            "citations": 0, "author_count": 1, "authors": "A",
            "venue": "V", "open_access": False, "top_concept": "C",
            "search_term": "ml", "ingested_at": "2024-01-01"
        }])
        result = _coerce_types(df)
        assert pd.isna(result["year"].iloc[0])

    def test_open_access_string_true_converts(self):
        df = pd.DataFrame([{
            "openalex_id": "X", "title": "T", "year": 2020,
            "citations": 0, "author_count": 1, "authors": "A",
            "venue": "V", "open_access": "True", "top_concept": "C",
            "search_term": "ml", "ingested_at": "2024-01-01"
        }])
        result = _coerce_types(df)
        assert result["open_access"].iloc[0] == True

    def test_empty_string_becomes_na(self):
        df = pd.DataFrame([{
            "openalex_id": "X", "title": "Test", "year": 2020,
            "citations": 0, "author_count": 1, "authors": "A",
            "venue": "", "open_access": False, "top_concept": "C",
            "search_term": "ml", "ingested_at": "2024-01-01"
        }])
        result = _coerce_types(df)
        assert pd.isna(result["venue"].iloc[0])


# ── Deduplication ──────────────────────────────────────────────────────────────

class TestRemoveDuplicates:
    def test_removes_exact_duplicate_ids(self, raw_df):
        df = _validate_schema(pd.concat([raw_df, raw_df]).reset_index(drop=True))
        df = _coerce_types(df)
        result = _remove_duplicates(df)
        assert len(result) == len(raw_df)

    def test_keeps_different_ids(self, raw_df):
        df = _validate_schema(raw_df)
        df = _coerce_types(df)
        result = _remove_duplicates(df)
        assert len(result) == len(raw_df)

    def test_deduplicates_by_title_and_year(self):
        rows = [
            {"openalex_id": "A1", "title": "Same Title", "year": 2020,
             "citations": 10, "author_count": 1, "authors": "X",
             "venue": "V", "open_access": False, "top_concept": "C",
             "search_term": "ml", "ingested_at": "2024-01-01"},
            {"openalex_id": "A2", "title": "Same Title", "year": 2020,
             "citations": 99, "author_count": 1, "authors": "Y",
             "venue": "V", "open_access": False, "top_concept": "C",
             "search_term": "ml", "ingested_at": "2024-01-01"},
        ]
        df = _coerce_types(_validate_schema(pd.DataFrame(rows)))
        result = _remove_duplicates(df)
        assert len(result) == 1
        assert result["openalex_id"].iloc[0] == "A1"  # keeps first


# ── Invalid row filtering ──────────────────────────────────────────────────────

class TestFilterInvalid:
    def _make_coerced(self, rows):
        return _coerce_types(_validate_schema(pd.DataFrame(rows)))

    def test_removes_empty_title(self):
        rows = [{"openalex_id": "X", "title": "", "year": 2020,
                 "citations": 5, "author_count": 1, "authors": "A",
                 "venue": "V", "open_access": False, "top_concept": "C",
                 "search_term": "ml", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 0

    def test_removes_implausible_year_too_old(self):
        rows = [{"openalex_id": "X", "title": "Old Paper", "year": 1800,
                 "citations": 5, "author_count": 1, "authors": "A",
                 "venue": "V", "open_access": False, "top_concept": "C",
                 "search_term": "ml", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 0

    def test_removes_future_year(self):
        rows = [{"openalex_id": "X", "title": "Future Paper", "year": 2099,
                 "citations": 0, "author_count": 1, "authors": "A",
                 "venue": "V", "open_access": False, "top_concept": "C",
                 "search_term": "ml", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 0

    def test_removes_negative_citations(self):
        rows = [{"openalex_id": "X", "title": "Bad Cites", "year": 2020,
                 "citations": -1, "author_count": 1, "authors": "A",
                 "venue": "V", "open_access": False, "top_concept": "C",
                 "search_term": "ml", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 0

    def test_keeps_valid_row(self):
        rows = [{"openalex_id": "X", "title": "Valid Paper", "year": 2020,
                 "citations": 10, "author_count": 2, "authors": "A; B",
                 "venue": "NeurIPS", "open_access": True, "top_concept": "ML",
                 "search_term": "deep learning", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 1

    def test_zero_citations_kept(self):
        rows = [{"openalex_id": "X", "title": "Uncited Paper", "year": 2023,
                 "citations": 0, "author_count": 1, "authors": "A",
                 "venue": "V", "open_access": False, "top_concept": "C",
                 "search_term": "ml", "ingested_at": "2024-01-01"}]
        result = _filter_invalid(self._make_coerced(rows))
        assert len(result) == 1


# ── Feature engineering ────────────────────────────────────────────────────────

class TestAddFeatures:
    def test_paper_age_calculated(self, raw_df):
        from pipeline.clean import CURRENT_YEAR
        df = clean(raw_df)
        for _, row in df.iterrows():
            assert row["paper_age"] == CURRENT_YEAR - int(row["year"])

    def test_citations_per_year_positive(self, raw_df):
        df = clean(raw_df)
        assert (df["citations_per_year"] >= 0).all()

    def test_impact_tier_uncited(self, raw_df):
        df = clean(raw_df)
        uncited = df[df["citations"] == 0]
        assert (uncited["impact_tier"] == "Uncited").all()

    def test_impact_tier_highly_cited(self, raw_df):
        df = clean(raw_df)
        high = df[df["citations"] > 200]
        assert (high["impact_tier"] == "Highly Cited").all()

    def test_decade_format(self, raw_df):
        df = clean(raw_df)
        assert df["decade"].str.match(r"^\d{4}s$").all()

    def test_is_collaborative_single_author(self, raw_df):
        df = clean(raw_df)
        solo = df[df["author_count"] == 1]
        assert (solo["is_collaborative"] == False).all()

    def test_is_collaborative_multi_author(self, raw_df):
        df = clean(raw_df)
        multi = df[df["author_count"] > 1]
        assert (multi["is_collaborative"] == True).all()

    def test_title_word_count_positive(self, raw_df):
        df = clean(raw_df)
        assert (df["title_word_count"] > 0).all()


# ── Full clean() integration ───────────────────────────────────────────────────

class TestCleanIntegration:
    def test_returns_dataframe(self, raw_df):
        assert isinstance(clean(raw_df), pd.DataFrame)

    def test_no_nulls_in_output(self, raw_df):
        df = clean(raw_df)
        assert df.isnull().sum().sum() == 0

    def test_empty_input_returns_empty(self):
        result = clean(pd.DataFrame())
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_all_invalid_returns_empty(self):
        bad = pd.DataFrame([{
            "openalex_id": "X", "title": "", "year": 1800,
            "citations": -5, "author_count": 1, "authors": "A",
            "venue": "V", "open_access": False, "top_concept": "C",
            "search_term": "ml", "ingested_at": "2024-01-01"
        }])
        result = clean(bad)
        assert len(result) == 0

    def test_output_row_count_lte_input(self, raw_df):
        result = clean(raw_df)
        assert len(result) <= len(raw_df)

    def test_derived_columns_present(self, raw_df):
        df = clean(raw_df)
        derived = ["paper_age", "citations_per_year", "impact_tier",
                   "decade", "is_collaborative", "title_word_count"]
        for col in derived:
            assert col in df.columns, f"Missing derived col: {col}"

    def test_idempotent(self, raw_df):
        """Cleaning the same data twice gives same result."""
        df1 = clean(raw_df)
        df2 = clean(raw_df)
        pd.testing.assert_frame_equal(df1.reset_index(drop=True),
                                      df2.reset_index(drop=True))
