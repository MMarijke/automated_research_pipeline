"""
tests/test_ingest.py
────────────────────
Unit tests for pipeline/ingest.py
"""

import sys
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))

from pipeline.ingest import (
    _generate_synthetic,
    _parse_work,
    _fetch_from_api,
    ingest,
    SEARCH_TERMS,
)


# ── _generate_synthetic ────────────────────────────────────────────────────────

class TestGenerateSynthetic:
    def test_returns_correct_row_count(self):
        df = _generate_synthetic(50)
        assert len(df) == 50

    def test_default_row_count(self):
        df = _generate_synthetic()
        assert len(df) == 150

    def test_has_all_required_columns(self):
        df = _generate_synthetic(10)
        required = {
            "openalex_id", "title", "year", "citations", "author_count",
            "authors", "venue", "open_access", "top_concept",
            "search_term", "ingested_at",
        }
        assert required.issubset(set(df.columns))

    def test_years_in_valid_range(self):
        df = _generate_synthetic(100)
        assert df["year"].between(2015, 2024).all()

    def test_citations_non_negative(self):
        df = _generate_synthetic(100)
        assert (df["citations"] >= 0).all()

    def test_search_terms_from_config(self):
        df = _generate_synthetic(100)
        assert set(df["search_term"].unique()).issubset(set(SEARCH_TERMS))

    def test_openalex_ids_unique(self):
        df = _generate_synthetic(100)
        assert df["openalex_id"].nunique() == 100

    def test_open_access_is_boolean(self):
        df = _generate_synthetic(20)
        assert df["open_access"].dtype == bool

    def test_reproducible_with_same_seed(self):
        """Random data (papers, years, citations) is seeded — exclude ingested_at which is a live timestamp."""
        df1 = _generate_synthetic(30).drop(columns=["ingested_at"])
        df2 = _generate_synthetic(30).drop(columns=["ingested_at"])
        pd.testing.assert_frame_equal(df1, df2)

    def test_zero_rows(self):
        df = _generate_synthetic(0)
        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)


# ── _parse_work ────────────────────────────────────────────────────────────────

class TestParseWork:
    def _make_raw_work(self, **overrides):
        base = {
            "id": "https://openalex.org/W123",
            "title": "Test Paper Title",
            "publication_year": 2022,
            "cited_by_count": 42,
            "authorships": [
                {"author": {"display_name": "Alice Smith"}},
                {"author": {"display_name": "Bob Jones"}},
            ],
            "primary_location": {
                "source": {"display_name": "NeurIPS"}
            },
            "open_access": {"is_oa": True},
            "concepts": [
                {"display_name": "Neural Networks", "score": 0.9},
                {"display_name": "Backpropagation", "score": 0.4},
            ],
        }
        base.update(overrides)
        return base

    def test_basic_fields_parsed(self):
        row = _parse_work(self._make_raw_work(), "deep learning")
        assert row["title"] == "Test Paper Title"
        assert row["year"] == 2022
        assert row["citations"] == 42
        assert row["venue"] == "NeurIPS"
        assert row["open_access"] is True
        assert row["top_concept"] == "Neural Networks"
        assert row["search_term"] == "deep learning"

    def test_author_count_correct(self):
        row = _parse_work(self._make_raw_work(), "ml")
        assert row["author_count"] == 2

    def test_top_concept_highest_score(self):
        raw = self._make_raw_work(concepts=[
            {"display_name": "Low Score", "score": 0.1},
            {"display_name": "High Score", "score": 0.95},
        ])
        row = _parse_work(raw, "ml")
        assert row["top_concept"] == "High Score"

    def test_missing_primary_location(self):
        raw = self._make_raw_work(primary_location=None)
        row = _parse_work(raw, "ml")
        assert row["venue"] == "Unknown"

    def test_missing_concepts(self):
        raw = self._make_raw_work(concepts=[])
        row = _parse_work(raw, "ml")
        assert row["top_concept"] == "Unknown"

    def test_no_authors(self):
        raw = self._make_raw_work(authorships=[])
        row = _parse_work(raw, "ml")
        assert row["author_count"] == 0
        assert row["authors"] == ""

    def test_max_five_authors_stored(self):
        raw = self._make_raw_work(authorships=[
            {"author": {"display_name": f"Author {i}"}} for i in range(10)
        ])
        row = _parse_work(raw, "ml")
        assert len(row["authors"].split("; ")) == 5

    def test_ingested_at_is_string(self):
        row = _parse_work(self._make_raw_work(), "ml")
        assert isinstance(row["ingested_at"], str)
        assert "T" in row["ingested_at"]  # ISO format


# ── ingest() (integration with mocking) ───────────────────────────────────────

class TestIngest:
    def test_returns_dataframe(self):
        df = ingest()
        assert isinstance(df, pd.DataFrame)

    def test_returns_non_empty(self):
        df = ingest()
        assert len(df) > 0

    def test_has_required_columns(self):
        df = ingest()
        required = {"openalex_id", "title", "year", "citations", "search_term"}
        assert required.issubset(set(df.columns))

    def test_falls_back_to_synthetic_on_api_failure(self):
        """When API raises, ingest() should return synthetic data."""
        import requests
        with patch("pipeline.ingest.requests.get") as mock_get:
            mock_get.side_effect = requests.ConnectionError("Network unreachable")
            df = ingest()
        assert len(df) == 150
        assert all(id_.startswith("synthetic_") for id_ in df["openalex_id"])

    def test_falls_back_when_api_returns_empty(self):
        """When API returns 0 results for all terms, use synthetic fallback."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"results": []}
        with patch("pipeline.ingest.requests.get", return_value=mock_resp):
            df = ingest()
        assert len(df) > 0
        assert all(id_.startswith("synthetic_") for id_ in df["openalex_id"])
