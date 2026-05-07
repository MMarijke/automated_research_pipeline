"""
tests/conftest.py
─────────────────
Shared pytest fixtures for the research pipeline test suite.
"""

import sys
import pytest
import tempfile
import sqlite3
import pandas as pd
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))


# ── Minimal valid raw DataFrame (mirrors ingest output schema) ─────────────────
RAW_ROWS = [
    {
        "openalex_id":  "W001",
        "title":        "Deep Learning for Image Recognition",
        "year":         2020,
        "citations":    350,
        "author_count": 3,
        "authors":      "A. Smith; B. Jones; C. Lee",
        "venue":        "NeurIPS",
        "open_access":  True,
        "top_concept":  "Neural Networks",
        "search_term":  "deep learning",
        "ingested_at":  "2024-01-01T00:00:00",
    },
    {
        "openalex_id":  "W002",
        "title":        "Transformer Architecture Survey",
        "year":         2021,
        "citations":    80,
        "author_count": 2,
        "authors":      "D. Brown; E. White",
        "venue":        "ICML",
        "open_access":  False,
        "top_concept":  "Attention Mechanism",
        "search_term":  "transformer architecture",
        "ingested_at":  "2024-01-01T00:00:00",
    },
    {
        "openalex_id":  "W003",
        "title":        "Reinforcement Learning in Robotics",
        "year":         2019,
        "citations":    5,
        "author_count": 1,
        "authors":      "F. Black",
        "venue":        "ICLR",
        "open_access":  True,
        "top_concept":  "Reinforcement Learning",
        "search_term":  "machine learning",
        "ingested_at":  "2024-01-01T00:00:00",
    },
    {
        "openalex_id":  "W004",
        "title":        "Graph Neural Networks: A Review",
        "year":         2022,
        "citations":    0,
        "author_count": 4,
        "authors":      "G. Green; H. Hill; I. Ing; J. Jacks",
        "venue":        "Nature Machine Intelligence",
        "open_access":  True,
        "top_concept":  "Graph Neural Networks",
        "search_term":  "deep learning",
        "ingested_at":  "2024-01-01T00:00:00",
    },
    {
        "openalex_id":  "W005",
        "title":        "Convolutional Networks for NLP",
        "year":         2018,
        "citations":    12,
        "author_count": 2,
        "authors":      "K. King; L. Lane",
        "venue":        "ACL",
        "open_access":  False,
        "top_concept":  "Convolutional Networks",
        "search_term":  "deep learning",
        "ingested_at":  "2024-01-01T00:00:00",
    },
]


@pytest.fixture
def raw_df():
    """Minimal valid raw DataFrame (as returned by ingest())."""
    return pd.DataFrame(RAW_ROWS)


@pytest.fixture
def clean_df(raw_df):
    """Cleaned DataFrame ready for storage."""
    from pipeline.clean import clean
    return clean(raw_df)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """
    Isolated in-memory SQLite path for each test.
    Monkeypatches config.DB_PATH so store.py uses the temp file.
    """
    import config
    db_file = tmp_path / "test_research.db"
    monkeypatch.setattr(config, "DB_PATH", db_file)

    # Also patch the store module's reference
    import pipeline.store as store_mod
    monkeypatch.setattr(store_mod, "DB_PATH", db_file)

    return db_file


@pytest.fixture
def populated_db(tmp_db, clean_df):
    """Temp DB already populated with 5 clean rows."""
    from pipeline.store import start_run, store, finish_run
    run_id = start_run({"fixture": True})
    ins, skip = store(clean_df, run_id)
    finish_run(run_id, "success", len(clean_df), ins, skip)
    return tmp_db
