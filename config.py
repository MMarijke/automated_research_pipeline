"""
Pipeline configuration — centralised settings for all modules.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"
LOGS_DIR    = BASE_DIR / "logs"
DB_PATH     = BASE_DIR / "research.db"

# ── Ingestion ──────────────────────────────────────────────────────────────────
# OpenAlex API — open bibliographic data, no key required
OPENALEX_BASE = "https://api.openalex.org"
SEARCH_TERMS  = ["machine learning", "deep learning", "transformer architecture"]
RECORDS_PER_TERM = 25          # results fetched per search term
REQUEST_TIMEOUT  = 15          # seconds

# ── Scheduling ─────────────────────────────────────────────────────────────────
PIPELINE_INTERVAL_SECONDS = 60  # mock-cron cadence for the scheduler loop
MAX_RUNS = 3                    # 0 = run forever

# ── Reporting ──────────────────────────────────────────────────────────────────
TOP_N_AUTHORS     = 10
TOP_N_VENUES      = 10
YEARLY_TREND_COLS = ["year", "paper_count", "avg_citations", "avg_authors"]
