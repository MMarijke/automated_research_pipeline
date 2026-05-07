# Research Data Pipeline

An end-to-end automated data pipeline that ingests, cleans, stores, and visualises academic research paper data. Built in production style with modular stages, structured logging, error handling, and scheduled execution.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR  (main.py)                     │
│          ingest → clean → store → visualize                     │
│          Scheduler loop / mock-cron / single-shot mode          │
└───────────┬─────────────┬────────────┬───────────────┬──────────┘
            │             │            │               │
     ┌──────▼──────┐ ┌────▼────┐ ┌────▼────┐  ┌──────▼──────┐
     │  INGEST     │ │  CLEAN  │ │  STORE  │  │  VISUALIZE  │
     │  ingest.py  │ │ clean.py│ │ store.py│  │ visualize.py│
     └──────┬──────┘ └────┬────┘ └────┬────┘  └──────┬──────┘
            │             │            │               │
     OpenAlex API   Validation    SQLite DB      HTML Dashboard
     + fallback     Dedup         (3 tables)     + CSV Reports
     synthetic      Features
     dataset        Engineering
```

---

## Project Structure

```
research_pipeline/
│
├── main.py                  # Orchestrator + scheduler
├── config.py                # Centralised settings
├── logger.py                # Shared structured logger
│
├── pipeline/
│   ├── __init__.py
│   ├── ingest.py            # Stage 1: Data ingestion
│   ├── clean.py             # Stage 2: Cleaning & transformation
│   ├── store.py             # Stage 3: SQLite storage
│   └── visualize.py         # Stage 4: Reports + dashboard
│
├── data/
│   └── cleaned_snapshot.csv # Debug snapshot after each clean run
│
├── reports/
│   ├── dashboard.html        # Self-contained HTML dashboard
│   ├── yearly_trend.csv
│   ├── top_venues.csv
│   ├── top_concepts.csv
│   ├── impact_distribution.csv
│   └── pipeline_runs.csv
│
├── logs/
│   └── pipeline_YYYYMMDD.log # Daily rotating log file
│
├── tests/
│   ├── conftest.py          # Shared fixtures and sample data
│   ├── test_ingest.py       # Ingest unit + integration tests
│   ├── test_clean.py        # Clean unit + integration tests
│   ├── test_store.py        # Store unit + integration tests
│   └── test_visualize_and_main.py  # Visualize + orchestrator tests
│
└── research.db              # SQLite database (runtime, not committed)
```

---

## Pipeline Stages

### Stage 1 — Ingest (`pipeline/ingest.py`)

Fetches bibliographic records from the **OpenAlex REST API** (open, no key required).

- Queries multiple configurable search terms sequentially (one per term, polite rate-limiting between calls)
- Flattens nested JSON (authors, venue, concepts) into tabular rows
- **Graceful fallback**: if the API is unreachable or returns no data, a realistic 150-record synthetic dataset is generated using a power-law citation distribution
- Returns a raw `pd.DataFrame`

### Stage 2 — Clean (`pipeline/clean.py`)

Applies a deterministic, ordered cleaning sequence:

| Step | Action |
|------|--------|
| Schema validation | Ensures all expected columns exist; drops unknown extras |
| Type coercion | Casts to nullable Int64 / bool / str with error→NA handling |
| Deduplication | By `openalex_id` first, then normalised title + year |
| Invalid-row removal | Drops rows with missing title, implausible year, negative citations |
| Imputation | Fills remaining NAs with domain-appropriate defaults |
| Feature engineering | Adds `paper_age`, `citations_per_year`, `impact_tier`, `decade`, `is_collaborative`, `title_word_count` |

Saves a `cleaned_snapshot.csv` after each run for reproducibility.

### Stage 3 — Store (`pipeline/store.py`)

Manages a **SQLite** database with three tables:

| Table | Purpose |
|-------|---------|
| `papers` | Core cleaned records (primary store) |
| `pipeline_runs` | Audit log of every execution with status + row counts |
| `yearly_summary` | Pre-aggregated yearly stats (materialised view pattern) |

Key design decisions:
- **Idempotent upserts** via SHA-1 row hashing — re-running the pipeline never produces duplicates
- WAL journal mode for concurrent read safety
- All indexes defined explicitly for query performance
- `yearly_summary` is fully refreshed on every run (cheap at this scale)

### Stage 4 — Visualize (`pipeline/visualize.py`)

Generates outputs from the database:

**CSV Reports:**
- `yearly_trend.csv` — paper count, avg citations, open access % per year
- `top_venues.csv` — ranked publication venues
- `top_concepts.csv` — ranked research concepts  
- `impact_distribution.csv` — citation tier breakdown
- `pipeline_runs.csv` — execution history

**HTML Dashboard** (`reports/dashboard.html`):
- Fully self-contained (no server needed — open in any browser)
- 5 interactive Chart.js charts: annual trend (dual-axis bar+line), top venues (horizontal bar), concept distribution (doughnut), impact tiers, open-access trend
- Summary stat cards and data tables
- Dark theme, responsive layout

---

## Database Schema

```sql
-- Core data table
CREATE TABLE papers (
    row_hash          TEXT PRIMARY KEY,   -- SHA-1 of id|title|year
    openalex_id       TEXT NOT NULL,
    title             TEXT NOT NULL,
    year              INTEGER,
    citations         INTEGER,
    author_count      INTEGER,
    authors           TEXT,               -- semicolon-delimited, first 5
    venue             TEXT,
    open_access       INTEGER,            -- 0/1
    top_concept       TEXT,
    search_term       TEXT,
    paper_age         INTEGER,            -- derived
    citations_per_year REAL,             -- derived
    impact_tier       TEXT,              -- Uncited/Low/Medium/High/Highly Cited
    decade            TEXT,              -- e.g. '2020s'
    is_collaborative  INTEGER,           -- 0/1
    title_word_count  INTEGER,
    ingested_at       TEXT,
    stored_at         TEXT
);

-- Execution audit log
CREATE TABLE pipeline_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT,
    finished_at   TEXT,
    status        TEXT,   -- running / success / failed / aborted
    rows_ingested INTEGER,
    rows_stored   INTEGER,
    rows_skipped  INTEGER,
    error_message TEXT,
    metadata      TEXT    -- JSON blob
);

-- Pre-aggregated analytics
CREATE TABLE yearly_summary (
    year             INTEGER PRIMARY KEY,
    paper_count      INTEGER,
    avg_citations    REAL,
    median_citations REAL,
    avg_authors      REAL,
    open_access_pct  REAL,
    top_concept      TEXT,
    top_venue        TEXT,
    refreshed_at     TEXT
);
```

---

## Installation & Usage

### Requirements

```
Python 3.8+  (tested on 3.12)
pandas >= 1.0
requests
```

Install dependencies:
```bash
pip install pandas requests
```

### Run once
```bash
python main.py --once
```

### Run scheduled loop (default: every 60 s, 3 iterations)
```bash
python main.py
```

### Custom schedule
```bash
python main.py --interval 120 --max-runs 10   # every 2 min, 10 runs
python main.py --interval 3600 --max-runs 0   # every hour, run forever
```

### View the dashboard
```bash
open reports/dashboard.html       # macOS
xdg-open reports/dashboard.html  # Linux
start reports/dashboard.html      # Windows
```

### Query the database directly
```python
import sqlite3, pandas as pd

conn = sqlite3.connect("research.db")
df = pd.read_sql("SELECT * FROM yearly_summary ORDER BY year", conn)
print(df)
```

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `SEARCH_TERMS` | `["machine learning", ...]` | Terms to query OpenAlex for |
| `RECORDS_PER_TERM` | `25` | API results per term |
| `PIPELINE_INTERVAL_SECONDS` | `60` | Scheduler cadence |
| `MAX_RUNS` | `3` | Scheduler iterations (0 = ∞) |
| `TOP_N_VENUES` | `10` | Venues shown in reports |

---

## Error Handling & Logging

- Every stage is wrapped in try/except; failures update `pipeline_runs` with status `"failed"` and the error message
- The ingest stage has a **two-level fallback**: API → synthetic data
- All log output goes to both stdout (INFO+) and a daily-rotating file in `logs/` (DEBUG+)
- Log format: `TIMESTAMP  LEVEL  [module]  message`

---

## Running the Tests

Install pytest, then run from the project root:

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
```

With coverage report:
```bash
python -m pytest tests/ --cov=pipeline --cov=main --cov-report=term-missing
```

The suite covers 99 tests across all four pipeline stages at ~90% line coverage.
Each test module uses isolated temporary databases via pytest fixtures — no
shared state between tests.

---

## Extending the Pipeline

**Add a new data source:** implement a function in `ingest.py` that returns a DataFrame matching `EXPECTED_COLS` in `clean.py`.

**Add a new feature:** add a column derivation step inside `_add_features()` in `clean.py` and add the column to `_PAPER_COLS` in `store.py`.

**Add a new report:** add a `_my_report()` function in `visualize.py` and call it from `visualize()`.

**Production scheduling:** replace the mock loop in `main.py` with a real cron entry or a task scheduler (APScheduler, Airflow, Prefect):
```
# crontab example — run every hour
0 * * * * cd /path/to/research_pipeline && python main.py --once
```
