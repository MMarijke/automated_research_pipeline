"""
main.py
───────
Pipeline Orchestrator

Wires together: ingest → clean → store → visualize
and simulates cron scheduling via a configurable run loop.

Usage
─────
    # Single run
    python main.py --once

    # Scheduled loop (default: every 60 s, 3 iterations)
    python main.py

    # Custom schedule
    python main.py --interval 120 --max-runs 5
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── Make sure project root is on sys.path ──────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipeline"))

from config import PIPELINE_INTERVAL_SECONDS, MAX_RUNS
from logger import get_logger
from pipeline.ingest    import ingest
from pipeline.clean     import clean
from pipeline.store     import store, start_run, finish_run
from pipeline.visualize import visualize

log = get_logger("orchestrator")

# ── ASCII banner ───────────────────────────────────────────────────────────────
BANNER = r"""
 ____                                 _      _____  _             _ _
|  _ \ ___  ___  ___  __ _ _ __ ___| |__  |  __ \(_)_ __   ___| (_)_ __   ___
| |_) / _ \/ __|/ _ \/ _` | '__/ __| '_ \ | |__) | | '_ \ / _ \ | | '_ \ / _ \
|  _ <  __/\__ \  __/ (_| | | | (__| | | ||  ___/| | |_) |  __/ | | | | |  __/
|_| \_\___||___/\___|\__,_|_|  \___|_| |_||_|    |_| .__/ \___|_|_|_| |_|\___|
                                                     |_|
"""


# ── Single pipeline run ────────────────────────────────────────────────────────

def run_pipeline(run_number: int = 1) -> bool:
    """
    Execute one complete ingest → clean → store → visualize cycle.

    Returns True on success, False on failure.
    """
    sep = "─" * 60
    log.info(sep)
    log.info("PIPELINE RUN #%d  started at %s", run_number, datetime.now(timezone.utc).isoformat())
    log.info(sep)

    run_id = start_run(meta={"run_number": run_number})
    rows_ingested = rows_stored = rows_skipped = 0

    try:
        # ── Stage 1: Ingest ────────────────────────────────────────────────────
        raw_df = ingest()
        rows_ingested = len(raw_df)

        if raw_df.empty:
            log.warning("Ingest returned empty DataFrame — aborting run.")
            finish_run(run_id, "aborted", 0, 0, 0, "Empty ingest result")
            return False

        # ── Stage 2: Clean ─────────────────────────────────────────────────────
        clean_df = clean(raw_df)

        # ── Stage 3: Store ─────────────────────────────────────────────────────
        rows_stored, rows_skipped = store(clean_df, run_id)

        # ── Stage 4: Visualize ─────────────────────────────────────────────────
        dashboard_path = visualize()

        finish_run(run_id, "success", rows_ingested, rows_stored, rows_skipped)

        log.info(sep)
        log.info(
            "RUN #%d COMPLETE — ingested=%d  stored=%d  skipped=%d",
            run_number, rows_ingested, rows_stored, rows_skipped,
        )
        log.info("Dashboard → %s", dashboard_path)
        log.info(sep)
        return True

    except KeyboardInterrupt:
        log.info("Run #%d interrupted by user.", run_number)
        finish_run(run_id, "interrupted", rows_ingested, rows_stored, rows_skipped)
        raise

    except Exception as exc:
        tb = traceback.format_exc()
        log.error("RUN #%d FAILED: %s\n%s", run_number, exc, tb)
        finish_run(run_id, "failed", rows_ingested, rows_stored, rows_skipped, str(exc))
        return False


# ── Scheduler loop ─────────────────────────────────────────────────────────────

def run_scheduler(interval: int, max_runs: int) -> None:
    """
    Mock-cron loop — runs the pipeline repeatedly.

    Args:
        interval:  Seconds between runs.
        max_runs:  0 = run forever; N = stop after N successful+failed runs.
    """
    log.info("Scheduler started | interval=%ds | max_runs=%s",
             interval, max_runs if max_runs else "∞")

    run_number  = 0
    successes   = 0
    failures    = 0

    try:
        while True:
            run_number += 1
            success = run_pipeline(run_number)
            if success:
                successes += 1
            else:
                failures += 1

            if max_runs and run_number >= max_runs:
                log.info("Reached max_runs=%d — scheduler stopping.", max_runs)
                break

            next_run = datetime.now(timezone.utc).strftime("%H:%M:%S")
            log.info(
                "Next run in %ds  (successes=%d  failures=%d) …",
                interval, successes, failures,
            )
            time.sleep(interval)

    except KeyboardInterrupt:
        log.info("Scheduler stopped by user. Runs: %d success / %d failed.",
                 successes, failures)

    log.info("Scheduler finished. Total runs: %d", run_number)


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Research Data Pipeline Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run the pipeline exactly once and exit.",
    )
    parser.add_argument(
        "--interval", type=int, default=PIPELINE_INTERVAL_SECONDS,
        help="Seconds between scheduled runs.",
    )
    parser.add_argument(
        "--max-runs", type=int, default=MAX_RUNS,
        help="Maximum number of runs (0 = infinite).",
    )
    return parser.parse_args()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(BANNER)
    args = _parse_args()

    if args.once:
        success = run_pipeline(run_number=1)
        sys.exit(0 if success else 1)
    else:
        run_scheduler(interval=args.interval, max_runs=args.max_runs)
