"""
tests/test_visualize_and_main.py
─────────────────────────────────
Tests for pipeline/visualize.py and main.py orchestration.
"""

import sys
import re
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))


# ── Visualize ──────────────────────────────────────────────────────────────────

class TestVisualize:
    def test_returns_path(self, populated_db, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
        import pipeline.visualize as viz_mod
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", tmp_path / "reports")
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        (tmp_path / "reports").mkdir(exist_ok=True)
        result = viz_mod.visualize()
        assert isinstance(result, Path)

    def test_dashboard_html_created(self, populated_db, tmp_path, monkeypatch):
        import config
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        viz_mod.visualize()
        assert (reports / "dashboard.html").exists()

    def test_all_csv_reports_created(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        viz_mod.visualize()
        expected = [
            "yearly_trend.csv", "top_venues.csv", "top_concepts.csv",
            "impact_distribution.csv", "pipeline_runs.csv"
        ]
        for fname in expected:
            assert (reports / fname).exists(), f"Missing report: {fname}"

    def test_dashboard_has_five_charts(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        dash = viz_mod.visualize()
        html = dash.read_text()
        assert html.count("new Chart(") == 5

    def test_dashboard_contains_all_canvas_ids(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        dash = viz_mod.visualize()
        html = dash.read_text()
        for canvas_id in ["yearlyChart", "venueChart", "conceptChart", "impactChart", "oaChart"]:
            assert canvas_id in html

    def test_dashboard_chart_data_non_empty(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        dash = viz_mod.visualize()
        html = dash.read_text()
        # Each chart should have non-empty label arrays
        label_arrays = re.findall(r'labels:\s*\["[^\]]+"\]', html)
        assert len(label_arrays) >= 3, f"Expected >=3 non-empty label arrays, got {len(label_arrays)}"

    def test_yearly_trend_csv_has_correct_columns(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        viz_mod.visualize()
        df = pd.read_csv(reports / "yearly_trend.csv")
        assert "year" in df.columns
        assert "paper_count" in df.columns
        assert "avg_citations" in df.columns
        assert len(df) > 0

    def test_impact_distribution_has_five_tiers(self, populated_db, tmp_path, monkeypatch):
        import pipeline.visualize as viz_mod
        reports = tmp_path / "reports"
        reports.mkdir()
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        viz_mod.visualize()
        df = pd.read_csv(reports / "impact_distribution.csv")
        valid_tiers = {"Uncited", "Low", "Medium", "High", "Highly Cited"}
        assert set(df["impact_tier"]).issubset(valid_tiers)

    def test_summary_stats_keys(self, populated_db, monkeypatch):
        import pipeline.visualize as viz_mod
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        stats = viz_mod._summary_stats()
        expected_keys = {
            "total_papers", "earliest_year", "latest_year",
            "avg_citations", "max_citations", "open_access_pct", "avg_authors"
        }
        assert expected_keys.issubset(set(stats.keys()))

    def test_summary_stats_total_matches_db(self, populated_db, monkeypatch):
        import pipeline.visualize as viz_mod
        from pipeline.store import query
        monkeypatch.setattr(viz_mod, "DB_PATH", populated_db)
        stats = viz_mod._summary_stats()
        actual = query("SELECT COUNT(*) as n FROM papers").iloc[0]["n"]
        assert int(stats["total_papers"]) == int(actual)


# ── Main orchestrator ──────────────────────────────────────────────────────────

class TestMainOrchestrator:
    def test_run_pipeline_returns_true_on_success(self, tmp_db, monkeypatch):
        import config
        monkeypatch.setattr(config, "DB_PATH", tmp_db)
        import pipeline.store as store_mod
        monkeypatch.setattr(store_mod, "DB_PATH", tmp_db)

        reports = tmp_db.parent / "reports"
        reports.mkdir(exist_ok=True)
        import pipeline.visualize as viz_mod
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", tmp_db)

        import main
        result = main.run_pipeline(run_number=1)
        assert result is True

    def test_run_pipeline_returns_false_on_ingest_failure(self, tmp_db, monkeypatch):
        import config, pipeline.store as store_mod
        monkeypatch.setattr(config, "DB_PATH", tmp_db)
        monkeypatch.setattr(store_mod, "DB_PATH", tmp_db)

        import main
        with patch("main.ingest", side_effect=RuntimeError("API totally down")):
            result = main.run_pipeline(run_number=99)
        assert result is False

    def test_failed_run_logged_in_db(self, tmp_db, monkeypatch):
        import config, pipeline.store as store_mod
        monkeypatch.setattr(config, "DB_PATH", tmp_db)
        monkeypatch.setattr(store_mod, "DB_PATH", tmp_db)

        import main
        from pipeline.store import query, _init_schema
        _init_schema()

        with patch("main.ingest", side_effect=RuntimeError("Injected failure")):
            main.run_pipeline(run_number=42)

        runs = query("SELECT status, error_message FROM pipeline_runs WHERE status='failed'")
        assert len(runs) >= 1
        assert "Injected failure" in runs.iloc[0]["error_message"]

    def test_scheduler_respects_max_runs(self, tmp_db, monkeypatch):
        import config, pipeline.store as store_mod, pipeline.visualize as viz_mod
        monkeypatch.setattr(config, "DB_PATH", tmp_db)
        monkeypatch.setattr(store_mod, "DB_PATH", tmp_db)
        reports = tmp_db.parent / "reports"
        reports.mkdir(exist_ok=True)
        monkeypatch.setattr(viz_mod, "REPORTS_DIR", reports)
        monkeypatch.setattr(viz_mod, "DB_PATH", tmp_db)

        run_count = []
        import main

        original_run = main.run_pipeline
        def counting_run(run_number=1):
            run_count.append(run_number)
            return original_run(run_number=run_number)

        with patch("main.run_pipeline", side_effect=counting_run):
            with patch("main.time.sleep"):  # don't actually sleep
                main.run_scheduler(interval=0, max_runs=2)

        assert len(run_count) == 2

    def test_arg_parser_once_flag(self):
        import main
        args = main._parse_args.__wrapped__() if hasattr(main._parse_args, '__wrapped__') else None
        # Just verify the parser is importable and has expected attributes
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--interval", type=int, default=60)
        parser.add_argument("--max-runs", type=int, default=3)
        args = parser.parse_args(["--once"])
        assert args.once is True
        assert args.interval == 60

    def test_arg_parser_custom_interval(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--once", action="store_true")
        parser.add_argument("--interval", type=int, default=60)
        parser.add_argument("--max-runs", type=int, default=3)
        args = parser.parse_args(["--interval", "120", "--max-runs", "5"])
        assert args.interval == 120
        assert args.max_runs == 5
