"""
pipeline/visualize.py
─────────────────────
Stage 4 — Reporting & Visualisation

Reads from the SQLite database and produces:
    • CSV summary reports  (reports/)
    • A self-contained HTML dashboard (reports/dashboard.html)

Public contract
───────────────
    visualize() -> Path   (path to the HTML dashboard)
"""

from __future__ import annotations

import textwrap
from datetime import datetime, timezone
from pathlib import Path
import sys
import json

import pandas as pd
import sqlite3

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import REPORTS_DIR, DB_PATH, TOP_N_AUTHORS, TOP_N_VENUES
from logger import get_logger

log = get_logger("visualize")

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Query helpers ──────────────────────────────────────────────────────────────

def _q(sql: str) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(sql, conn)


# ── Report generators ──────────────────────────────────────────────────────────

def _yearly_trend() -> pd.DataFrame:
    df = _q("""
        SELECT year, paper_count, avg_citations, avg_authors, open_access_pct
        FROM   yearly_summary
        ORDER  BY year
    """)
    df.to_csv(REPORTS_DIR / "yearly_trend.csv", index=False)
    log.info("Report saved: yearly_trend.csv")
    return df


def _top_venues() -> pd.DataFrame:
    df = _q(f"""
        SELECT   venue,
                 COUNT(*)              AS paper_count,
                 ROUND(AVG(citations),1) AS avg_citations,
                 ROUND(100.0*SUM(open_access)/COUNT(*),1) AS open_access_pct
        FROM     papers
        WHERE    venue != 'Unknown'
        GROUP BY venue
        ORDER BY paper_count DESC
        LIMIT    {TOP_N_VENUES}
    """)
    df.to_csv(REPORTS_DIR / "top_venues.csv", index=False)
    log.info("Report saved: top_venues.csv")
    return df


def _top_concepts() -> pd.DataFrame:
    df = _q("""
        SELECT   top_concept,
                 COUNT(*)                AS paper_count,
                 ROUND(AVG(citations),1) AS avg_citations
        FROM     papers
        WHERE    top_concept != 'Unknown'
        GROUP BY top_concept
        ORDER BY paper_count DESC
        LIMIT    12
    """)
    df.to_csv(REPORTS_DIR / "top_concepts.csv", index=False)
    log.info("Report saved: top_concepts.csv")
    return df


def _impact_distribution() -> pd.DataFrame:
    df = _q("""
        SELECT   impact_tier,
                 COUNT(*) AS paper_count,
                 ROUND(AVG(citations),1) AS avg_citations
        FROM     papers
        GROUP BY impact_tier
        ORDER BY avg_citations DESC
    """)
    df.to_csv(REPORTS_DIR / "impact_distribution.csv", index=False)
    log.info("Report saved: impact_distribution.csv")
    return df


def _pipeline_runs() -> pd.DataFrame:
    df = _q("""
        SELECT run_id, started_at, finished_at, status,
               rows_ingested, rows_stored, rows_skipped, error_message
        FROM   pipeline_runs
        ORDER  BY run_id DESC
        LIMIT  20
    """)
    df.to_csv(REPORTS_DIR / "pipeline_runs.csv", index=False)
    log.info("Report saved: pipeline_runs.csv")
    return df


def _summary_stats() -> dict:
    df = _q("""
        SELECT
            COUNT(*)                                   AS total_papers,
            MIN(year)                                  AS earliest_year,
            MAX(year)                                  AS latest_year,
            ROUND(AVG(citations),1)                    AS avg_citations,
            MAX(citations)                             AS max_citations,
            ROUND(100.0*SUM(open_access)/COUNT(*),1)   AS open_access_pct,
            ROUND(AVG(author_count),1)                 AS avg_authors
        FROM papers
    """)
    return df.iloc[0].to_dict() if not df.empty else {}


# ── HTML dashboard ─────────────────────────────────────────────────────────────

def _df_to_table_rows(df: pd.DataFrame, max_rows: int = 50) -> str:
    rows = []
    for _, r in df.head(max_rows).iterrows():
        cells = "".join(f"<td>{v}</td>" for v in r)
        rows.append(f"<tr>{cells}</tr>")
    return "\n".join(rows)


def _df_to_thead(df: pd.DataFrame) -> str:
    headers = "".join(f"<th>{c}</th>" for c in df.columns)
    return f"<tr>{headers}</tr>"


def _build_dashboard(
    stats: dict,
    yearly: pd.DataFrame,
    venues: pd.DataFrame,
    concepts: pd.DataFrame,
    impact: pd.DataFrame,
    runs: pd.DataFrame,
) -> str:
    """Render a self-contained HTML dashboard with embedded Chart.js charts."""

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Chart data (JSON) ──────────────────────────────────────────────────────
    yearly_labels  = json.dumps(yearly["year"].astype(str).tolist())
    yearly_papers  = json.dumps(yearly["paper_count"].tolist())
    yearly_cites   = json.dumps(yearly["avg_citations"].round(1).tolist())
    yearly_oa      = json.dumps(yearly["open_access_pct"].tolist())

    venue_labels   = json.dumps(venues["venue"].tolist())
    venue_counts   = json.dumps(venues["paper_count"].tolist())

    concept_labels = json.dumps(concepts["top_concept"].tolist())
    concept_counts = json.dumps(concepts["paper_count"].tolist())

    impact_order   = ["Uncited", "Low", "Medium", "High", "Highly Cited"]
    impact_sorted  = impact.set_index("impact_tier").reindex(impact_order).dropna()
    impact_labels  = json.dumps(impact_sorted.index.tolist())
    impact_counts  = json.dumps(impact_sorted["paper_count"].astype(int).tolist())

    # Stat cards
    def card(label, value, icon):
        return f"""
        <div class="stat-card">
          <div class="stat-icon">{icon}</div>
          <div class="stat-value">{value}</div>
          <div class="stat-label">{label}</div>
        </div>"""

    stat_cards = "".join([
        card("Total Papers",     int(stats.get("total_papers", 0)),    "📄"),
        card("Year Range",       f"{int(stats.get('earliest_year',0))}–{int(stats.get('latest_year',0))}", "📅"),
        card("Avg Citations",    stats.get("avg_citations", "—"),       "📊"),
        card("Max Citations",    int(stats.get("max_citations", 0)),    "🏆"),
        card("Open Access %",    f"{stats.get('open_access_pct','—')}%","🔓"),
        card("Avg Authors",      stats.get("avg_authors", "—"),         "👥"),
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Research Pipeline Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg:       #0d1117;
      --surface:  #161b22;
      --border:   #21262d;
      --accent:   #3fb950;
      --accent2:  #58a6ff;
      --accent3:  #f78166;
      --accent4:  #d2a8ff;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --mono:     'IBM Plex Mono', monospace;
      --sans:     'IBM Plex Sans', sans-serif;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      font-size: 14px;
      line-height: 1.6;
      min-height: 100vh;
    }}

    /* ── Header ── */
    header {{
      border-bottom: 1px solid var(--border);
      padding: 20px 32px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: var(--surface);
    }}
    .header-left h1 {{
      font-size: 20px;
      font-weight: 700;
      letter-spacing: -0.3px;
    }}
    .header-left h1 span {{ color: var(--accent); }}
    .header-meta {{
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      margin-top: 2px;
    }}
    .badge {{
      background: rgba(63,185,80,0.15);
      color: var(--accent);
      border: 1px solid rgba(63,185,80,0.3);
      padding: 4px 12px;
      border-radius: 20px;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 600;
    }}

    /* ── Main layout ── */
    main {{ padding: 28px 32px; max-width: 1400px; margin: 0 auto; }}

    /* ── Stat cards ── */
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 14px;
      margin-bottom: 28px;
    }}
    .stat-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px 20px;
      transition: border-color .2s;
    }}
    .stat-card:hover {{ border-color: var(--accent2); }}
    .stat-icon  {{ font-size: 20px; margin-bottom: 8px; }}
    .stat-value {{
      font-family: var(--mono);
      font-size: 22px;
      font-weight: 600;
      color: var(--accent2);
      line-height: 1.2;
    }}
    .stat-label {{ font-size: 11px; color: var(--muted); margin-top: 4px; text-transform: uppercase; letter-spacing: .5px; }}

    /* ── Section headings ── */
    .section-title {{
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--muted);
      margin-bottom: 14px;
      padding-bottom: 8px;
      border-bottom: 1px solid var(--border);
    }}

    /* ── Chart grid ── */
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 28px;
    }}
    @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}

    .chart-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
    }}
    .chart-title {{
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 16px;
      color: var(--text);
    }}
    .chart-container {{ position: relative; height: 240px; }}

    /* ── Full-width chart ── */
    .chart-card.full {{
      grid-column: 1 / -1;
    }}
    .chart-container.tall {{ height: 200px; }}

    /* ── Data tables ── */
    .table-card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      margin-bottom: 18px;
    }}
    .table-card-header {{
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      font-size: 13px;
      font-weight: 600;
    }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    thead tr {{ background: rgba(255,255,255,.03); }}
    th {{
      padding: 10px 14px;
      text-align: left;
      font-family: var(--mono);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: .5px;
      color: var(--muted);
      white-space: nowrap;
    }}
    td {{
      padding: 9px 14px;
      border-top: 1px solid var(--border);
      color: var(--text);
      max-width: 300px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    tr:hover td {{ background: rgba(255,255,255,.02); }}

    /* ── Footer ── */
    footer {{
      margin-top: 40px;
      padding: 20px 0;
      border-top: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      text-align: center;
    }}
  </style>
</head>
<body>

<header>
  <div class="header-left">
    <h1>Research Pipeline <span>Dashboard</span></h1>
    <div class="header-meta">Generated {generated} · SQLite · Python/Pandas</div>
  </div>
  <div class="badge">● LIVE</div>
</header>

<main>

  <!-- Stat cards -->
  <div class="stats-grid">
    {stat_cards}
  </div>

  <!-- Charts row 1 -->
  <div class="charts-grid">

    <div class="chart-card full">
      <div class="chart-title">📈 Annual Publication Volume &amp; Average Citations</div>
      <div class="chart-container tall">
        <canvas id="yearlyChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">🏛️ Top Venues by Paper Count</div>
      <div class="chart-container">
        <canvas id="venueChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">🧠 Top Research Concepts</div>
      <div class="chart-container">
        <canvas id="conceptChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">🏆 Impact Tier Distribution</div>
      <div class="chart-container">
        <canvas id="impactChart"></canvas>
      </div>
    </div>

    <div class="chart-card">
      <div class="chart-title">🔓 Open Access % Over Time</div>
      <div class="chart-container">
        <canvas id="oaChart"></canvas>
      </div>
    </div>

  </div>

  <!-- Venues table -->
  <div class="section-title">Top Venues</div>
  <div class="table-card">
    <div class="table-wrap">
      <table>
        <thead>{_df_to_thead(venues)}</thead>
        <tbody>{_df_to_table_rows(venues)}</tbody>
      </table>
    </div>
  </div>

  <!-- Pipeline runs table -->
  <div class="section-title">Pipeline Run History</div>
  <div class="table-card">
    <div class="table-wrap">
      <table>
        <thead>{_df_to_thead(runs)}</thead>
        <tbody>{_df_to_table_rows(runs)}</tbody>
      </table>
    </div>
  </div>

</main>

<footer>Research Data Pipeline · Modular ETL · Python {"{"}pandas · sqlite3 · Chart.js{"}"}</footer>

<script>
const C = (id) => document.getElementById(id).getContext('2d');
const GRID = {{ color: 'rgba(255,255,255,0.05)' }};
const TICK = {{ color: '#8b949e', font: {{ size: 11 }} }};

// ── Yearly trend (dual axis) ────────────────────────────────────────────────
new Chart(C('yearlyChart'), {{
  data: {{
    labels: {yearly_labels},
    datasets: [
      {{
        type: 'bar',
        label: 'Papers',
        data: {yearly_papers},
        backgroundColor: 'rgba(88,166,255,0.25)',
        borderColor: '#58a6ff',
        borderWidth: 1,
        yAxisID: 'y',
      }},
      {{
        type: 'line',
        label: 'Avg Citations',
        data: {yearly_cites},
        borderColor: '#3fb950',
        backgroundColor: 'rgba(63,185,80,0.1)',
        borderWidth: 2,
        pointRadius: 3,
        tension: 0.4,
        fill: true,
        yAxisID: 'y1',
      }}
    ]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ labels: {{ color: '#8b949e', font: {{ size: 11 }} }} }} }},
    scales: {{
      x: {{ grid: GRID, ticks: TICK }},
      y:  {{ grid: GRID, ticks: TICK, position: 'left',  title: {{ display: true, text: 'Papers', color: '#8b949e', font: {{size:11}} }} }},
      y1: {{ grid: {{ drawOnChartArea: false }}, ticks: TICK, position: 'right', title: {{ display: true, text: 'Avg Citations', color: '#8b949e', font: {{size:11}} }} }},
    }}
  }}
}});

// ── Venues (horizontal bar) ─────────────────────────────────────────────────
new Chart(C('venueChart'), {{
  type: 'bar',
  data: {{
    labels: {venue_labels},
    datasets: [{{ label: 'Papers', data: {venue_counts},
      backgroundColor: 'rgba(210,168,255,0.3)', borderColor: '#d2a8ff', borderWidth: 1 }}]
  }},
  options: {{
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: GRID, ticks: TICK }},
      y: {{ grid: {{ display: false }}, ticks: {{ color: '#8b949e', font: {{ size: 10 }} }} }}
    }}
  }}
}});

// ── Concepts (doughnut) ──────────────────────────────────────────────────────
new Chart(C('conceptChart'), {{
  type: 'doughnut',
  data: {{
    labels: {concept_labels},
    datasets: [{{ data: {concept_counts},
      backgroundColor: [
        'rgba(63,185,80,.7)','rgba(88,166,255,.7)','rgba(210,168,255,.7)',
        'rgba(247,129,102,.7)','rgba(255,214,51,.7)','rgba(79,209,197,.7)',
        'rgba(255,107,139,.7)','rgba(134,239,172,.7)','rgba(165,180,252,.7)',
        'rgba(251,191,36,.7)','rgba(196,181,253,.7)','rgba(52,211,153,.7)',
      ],
      borderWidth: 0 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'right', labels: {{ color: '#8b949e', font: {{ size: 10 }}, boxWidth: 12 }} }} }}
  }}
}});

// ── Impact tiers (bar) ───────────────────────────────────────────────────────
new Chart(C('impactChart'), {{
  type: 'bar',
  data: {{
    labels: {impact_labels},
    datasets: [{{ label: 'Papers', data: {impact_counts},
      backgroundColor: ['rgba(139,148,158,.4)','rgba(88,166,255,.4)',
        'rgba(63,185,80,.4)','rgba(210,168,255,.4)','rgba(247,129,102,.7)'],
      borderColor:     ['#8b949e','#58a6ff','#3fb950','#d2a8ff','#f78166'],
      borderWidth: 1 }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{ x: {{ grid: {{ display: false }}, ticks: TICK }}, y: {{ grid: GRID, ticks: TICK }} }}
  }}
}});

// ── Open Access % line ───────────────────────────────────────────────────────
new Chart(C('oaChart'), {{
  type: 'line',
  data: {{
    labels: {yearly_labels},
    datasets: [{{ label: 'Open Access %', data: {yearly_oa},
      borderColor: '#f78166', backgroundColor: 'rgba(247,129,102,0.1)',
      borderWidth: 2, pointRadius: 3, tension: 0.4, fill: true }}]
  }},
  options: {{
    responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      x: {{ grid: GRID, ticks: TICK }},
      y: {{ grid: GRID, ticks: TICK, min: 0, max: 100,
            title: {{ display: true, text: '%', color: '#8b949e', font: {{size:11}} }} }}
    }}
  }}
}});
</script>

</body>
</html>"""
    return html


# ── Public entry point ─────────────────────────────────────────────────────────

def visualize() -> Path:
    """
    Generate all CSV reports and the HTML dashboard.
    Returns the path to the dashboard HTML file.
    """
    log.info("=== VISUALIZE stage started ===")

    stats    = _summary_stats()
    yearly   = _yearly_trend()
    venues   = _top_venues()
    concepts = _top_concepts()
    impact   = _impact_distribution()
    runs     = _pipeline_runs()

    dash_path = REPORTS_DIR / "dashboard.html"
    html = _build_dashboard(stats, yearly, venues, concepts, impact, runs)
    dash_path.write_text(html, encoding="utf-8")
    log.info("Dashboard saved → %s", dash_path)

    log.info("=== VISUALIZE stage finished ===")
    return dash_path


if __name__ == "__main__":
    p = visualize()
    print(f"Dashboard → {p}")
