"""Reporter agent - the 4th and final LLM call.

Loads Gold parquet into DuckDB, asks the LLM for SQL + narrative + chart specs,
executes the SQL deterministically, renders Plotly charts, and assembles a
self-contained HTML report.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px

from core.config import REPORTS_DIR
from core.audit import AuditLogger
from agents.sttm_generator import _call_llm


def _schema_context(conn: duckdb.DuckDBPyConnection, table_names: list[str]) -> str:
    lines = []
    for t in table_names:
        df = conn.execute(f"SELECT * FROM {t} LIMIT 3").fetchdf()
        cols = conn.execute(f"DESCRIBE {t}").fetchdf()
        lines.append(f"\nTable: {t}")
        lines.append(f"  columns: {list(zip(cols['column_name'], cols['column_type']))}")
        lines.append(f"  sample rows:\n{df.to_string(index=False)}")
    return "\n".join(lines)


def _extract_sql(text: str) -> str:
    fence = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip().rstrip(";")
    m = re.search(r"(SELECT.*?)(?:NARRATIVE:|CHARTS:|$)", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().rstrip(";")
    return ""


def _extract_narrative(text: str) -> str:
    m = re.search(r"NARRATIVE:\s*(.*?)(?:CHARTS:|$)", text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else ""


def _extract_json_block(text: str) -> list[dict]:
    m = re.search(r"CHARTS:\s*```json\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not m:
        m = re.search(r"CHARTS:\s*(\[.*?\])", text, re.DOTALL | re.IGNORECASE)
    if not m:
        return []
    try:
        return json.loads(m.group(1).strip())
    except json.JSONDecodeError:
        return []


def _render_charts(df: pd.DataFrame, chart_specs: list[dict]) -> list[str]:
    html_blocks = []
    for spec in chart_specs:
        ctype = spec.get("type", "bar")
        x, y, title = spec.get("x"), spec.get("y"), spec.get("title", "")
        if x not in df.columns or y not in df.columns:
            continue
        try:
            if ctype == "pie":
                fig = px.pie(df, names=x, values=y, title=title)
            elif ctype == "line":
                fig = px.line(df, x=x, y=y, title=title)
            else:
                fig = px.bar(df, x=x, y=y, title=title)
            html_blocks.append(fig.to_html(full_html=False, include_plotlyjs=False))
        except Exception as e:
            print(f"[reporter] chart render failed for {spec}: {e}")
    return html_blocks


def generate_report(gold_paths: list[str], business_intent: str, run_id: str) -> str:
    audit = AuditLogger(run_id)
    conn = duckdb.connect(database=":memory:")

    table_names = []
    for fp in gold_paths:
        table_name = re.sub(rf"_{re.escape(run_id)}$", "", Path(fp).stem)
        conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{fp}')")
        table_names.append(table_name)

    schema_context = _schema_context(conn, table_names)

    prompt = f"""You are a data analyst. Given the Gold-layer table(s) below and a
business question, write a DuckDB SQL query that answers it, a short executive
narrative, and 1-2 chart specs.

Business question: {business_intent}

Available tables:
{schema_context}

Respond in EXACTLY this format:

SQL:
```sql
<a single DuckDB SELECT statement using only the table(s) above>
```

NARRATIVE:
<2-3 paragraph executive summary answering the business question, citing numbers>

CHARTS:
```json
[{{"type": "bar", "x": "<column>", "y": "<column>", "title": "<title>"}}]
```
"""

    response = _call_llm(prompt) or ""
    sql = _extract_sql(response)
    narrative = _extract_narrative(response)
    chart_specs = _extract_json_block(response)

    result_df = None
    if sql:
        try:
            result_df = conn.execute(sql).fetchdf()
        except Exception as e:
            print(f"[reporter] SQL failed, retrying once: {e}")
            retry_prompt = prompt + f"\n\nYour previous SQL failed with error: {e}\nFix it and respond in the same format."
            response2 = _call_llm(retry_prompt) or ""
            sql2 = _extract_sql(response2)
            if sql2:
                try:
                    result_df = conn.execute(sql2).fetchdf()
                    sql = sql2
                    if _extract_narrative(response2):
                        narrative = _extract_narrative(response2)
                    if _extract_json_block(response2):
                        chart_specs = _extract_json_block(response2)
                except Exception as e2:
                    print(f"[reporter] retry SQL also failed: {e2}")

    if result_df is None:
        sql = f"SELECT * FROM {table_names[0]} LIMIT 20"
        result_df = conn.execute(sql).fetchdf()
        if not narrative:
            narrative = "The LLM-generated SQL could not be executed, so this report shows the raw Gold table instead."

    chart_html_blocks = _render_charts(result_df, chart_specs)
    table_html = result_df.to_html(index=False, classes="data-table")

    generated_at = datetime.now(timezone.utc).isoformat()
    charts_section = (
        "\n".join(f'<div class="chart-card">{c}</div>' for c in chart_html_blocks)
        if chart_html_blocks
        else '<p class="muted">No charts were generated for this run.</p>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>IDAMP Report - {run_id}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; background:#f3f4f6; color:#374151; margin:0; }}
  header {{ background:linear-gradient(135deg,#1e3a8a,#2563eb); color:white; padding:32px 40px; }}
  header h1 {{ margin:0 0 6px; font-size:1.5rem; }}
  header p {{ margin:0; opacity:0.9; }}
  .container {{ max-width:1000px; margin:0 auto; padding:24px; }}
  .card {{ background:white; border-radius:10px; padding:20px 24px; margin-bottom:20px; box-shadow:0 1px 4px rgba(0,0,0,0.08); }}
  .card h2 {{ font-size:0.9rem; text-transform:uppercase; letter-spacing:0.5px; color:#6b7280; margin:0 0 12px; }}
  .charts-grid {{ display:grid; grid-template-columns:1fr; gap:16px; }}
  table.data-table {{ width:100%; border-collapse:collapse; font-size:0.85rem; }}
  table.data-table th {{ background:#f1f5f9; text-align:left; padding:8px 10px; border-bottom:2px solid #e5e7eb; }}
  table.data-table td {{ padding:6px 10px; border-bottom:1px solid #e5e7eb; }}
  pre {{ background:#0f172a; color:#e2e8f0; padding:14px 16px; border-radius:8px; overflow-x:auto; font-size:0.8rem; }}
  footer {{ text-align:center; color:#9ca3af; font-size:0.78rem; padding:24px; }}
  .muted {{ color:#9ca3af; }}
</style>
</head>
<body>
<header>
  <h1>IDAMP Analytics Report</h1>
  <p>{business_intent}</p>
</header>
<div class="container">
  <div class="card">
    <h2>Executive Answer</h2>
    <p>{narrative}</p>
  </div>
  <div class="card">
    <h2>Charts</h2>
    <div class="charts-grid">{charts_section}</div>
  </div>
  <div class="card">
    <h2>Data Table</h2>
    {table_html}
  </div>
  <div class="card">
    <h2>SQL</h2>
    <pre><code>{sql}</code></pre>
  </div>
</div>
<footer>run_id: {run_id} &middot; generated {generated_at}</footer>
</body>
</html>"""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = REPORTS_DIR / f"report_{run_id}.html"
    html_path.write_text(html, encoding="utf-8")

    json_summary = {
        "run_id": run_id,
        "business_intent": business_intent,
        "sql": sql,
        "narrative": narrative,
        "chart_specs": chart_specs,
        "row_count": int(len(result_df)),
        "generated_at": generated_at,
    }
    json_path = REPORTS_DIR / f"report_{run_id}.json"
    json_path.write_text(json.dumps(json_summary, indent=2, default=str), encoding="utf-8")

    audit.log(agent="reporter", action="completed", html_path=str(html_path), json_path=str(json_path))
    return str(html_path)
