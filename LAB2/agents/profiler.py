"""Profiler agent - scans raw CSVs and computes quality stats.

Pure pandas. No LLM. Output feeds the STTM generator (Step 4).
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from core.config import PROFILES_DIR
from core.audit import AuditLogger

MIXED_DATE_THRESHOLD = 0.10
ABBREVIATION_AVG_LEN = 6


def _mixed_date_flag(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    mmddyyyy = non_null.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    yyyymmdd = non_null.str.match(r"^\d{4}-\d{1,2}-\d{1,2}$")
    n = len(non_null)
    return (mmddyyyy.sum() / n > MIXED_DATE_THRESHOLD) and (
        yyyymmdd.sum() / n > MIXED_DATE_THRESHOLD
    )


def _abbreviation_flag(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str)
    if non_null.empty:
        return False
    avg_len = non_null.str.len().mean()
    return bool(avg_len < ABBREVIATION_AVG_LEN)


def _profile_column(series: pd.Series) -> dict:
    dtype = str(series.dtype)
    null_count = int(series.isna().sum())
    null_pct = round(null_count / len(series), 4) if len(series) else 0.0
    unique_count = int(series.nunique(dropna=True))
    sample_values = [str(v) for v in series.dropna().unique()[:5]]

    col = {
        "dtype": dtype,
        "null_count": null_count,
        "null_pct": null_pct,
        "unique_count": unique_count,
        "sample_values": sample_values,
        "quality_flags": [],
    }

    if pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        if not numeric.empty:
            col["min"] = float(numeric.min())
            col["max"] = float(numeric.max())
            col["mean"] = float(numeric.mean())
    else:
        # covers both legacy "object" string columns and pandas>=3's "str" dtype
        if _mixed_date_flag(series):
            col["quality_flags"].append("mixed_date_formats")
        if _abbreviation_flag(series):
            col["quality_flags"].append("possible_abbreviations")

    return col


def profile(file_paths: list[str], run_id: str) -> str:
    tables: dict[str, dict] = {}
    column_to_tables: dict[str, list[str]] = defaultdict(list)

    for fp in file_paths:
        table_name = Path(fp).stem
        df = pd.read_csv(fp, low_memory=False)

        columns = {}
        for col_name in df.columns:
            columns[col_name] = _profile_column(df[col_name])
            if col_name.endswith("_id"):
                column_to_tables[col_name].append(table_name)

        tables[table_name] = {
            "row_count": int(len(df)),
            "columns": columns,
        }

    candidate_join_keys = {
        col: table_list
        for col, table_list in column_to_tables.items()
        if len(table_list) >= 2
    }

    result = {
        "run_id": run_id,
        "tables": tables,
        "candidate_join_keys": candidate_join_keys,
    }

    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROFILES_DIR / f"profile_combined_{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json

        json.dump(result, f, indent=2, default=str)

    AuditLogger(run_id).log(
        agent="profiler",
        action="completed",
        tables=list(tables.keys()),
        output_path=str(out_path),
    )

    return str(out_path)
