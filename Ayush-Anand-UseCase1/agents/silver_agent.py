"""Silver agent - executes the approved Silver STTM in pandas.

Quality layer: null handling, date/text standardisation, dedup, surrogate key.
No LLM. Reads each rule's transformation_logic keyword and runs the matching
pandas operation.
"""

import re
from pathlib import Path

import pandas as pd

from core.config import SILVER_DIR
from core.audit import AuditLogger


def _apply_rule(df: pd.DataFrame, col: str, logic: str) -> pd.DataFrame:
    logic_l = (logic or "").lower()

    if "drop" in logic_l and "null" in logic_l:
        df = df.dropna(subset=[col])
    elif "fill" in logic_l and "null" in logic_l:
        if col in df.columns:
            if "mean" in logic_l:
                df[col] = df[col].fillna(df[col].mean())
            elif "median" in logic_l:
                df[col] = df[col].fillna(df[col].median())
            elif "mode" in logic_l:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else 0)
            else:
                df[col] = df[col].fillna(0)
    elif "date" in logic_l or "datetime" in logic_l:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")
    elif any(k in logic_l for k in ("integer", "float", "numeric")):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    elif "lowercase" in logic_l:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower()
    elif "uppercase" in logic_l:
        if col in df.columns:
            df[col] = df[col].astype(str).str.upper()
    elif "title" in logic_l:
        if col in df.columns:
            df[col] = df[col].astype(str).str.title()
    elif "strip" in logic_l or "trim" in logic_l:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    elif "deduplic" in logic_l:
        df = df.drop_duplicates()

    return df


def _match_rules(sttm: pd.DataFrame, base_name: str, run_id: str) -> pd.DataFrame:
    bronze_table_name = f"{base_name}_bronze"
    stripped_suffix = f"_bronze_{run_id}"
    candidates = {base_name, bronze_table_name}
    return sttm[sttm["source_table"].isin(candidates) | sttm["source_table"].str.replace(stripped_suffix, "", regex=False).isin(candidates)]


def run(bronze_paths: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    audit = AuditLogger(run_id)
    written = []

    for fp in bronze_paths:
        stem = Path(fp).stem  # e.g. "sales_data_bronze_20260817_130818"
        base_name = re.sub(rf"_bronze_{re.escape(run_id)}$", "", stem)

        rules = _match_rules(sttm, base_name, run_id)
        if rules.empty:
            continue

        df = pd.read_parquet(fp)
        input_shape = df.shape

        kept_columns = []
        for _, rule in rules.iterrows():
            ttype = str(rule.get("transformation_type", "")).strip().lower()
            src_col = rule.get("source_column")
            tgt_col = rule.get("target_column") or src_col
            logic = rule.get("transformation_logic", "")

            if ttype == "deduplicate":
                df = _apply_rule(df, None, "deduplic")
                continue

            if src_col and src_col in df.columns and tgt_col and src_col != tgt_col:
                df = df.rename(columns={src_col: tgt_col})

            col = tgt_col if tgt_col in df.columns else src_col
            if col and col in df.columns:
                df = _apply_rule(df, col, logic)
                kept_columns.append(col)

        kept_columns = [c for c in dict.fromkeys(kept_columns) if c in df.columns]
        if kept_columns:
            df = df[kept_columns]

        df.insert(0, f"pk_{base_name}_silver_id", range(1, len(df) + 1))

        out_path = SILVER_DIR / f"{base_name}_silver_{run_id}.parquet"
        df.to_parquet(out_path, index=False)
        written.append(str(out_path))

        audit.log(
            agent="silver_agent",
            action="table_written",
            table=base_name,
            input_shape=list(input_shape),
            output_shape=list(df.shape),
            output_path=str(out_path),
        )

    audit.log(agent="silver_agent", action="completed", written=written)
    return written
