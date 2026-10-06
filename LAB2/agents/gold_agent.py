"""Gold agent - executes the approved Gold STTM in pandas.

Analytics layer: joins Silver tables, then group-by + aggregate.
No LLM. Result sets are small (typically <50 rows).
"""

import re
from pathlib import Path

import pandas as pd

from core.config import GOLD_DIR
from core.audit import AuditLogger

_AGG_FUNCS = {"SUM": "sum", "AVG": "mean", "COUNT": "count", "MAX": "max", "MIN": "min"}


def _load_silver_tables(silver_paths: list[str], run_id: str) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for fp in silver_paths:
        stem = Path(fp).stem  # e.g. "sales_data_silver_20260817_130818"
        base_no_run = re.sub(rf"_{re.escape(run_id)}$", "", stem)  # "sales_data_silver"
        base_no_layer = re.sub(r"_silver$", "", base_no_run)  # "sales_data"

        df = pd.read_parquet(fp)
        for key in {stem, base_no_run, base_no_layer}:
            tables[key] = df
    return tables


def _resolve_table(tables: dict[str, pd.DataFrame], name: str) -> pd.DataFrame:
    name = name.strip()
    if name in tables:
        return tables[name]
    for key, df in tables.items():
        if key.startswith(name) or name.startswith(key):
            return df
    raise KeyError(f"Gold join references unknown table: {name}")


def _parse_agg(logic: str):
    """'SUM(total_amount) AS total_revenue' -> ('total_amount', 'sum', 'total_revenue')"""
    m = re.match(r"\s*(\w+)\((\w+)\)\s*(?:AS\s+(\w+))?\s*$", logic, re.IGNORECASE)
    if not m:
        return None
    func, col, alias = m.groups()
    func = func.upper()
    pandas_func = _AGG_FUNCS.get(func, "sum")
    return col, pandas_func, alias or f"{col}_{pandas_func}"


def _find_col(df: pd.DataFrame, source_table, col, right_suffixes: list[str]):
    """Resolve a column that may have been suffixed by a join collision.

    Two silver tables can both carry a same-named column (e.g. "category" on
    both a fact and a dimension table). Joins below keep the left side's
    column names untouched and only suffix the right side's colliding
    columns as "<col>__<right_table>", so a bare lookup can silently miss
    the intended column after a join. This tries the bare name first, then
    the suffixed variant that matches the rule's own source_table, then any
    suffixed variant as a last resort.
    """
    if not col or pd.isna(col):
        return None
    col = str(col)
    if col in df.columns:
        return col
    source_table = str(source_table).strip() if source_table and not pd.isna(source_table) else ""
    for right_name in right_suffixes:
        candidate = f"{col}__{right_name}"
        if candidate in df.columns and (
            right_name == source_table
            or right_name.startswith(source_table)
            or source_table.startswith(right_name)
        ):
            return candidate
    for right_name in right_suffixes:
        candidate = f"{col}__{right_name}"
        if candidate in df.columns:
            return candidate
    return None


def run(
    silver_paths: list[str], sttm_path: str, business_intent: str, run_id: str
) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    tables = _load_silver_tables(silver_paths, run_id)
    audit = AuditLogger(run_id)
    written = []

    for target_table, rules in sttm.groupby("target_table"):
        df = None
        right_suffixes: list[str] = []

        join_rules = rules[rules["transformation_type"].str.lower() == "join"]
        for _, rule in join_rules.iterrows():
            logic = str(rule["transformation_logic"])
            parts = logic.split(":")
            if len(parts) != 4 or parts[0].lower() != "join_left":
                continue
            _, left_name, right_name, key = parts
            left = _resolve_table(tables, left_name) if df is None else df
            right = _resolve_table(tables, right_name)
            # Keep the left side's column names untouched; only the right
            # side's colliding columns get a "__<right_name>" suffix, so
            # group_by/aggregate rules can still be resolved by name.
            df = left.merge(right, on=key, how="left", suffixes=("", f"__{right_name}"))
            right_suffixes.append(right_name)

        if df is None:
            # no join rule: fall back to the first referenced silver table
            first_key = next(iter(tables))
            df = tables[first_key]

        rename_rules = rules[
            rules["transformation_type"].str.lower().isin(["passthrough", "rename"])
        ]
        for _, rule in rename_rules.iterrows():
            src_col = _find_col(df, rule.get("source_table"), rule.get("source_column"), right_suffixes)
            tgt_col = rule.get("target_column")
            if src_col and tgt_col and not pd.isna(tgt_col) and src_col != tgt_col:
                df = df.rename(columns={src_col: tgt_col})

        group_col_pairs = []  # (actual_column_in_df, target_column)
        for _, r in rules[rules["transformation_type"].str.lower() == "group_by"].iterrows():
            target = r.get("target_column")
            actual = _find_col(df, r.get("source_table"), r.get("source_column"), right_suffixes)
            if not actual:
                actual = _find_col(df, r.get("source_table"), target, right_suffixes)
            if actual:
                group_col_pairs.append((actual, target if target and not pd.isna(target) else actual))
        group_actual_cols = [actual for actual, _ in group_col_pairs]
        group_rename_map = {actual: target for actual, target in group_col_pairs if actual != target}

        agg_map = {}
        alias_map = {}
        for _, rule in rules[rules["transformation_type"].str.lower() == "aggregate"].iterrows():
            parsed = _parse_agg(str(rule["transformation_logic"]))
            if parsed:
                col, func, alias = parsed
                actual = _find_col(df, rule.get("source_table"), col, right_suffixes)
                if actual:
                    agg_map[actual] = func
                    alias_map[actual] = alias

        if group_actual_cols and agg_map:
            df = df.groupby(group_actual_cols, dropna=False).agg(agg_map).reset_index()
            df = df.rename(columns={**group_rename_map, **alias_map})
        elif agg_map:
            summary = {alias_map[c]: [df[c].agg(f)] for c, f in agg_map.items()}
            df = pd.DataFrame(summary)

        group_target_cols = [target for _, target in group_col_pairs]
        keep_cols = list(dict.fromkeys(group_target_cols + list(alias_map.values())))
        keep_cols = [c for c in keep_cols if c in df.columns]
        if keep_cols:
            df = df[keep_cols]

        df.insert(0, "pk_gold_id", range(1, len(df) + 1))

        out_path = GOLD_DIR / f"{target_table}_{run_id}.parquet"
        df.to_parquet(out_path, index=False)
        written.append(str(out_path))

        audit.log(
            agent="gold_agent",
            action="table_written",
            table=target_table,
            output_shape=list(df.shape),
            output_path=str(out_path),
        )

    audit.log(agent="gold_agent", action="completed", written=written)
    return written
