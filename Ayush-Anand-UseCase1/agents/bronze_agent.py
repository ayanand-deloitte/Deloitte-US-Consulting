"""Bronze agent - executes the approved Bronze STTM in pandas.

Fidelity layer only: type casting, renaming, metadata injection.
No LLM, no business logic, no joins. Row count in = row count out.
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import BRONZE_DIR
from core.audit import AuditLogger


def _cast(series: pd.Series, logic: str) -> pd.Series:
    logic = (logic or "").strip().lower()
    if logic == "datetime":
        return pd.to_datetime(series, errors="coerce")
    if logic == "float":
        return pd.to_numeric(series, errors="coerce")
    if logic == "int":
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    if logic == "str":
        return series.astype(str)
    return series


def run(input_files: list[str], sttm_path: str, run_id: str) -> list[str]:
    sttm = pd.read_csv(sttm_path)
    audit = AuditLogger(run_id)
    written = []

    for fp in input_files:
        table_name = Path(fp).stem
        rules = sttm[sttm["source_table"] == table_name]
        if rules.empty:
            continue

        src = pd.read_csv(fp)
        input_shape = src.shape
        out = pd.DataFrame(index=src.index)

        for _, rule in rules.iterrows():
            ttype = str(rule.get("transformation_type", "")).strip().lower()
            src_col = rule.get("source_column")
            tgt_col = rule.get("target_column")

            if ttype == "metadata_inject":
                continue  # handled after the loop

            if src_col not in src.columns:
                continue

            if ttype == "type_cast":
                out[tgt_col] = _cast(src[src_col], rule.get("transformation_logic"))
            elif ttype == "passthrough":
                out[tgt_col] = src[src_col]
            else:
                out[tgt_col] = src[src_col]

        out["_load_timestamp"] = datetime.now(timezone.utc).isoformat()
        out["_source_file"] = str(fp)

        out_path = BRONZE_DIR / f"{table_name}_bronze_{run_id}.parquet"
        out.to_parquet(out_path, index=False)
        written.append(str(out_path))

        audit.log(
            agent="bronze_agent",
            action="table_written",
            table=table_name,
            input_shape=list(input_shape),
            output_shape=list(out.shape),
            output_path=str(out_path),
        )

    audit.log(agent="bronze_agent", action="completed", written=written)
    return written
