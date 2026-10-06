"""STTM Generator - the LLM writes the transformation contract (Bronze/Silver/Gold rules).

LLM calls 1-3 of the pipeline's 4 total calls. Everything downstream (bronze_agent,
silver_agent, gold_agent) is deterministic Python that just executes these rules.
"""

import io
import json
import re

import pandas as pd

from core.config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GITHUB_TOKEN, GITHUB_BASE_URL, GITHUB_MODEL
from core.audit import AuditLogger

REQUIRED_COLUMNS = [
    "source_schema",
    "source_table",
    "source_column",
    "target_schema",
    "target_table",
    "target_column",
    "transformation_type",
    "transformation_logic",
]

_llm = None


def _make_llm():
    """Build the chat model for LLM_PROVIDER (default: groq).

    Groq exposes an OpenAI-compatible endpoint but we use langchain-groq's
    native ChatGroq client. Swapping providers only means adding a branch
    here - the rest of the pipeline never changes.
    """
    global _llm
    if _llm is not None:
        return _llm

    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        if not GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a free key at "
                "https://console.groq.com/keys and put it in .env"
            )
        _llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, temperature=0)

    elif LLM_PROVIDER == "github":
        from langchain_openai import ChatOpenAI

        if not GITHUB_TOKEN:
            raise RuntimeError("GITHUB_TOKEN is not set in .env")
        _llm = ChatOpenAI(
            api_key=GITHUB_TOKEN,
            base_url=GITHUB_BASE_URL,
            model=GITHUB_MODEL,
            temperature=0,
        )

    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

    return _llm


def _call_llm(prompt: str, stage: str | None = None) -> str:
    """Call the configured model and return its textual response content."""
    response = _make_llm().invoke(prompt)
    content = response.content

    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = [
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str)
            or (
                isinstance(block, dict)
                and block.get("type", "text") == "text"
                and isinstance(block.get("text"), str)
            )
        ]
        text = "\n".join(text_parts).strip()
    else:
        text = ""

    if not text and stage:
        model = GROQ_MODEL if LLM_PROVIDER == "groq" else GITHUB_MODEL
        metadata = getattr(response, "response_metadata", None) or {}
        finish_reason = metadata.get("finish_reason")
        reason = f", finish_reason={finish_reason}" if finish_reason else ""
        raise ValueError(
            f"{stage} STTM generation received an empty response "
            f"(provider={LLM_PROVIDER}, model={model}{reason})."
        )

    return text


def _extract_csv(text: str) -> str:
    """Pull CSV text out of an LLM response: a ```csv fence first, else raw text."""
    if not text:
        return ""
    fence = re.search(r"```csv\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    fence = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def _validate_and_save(csv_text: str, out_path, stage: str) -> str:
    if not csv_text.strip():
        raise ValueError(f"{stage} STTM response contained no CSV content.")

    df = pd.read_csv(io.StringIO(csv_text))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"STTM is missing required columns: {missing}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


def _table_schema_context(profile_tables: dict) -> str:
    lines = []
    for table_name, table in profile_tables.items():
        lines.append(f"\nTable: {table_name} ({table['row_count']} rows)")
        for col_name, col in table["columns"].items():
            flags = f" flags={col['quality_flags']}" if col.get("quality_flags") else ""
            lines.append(
                f"  - {col_name}: dtype={col['dtype']} null_pct={col['null_pct']} "
                f"samples={col['sample_values']}{flags}"
            )
    return "\n".join(lines)


def _parquet_schema_context(paths: list[str]) -> str:
    lines = []
    for p in paths:
        df = pd.read_parquet(p)
        lines.append(f"\nTable: {p}")
        lines.append(f"  columns: {list(df.columns)}")
        lines.append(f"  dtypes: {df.dtypes.astype(str).to_dict()}")
        lines.append(f"  sample rows:\n{df.head(3).to_string(index=False)}")
    return "\n".join(lines)


# ── Phase 1b: Bronze STTM (LLM call 1) ──────────────────────────────────
def generate_bronze_sttm(profile_path: str, business_intent: str, run_id: str) -> str:
    from core.config import STTM_DIR

    with open(profile_path, "r", encoding="utf-8") as f:
        profile_data = json.load(f)

    schema_context = _table_schema_context(profile_data["tables"])

    prompt = f"""You are a data engineer generating a Bronze-layer Source-to-Target
Mapping (STTM) for a medallion data pipeline. Bronze is the fidelity layer:
type casting, column renaming, and metadata injection only. NO business logic.
Bronze is intent-agnostic - map EVERY column of every table below.

Business intent (context only, does not change Bronze): {business_intent}

Source table profiles:
{schema_context}

Produce a CSV with EXACTLY these 8 columns, no extra columns, no explanation:
source_schema,source_table,source_column,target_schema,target_table,target_column,transformation_type,transformation_logic

Rules:
- One row per source column with transformation_type=type_cast. Infer the
  target type from dtype/quality flags. transformation_logic must be one of:
  datetime, float, int, str.
- source_schema is always "landing". target_schema is always "bronze".
- target_table is "{{source_table}}_bronze".
- target_column is normally the same as source_column unless it needs cleanup.
- After the type_cast rows for a table, add exactly 2 metadata_inject rows:
  one for target_column=_load_timestamp, one for target_column=_source_file
  (source_column can be blank or "*", transformation_logic can be "system").

Return ONLY a ```csv fenced code block with the CSV content. No prose."""

    response = _call_llm(prompt, "Bronze")
    csv_text = _extract_csv(response)
    out_path = STTM_DIR / f"sttm_bronze_{run_id}.csv"
    result_path = _validate_and_save(csv_text, out_path, "Bronze")

    AuditLogger(run_id).log(agent="sttm_generator", action="bronze_sttm_generated", path=result_path)
    return result_path


# ── Phase 2b: Silver STTM (LLM call 2) ──────────────────────────────────
def generate_silver_sttm(
    bronze_paths: list[str], bronze_sttm_path: str, business_intent: str, run_id: str
) -> str:
    from core.config import STTM_DIR

    schema_context = _parquet_schema_context(bronze_paths)
    bronze_sttm_text = pd.read_csv(bronze_sttm_path).to_csv(index=False)

    prompt = f"""You are a data engineer generating a Silver-layer Source-to-Target
Mapping (STTM). Silver is the quality layer: null handling, date standardisation,
text normalisation, deduplication, and a surrogate key. NO joins here.

Business intent (context only): {business_intent}

Bronze parquet schemas:
{schema_context}

The Bronze STTM used to produce these tables (for context):
{bronze_sttm_text}

Produce a CSV with EXACTLY these 8 columns, no extra columns, no explanation:
source_schema,source_table,source_column,target_schema,target_table,target_column,transformation_type,transformation_logic

Rules:
- source_schema is "bronze", source_table is the bronze table name (without the
  run_id suffix), target_schema is "silver", target_table is "{{table}}_silver".
- For date/datetime columns: transformation_type=date, transformation_logic must
  contain the word "date" (e.g. "standardize date to YYYY-MM-DD").
- For text columns with mixed case/whitespace: transformation_type=cleanse,
  transformation_logic should contain one of: lowercase, uppercase, title case, strip.
- For numeric columns with nulls: transformation_type=null_handling,
  transformation_logic should contain "fill null" + mean/median/mode/0, or
  "drop null" if the column is a required key.
- Add one transformation_type=deduplicate row per table (transformation_logic="deduplicate").
- Map every column you keep with a passthrough or cleanse rule; you do not need
  to emit rows for the metadata columns (_load_timestamp, _source_file).

Return ONLY a ```csv fenced code block with the CSV content. No prose."""

    response = _call_llm(prompt, "Silver")
    csv_text = _extract_csv(response)
    out_path = STTM_DIR / f"sttm_silver_{run_id}.csv"
    result_path = _validate_and_save(csv_text, out_path, "Silver")

    AuditLogger(run_id).log(agent="sttm_generator", action="silver_sttm_generated", path=result_path)
    return result_path


# ── Phase 3b: Gold STTM (LLM call 3) ────────────────────────────────────
def generate_gold_sttm(
    silver_paths: list[str], silver_sttm_path: str, business_intent: str, run_id: str
) -> str:
    from core.config import STTM_DIR

    schema_context = _parquet_schema_context(silver_paths)
    silver_sttm_text = pd.read_csv(silver_sttm_path).to_csv(index=False)

    prompt = f"""You are a data engineer generating a Gold-layer Source-to-Target
Mapping (STTM). Gold is the analytics layer, shaped entirely by the business
intent below: joins, group-by, and aggregations.

Business intent: {business_intent}

Silver parquet schemas:
{schema_context}

The Silver STTM used to produce these tables (for context):
{silver_sttm_text}

Produce a CSV with EXACTLY these 8 columns, no extra columns, no explanation:
source_schema,source_table,source_column,target_schema,target_table,target_column,transformation_type,transformation_logic

Rules:
- source_schema is "silver". target_schema is "gold". Pick ONE target_table
  name that answers the business intent, e.g. "sales_by_category_gold".
- For each join needed: transformation_type=join,
  transformation_logic="join_left:<left_table>:<right_table>:<key_column>"
  where <left_table>/<right_table> are the silver table names (without the
  "_silver_<run_id>" suffix, i.e. the base names like "sales_data_silver").
- For the grouping column(s): transformation_type=group_by, target_column=<col>.
- For each aggregation: transformation_type=aggregate,
  transformation_logic="SUM(<col>) AS <alias>" (or AVG/COUNT/MAX/MIN),
  target_column=<alias>.
- Keep the rule set small and directly answering the business intent.

Return ONLY a ```csv fenced code block with the CSV content. No prose."""

    response = _call_llm(prompt, "Gold")
    csv_text = _extract_csv(response)
    out_path = STTM_DIR / f"sttm_gold_{run_id}.csv"
    result_path = _validate_and_save(csv_text, out_path, "Gold")

    AuditLogger(run_id).log(agent="sttm_generator", action="gold_sttm_generated", path=result_path)
    return result_path
