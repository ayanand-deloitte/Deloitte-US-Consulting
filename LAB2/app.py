"""Streamlit UI for the IDAMP medallion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from agents import bronze_agent, gold_agent, profiler, reporter, silver_agent, sttm_generator as sttm_gen
from core.audit import AuditLogger, new_run_id
from core.config import LANDING_DIR, ensure_dirs
from core.state import PipelineState

st.set_page_config(page_title="IDAMP", page_icon="📊", layout="wide")


def _save_uploaded_files(files: Iterable[st.runtime.uploaded_file.UploadedFile]) -> list[str]:
    """Copy uploaded CSV files into the landing directory."""
    saved: list[str] = []
    for uploaded in files:
        if not uploaded.name.lower().endswith(".csv"):
            st.warning(f"Skipped {uploaded.name}: only CSV files are supported.")
            continue

        destination = LANDING_DIR / uploaded.name
        destination.write_bytes(uploaded.getvalue())
        saved.append(str(destination))

    return saved


def _show_sttm_editor(sttm_path: str, layer: str) -> bool | None:
    """Display an STTM and return whether the user approved it."""
    st.subheader(f"{layer} STTM review")
    st.caption(f"Review and edit the generated rules before continuing.")

    with open(sttm_path, encoding="utf-8") as f:
        edited = st.text_area(
            f"{layer} STTM CSV",
            f.read(),
            height=420,
            help="Edit the CSV and click Approve when it is ready.",
        )

    if st.button(f"Approve {layer} STTM", key=f"approve-{layer.lower()}", use_container_width=True):
        Path(sttm_path).write_text(edited, encoding="utf-8")
        return True

    if st.button(f"Abort at {layer} stage", key=f"abort-{layer.lower()}", type="secondary"):
        return False

    return None


def _load_sample_files() -> list[str]:
    candidates = [
        LANDING_DIR / "sales_data_2.csv",
        LANDING_DIR / "products_3.csv",
        LANDING_DIR / "stores_3.csv",
    ]
    return [str(path) for path in candidates if path.exists()]


def run_pipeline_ui(files: list[str], intent: str) -> PipelineState | None:
    """Initialize a run and generate the Bronze STTM for its first review gate."""
    ensure_dirs()
    run_id = new_run_id()
    audit = AuditLogger(run_id)
    state = PipelineState(run_id=run_id, input_files=[], business_intent=intent)
    state.input_files = [str(Path(f)) for f in files]

    st.session_state.current_run = run_id
    st.session_state.pipeline_state = state
    st.session_state.pipeline_stage = "bronze_review"

    with st.status("Starting IDAMP run", expanded=True):
        st.write(f"Run ID: {run_id}")
        st.write(f"Business intent: {intent}")

        st.write("Phase 1: profiling input tables")
        state.profile_path = profiler.profile(files, run_id)
        st.write(f"Profile saved to {state.profile_path}")

        st.write("Phase 1: generating Bronze STTM")
        state.bronze_sttm_path = sttm_gen.generate_bronze_sttm(state.profile_path, intent, run_id)
        audit.log(agent="ui", action="awaiting_approval", stage="bronze")

    return state


def _continue_pipeline_ui() -> None:
    """Render the active STTM gate and resume processing after each approval."""
    state: PipelineState = st.session_state.pipeline_state
    stage = st.session_state.get("pipeline_stage")
    audit = AuditLogger(state.run_id)

    while stage in {"bronze_review", "silver_review", "gold_review"}:
        layer, sttm_path = {
            "bronze_review": ("Bronze", state.bronze_sttm_path),
            "silver_review": ("Silver", state.silver_sttm_path),
            "gold_review": ("Gold", state.gold_sttm_path),
        }[stage]
        if not sttm_path:
            raise RuntimeError(f"{layer} STTM path is missing for this pipeline stage.")

        approval = _show_sttm_editor(sttm_path, layer)
        if approval is None:
            st.info(f"Review the {layer} STTM and approve it to continue.")
            return
        if approval is False:
            st.session_state.pipeline_stage = "aborted"
            audit.log(agent="ui", action="aborted", stage=layer.lower(), reason="user_rejected")
            st.info(f"Pipeline aborted at the {layer} approval gate.")
            return

        with st.status(f"Executing {layer} stage", expanded=True) as status:
            if stage == "bronze_review":
                state.bronze_paths = bronze_agent.run(
                    state.input_files, state.bronze_sttm_path, state.run_id
                )
                status.write(f"Bronze tables written: {len(state.bronze_paths)}")
                state.silver_sttm_path = sttm_gen.generate_silver_sttm(
                    state.bronze_paths,
                    state.bronze_sttm_path,
                    state.business_intent,
                    state.run_id,
                )
                stage = "silver_review"
            elif stage == "silver_review":
                state.silver_paths = silver_agent.run(
                    state.bronze_paths, state.silver_sttm_path, state.run_id
                )
                status.write(f"Silver tables written: {len(state.silver_paths)}")
                state.gold_sttm_path = sttm_gen.generate_gold_sttm(
                    state.silver_paths,
                    state.silver_sttm_path,
                    state.business_intent,
                    state.run_id,
                )
                stage = "gold_review"
            else:
                state.gold_paths = gold_agent.run(
                    state.silver_paths,
                    state.gold_sttm_path,
                    state.business_intent,
                    state.run_id,
                )
                status.write(f"Gold tables written: {len(state.gold_paths)}")
                state.report_path = reporter.generate_report(
                    state.gold_paths, state.business_intent, state.run_id
                )
                stage = "completed"

            st.session_state.pipeline_stage = stage
            if stage.endswith("_review"):
                audit.log(agent="ui", action="awaiting_approval", stage=stage.removesuffix("_review"))
            status.update(
                label=f"{layer} stage complete" if stage != "completed" else "Pipeline completed",
                state="complete",
            )

    if stage == "completed":
        st.success(f"Pipeline completed. Report generated: {state.report_path}")
    elif stage == "aborted":
        st.info("This pipeline run was aborted at an approval gate.")


def main() -> None:
    st.title("IDAMP Pipeline")
    st.caption("Run the existing medallion pipeline with a human approval gate for each STTM.")

    with st.sidebar:
        st.header("Inputs")
        uploaded_files = st.file_uploader(
            "Upload CSV files",
            type="csv",
            accept_multiple_files=True,
        )
        use_samples = st.checkbox("Use the project sample files", value=True)
        intent = st.text_area(
            "Business intent",
            value="Which product category generated the highest total sales revenue?",
            height=100,
        )

        run_button = st.button("Run pipeline", type="primary", use_container_width=True)

    if uploaded_files:
        available = _save_uploaded_files(uploaded_files)
    elif use_samples:
        available = _load_sample_files()
    else:
        available = []

    if available:
        st.write("Selected files:")
        for path in available:
            st.code(path)

    if run_button:
        if not available:
            st.error("Select at least one CSV file or enable the sample files.")
            return
        if not intent.strip():
            st.error("Enter a business intent.")
            return

        try:
            run_pipeline_ui(available, intent.strip())
        except RuntimeError as exc:
            st.error(str(exc))
        except Exception as exc:  # pragma: no cover - UI-level protection
            st.error(f"Pipeline failed: {exc}")

    if "pipeline_state" in st.session_state:
        if (
            "pipeline_stage" not in st.session_state
            and st.session_state.pipeline_state.bronze_sttm_path
        ):
            st.session_state.pipeline_stage = "bronze_review"

        if st.session_state.get("pipeline_stage") in {
            "bronze_review",
            "silver_review",
            "gold_review",
            "completed",
            "aborted",
        }:
            try:
                _continue_pipeline_ui()
            except RuntimeError as exc:
                st.error(str(exc))
            except Exception as exc:  # pragma: no cover - UI-level protection
                st.error(f"Pipeline failed: {exc}")

    if "pipeline_state" in st.session_state:
        state: PipelineState = st.session_state.pipeline_state
        st.divider()
        st.subheader("Run summary")
        st.write(f"Run ID: {state.run_id}")
        st.write(f"Input files: {len(state.input_files)}")

        if state.profile_path:
            st.write(f"Profile: {state.profile_path}")
        if state.bronze_sttm_path:
            st.write(f"Bronze STTM: {state.bronze_sttm_path}")
        if state.silver_sttm_path:
            st.write(f"Silver STTM: {state.silver_sttm_path}")
        if state.gold_sttm_path:
            st.write(f"Gold STTM: {state.gold_sttm_path}")
        if state.report_path:
            st.link_button("Open generated report", state.report_path)


if __name__ == "__main__":
    main()
