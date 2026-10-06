"""IDAMP orchestrator - interactive CLI with human-in-the-loop approval gates.

Wires the 4 pipeline phases together:
  Phase 1: Profile (Python)          + Bronze STTM (LLM 1) -> gate
  Phase 2: Bronze execution (Python) + Silver STTM (LLM 2) -> gate
  Phase 3: Silver execution (Python) + Gold STTM   (LLM 3) -> gate
  Phase 4: Gold execution (Python)   + Report      (LLM 4)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

# Windows terminals often default to a legacy codepage (cp1252) that can't
# render tabulate's Unicode box-drawing characters. Force UTF-8 stdout/stderr
# where the runtime supports it so the STTM tables print cleanly everywhere.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

from core.config import LANDING_DIR, ensure_dirs
from core.state import PipelineState
from core.audit import new_run_id, AuditLogger

from agents import profiler
from agents import sttm_generator as sttm_gen
from agents import bronze_agent
from agents import silver_agent
from agents import gold_agent
from agents import reporter


def banner(text: str) -> None:
    line = "=" * max(len(text) + 4, 40)
    print(f"\n{line}\n  {text}\n{line}")


def display_sttm(sttm_path: str, layer: str) -> None:
    df = pd.read_csv(sttm_path)
    print(f"\n--- {layer} STTM ({sttm_path}) - {len(df)} rules ---")
    print(tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False))


def hitl_gate(layer: str, sttm_path: str) -> bool:
    display_sttm(sttm_path, layer)
    while True:
        choice = input(f"\n[{layer}] [y]es / [e]dit then re-review / [n]o abort > ").strip().lower()
        if choice == "y":
            return True
        if choice == "n":
            print(f"Aborted at the {layer} gate.")
            return False
        if choice == "e":
            editor = os.environ.get("EDITOR")
            if editor:
                subprocess.run([editor, sttm_path])
            elif os.name == "nt":
                subprocess.run(["notepad", sttm_path])
            else:
                subprocess.run(["nano", sttm_path])
            display_sttm(sttm_path, layer)
            continue
        print("Please enter y, e, or n.")


def run_pipeline(files: list[str], intent: str) -> PipelineState:
    ensure_dirs()
    run_id = new_run_id()
    audit = AuditLogger(run_id)
    state = PipelineState(run_id=run_id, input_files=[], business_intent=intent)

    banner(f"IDAMP run {run_id}")
    print(f"Business intent: {intent}")

    landed_files = []
    for f in files:
        src = Path(f)
        dest = LANDING_DIR / src.name
        if src.resolve() != dest.resolve():
            shutil.copy(src, dest)
        landed_files.append(str(dest))
    state.input_files = landed_files

    # ── Phase 1: Profile (Python) + Bronze STTM (LLM 1) ──
    banner("Phase 1 - Profiling + Bronze STTM")
    state.profile_path = profiler.profile(landed_files, run_id)
    print(f"Profile written to {state.profile_path}")
    state.bronze_sttm_path = sttm_gen.generate_bronze_sttm(state.profile_path, intent, run_id)
    if not hitl_gate("Bronze", state.bronze_sttm_path):
        return state

    # ── Phase 2: Bronze execution (Python) + Silver STTM (LLM 2) ──
    banner("Phase 2 - Bronze execution + Silver STTM")
    state.bronze_paths = bronze_agent.run(landed_files, state.bronze_sttm_path, run_id)
    print(f"Bronze tables written: {len(state.bronze_paths)}")
    state.silver_sttm_path = sttm_gen.generate_silver_sttm(
        state.bronze_paths, state.bronze_sttm_path, intent, run_id
    )
    if not hitl_gate("Silver", state.silver_sttm_path):
        return state

    # ── Phase 3: Silver execution (Python) + Gold STTM (LLM 3) ──
    banner("Phase 3 - Silver execution + Gold STTM")
    state.silver_paths = silver_agent.run(state.bronze_paths, state.silver_sttm_path, run_id)
    print(f"Silver tables written: {len(state.silver_paths)}")
    state.gold_sttm_path = sttm_gen.generate_gold_sttm(
        state.silver_paths, state.silver_sttm_path, intent, run_id
    )
    if not hitl_gate("Gold", state.gold_sttm_path):
        return state

    # ── Phase 4: Gold execution (Python) + Report (LLM 4) ──
    banner("Phase 4 - Gold execution + Report")
    state.gold_paths = gold_agent.run(state.silver_paths, state.gold_sttm_path, intent, run_id)
    print(f"Gold tables written: {len(state.gold_paths)}")
    state.report_path = reporter.generate_report(state.gold_paths, intent, run_id)

    banner("Done")
    print(f"run_id:      {run_id}")
    print(f"report:      {state.report_path}")
    print(f"audit log:   {audit.log_path}")
    return state


def main():
    parser = argparse.ArgumentParser(description="IDAMP - Intent-Driven Agentic Medallion Pipeline")
    parser.add_argument("--files", nargs="+", help="Paths to input CSV files")
    parser.add_argument("--intent", type=str, help="The business question to answer")
    args = parser.parse_args()

    files = args.files
    if not files:
        raw = input("Enter input CSV paths (space-separated): ").strip()
        files = raw.split()

    intent = args.intent
    if not intent:
        intent = input("Enter your business intent (question to answer): ").strip()

    if not files or not intent:
        print("Both --files and --intent are required.", file=sys.stderr)
        sys.exit(1)

    run_pipeline(files, intent)


if __name__ == "__main__":
    main()
