"""PipelineState - the baton that carries file paths between pipeline phases."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PipelineState:
    run_id: str
    input_files: list[str] = field(default_factory=list)
    business_intent: str = ""

    profile_path: Optional[str] = None

    bronze_sttm_path: Optional[str] = None
    bronze_paths: list[str] = field(default_factory=list)

    silver_sttm_path: Optional[str] = None
    silver_paths: list[str] = field(default_factory=list)

    gold_sttm_path: Optional[str] = None
    gold_paths: list[str] = field(default_factory=list)

    report_path: Optional[str] = None
