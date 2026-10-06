"""Append-only JSONL audit logger, plus the run-id helper."""

import json
from datetime import datetime, timezone

from core.config import AUDIT_DIR


def new_run_id() -> str:
    """Human-readable run id based on local time, e.g. '20260817_130455'."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class AuditLogger:
    def __init__(self, run_id: str | None = None):
        self.run_id = run_id or new_run_id()
        AUDIT_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent: str, action: str, **kwargs) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
            **kwargs,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_logs(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        with open(self.log_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
