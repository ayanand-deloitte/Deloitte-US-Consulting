"""Central configuration for IDAMP: paths, directories, and LLM provider settings."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

LANDING_DIR = BASE_DIR / "data" / "landing"
PROFILES_DIR = BASE_DIR / "data" / "profiles"
STTM_DIR = BASE_DIR / "data" / "sttm"
BRONZE_DIR = BASE_DIR / "data" / "bronze_layer"
SILVER_DIR = BASE_DIR / "data" / "silver_layer"
GOLD_DIR = BASE_DIR / "data" / "gold_layer"
TRACES_DIR = BASE_DIR / "data" / "traces"
REPORTS_DIR = BASE_DIR / "reports"
AUDIT_DIR = BASE_DIR / "audit_logs"

# ── LLM provider ─────────────────────────────────────────────────────────
# Default provider is Groq (https://console.groq.com) - free, fast inference
# behind an OpenAI-compatible endpoint. GitHub Models is kept as a fallback
# so the "swap the provider" extension challenge (Step 10) has somewhere to go.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()

# Groq
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# GitHub Models (OpenAI-compatible, needs a publisher-prefixed model id)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_BASE_URL = os.getenv("GITHUB_BASE_URL", "https://models.github.ai/inference")
GITHUB_MODEL = os.getenv("GITHUB_MODEL", "gpt-4.1-mini")
if GITHUB_MODEL and "/" not in GITHUB_MODEL:
    GITHUB_MODEL = f"openai/{GITHUB_MODEL}"


def ensure_dirs() -> None:
    """Create every pipeline directory if it does not already exist."""
    for d in (
        LANDING_DIR,
        PROFILES_DIR,
        STTM_DIR,
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR,
        TRACES_DIR,
        REPORTS_DIR,
        AUDIT_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
