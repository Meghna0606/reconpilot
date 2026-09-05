from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
DATA_DIR = ROOT_DIR / "data"
BENCHMARK_DIR = ROOT_DIR / "benchmark"


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or not value.strip() else value.strip()


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))

DB_PATH = Path(_env("RECONPILOT_DB_PATH", str(DATA_DIR / "reconpilot.db")))
if not DB_PATH.is_absolute():
    DB_PATH = ROOT_DIR / DB_PATH
DATABASE_URL = f"sqlite:///{DB_PATH}"

AI_MODE = _env("RECONPILOT_AI_MODE", "live").lower()
LITELLM_MODEL = _env("LITELLM_MODEL", "gpt-4o-mini")
LLM_TIMEOUT_SECONDS = _env_int("LLM_TIMEOUT_SECONDS", 20)
DETERMINISTIC_CONFIDENCE_MIN = _env_float("DETERMINISTIC_CONFIDENCE_MIN", 0.95)
AI_CONFIDENCE_MIN = _env_float("AI_CONFIDENCE_MIN", 0.75)
FEE_RATE = _env_float("FEE_RATE", 0.02)
DATE_WINDOW_DAYS = _env_int("DATE_WINDOW_DAYS", 2)
