"""Central configuration. Everything is overridable through environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.getenv("FP_DATA_DIR", ROOT / "data")).resolve()
DB_PATH = DATA_DIR / "factorypulse.db"
MODEL_DIR = DATA_DIR / "models"
RISK_MODEL_PATH = MODEL_DIR / "failure_model.pkl"

# Local LLM (IBM Granite 4.0 1B, GGUF, served by llama.cpp)
LLM_MODEL_PATH = Path(
    os.getenv(
        "FP_LLM_MODEL_PATH",
        r"C:\Users\ota3kor\Downloads\Models\ibm-granite-4.0-1b-Q4_K_M.gguf",
    )
)
LLM_CTX = int(os.getenv("FP_LLM_CTX", "4096"))
LLM_THREADS = int(os.getenv("FP_LLM_THREADS", "8"))
LLM_MAX_TOKENS = int(os.getenv("FP_LLM_MAX_TOKENS", "600"))

# Synthetic data
SEED = 42
N_DAYS = 30
SAMPLE_MINUTES = 5

# Prediction / alerting
PREDICTION_HORIZON_H = 24
RISK_CRITICAL = 0.70
RISK_WARNING = 0.40
# Hard engineering limits used by the rule-based safety net (ISO 10816 zone C/D style)
SENSOR_LIMITS = {"vibration_mm_s": 7.1, "temperature_c": 85.0}

SHIFTS = [("S1", 6), ("S2", 14), ("S3", 22)]  # shift name, start hour (8h shifts)
