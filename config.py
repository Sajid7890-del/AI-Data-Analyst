"""
config.py
---------
Central configuration for the AI Data Analyst application.
All constants, paths, and environment-based settings live here so the
rest of the codebase never hard-codes a path or key.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Base paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
REPORTS_DIR = BASE_DIR / "reports"
DATABASE_DIR = BASE_DIR / "database"
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"

for _dir in (UPLOAD_DIR, REPORTS_DIR, DATABASE_DIR, ASSETS_DIR, MODELS_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DATABASE_PATH = str(DATABASE_DIR / "app_data.db")

# ---------------------------------------------------------------------------
# File upload settings
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls"}
MAX_FILE_SIZE_MB = 200
MAX_ROWS_PREVIEW = 100

# ---------------------------------------------------------------------------
# AI / LLM settings
# ---------------------------------------------------------------------------
# Reads keys from environment variables (or Streamlit secrets at runtime).
# The app works WITHOUT any key using a built-in rule-based analytical
# engine, so it is fully demonstrable out of the box.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Which provider to prefer if both are configured: "openai" | "gemini" | "auto"
AI_PROVIDER = os.environ.get("AI_PROVIDER", "auto")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

# ---------------------------------------------------------------------------
# App metadata
# ---------------------------------------------------------------------------
APP_NAME = "AI Data Analyst"
APP_ICON = "📊"
APP_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
# Used to salt/hash passwords. In production, override via environment
# variable so it isn't checked into source control.
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "change-this-secret-in-production")

# Session expects a logged-in user in st.session_state["user"]
SESSION_USER_KEY = "user"

# ---------------------------------------------------------------------------
# Charting / ML thresholds
# ---------------------------------------------------------------------------
OUTLIER_IQR_MULTIPLIER = 1.5
CORRELATION_THRESHOLD = 0.5
MAX_CATEGORIES_FOR_BAR = 20
