"""
utils.py
--------
Data loading, validation, cleaning, and statistical profiling helpers.
Pure pandas/numpy logic with no Streamlit or AI dependencies, so it is
easy to unit test in isolation.
"""

import os
import numpy as np
import pandas as pd

from config import ALLOWED_EXTENSIONS, OUTLIER_IQR_MULTIPLIER


# ---------------------------------------------------------------------------
# File validation & loading
# ---------------------------------------------------------------------------
def validate_file(filename: str) -> tuple[bool, str]:
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type '{ext}'. Please upload CSV or Excel files."
    return True, "OK"


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Loads a CSV or Excel file into a DataFrame, trying a few encodings."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        for encoding in ("utf-8", "latin1", "cp1252"):
            try:
                return pd.read_csv(filepath, encoding=encoding)
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        raise ValueError("Could not parse CSV file with common encodings.")
    elif ext in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def detect_column_types(df: pd.DataFrame) -> dict:
    """Classifies each column as numeric, datetime, categorical, or text."""
    types = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime"
        else:
            # Try to parse as date
            sample = series.dropna().astype(str).head(20)
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            if len(sample) > 0 and parsed.notna().mean() > 0.8:
                types[col] = "datetime"
            elif series.nunique(dropna=True) <= max(20, len(series) * 0.05):
                types[col] = "categorical"
            else:
                types[col] = "text"
    return types


def handle_missing_values(df: pd.DataFrame, strategy: str = "report_only") -> pd.DataFrame:
    """
    strategy:
      - "report_only": leaves data untouched (default, safest)
      - "drop": drops rows with any missing values
      - "fill_mean": fills numeric NaNs with column mean, categorical with mode
    """
    if strategy == "drop":
        return df.dropna()
    if strategy == "fill_mean":
        df = df.copy()
        for col in df.columns:
            if df[col].isna().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    mode = df[col].mode(dropna=True)
                    fill_val = mode.iloc[0] if not mode.empty else "Unknown"
                    df[col] = df[col].fillna(fill_val)
        return df
    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Returns the duplicate rows found in the dataset."""
    return df[df.duplicated(keep=False)]


# ---------------------------------------------------------------------------
# Profiling / statistics
# ---------------------------------------------------------------------------
def get_basic_info(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 ** 2), 3),
        "column_types": detect_column_types(df),
    }


def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    pct = (missing / len(df) * 100).round(2)
    report = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return report[report["missing_count"] > 0].sort_values("missing_count", ascending=False)


def get_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return pd.DataFrame()
    return numeric_df.describe().T.round(3)


def get_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return pd.DataFrame()
    return numeric_df.corr().round(3)


def detect_outliers(df: pd.DataFrame) -> dict:
    """IQR-based outlier detection for every numeric column."""
    results = {}
    numeric_df = df.select_dtypes(include=[np.number])
    for col in numeric_df.columns:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + OUTLIER_IQR_MULTIPLIER * iqr
        outliers = series[(series < lower) | (series > upper)]
        if len(outliers) > 0:
            results[col] = {
                "count": len(outliers),
                "pct": round(len(outliers) / len(series) * 100, 2),
                "lower_bound": round(lower, 3),
                "upper_bound": round(upper, 3),
                "min_outlier": round(outliers.min(), 3),
                "max_outlier": round(outliers.max(), 3),
            }
    return results


def get_numeric_columns(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_columns(df: pd.DataFrame) -> list:
    types = detect_column_types(df)
    return [c for c, t in types.items() if t == "categorical"]


def get_datetime_columns(df: pd.DataFrame) -> list:
    types = detect_column_types(df)
    return [c for c, t in types.items() if t == "datetime"]


def find_likely_date_column(df: pd.DataFrame):
    candidates = get_datetime_columns(df)
    if candidates:
        return candidates[0]
    for col in df.columns:
        if any(k in col.lower() for k in ("date", "time", "month", "year")):
            return col
    return None


def find_likely_amount_column(df: pd.DataFrame):
    """
    Picks the numeric column most likely to represent a monetary total.
    Checked in priority tiers so e.g. 'Sales' wins over 'UnitPrice' even
    though both loosely relate to money.
    """
    numeric_cols = get_numeric_columns(df)
    priority_tiers = [
        ("sales", "revenue", "total"),
        ("amount", "spent", "income", "profit"),
        ("price", "value", "cost"),
    ]
    for tier in priority_tiers:
        for col in numeric_cols:
            if any(k in col.lower() for k in tier):
                return col
    return numeric_cols[0] if numeric_cols else None


def format_bytes(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"
