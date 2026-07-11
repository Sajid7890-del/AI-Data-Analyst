"""
ai_engine.py
------------
Natural-language question answering over a pandas DataFrame.

Two modes, chosen automatically:
  1. LLM mode  - if an OpenAI or Gemini API key is configured, the
     dataset's schema + a small statistical profile is sent to the
     model along with the user's question, and the model is asked to
     respond in plain English (never given raw data row-by-row, to
     keep prompts small and avoid leaking full datasets unnecessarily).
  2. Rule-based fallback - a lightweight intent-matching engine that
     covers the exact example questions in the spec (totals, top
     sellers, monthly trends, top customers, forecasts, top-N,
     trends, recommendations) using pure pandas. This means the app
     is fully functional and demoable with ZERO API key.

Both modes also drive automatic business-insight generation.
"""

import re
import json
import numpy as np
import pandas as pd

from config import (
    OPENAI_API_KEY, GEMINI_API_KEY, AI_PROVIDER, OPENAI_MODEL, GEMINI_MODEL,
)
from utils import (
    get_basic_info, get_numeric_columns, get_categorical_columns,
    find_likely_date_column, find_likely_amount_column, detect_outliers,
    get_correlation_matrix,
)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------
def available_provider():
    if AI_PROVIDER in ("openai", "gemini"):
        return AI_PROVIDER if _has_key(AI_PROVIDER) else None
    if OPENAI_API_KEY:
        return "openai"
    if GEMINI_API_KEY:
        return "gemini"
    return None


def _has_key(provider):
    return bool(OPENAI_API_KEY) if provider == "openai" else bool(GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Dataset profile (used both for LLM context and rule-based reasoning)
# ---------------------------------------------------------------------------
def build_dataset_profile(df: pd.DataFrame) -> str:
    info = get_basic_info(df)
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)
    profile = {
        "rows": info["rows"],
        "columns": info["columns"],
        "column_names_and_types": info["column_types"],
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "sample_rows": df.head(5).to_dict(orient="records"),
    }
    if numeric_cols:
        profile["numeric_summary"] = df[numeric_cols].describe().round(2).to_dict()
    return json.dumps(profile, default=str)


# ---------------------------------------------------------------------------
# LLM-backed Q&A
# ---------------------------------------------------------------------------
def ask_llm(question: str, df: pd.DataFrame) -> str:
    provider = available_provider()
    profile_json = build_dataset_profile(df)

    system_prompt = (
        "You are an expert data analyst. You are given a JSON profile of a "
        "user's dataset (schema, types, sample rows, and numeric summary "
        "statistics) and a question in plain English. Answer clearly and "
        "concisely using simple, non-technical English, referencing actual "
        "numbers from the profile where possible. If the question cannot be "
        "answered from the given profile alone, say what additional "
        "aggregation would be needed instead of guessing."
    )
    user_prompt = f"Dataset profile:\n{profile_json}\n\nQuestion: {question}"

    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return resp.choices[0].message.content.strip()

        elif provider == "gemini":
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(f"{system_prompt}\n\n{user_prompt}")
            return resp.text.strip()

    except Exception as e:
        return f"⚠️ AI provider error ({provider}): {e}\n\nFalling back to rule-based answer:\n\n" + rule_based_answer(question, df)

    return rule_based_answer(question, df)


# ---------------------------------------------------------------------------
# Rule-based fallback engine (no API key required)
# ---------------------------------------------------------------------------
def rule_based_answer(question: str, df: pd.DataFrame) -> str:
    q = question.lower().strip()
    amount_col = find_likely_amount_column(df)
    date_col = find_likely_date_column(df)
    categorical_cols = get_categorical_columns(df)
    numeric_cols = get_numeric_columns(df)

    # --- total / sum questions -------------------------------------------------
    if re.search(r"\btotal\b|\bsum\b", q):
        target = _match_column(q, numeric_cols) or amount_col
        if target:
            total = df[target].sum()
            return f"The total {target} is **{total:,.2f}**."
        return "I couldn't find a numeric column to total. Try naming one explicitly."

    # --- "which X sold the most / top" -----------------------------------------
    if re.search(r"most|top|best|highest", q) and re.search(r"sold|selling|sale|product|item", q):
        cat = _match_column(q, categorical_cols) or (categorical_cols[0] if categorical_cols else None)
        if cat and amount_col:
            grouped = df.groupby(cat)[amount_col].sum().sort_values(ascending=False)
            if not grouped.empty:
                top = grouped.index[0]
                return f"**{top}** sold the most, with a total {amount_col} of **{grouped.iloc[0]:,.2f}**."
        return "I need both a category column (e.g., product) and a numeric amount column to answer this."

    # --- top N ---------------------------------------------------------------
    n_match = re.search(r"top\s+(\d+)", q)
    if n_match:
        n = int(n_match.group(1))
        cat = _match_column(q, categorical_cols) or (categorical_cols[0] if categorical_cols else None)
        if cat and amount_col:
            grouped = df.groupby(cat)[amount_col].sum().sort_values(ascending=False).head(n)
            lines = "\n".join(f"{i+1}. {idx} — {val:,.2f}" for i, (idx, val) in enumerate(grouped.items()))
            return f"Here are the top {n} {cat} by {amount_col}:\n\n{lines}"
        return "I need a category and a numeric column to rank the top items."

    # --- predict / forecast (checked BEFORE the generic trend branch, since
    # phrases like "predict next month's sales" also contain "month") ----------
    if re.search(r"predict|forecast|next month|next quarter|next year", q):
        if date_col and amount_col:
            from ml_engine import simple_forecast
            forecast_value, trend_desc = simple_forecast(df, date_col, amount_col)
            if forecast_value is not None:
                return (
                    f"Based on the historical trend, projected {amount_col} for the next period "
                    f"is approximately **{forecast_value:,.2f}** ({trend_desc}).\n\n"
                    f"_For a more rigorous forecast, use the Machine Learning tab, which fits "
                    f"a proper regression model with accuracy metrics._"
                )
        return "I need a date column and a numeric column to forecast. Try the Machine Learning tab for a full model."

    # --- monthly / trend over time ---------------------------------------------
    if re.search(r"month|trend|over time|by date|time series|weekly|yearly|quarterly", q):
        if date_col and amount_col:
            temp = df.copy()
            temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
            temp = temp.dropna(subset=[date_col])
            freq = "ME"
            if "week" in q:
                freq = "W"
            elif "year" in q:
                freq = "YE"
            elif "quarter" in q:
                freq = "QE"
            grouped = temp.set_index(date_col).resample(freq)[amount_col].sum()
            lines = "\n".join(f"- {idx.strftime('%Y-%m-%d')}: {val:,.2f}" for idx, val in grouped.items())
            trend_word = "increasing 📈" if grouped.iloc[-1] > grouped.iloc[0] else "decreasing 📉"
            return f"{amount_col} by period (overall trend is {trend_word}):\n\n{lines}"
        return "I need a date column and a numeric column to show a trend."

    # --- top customer / who spent the most --------------------------------------
    if re.search(r"customer|client|buyer", q) and re.search(r"most|top|highest|best", q):
        cust_col = _match_column(q, categorical_cols, keywords=["customer", "client", "buyer", "name"])
        if cust_col and amount_col:
            grouped = df.groupby(cust_col)[amount_col].sum().sort_values(ascending=False)
            top = grouped.index[0]
            return f"**{top}** is the top customer, having spent a total of **{grouped.iloc[0]:,.2f}**."
        return "I couldn't find a customer column and an amount column to answer this."

    # --- trends / findings -------------------------------------------------------
    if re.search(r"find trends|trends|patterns", q):
        return generate_key_findings(df)

    # --- recommendations / business insights ------------------------------------
    if re.search(r"recommend|advice|suggest|opportunit|risk", q):
        return generate_recommendations(df)

    # --- average / mean -----------------------------------------------------------
    if re.search(r"average|mean", q):
        target = _match_column(q, numeric_cols) or amount_col
        if target:
            return f"The average {target} is **{df[target].mean():,.2f}**."

    # --- min / max ---------------------------------------------------------------
    if re.search(r"maximum|highest value|max\b", q):
        target = _match_column(q, numeric_cols) or amount_col
        if target:
            return f"The maximum {target} is **{df[target].max():,.2f}**."
    if re.search(r"minimum|lowest value|min\b", q):
        target = _match_column(q, numeric_cols) or amount_col
        if target:
            return f"The minimum {target} is **{df[target].min():,.2f}**."

    # --- count / how many rows ----------------------------------------------------
    if re.search(r"how many rows|row count|number of records|count of rows", q):
        return f"The dataset contains **{len(df):,}** rows."

    # --- fallback: general profile summary -----------------------------------------
    return (
        "I'm not fully sure how to answer that with the rule-based engine. "
        "Here's a quick profile of your data instead:\n\n" + generate_key_findings(df) +
        "\n\n_Tip: add an OpenAI or Gemini API key in your environment for fully open-ended Q&A._"
    )


def _match_column(question: str, candidates: list, keywords: list = None) -> str:
    """Finds the candidate column whose name (or given keywords) appears in the question."""
    q = question.lower()
    for col in candidates:
        if col.lower() in q:
            return col
    if keywords:
        for col in candidates:
            if any(k in col.lower() for k in keywords):
                return col
    return None


# ---------------------------------------------------------------------------
# Insights (used by both chat "find trends"/"recommendations" and the
# Business Insights tab)
# ---------------------------------------------------------------------------
def generate_key_findings(df: pd.DataFrame) -> str:
    info = get_basic_info(df)
    findings = [
        f"- Dataset has **{info['rows']:,} rows** and **{info['columns']} columns**.",
        f"- **{info['missing_values']:,}** missing values and **{info['duplicate_rows']:,}** duplicate rows detected.",
    ]

    corr = get_correlation_matrix(df)
    if not corr.empty:
        strong_pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.7:
                    strong_pairs.append(f"{cols[i]} & {cols[j]} (r={val:.2f})")
        if strong_pairs:
            findings.append("- Strong correlations found: " + ", ".join(strong_pairs) + ".")

    outliers = detect_outliers(df)
    if outliers:
        top_outlier_col = max(outliers, key=lambda c: outliers[c]["count"])
        findings.append(
            f"- **{top_outlier_col}** has the most outliers "
            f"({outliers[top_outlier_col]['count']} values, {outliers[top_outlier_col]['pct']}% of data)."
        )

    amount_col = find_likely_amount_column(df)
    date_col = find_likely_date_column(df)
    if amount_col and date_col:
        temp = df.copy()
        temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
        temp = temp.dropna(subset=[date_col]).sort_values(date_col)
        if len(temp) > 1:
            monthly = temp.set_index(date_col).resample("ME")[amount_col].sum()
            if len(monthly) > 1:
                pct_change = ((monthly.iloc[-1] - monthly.iloc[0]) / max(abs(monthly.iloc[0]), 1e-9)) * 100
                direction = "increased" if pct_change > 0 else "decreased"
                findings.append(f"- {amount_col} has **{direction} by {abs(pct_change):.1f}%** from the first to the latest period.")

    return "\n".join(findings)


def generate_recommendations(df: pd.DataFrame) -> str:
    amount_col = find_likely_amount_column(df)
    categorical_cols = get_categorical_columns(df)
    outliers = detect_outliers(df)
    info = get_basic_info(df)

    recs = []
    if info["duplicate_rows"] > 0:
        recs.append(f"**Data quality:** Remove {info['duplicate_rows']} duplicate rows before reporting to stakeholders.")
    if info["missing_values"] > 0:
        recs.append(f"**Data quality:** Investigate {info['missing_values']} missing values — consider imputation or source-system fixes.")
    if outliers:
        col = max(outliers, key=lambda c: outliers[c]["count"])
        recs.append(f"**Risk:** {col} contains {outliers[col]['count']} outliers — verify these aren't data-entry errors before using them in models.")
    if amount_col and categorical_cols:
        grouped = df.groupby(categorical_cols[0])[amount_col].sum().sort_values(ascending=False)
        if len(grouped) > 1:
            top_share = grouped.iloc[0] / grouped.sum() * 100
            if top_share > 40:
                recs.append(
                    f"**Opportunity/Risk:** '{grouped.index[0]}' drives {top_share:.1f}% of total {amount_col} — "
                    f"a concentration risk, but also a clear opportunity to double down on what's working."
                )
            recs.append(f"**Opportunity:** Focus marketing/sales effort on the top {min(3, len(grouped))} performing {categorical_cols[0]} segments.")
    if not recs:
        recs.append("Data looks clean and well-distributed — consider building a forecasting model to plan ahead (see the Machine Learning tab).")

    return "\n".join(f"- {r}" for r in recs)


def generate_executive_summary(df: pd.DataFrame) -> str:
    info = get_basic_info(df)
    amount_col = find_likely_amount_column(df)
    summary_lines = [
        f"This dataset contains {info['rows']:,} records across {info['columns']} columns, "
        f"with {info['missing_values']:,} missing values and {info['duplicate_rows']:,} duplicate rows."
    ]
    if amount_col:
        summary_lines.append(
            f"Total {amount_col} across all records is {df[amount_col].sum():,.2f}, "
            f"averaging {df[amount_col].mean():,.2f} per record."
        )
    summary_lines.append("Key findings:\n" + generate_key_findings(df))
    summary_lines.append("Recommendations:\n" + generate_recommendations(df))
    return "\n\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Unified entrypoint used by app.py
# ---------------------------------------------------------------------------
def answer_question(question: str, df: pd.DataFrame) -> str:
    if available_provider():
        return ask_llm(question, df)
    return rule_based_answer(question, df)
