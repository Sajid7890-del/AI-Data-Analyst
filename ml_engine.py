"""
ml_engine.py
------------
Lightweight machine-learning layer built on scikit-learn:
  - Linear Regression
  - Random Forest (regression & classification)
  - Simple time-based sales forecasting
  - Classification with accuracy / precision / recall
  - Feature importance extraction

Designed to work generically on any uploaded dataset: the caller picks
a target column and the module figures out reasonable defaults.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
)

from utils import get_numeric_columns, get_categorical_columns


def _prepare_features(df: pd.DataFrame, target_col: str, feature_cols: list):
    """Encodes categorical features numerically and drops rows with NaNs."""
    data = df[feature_cols + [target_col]].dropna().copy()
    encoders = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(data[col]):
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le
    X = data[feature_cols]
    y = data[target_col]
    return X, y, encoders


def run_linear_regression(df: pd.DataFrame, target_col: str, feature_cols: list = None):
    feature_cols = feature_cols or [c for c in get_numeric_columns(df) if c != target_col]
    if not feature_cols:
        return {"error": "No numeric feature columns available for regression."}

    X, y, _ = _prepare_features(df, target_col, feature_cols)
    if len(X) < 10:
        return {"error": "Not enough clean rows (need at least 10) to train a model."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "model_name": "Linear Regression",
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "feature_importance": dict(zip(feature_cols, np.round(np.abs(model.coef_), 4))),
        "features_used": feature_cols,
    }


def run_random_forest_regression(df: pd.DataFrame, target_col: str, feature_cols: list = None):
    feature_cols = feature_cols or [c for c in get_numeric_columns(df) if c != target_col]
    if not feature_cols:
        return {"error": "No numeric feature columns available for regression."}

    X, y, encoders = _prepare_features(df, target_col, feature_cols)
    if len(X) < 10:
        return {"error": "Not enough clean rows (need at least 10) to train a model."}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    return {
        "model": model,
        "model_name": "Random Forest Regressor",
        "r2": round(r2_score(y_test, preds), 4),
        "mae": round(mean_absolute_error(y_test, preds), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
        "feature_importance": dict(zip(feature_cols, np.round(model.feature_importances_, 4))),
        "features_used": feature_cols,
    }


def run_classification(df: pd.DataFrame, target_col: str, feature_cols: list = None):
    """Random Forest classifier. Target column is label-encoded automatically."""
    feature_cols = feature_cols or [c for c in df.columns if c != target_col]
    feature_cols = [c for c in feature_cols if c != target_col]
    if not feature_cols:
        return {"error": "No feature columns available for classification."}

    data = df[feature_cols + [target_col]].dropna().copy()
    if len(data) < 10:
        return {"error": "Not enough clean rows (need at least 10) to train a model."}

    X, y_raw, _ = _prepare_features(data, target_col, feature_cols)
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw.astype(str))

    if len(np.unique(y)) < 2:
        return {"error": f"'{target_col}' has only one unique value — classification needs at least 2 classes."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    avg = "binary" if len(np.unique(y)) == 2 else "weighted"
    return {
        "model": model,
        "model_name": "Random Forest Classifier",
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds, average=avg, zero_division=0), 4),
        "recall": round(recall_score(y_test, preds, average=avg, zero_division=0), 4),
        "f1": round(f1_score(y_test, preds, average=avg, zero_division=0), 4),
        "feature_importance": dict(zip(feature_cols, np.round(model.feature_importances_, 4))),
        "features_used": feature_cols,
        "classes": list(target_encoder.classes_),
    }


def simple_forecast(df: pd.DataFrame, date_col: str, value_col: str, periods_ahead: int = 1):
    """
    Quick linear-trend forecast used by the chat engine's 'predict next month'
    answers. Returns (forecast_value, trend_description) or (None, None).
    """
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    if len(temp) < 3:
        return None, None

    monthly = temp.set_index(date_col).resample("ME")[value_col].sum().reset_index()
    if len(monthly) < 2:
        return None, None

    monthly["t"] = np.arange(len(monthly))
    model = LinearRegression()
    model.fit(monthly[["t"]], monthly[value_col])
    next_t = len(monthly) + periods_ahead - 1
    forecast_value = float(model.predict([[next_t]])[0])

    slope = model.coef_[0]
    trend_desc = "an upward trend" if slope > 0 else "a downward trend" if slope < 0 else "a flat trend"
    return forecast_value, trend_desc


def forecast_series(df: pd.DataFrame, date_col: str, value_col: str, periods_ahead: int = 6):
    """
    Fits a linear trend on historical monthly totals and projects forward
    `periods_ahead` months. Returns a DataFrame with both historical and
    forecast rows plus a `type` column, ready for charting.
    """
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col, value_col]).sort_values(date_col)
    monthly = temp.set_index(date_col).resample("ME")[value_col].sum().reset_index()
    if len(monthly) < 2:
        return None, {"error": "Need at least two time periods of data to forecast."}

    monthly["t"] = np.arange(len(monthly))
    model = LinearRegression()
    model.fit(monthly[["t"]], monthly[value_col])
    preds_in_sample = model.predict(monthly[["t"]])
    r2 = round(r2_score(monthly[value_col], preds_in_sample), 4)

    last_date = monthly[date_col].max()
    future_dates = pd.date_range(last_date, periods=periods_ahead + 1, freq="ME")[1:]
    future_t = np.arange(len(monthly), len(monthly) + periods_ahead)
    future_values = model.predict(future_t.reshape(-1, 1))

    hist = monthly[[date_col, value_col]].rename(columns={value_col: "value"})
    hist["type"] = "Historical"
    future = pd.DataFrame({date_col: future_dates, "value": future_values, "type": "Forecast"})
    combined = pd.concat([hist, future], ignore_index=True)

    metrics = {"r2": r2, "trend_slope": round(float(model.coef_[0]), 4)}
    return combined, metrics
