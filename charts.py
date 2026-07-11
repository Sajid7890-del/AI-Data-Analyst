"""
charts.py
---------
All chart-generation logic, built with Plotly so charts are interactive
inside Streamlit and easy to export as PNG (via kaleido).

Every function returns a `plotly.graph_objects.Figure` (or None if the
requested chart isn't meaningful for the given data), so app.py can
simply call `st.plotly_chart(fig)`.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import get_numeric_columns, get_categorical_columns, detect_outliers

TEMPLATE = "plotly_white"


def bar_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str = None, top_n: int = 15):
    grouped = (
        df.groupby(category_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.bar(
        grouped, x=category_col, y=value_col,
        title=title or f"{value_col} by {category_col}",
        template=TEMPLATE, color=value_col, color_continuous_scale="Blues",
        text_auto=".2s",
    )
    fig.update_layout(showlegend=False)
    return fig


def line_chart(df: pd.DataFrame, date_col: str, value_col: str, title: str = None, freq: str = "ME"):
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col])
    if temp.empty:
        return None
    grouped = temp.set_index(date_col).resample(freq)[value_col].sum().reset_index()
    fig = px.line(
        grouped, x=date_col, y=value_col, markers=True,
        title=title or f"{value_col} Over Time",
        template=TEMPLATE,
    )
    return fig


def pie_chart(df: pd.DataFrame, category_col: str, value_col: str, title: str = None, top_n: int = 10):
    grouped = (
        df.groupby(category_col)[value_col]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    fig = px.pie(
        grouped, names=category_col, values=value_col,
        title=title or f"Share of {value_col} by {category_col}",
        template=TEMPLATE, hole=0.35,
    )
    return fig


def histogram(df: pd.DataFrame, col: str, title: str = None, bins: int = 30):
    fig = px.histogram(
        df, x=col, nbins=bins,
        title=title or f"Distribution of {col}",
        template=TEMPLATE, color_discrete_sequence=["#4C78A8"],
    )
    return fig


def scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None, title: str = None):
    """
    Note: intentionally does NOT use Plotly's trendline="ols" option, since
    that silently requires the extra `statsmodels` package and crashes the
    app with a ModuleNotFoundError if it isn't installed. If you want a
    trendline and have statsmodels installed, pass trendline='ols' yourself.
    """
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title or f"{y_col} vs {x_col}",
        template=TEMPLATE,
    )
    return fig


def box_plot(df: pd.DataFrame, col: str, group_col: str = None, title: str = None):
    fig = px.box(
        df, x=group_col, y=col,
        title=title or f"Box Plot of {col}",
        template=TEMPLATE, color=group_col,
    )
    return fig


def correlation_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap"):
    numeric_cols = get_numeric_columns(df)
    if len(numeric_cols) < 2:
        return None
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        title=title, template=TEMPLATE, zmin=-1, zmax=1,
    )
    return fig


def outlier_chart(df: pd.DataFrame, col: str, title: str = None):
    """Box plot highlighting outlier bounds for a single numeric column."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[col], name=col, boxpoints="outliers", marker_color="#E45756"))
    fig.update_layout(title=title or f"Outliers in {col}", template=TEMPLATE)
    return fig


def auto_chart_suite(df: pd.DataFrame, date_col: str = None, amount_col: str = None):
    """
    Generates a sensible default set of charts based on the dataset's
    shape, used for the automatic dashboard view.
    Returns a dict of {chart_title: plotly Figure}.
    """
    charts = {}
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    if date_col and amount_col:
        fig = line_chart(df, date_col, amount_col, title=f"{amount_col} Trend Over Time")
        if fig:
            charts["Trend Over Time"] = fig

    if categorical_cols and amount_col:
        cat = categorical_cols[0]
        charts[f"{amount_col} by {cat}"] = bar_chart(df, cat, amount_col)
        charts[f"Share of {amount_col} by {cat}"] = pie_chart(df, cat, amount_col)

    if numeric_cols:
        charts[f"Distribution of {numeric_cols[0]}"] = histogram(df, numeric_cols[0])

    if len(numeric_cols) >= 2:
        charts["Correlation Heatmap"] = correlation_heatmap(df)
        charts[f"{numeric_cols[1]} vs {numeric_cols[0]}"] = scatter_plot(df, numeric_cols[0], numeric_cols[1])

    outliers = detect_outliers(df)
    if outliers:
        first_outlier_col = list(outliers.keys())[0]
        charts[f"Outliers in {first_outlier_col}"] = outlier_chart(df, first_outlier_col)

    return {k: v for k, v in charts.items() if v is not None}


def fig_to_png_bytes(fig):
    """Exports a Plotly figure to PNG bytes (requires the `kaleido` package)."""
    try:
        return fig.to_image(format="png", width=1000, height=600, scale=2)
    except Exception:
        return None
