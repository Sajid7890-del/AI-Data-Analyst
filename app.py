"""
app.py
------
Main Streamlit entrypoint for the AI Data Analyst application.

Run with:  streamlit run app.py
"""

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

import database as db
import auth
import utils
import charts
import ai_engine
import ml_engine
import report_generator
from config import APP_NAME, APP_ICON, UPLOAD_DIR, MAX_FILE_SIZE_MB, MAX_ROWS_PREVIEW

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ---------------------------------------------------------------------------
# Theme / CSS
# ---------------------------------------------------------------------------
def inject_css(dark_mode: bool):
    if dark_mode:
        bg, card, text, accent = "#0E1117", "#1A1D24", "#FAFAFA", "#4C78A8"
    else:
        bg, card, text, accent = "#FFFFFF", "#F7F9FC", "#111111", "#4C78A8"

    st.markdown(f"""
        <style>
        .stApp {{ background-color: {bg}; color: {text}; }}
        .metric-card {{
            background-color: {card};
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid rgba(128,128,128,0.15);
            text-align: center;
        }}
        .metric-card h2 {{ margin: 0; color: {accent}; font-size: 1.8rem; }}
        .metric-card p {{ margin: 0; opacity: 0.7; font-size: 0.85rem; }}
        .stButton>button {{
            border-radius: 8px;
            border: 1px solid {accent};
            font-weight: 600;
        }}
        .chat-bubble-user {{
            background-color: {accent}22;
            padding: 0.7rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.4rem;
        }}
        .chat-bubble-ai {{
            background-color: {card};
            padding: 0.7rem 1rem;
            border-radius: 12px;
            margin-bottom: 0.8rem;
            border-left: 3px solid {accent};
        }}
        section[data-testid="stSidebar"] {{ border-right: 1px solid rgba(128,128,128,0.15); }}
        </style>
    """, unsafe_allow_html=True)


if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
inject_css(st.session_state.dark_mode)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in [
    ("df", None), ("dataset_id", None), ("dataset_name", None),
    ("chat_log", []), ("auth_mode", "Login"),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---------------------------------------------------------------------------
# Authentication screen
# ---------------------------------------------------------------------------
def render_auth_screen():
    st.markdown(f"<h1 style='text-align:center'>{APP_ICON} {APP_NAME}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; opacity:0.7'>Upload data. Ask questions. Get instant AI-powered insights.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login", use_container_width=True)
                if submitted:
                    with st.spinner("Signing you in..."):
                        time.sleep(0.3)
                        success, message = auth.login_user(username, password)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)

        with tab_register:
            with st.form("register_form"):
                new_username = st.text_input("Choose a username")
                new_email = st.text_input("Email")
                new_password = st.text_input("Choose a password", type="password")
                confirm_password = st.text_input("Confirm password", type="password")
                submitted = st.form_submit_button("Create account", use_container_width=True)
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        with st.spinner("Creating your account..."):
                            time.sleep(0.3)
                            success, message = auth.register_user(new_username, new_email, new_password)
                        if success:
                            st.success(message)
                        else:
                            st.error(message)

        st.info("💡 Demo tip: register any username/password to explore the app — everything runs locally in SQLite.")

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar():
    user = auth.get_current_user()
    with st.sidebar:
        st.markdown(f"### {APP_ICON} {APP_NAME}")
        st.caption(f"Logged in as **{user['username']}**")

        st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)

        st.divider()
        page = st.radio(
            "Navigate",
            ["📁 Upload Data", "📊 Dashboard", "💬 AI Chat", "🧠 Business Insights",
             "🤖 Machine Learning", "📜 History", "📤 Reports"],
            label_visibility="collapsed",
        )

        st.divider()
        past_datasets = db.get_datasets_for_user(user["id"])
        if past_datasets:
            st.caption("Previously uploaded datasets")
            options = {f"{d['filename']} ({d['rows']}x{d['columns']})": d for d in past_datasets}
            choice = st.selectbox("Load a previous dataset", ["-- select --"] + list(options.keys()))
            if choice != "-- select --":
                if st.button("Load selected dataset", use_container_width=True):
                    d = options[choice]
                    try:
                        st.session_state.df = utils.load_dataframe(d["filepath"])
                        st.session_state.dataset_id = d["id"]
                        st.session_state.dataset_name = d["filename"]
                        st.success(f"Loaded {d['filename']}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load dataset: {e}")

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            auth.logout_user()
            st.rerun()

    return page


# ---------------------------------------------------------------------------
# Upload page
# ---------------------------------------------------------------------------
def render_upload_page():
    st.header("📁 Upload Your Dataset")
    st.write("Upload a CSV or Excel file to begin analysis.")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    missing_strategy = st.selectbox(
        "How should missing values be handled?",
        ["Report only (keep data as-is)", "Drop rows with missing values", "Fill with mean/mode"],
    )
    strategy_map = {
        "Report only (keep data as-is)": "report_only",
        "Drop rows with missing values": "drop",
        "Fill with mean/mode": "fill_mean",
    }

    if uploaded_file is not None:
        valid, message = utils.validate_file(uploaded_file.name)
        if not valid:
            st.error(message)
            return

        size_mb = uploaded_file.size / (1024 ** 2)
        if size_mb > MAX_FILE_SIZE_MB:
            st.error(f"File is {size_mb:.1f} MB, which exceeds the {MAX_FILE_SIZE_MB} MB limit.")
            return

        if st.button("🚀 Analyze this file", type="primary"):
            with st.spinner("Reading and validating file..."):
                save_path = os.path.join(UPLOAD_DIR, f"{int(time.time())}_{uploaded_file.name}")
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                try:
                    df = utils.load_dataframe(save_path)
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    return

            with st.spinner("Cleaning data..."):
                df = utils.handle_missing_values(df, strategy_map[missing_strategy])

            user = auth.get_current_user()
            dataset_id = db.save_dataset(user["id"], uploaded_file.name, save_path, len(df), len(df.columns))

            st.session_state.df = df
            st.session_state.dataset_id = dataset_id
            st.session_state.dataset_name = uploaded_file.name
            st.session_state.chat_log = []

            st.success(f"✅ Loaded **{uploaded_file.name}** — {len(df):,} rows × {len(df.columns)} columns")
            st.balloons()

    if st.session_state.df is not None:
        st.divider()
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head(MAX_ROWS_PREVIEW), use_container_width=True)

        info = utils.get_basic_info(st.session_state.df)
        c1, c2, c3, c4 = st.columns(4)
        for col, label, value in zip(
            (c1, c2, c3, c4),
            ("Rows", "Columns", "Missing Values", "Duplicate Rows"),
            (info["rows"], info["columns"], info["missing_values"], info["duplicate_rows"]),
        ):
            col.markdown(f"<div class='metric-card'><h2>{value:,}</h2><p>{label}</p></div>", unsafe_allow_html=True)

        with st.expander("Column data types"):
            types_df = pd.DataFrame(list(info["column_types"].items()), columns=["Column", "Detected Type"])
            st.dataframe(types_df, use_container_width=True, hide_index=True)

        dup_rows = utils.detect_duplicates(st.session_state.df)
        if not dup_rows.empty:
            with st.expander(f"⚠️ {len(dup_rows)} duplicate rows found"):
                st.dataframe(dup_rows, use_container_width=True)

# ---------------------------------------------------------------------------
# Dashboard page
# ---------------------------------------------------------------------------
def render_dashboard_page():
    st.header("📊 Analysis Dashboard")
    df = st.session_state.df
    if df is None:
        st.info("👈 Upload a dataset first from the 'Upload Data' page.")
        return

    info = utils.get_basic_info(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value in zip(
        (c1, c2, c3, c4, c5),
        ("Rows", "Columns", "Missing", "Duplicates", "Memory"),
        (f"{info['rows']:,}", info["columns"], f"{info['missing_values']:,}",
         f"{info['duplicate_rows']:,}", f"{info['memory_usage_mb']} MB"),
    ):
        col.markdown(f"<div class='metric-card'><h2>{value}</h2><p>{label}</p></div>", unsafe_allow_html=True)

    st.divider()
    tab_stats, tab_corr, tab_outliers, tab_charts = st.tabs(
        ["📈 Statistical Summary", "🔗 Correlation Matrix", "🎯 Outlier Detection", "📊 Auto Charts"]
    )

    with tab_stats:
        stats = utils.get_statistical_summary(df)
        if stats.empty:
            st.warning("No numeric columns found for statistical summary.")
        else:
            st.dataframe(stats, use_container_width=True)

        missing_report = utils.get_missing_value_report(df)
        if not missing_report.empty:
            st.subheader("Missing Values by Column")
            st.dataframe(missing_report, use_container_width=True)

    with tab_corr:
        corr = utils.get_correlation_matrix(df)
        if corr.empty:
            st.warning("Need at least 2 numeric columns for a correlation matrix.")
        else:
            fig = charts.correlation_heatmap(df)
            st.plotly_chart(fig, use_container_width=True)

    with tab_outliers:
        outliers = utils.detect_outliers(df)
        if not outliers:
            st.success("No significant outliers detected (using the IQR method).")
        else:
            for col, stats_ in outliers.items():
                with st.expander(f"⚠️ {col} — {stats_['count']} outliers ({stats_['pct']}%)"):
                    st.write(
                        f"Normal range: **{stats_['lower_bound']}** to **{stats_['upper_bound']}**  \n"
                        f"Outlier range found: **{stats_['min_outlier']}** to **{stats_['max_outlier']}**"
                    )
                    st.plotly_chart(charts.outlier_chart(df, col), use_container_width=True)

    with tab_charts:
        date_col = utils.find_likely_date_column(df)
        amount_col = utils.find_likely_amount_column(df)
        c1, c2 = st.columns(2)
        date_col = c1.selectbox("Date column (optional)", [None] + list(df.columns),
                                 index=(list(df.columns).index(date_col) + 1) if date_col in df.columns else 0)
        amount_col = c2.selectbox("Amount / value column", [None] + utils.get_numeric_columns(df),
                                   index=(utils.get_numeric_columns(df).index(amount_col) + 1) if amount_col in utils.get_numeric_columns(df) else 0)

        auto_charts = charts.auto_chart_suite(df, date_col=date_col, amount_col=amount_col)
        if not auto_charts:
            st.info("Not enough structured data to auto-generate charts. Try picking columns above.")
        for title, fig in auto_charts.items():
            st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("🛠️ Build a Custom Chart")
        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Pie", "Histogram", "Scatter", "Box Plot"])
        numeric_cols = utils.get_numeric_columns(df)
        cat_cols = utils.get_categorical_columns(df) or list(df.select_dtypes(include="object").columns)

        try:
            if chart_type == "Bar" and cat_cols and numeric_cols:
                x = st.selectbox("Category", cat_cols, key="bar_x")
                y = st.selectbox("Value", numeric_cols, key="bar_y")
                st.plotly_chart(charts.bar_chart(df, x, y), use_container_width=True)
            elif chart_type == "Line" and numeric_cols:
                x = st.selectbox("Date/X-axis column", list(df.columns), key="line_x")
                y = st.selectbox("Value", numeric_cols, key="line_y")
                fig = charts.line_chart(df, x, y)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Selected X-axis column could not be parsed as dates.")
            elif chart_type == "Pie" and cat_cols and numeric_cols:
                x = st.selectbox("Category", cat_cols, key="pie_x")
                y = st.selectbox("Value", numeric_cols, key="pie_y")
                st.plotly_chart(charts.pie_chart(df, x, y), use_container_width=True)
            elif chart_type == "Histogram" and numeric_cols:
                x = st.selectbox("Column", numeric_cols, key="hist_x")
                st.plotly_chart(charts.histogram(df, x), use_container_width=True)
            elif chart_type == "Scatter" and len(numeric_cols) >= 2:
                x = st.selectbox("X-axis", numeric_cols, key="sc_x")
                y = st.selectbox("Y-axis", numeric_cols, key="sc_y", index=min(1, len(numeric_cols) - 1))
                color = st.selectbox("Color by (optional)", [None] + cat_cols, key="sc_c")
                st.plotly_chart(charts.scatter_plot(df, x, y, color), use_container_width=True)
            elif chart_type == "Box Plot" and numeric_cols:
                x = st.selectbox("Numeric column", numeric_cols, key="box_x")
                group = st.selectbox("Group by (optional)", [None] + cat_cols, key="box_g")
                st.plotly_chart(charts.box_plot(df, x, group), use_container_width=True)
            else:
                st.info("Not enough compatible columns for this chart type.")
        except Exception as e:
            st.error(f"Couldn't build that chart: {e}")

# ---------------------------------------------------------------------------
# AI Chat page
# ---------------------------------------------------------------------------
def render_chat_page():
    st.header("💬 AI Chat — Ask Questions About Your Data")
    df = st.session_state.df
    if df is None:
        st.info("👈 Upload a dataset first from the 'Upload Data' page.")
        return

    provider = ai_engine.available_provider()
    if provider:
        st.success(f"✅ Connected to **{provider.upper()}** for open-ended AI answers.")
    else:
        st.warning(
            "⚡ No AI API key configured — using the built-in rule-based engine, which "
            "handles totals, top-N, trends, forecasts, and recommendations. "
            "Add `OPENAI_API_KEY` or `GEMINI_API_KEY` for fully open-ended answers."
        )

    st.write("Try: *What is the total sales?* · *Which product sold the most?* · *Show monthly sales* · *Predict next month's sales* · *Give business recommendations*")

    for entry in st.session_state.chat_log:
        st.markdown(f"<div class='chat-bubble-user'>🧑 <b>You:</b> {entry['question']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble-ai'>🤖 <b>AI:</b><br>{entry['answer']}</div>", unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your data...")
    if question:
        with st.spinner("Analyzing your data..."):
            answer = ai_engine.answer_question(question, df)
        st.session_state.chat_log.append({"question": question, "answer": answer})

        user = auth.get_current_user()
        db.log_chat(user["id"], st.session_state.dataset_id, question, answer)
        st.rerun()

    quick_qs = ["What is the total?", "Show top 10", "Find trends", "Give business recommendations"]
    st.caption("Quick questions:")
    qcols = st.columns(len(quick_qs))
    for c, q in zip(qcols, quick_qs):
        if c.button(q, use_container_width=True):
            with st.spinner("Analyzing your data..."):
                answer = ai_engine.answer_question(q, df)
            st.session_state.chat_log.append({"question": q, "answer": answer})
            user = auth.get_current_user()
            db.log_chat(user["id"], st.session_state.dataset_id, q, answer)
            st.rerun()


# ---------------------------------------------------------------------------
# Business Insights page
# ---------------------------------------------------------------------------
def render_insights_page():
    st.header("🧠 Business Insights")
    df = st.session_state.df
    if df is None:
        st.info("👈 Upload a dataset first from the 'Upload Data' page.")
        return

    with st.spinner("Generating insights..."):
        findings = ai_engine.generate_key_findings(df)
        recommendations = ai_engine.generate_recommendations(df)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔍 Key Findings & Trends")
        st.markdown(findings)
    with col2:
        st.subheader("💡 Recommendations")
        st.markdown(recommendations)

    st.divider()
    st.subheader("📋 Executive Summary")
    with st.spinner("Compiling executive summary..."):
        summary = ai_engine.generate_executive_summary(df)
    st.markdown(summary)

    user = auth.get_current_user()
    db.log_analysis(user["id"], st.session_state.dataset_id, summary[:2000])


# ---------------------------------------------------------------------------
# Machine Learning page
# ---------------------------------------------------------------------------
def render_ml_page():
    st.header("🤖 Machine Learning")
    df = st.session_state.df
    if df is None:
        st.info("👈 Upload a dataset first from the 'Upload Data' page.")
        return

    task = st.selectbox("Choose a task", [
        "Sales Forecasting", "Linear Regression", "Random Forest Regression", "Classification",
    ])

    numeric_cols = utils.get_numeric_columns(df)

    if task == "Sales Forecasting":
        date_col = st.selectbox("Date column", [utils.find_likely_date_column(df)] + list(df.columns))
        value_col = st.selectbox("Value column to forecast", numeric_cols)
        periods = st.slider("Months ahead to forecast", 1, 12, 6)

        if st.button("Run Forecast", type="primary"):
            with st.spinner("Fitting forecast model..."):
                combined, metrics = ml_engine.forecast_series(df, date_col, value_col, periods)
            if combined is None:
                st.error(metrics["error"])
            else:
                st.success(f"Model fit R² = {metrics['r2']} | Trend slope = {metrics['trend_slope']}")
                import plotly.express as px
                fig = px.line(combined, x=date_col, y="value", color="type", markers=True,
                               title=f"{value_col} Forecast — Next {periods} Months", template="plotly_white")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(combined.tail(periods + 3), use_container_width=True)

    elif task in ("Linear Regression", "Random Forest Regression"):
        target = st.selectbox("Target column (what to predict)", numeric_cols)
        features = st.multiselect("Feature columns", [c for c in numeric_cols if c != target],
                                   default=[c for c in numeric_cols if c != target][:5])
        if st.button("Train Model", type="primary"):
            if not features:
                st.error("Select at least one feature column.")
            else:
                with st.spinner("Training model..."):
                    fn = ml_engine.run_linear_regression if task == "Linear Regression" else ml_engine.run_random_forest_regression
                    result = fn(df, target, features)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"{result['model_name']} trained successfully.")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("R² Score", result["r2"])
                    c2.metric("MAE", result["mae"])
                    c3.metric("RMSE", result["rmse"])

                    st.subheader("Feature Importance")
                    imp_df = pd.DataFrame(list(result["feature_importance"].items()), columns=["Feature", "Importance"])
                    imp_df = imp_df.sort_values("Importance", ascending=False)
                    st.plotly_chart(charts.bar_chart(imp_df, "Feature", "Importance", title="Feature Importance"), use_container_width=True)

    elif task == "Classification":
        target = st.selectbox("Target column (category to predict)", utils.get_categorical_columns(df) or list(df.columns))
        candidate_features = [c for c in df.columns if c != target]
        features = st.multiselect("Feature columns", candidate_features, default=candidate_features[:5])
        if st.button("Train Classifier", type="primary"):
            if not features:
                st.error("Select at least one feature column.")
            else:
                with st.spinner("Training classifier..."):
                    result = ml_engine.run_classification(df, target, features)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success(f"{result['model_name']} trained successfully.")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Accuracy", result["accuracy"])
                    c2.metric("Precision", result["precision"])
                    c3.metric("Recall", result["recall"])
                    c4.metric("F1 Score", result["f1"])

                    st.subheader("Feature Importance")
                    imp_df = pd.DataFrame(list(result["feature_importance"].items()), columns=["Feature", "Importance"])
                    imp_df = imp_df.sort_values("Importance", ascending=False)
                    st.plotly_chart(charts.bar_chart(imp_df, "Feature", "Importance", title="Feature Importance"), use_container_width=True)
                    st.caption(f"Classes: {', '.join(map(str, result['classes']))}")

# ---------------------------------------------------------------------------
# History page
# ---------------------------------------------------------------------------
def render_history_page():
    st.header("📜 History")
    user = auth.get_current_user()

    tab_datasets, tab_analysis, tab_chat = st.tabs(["📁 Datasets", "📊 Analysis Log", "💬 Chat Log"])

    with tab_datasets:
        datasets = db.get_datasets_for_user(user["id"])
        if not datasets:
            st.info("No datasets uploaded yet.")
        else:
            st.dataframe(pd.DataFrame(datasets)[["filename", "rows", "columns", "uploaded_at"]],
                         use_container_width=True, hide_index=True)

    with tab_analysis:
        history = db.get_analysis_history(user["id"])
        if not history:
            st.info("No analyses run yet.")
        else:
            for h in history:
                with st.expander(f"Analysis on {h['created_at']}"):
                    st.write(h["summary"])

    with tab_chat:
        chats = db.get_chat_history(user["id"])
        if not chats:
            st.info("No chat history yet.")
        else:
            for c in chats:
                st.markdown(f"**Q ({c['created_at']}):** {c['question']}")
                st.markdown(f"**A:** {c['answer']}")
                st.divider()


# ---------------------------------------------------------------------------
# Reports page
# ---------------------------------------------------------------------------
def render_reports_page():
    st.header("📤 Download Reports")
    df = st.session_state.df
    if df is None:
        st.info("👈 Upload a dataset first from the 'Upload Data' page.")
        return

    st.write("Generate a downloadable report of your current dataset and analysis.")
    dataset_name = st.session_state.dataset_name or "dataset"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📄 CSV")
        st.caption("Cleaned dataset export")
        if st.button("Generate CSV", use_container_width=True):
            with st.spinner("Building CSV..."):
                data = report_generator.generate_csv_report(df)
            st.download_button("⬇️ Download CSV", data, file_name=f"{dataset_name}_report.csv",
                                mime="text/csv", use_container_width=True)

    with col2:
        st.subheader("📊 Excel")
        st.caption("Multi-sheet workbook: data, stats, correlation, outliers")
        if st.button("Generate Excel", use_container_width=True):
            with st.spinner("Building Excel workbook..."):
                data = report_generator.generate_excel_report(df)
            st.download_button("⬇️ Download Excel", data, file_name=f"{dataset_name}_report.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True)

    with col3:
        st.subheader("📕 PDF")
        st.caption("Executive report with findings & charts")
        include_charts = st.checkbox("Include charts (slower)", value=True)
        if st.button("Generate PDF", use_container_width=True):
            with st.spinner("Building PDF report... this can take a few seconds"):
                chart_images = None
                if include_charts:
                    auto_charts = charts.auto_chart_suite(
                        df, date_col=utils.find_likely_date_column(df),
                        amount_col=utils.find_likely_amount_column(df),
                    )
                    chart_images = {t: charts.fig_to_png_bytes(f) for t, f in list(auto_charts.items())[:4]}
                data = report_generator.generate_pdf_report(df, dataset_name, chart_images)
            st.download_button("⬇️ Download PDF", data, file_name=f"{dataset_name}_report.pdf",
                                mime="application/pdf", use_container_width=True)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------
def main():
    if not auth.is_authenticated():
        render_auth_screen()
        return

    page = render_sidebar()

    if page == "📁 Upload Data":
        render_upload_page()
    elif page == "📊 Dashboard":
        render_dashboard_page()
    elif page == "💬 AI Chat":
        render_chat_page()
    elif page == "🧠 Business Insights":
        render_insights_page()
    elif page == "🤖 Machine Learning":
        render_ml_page()
    elif page == "📜 History":
        render_history_page()
    elif page == "📤 Reports":
        render_reports_page()


if __name__ == "__main__":
    main()
