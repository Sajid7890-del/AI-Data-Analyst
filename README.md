# 📊 AI Data Analyst

A production-ready Streamlit web application that lets anyone upload a CSV or Excel file,
explore it visually, ask questions in plain English, and get AI-generated business insights,
charts, forecasts, and downloadable reports — no coding required.

**Live demo-ready. Zero API key required to run** (a built-in rule-based NL engine handles
common questions out of the box; plug in an OpenAI or Gemini key for fully open-ended Q&A).

---

## ✨ Features

| Area | What it does |
|---|---|
| **Auth** | Register / login / logout with PBKDF2-SHA256 password hashing (no plaintext passwords, ever) |
| **Upload** | CSV & Excel upload, file-type validation, missing-value strategy, duplicate detection, automatic column type detection |
| **Dashboard** | Row/column counts, missing values, duplicates, statistical summary, correlation matrix, IQR outlier detection |
| **AI Chat** | Ask natural-language questions ("What is the total sales?", "Which product sold the most?", "Predict next month's sales") and get plain-English answers |
| **Auto Charts** | Bar, line, pie, histogram, scatter, box plot, and correlation heatmap — generated automatically based on your data's shape |
| **Business Insights** | Key findings, trends, opportunities, risks, recommendations, and an executive summary |
| **Machine Learning** | Linear Regression, Random Forest (regression & classification), sales forecasting, accuracy metrics, feature importance |
| **Reports** | Download CSV, multi-sheet Excel, or a formatted PDF report with embedded charts |
| **Database** | SQLite storage for users, uploaded datasets, analysis history, and chat history |
| **UI** | Dark/light mode toggle, sidebar navigation, responsive layout, loading spinners |

---

## 🗂️ Project Structure

```
ai-data-analyst/
├── app.py                 # Main Streamlit application (UI + page routing)
├── config.py               # Central configuration (paths, keys, thresholds)
├── database.py              # SQLite persistence layer
├── auth.py                   # Registration, login, password hashing
├── utils.py                   # Data loading, cleaning, profiling/statistics
├── charts.py                   # Plotly chart-generation functions
├── ai_engine.py                 # NL Q&A (LLM + rule-based fallback), insights
├── ml_engine.py                  # Regression, classification, forecasting
├── report_generator.py            # CSV / Excel / PDF report builders
├── requirements.txt
├── README.md
├── .gitignore
├── uploads/                        # Uploaded files land here (gitignored)
├── reports/                         # Generated reports (gitignored)
├── database/                         # app_data.db lives here (gitignored)
├── assets/                            # Sample dataset, static assets
│   └── sample_sales_data.csv
└── models/                             # Reserved for persisted ML models
```

---

## 🚀 Quick Start (Local)

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/ai-data-analyst.git
cd ai-data-analyst
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. (Optional) Enable full AI chat

The app works immediately with **no API key** using a built-in rule-based engine.
To enable fully open-ended AI conversation, set one of:

```bash
# macOS/Linux
export OPENAI_API_KEY="sk-..."
# or
export GEMINI_API_KEY="..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
```

Or create a `.env` file in the project root:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=
AI_PROVIDER=auto
```

### 3. Run

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), register an account,
and upload `assets/sample_sales_data.csv` to try every feature immediately.

---

## 🔑 Authentication Notes

- Passwords are hashed with **PBKDF2-HMAC-SHA256** (200,000 iterations) plus a random
  per-user salt — never stored in plaintext.
- All auth data lives in the local SQLite database (`database/app_data.db`), created
  automatically on first run.
- This is a demo-grade auth system suitable for a portfolio project; for real production
  use, pair it with HTTPS, rate limiting, and a managed identity provider.

---

## 🤖 How the AI Chat Works

1. **If an API key is set** (`OPENAI_API_KEY` or `GEMINI_API_KEY`), the app builds a compact
   JSON profile of your dataset (schema, types, sample rows, summary stats) and sends it
   with your question to the LLM, which answers in plain English.
2. **If no key is set**, a rule-based intent engine (regex + pandas) directly answers common
   analytical questions: totals, top-N rankings, monthly/weekly/yearly trends, top customers,
   simple linear-trend forecasts, and business recommendations — so the app is fully
   functional and demoable with zero setup.

---

## 📈 Machine Learning Tab

- **Sales Forecasting** — fits a linear trend on monthly aggregates and projects N months ahead, reporting R².
- **Linear Regression / Random Forest Regression** — pick any numeric target + features, get R², MAE, RMSE, and feature importance.
- **Classification** — Random Forest classifier over any categorical target, with accuracy, precision, recall, F1, and feature importance.

---

## 📤 Reports

From the **Reports** tab you can generate:
- **CSV** — the cleaned dataset
- **Excel** — multi-sheet workbook (Data / Summary / Statistics / Correlation / Outliers)
- **PDF** — an executive report with key findings, recommendations, a statistics table, and embedded charts (via ReportLab + Kaleido)

---

## 🧪 Sample Dataset

`assets/sample_sales_data.csv` contains ~600 synthetic e-commerce orders (product, region,
customer, quantity, unit price, sales, date) with realistic missing values and duplicate rows
baked in, so every feature (including data-quality detection) has something to show.

---

## ☁️ Deployment

### Deploy to GitHub
```bash
git init
git add .
git commit -m "Initial commit: AI Data Analyst"
git branch -M main
git remote add origin https://github.com/<your-username>/ai-data-analyst.git
git push -u origin main
```
Your `.gitignore` already excludes the local database, uploads, and generated reports, so
only source code and the sample dataset are committed.

### Deploy to Streamlit Community Cloud
1. Push the repo to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repo/branch, and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add (optional):
   ```toml
   OPENAI_API_KEY = "sk-..."
   GEMINI_API_KEY = ""
   ```
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt` automatically.

> Note: Streamlit Cloud's filesystem is ephemeral — uploaded files and the SQLite DB reset
> on redeploy/restart. For persistent storage in production, swap SQLite for a hosted DB
> (e.g., Postgres) and file storage for S3/GCS.

### Deploy to Render
1. Push the repo to GitHub.
2. In the Render dashboard, click **New → Web Service** and connect your repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add environment variables (`OPENAI_API_KEY`, etc.) under **Environment**.
5. Click **Create Web Service**.

---

## 🎁 Bonus / Extension Ideas

These aren't wired up by default but the codebase is structured to make them straightforward:

- **Voice input/output** — add `streamlit-webrtc` or `speech_recognition` for mic input, and
  `pyttsx3`/browser `SpeechSynthesis` to read AI answers aloud.
- **AI-generated dashboard summary** — call `ai_engine.generate_executive_summary(df)` on the
  Dashboard page and render it at the top (already implemented on the Business Insights page).
- **Export charts as PNG** — already supported via `charts.fig_to_png_bytes()`; wire a
  download button next to any `st.plotly_chart`.
- **Download dashboard as PDF** — `report_generator.generate_pdf_report()` already embeds
  charts; call it from the Dashboard page with the current chart set.
- **Email report** — use `smtplib`/`sendgrid` to email the bytes returned by
  `report_generator.generate_pdf_report()`.
- **Multi-language support** — wrap UI strings with `gettext` or a simple dict-based i18n
  layer, and instruct the LLM prompt in `ai_engine.py` to answer in the selected language.

---

## 🛠️ Tech Stack

Python · Streamlit · Pandas · NumPy · Plotly · Matplotlib · OpenPyXL · scikit-learn ·
ReportLab · SQLite · OpenAI API / Google Gemini API

---

## 📄 License

This project is provided as a portfolio/learning template — use and adapt freely.
