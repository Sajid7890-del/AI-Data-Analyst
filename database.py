"""
database.py
-----------
Handles all SQLite persistence for the AI Data Analyst app:
  - users               (authentication)
  - datasets             (metadata about uploaded files)
  - analysis_history     (log of analyses run)
  - chat_history          (AI Q&A log)

Every function opens its own short-lived connection so the module is
safe to use from Streamlit's rerun-heavy execution model.
"""

import sqlite3
import json
from datetime import datetime
from contextlib import contextmanager

from config import DATABASE_PATH


@contextmanager
def get_connection():
    """Context manager that yields a SQLite connection and always closes it."""
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create all tables if they do not already exist. Call once at app start."""
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                uploaded_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_id INTEGER,
                summary TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id),
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            )
        """)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------
def create_user(username, email, password_hash, salt):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users (username, email, password_hash, salt, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (username, email, password_hash, salt, datetime.utcnow().isoformat()),
        )


def get_user_by_username(username):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_user_by_email(email):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def save_dataset(user_id, filename, filepath, rows, columns):
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO datasets (user_id, filename, filepath, rows, columns, uploaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, filename, filepath, rows, columns, datetime.utcnow().isoformat()),
        )
        return cur.lastrowid


def get_datasets_for_user(user_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM datasets WHERE user_id = ? ORDER BY uploaded_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Analysis history
# ---------------------------------------------------------------------------
def log_analysis(user_id, dataset_id, summary):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO analysis_history (user_id, dataset_id, summary, created_at) "
            "VALUES (?, ?, ?, ?)",
            (user_id, dataset_id, summary, datetime.utcnow().isoformat()),
        )


def get_analysis_history(user_id):
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------
def log_chat(user_id, dataset_id, question, answer):
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (user_id, dataset_id, question, answer, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, dataset_id, question, answer, datetime.utcnow().isoformat()),
        )


def get_chat_history(user_id, dataset_id=None, limit=50):
    with get_connection() as conn:
        if dataset_id:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE user_id = ? AND dataset_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (user_id, dataset_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
