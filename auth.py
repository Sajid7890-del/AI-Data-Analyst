"""
auth.py
-------
Simple, dependency-free authentication:
  - PBKDF2-HMAC-SHA256 password hashing with per-user random salt
  - register / login / logout helpers
  - Streamlit session-state integration

No plaintext passwords are ever stored.
"""

import hashlib
import os
import re

import streamlit as st

from config import SESSION_USER_KEY
import database as db

PBKDF2_ITERATIONS = 200_000


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: bytes) -> str:
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return dk.hex()


def hash_new_password(password: str):
    """Returns (password_hash_hex, salt_hex) for a brand-new password."""
    salt = os.urandom(16)
    return _hash_password(password, salt), salt.hex()


def verify_password(password: str, password_hash: str, salt_hex: str) -> bool:
    salt = bytes.fromhex(salt_hex)
    return _hash_password(password, salt) == password_hash


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def is_valid_email(email: str) -> bool:
    return re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email) is not None


def is_valid_username(username: str) -> bool:
    return re.match(r"^[A-Za-z0-9_]{3,20}$", username) is not None


def password_strength_ok(password: str) -> bool:
    return len(password) >= 6


# ---------------------------------------------------------------------------
# Registration / Login
# ---------------------------------------------------------------------------
def register_user(username: str, email: str, password: str):
    """Returns (success: bool, message: str)."""
    if not is_valid_username(username):
        return False, "Username must be 3-20 characters (letters, numbers, underscore)."
    if not is_valid_email(email):
        return False, "Please enter a valid email address."
    if not password_strength_ok(password):
        return False, "Password must be at least 6 characters long."
    if db.get_user_by_username(username):
        return False, "That username is already taken."
    if db.get_user_by_email(email):
        return False, "An account with that email already exists."

    password_hash, salt = hash_new_password(password)
    db.create_user(username, email, password_hash, salt)
    return True, "Account created successfully! Please log in."


def login_user(username: str, password: str):
    """Returns (success: bool, message: str)."""
    user = db.get_user_by_username(username)
    if not user:
        return False, "No account found with that username."
    if not verify_password(password, user["password_hash"], user["salt"]):
        return False, "Incorrect password."

    st.session_state[SESSION_USER_KEY] = {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
    }
    return True, f"Welcome back, {user['username']}!"


def logout_user():
    if SESSION_USER_KEY in st.session_state:
        del st.session_state[SESSION_USER_KEY]


def get_current_user():
    return st.session_state.get(SESSION_USER_KEY)


def is_authenticated() -> bool:
    return SESSION_USER_KEY in st.session_state
