"""
database.py — Persistent storage for CareRights AI.

Two modes, chosen automatically:

1. DATABASE_URL is NOT set  -> uses a local SQLite file (carerights.db). Great for running
   on your own computer. NOTE: on Render's free plan the disk is wiped every time the
   server restarts, redeploys or wakes up from sleep, so accounts saved in this file
   disappear. That is what caused "I registered, but later login says my id is wrong".

2. DATABASE_URL IS set      -> uses that PostgreSQL database (e.g. a free Neon or Supabase
   database). The data lives outside Render, so accounts, sessions and cases survive
   restarts. This is what you should use when deployed.

Passwords are hashed with bcrypt, never stored in plain text. Emails are always stored
lowercase with spaces removed, so "Smitha@Gmail.com " and "smitha@gmail.com" are the same
account.
"""

import os
import sqlite3
import json
import secrets
import bcrypt
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "carerights.db"

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
if DATABASE_URL.startswith("postgres://"):  # some providers still give the old prefix
    DATABASE_URL = "postgresql://" + DATABASE_URL[len("postgres://"):]
USE_PG = bool(DATABASE_URL)

if USE_PG:
    import psycopg
    from psycopg.rows import dict_row
    IntegrityError = psycopg.errors.UniqueViolation
else:
    IntegrityError = sqlite3.IntegrityError


def backend_name() -> str:
    return "postgresql" if USE_PG else "sqlite"


def get_connection():
    if USE_PG:
        return psycopg.connect(DATABASE_URL, row_factory=dict_row)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sql(query: str) -> str:
    """Queries are written with SQLite-style '?' placeholders; PostgreSQL wants '%s'."""
    return query.replace("?", "%s") if USE_PG else query


def _execute(conn, query: str, params: tuple = ()):
    if USE_PG:
        return conn.execute(_sql(query), params)
    return conn.execute(query, params)


def _insert(conn, query: str, params: tuple) -> int:
    """Runs an INSERT and returns the new row's id on both databases."""
    if USE_PG:
        cur = conn.execute(_sql(query) + " RETURNING id", params)
        return cur.fetchone()["id"]
    cur = conn.execute(query, params)
    return cur.lastrowid


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def init_db():
    pk = "SERIAL PRIMARY KEY" if USE_PG else "INTEGER PRIMARY KEY AUTOINCREMENT"
    conn = get_connection()
    try:
        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS users (
                id {pk},
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        _execute(conn, """
            CREATE TABLE IF NOT EXISTS sessions (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS cases (
                id {pk},
                user_id INTEGER NOT NULL,
                case_text TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        _execute(conn, f"""
            CREATE TABLE IF NOT EXISTS feedback (
                id {pk},
                user_id INTEGER NOT NULL,
                case_id INTEGER NOT NULL,
                agent TEXT NOT NULL,
                rating TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (case_id) REFERENCES cases(id)
            )
        """)
        conn.commit()
        _normalize_existing_emails(conn)
    finally:
        conn.close()


def _normalize_existing_emails(conn):
    """One-time tidy-up for accounts created before emails were normalised."""
    rows = _execute(conn, "SELECT id, email FROM users").fetchall()
    for r in rows:
        clean = normalize_email(r["email"])
        if clean != r["email"]:
            try:
                _execute(conn, "UPDATE users SET email = ? WHERE id = ?", (clean, r["id"]))
                conn.commit()
            except IntegrityError:
                conn.rollback()  # two accounts differ only by capitals; leave both alone


# ---------- users ----------

def create_user(name: str, email: str, password: str) -> dict:
    email = normalize_email(email)
    name = name.strip()
    conn = get_connection()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user_id = _insert(
            conn,
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        return {"id": user_id, "name": name, "email": email}
    except IntegrityError:
        conn.rollback()
        raise ValueError("An account with this email already exists.")
    finally:
        conn.close()


def verify_user(email: str, password: str) -> dict | None:
    email = normalize_email(email)
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def get_profile(user_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        count = _execute(conn, "SELECT COUNT(*) AS n FROM cases WHERE user_id = ?", (user_id,)).fetchone()["n"]
    finally:
        conn.close()
    return {"id": row["id"], "name": row["name"], "email": row["email"],
            "created_at": row["created_at"], "case_count": count}


def change_password(user_id: int, current_password: str, new_password: str, keep_token: str | None = None) -> bool:
    """Returns False if the current password is wrong. On success, every other login
    session of this user is signed out (the one passed as keep_token stays)."""
    conn = get_connection()
    try:
        row = _execute(conn, "SELECT password_hash FROM users WHERE id = ?", (user_id,)).fetchone()
        if row is None:
            return False
        if not bcrypt.checkpw(current_password.encode("utf-8"), row["password_hash"].encode("utf-8")):
            return False
        new_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        _execute(conn, "UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
        if keep_token:
            _execute(conn, "DELETE FROM sessions WHERE user_id = ? AND token != ?", (user_id, keep_token))
        else:
            _execute(conn, "DELETE FROM sessions WHERE user_id = ?", (user_id,))
        conn.commit()
        return True
    finally:
        conn.close()


# ---------- sessions (simple token-based auth) ----------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn = get_connection()
    try:
        _execute(conn, "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
                 (token, user_id, datetime.utcnow().isoformat()))
        conn.commit()
    finally:
        conn.close()
    return token


def get_user_from_token(token: str) -> dict | None:
    conn = get_connection()
    try:
        row = _execute(conn, """
            SELECT users.id, users.name, users.email FROM sessions
            JOIN users ON sessions.user_id = users.id
            WHERE sessions.token = ?
        """, (token,)).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def delete_session(token: str):
    conn = get_connection()
    try:
        _execute(conn, "DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
    finally:
        conn.close()


# ---------- cases ----------

def save_case(user_id: int, case_text: str, result: dict) -> dict:
    conn = get_connection()
    try:
        case_id = _insert(
            conn,
            "INSERT INTO cases (user_id, case_text, result_json, created_at) VALUES (?, ?, ?, ?)",
            (user_id, case_text, json.dumps(result), datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"id": case_id, "case_text": case_text, "result": result}


def get_cases_for_user(user_id: int) -> list[dict]:
    conn = get_connection()
    try:
        rows = _execute(
            conn,
            "SELECT id, case_text, result_json, created_at FROM cases WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()
    return [
        {"id": r["id"], "case_text": r["case_text"], "result": json.loads(r["result_json"]), "created_at": r["created_at"]}
        for r in rows
    ]


def get_case_by_id(case_id: int, user_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = _execute(
            conn,
            "SELECT id, case_text, result_json, created_at FROM cases WHERE id = ? AND user_id = ?",
            (case_id, user_id),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return {"id": row["id"], "case_text": row["case_text"], "result": json.loads(row["result_json"]), "created_at": row["created_at"]}


# ---------- feedback ----------

def save_feedback(user_id: int, case_id: int, agent: str, rating: str):
    conn = get_connection()
    try:
        _execute(
            conn,
            "INSERT INTO feedback (user_id, case_id, agent, rating, created_at) VALUES (?, ?, ?, ?, ?)",
            (user_id, case_id, agent, rating, datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
