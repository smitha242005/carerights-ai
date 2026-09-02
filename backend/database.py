"""
database.py — Real persistent storage for CareRights AI using SQLite.

Not in-memory, not mocked — this creates an actual .db file on disk that survives
server restarts. Passwords are hashed with bcrypt, never stored in plain text.
"""

import sqlite3
import json
import secrets
import bcrypt
from pathlib import Path
from datetime import datetime

DB_PATH = Path(__file__).parent / "carerights.db"


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            case_text TEXT NOT NULL,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    conn.close()


# ---------- users ----------

def create_user(name: str, email: str, password: str) -> dict:
    conn = get_connection()
    try:
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, datetime.utcnow().isoformat()),
        )
        conn.commit()
        user_id = cursor.lastrowid
        return {"id": user_id, "name": name, "email": email}
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists.")
    finally:
        conn.close()


def verify_user(email: str, password: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if row is None:
        return None
    if not bcrypt.checkpw(password.encode("utf-8"), row["password_hash"].encode("utf-8")):
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


# ---------- sessions (simple token-based auth) ----------

def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn = get_connection()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, user_id, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return token


def get_user_from_token(token: str) -> dict | None:
    conn = get_connection()
    row = conn.execute("""
        SELECT users.id, users.name, users.email FROM sessions
        JOIN users ON sessions.user_id = users.id
        WHERE sessions.token = ?
    """, (token,)).fetchone()
    conn.close()
    if row is None:
        return None
    return {"id": row["id"], "name": row["name"], "email": row["email"]}


def delete_session(token: str):
    conn = get_connection()
    conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ---------- cases ----------

def save_case(user_id: int, case_text: str, result: dict) -> dict:
    conn = get_connection()
    cursor = conn.execute(
        "INSERT INTO cases (user_id, case_text, result_json, created_at) VALUES (?, ?, ?, ?)",
        (user_id, case_text, json.dumps(result), datetime.utcnow().isoformat()),
    )
    conn.commit()
    case_id = cursor.lastrowid
    conn.close()
    return {"id": case_id, "case_text": case_text, "result": result}


def get_cases_for_user(user_id: int) -> list[dict]:
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, case_text, result_json, created_at FROM cases WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return [
        {"id": r["id"], "case_text": r["case_text"], "result": json.loads(r["result_json"]), "created_at": r["created_at"]}
        for r in rows
    ]


def get_case_by_id(case_id: int, user_id: int) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT id, case_text, result_json, created_at FROM cases WHERE id = ? AND user_id = ?",
        (case_id, user_id),
    ).fetchone()
    conn.close()
    if row is None:
        return None
    return {"id": row["id"], "case_text": row["case_text"], "result": json.loads(row["result_json"]), "created_at": row["created_at"]}


# ---------- feedback ----------

def save_feedback(user_id: int, case_id: int, agent: str, rating: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO feedback (user_id, case_id, agent, rating, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, case_id, agent, rating, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
