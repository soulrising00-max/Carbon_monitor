"""
SQLite persistence for analysis runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from configs.settings import settings


def _db_path() -> Path:
    return Path(settings.RUNS_DB_PATH)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=MEMORY")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def init_db() -> None:
    _db_path().parent.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                status TEXT NOT NULL,
                risk_score REAL,
                warnings TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_analysis_runs_project_created
            ON analysis_runs (project_id, created_at DESC)
            """
        )


def create_run(run_id: str, project_id: str, status: str) -> str:
    created_at = _now_iso()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (run_id, project_id, status, risk_score, warnings, created_at)
            VALUES (?, ?, ?, NULL, '[]', ?)
            """,
            (run_id, project_id, status, created_at),
        )
    return created_at


def update_run(
    run_id: str,
    *,
    project_id: str | None = None,
    status: str | None = None,
    risk_score: float | None = None,
    warnings: list[str] | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []

    if project_id is not None:
        fields.append("project_id = ?")
        values.append(project_id)
    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if risk_score is not None:
        fields.append("risk_score = ?")
        values.append(float(risk_score))
    if warnings is not None:
        fields.append("warnings = ?")
        values.append(json.dumps(warnings))

    if not fields:
        return

    values.append(run_id)
    with _connect() as conn:
        conn.execute(
            f"UPDATE analysis_runs SET {', '.join(fields)} WHERE run_id = ?",
            values,
        )


def get_latest_run_for_project(project_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            """
            SELECT run_id, project_id, status, risk_score, warnings, created_at
            FROM analysis_runs
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (project_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "run_id": row["run_id"],
        "project_id": row["project_id"],
        "status": row["status"],
        "risk_score": row["risk_score"],
        "warnings": json.loads(row["warnings"] or "[]"),
        "created_at": row["created_at"],
    }
