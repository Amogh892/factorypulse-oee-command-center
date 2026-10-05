"""SQLite persistence. In the local MVP SQLite stands in for Snowflake; snowflake/ holds the cloud DDL."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager

import pandas as pd

from . import config

MUTABLE_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_id TEXT NOT NULL,
    alert_type TEXT NOT NULL,            -- PREDICTED_FAILURE | VIBRATION_LIMIT | TEMPERATURE_LIMIT
    severity TEXT NOT NULL,              -- CRITICAL | WARNING
    risk_score REAL,
    message TEXT,
    status TEXT NOT NULL DEFAULT 'OPEN', -- OPEN | ACKNOWLEDGED | RESOLVED
    created_ts TEXT NOT NULL,
    updated_ts TEXT NOT NULL,
    work_order_id TEXT
);
CREATE TABLE IF NOT EXISTS risk_scores (
    asset_id TEXT NOT NULL, ts TEXT NOT NULL, risk_score REAL NOT NULL,
    top_drivers TEXT, PRIMARY KEY (asset_id, ts)
);
CREATE TABLE IF NOT EXISTS model_metrics (
    trained_ts TEXT, model TEXT, roc_auc REAL, precision_ REAL, recall REAL, f1 REAL,
    n_train INTEGER, n_test INTEGER, positive_rate REAL, feature_importance TEXT
);
CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_ts TEXT, step TEXT, status TEXT, detail TEXT, duration_s REAL
);
"""


def connect(readonly: bool = False) -> sqlite3.Connection:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if readonly:
        return sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.executescript(MUTABLE_DDL)
    return conn


@contextmanager
def session(readonly: bool = False):
    conn = connect(readonly)
    try:
        yield conn
        if not readonly:
            conn.commit()
    finally:
        conn.close()


def write_frames(conn: sqlite3.Connection, frames: dict[str, pd.DataFrame], if_exists: str = "replace") -> None:
    for name, df in frames.items():
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_sql(name, conn, if_exists=if_exists, index=False)
    if table_exists(conn, "sensor_readings"):
        conn.execute("CREATE INDEX IF NOT EXISTS ix_sensor_asset_ts ON sensor_readings(asset_id, ts)")
    conn.commit()


def query(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def log_run(conn: sqlite3.Connection, step: str, status: str, detail: str, duration_s: float) -> None:
    conn.execute("INSERT INTO pipeline_runs VALUES (?,?,?,?,?)",
                 (now_str(), step, status, detail, round(duration_s, 2)))
    conn.commit()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
