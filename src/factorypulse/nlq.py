"""Natural-language questions over the converged IT/OT data.

Two tiers, mirroring Snowflake Cortex Analyst's "verified queries" pattern:
  1. Verified queries - curated SQL matched by intent keywords (deterministic, always correct).
  2. LLM text-to-SQL  - Granite writes SQL against a schema description; the SQL passes a strict
     read-only guard and runs on a read-only connection, with one self-correction retry on error.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

from . import config
from .llm import LocalLLM

SCHEMA_DOC = """
Tables (SQLite):
assets(asset_id, asset_name, asset_type, line_id, plant_id, criticality, ideal_cycle_sec, install_date, manufacturer)
sensor_readings(ts, asset_id, vibration_mm_s, temperature_c, rpm, machine_status)  -- 5-minute OT data; status RUNNING/DOWN_UNPLANNED/DOWN_PLANNED
production_orders(order_id, asset_id, shift_date, shift, product_code, planned_minutes, unplanned_downtime_minutes, run_minutes, planned_qty, total_count, good_count, scrap_count)  -- ERP
work_orders(wo_id, asset_id, wo_type, status, priority, created_ts, closed_ts, title, notes, parts, cost_eur, technician, source, alert_id)  -- wo_type CORRECTIVE/PREVENTIVE/PREDICTIVE
failure_events(asset_id, failure_ts, failure_mode, downtime_minutes, restored_ts, observed)  -- observed=1 for past failures
alerts(alert_id, asset_id, alert_type, severity, risk_score, message, status, created_ts, updated_ts, work_order_id)
risk_scores(asset_id, ts, risk_score, top_drivers)
oee_daily(plant_id, line_id, asset_id, shift_date, planned_minutes, run_minutes, unplanned_downtime_minutes, total_count, good_count, scrap_count, ideal_cycle_sec, availability, performance, quality, oee)
Timestamps are text 'YYYY-MM-DD HH:MM:SS'. Use datetime('now', '-7 days') style for relative dates.
"""


@dataclass
class VerifiedQuery:
    name: str
    keywords: tuple[str, ...]
    sql: str


MIN_MATCH_SCORE = 3  # total matched-keyword characters needed before a verified query is trusted ("oee" = 3)

VERIFIED_QUERIES = [
    VerifiedQuery("Assets ranked by current failure risk", ("likely to fail", "most likely", "risk", "riskiest", "at risk"),
                  "SELECT r.asset_id, a.asset_name, a.line_id, ROUND(r.risk_score,3) AS risk_score, r.ts AS as_of "
                  "FROM risk_scores r JOIN assets a USING(asset_id) "
                  "JOIN (SELECT asset_id, MAX(ts) ts FROM risk_scores GROUP BY asset_id) m ON r.asset_id=m.asset_id AND r.ts=m.ts "
                  "ORDER BY r.risk_score DESC"),
    VerifiedQuery("OEE by asset, last 7 days", ("oee", "effectiveness", "availability", "performance", "quality"),
                  "SELECT asset_id, ROUND(SUM(run_minutes)*1.0/SUM(planned_minutes),3) availability, "
                  "ROUND(MIN(1.0, SUM(total_count*ideal_cycle_sec/60.0)/SUM(run_minutes)),3) performance, "
                  "ROUND(SUM(good_count)*1.0/SUM(total_count),3) quality, "
                  "ROUND(SUM(run_minutes)*1.0/SUM(planned_minutes) * MIN(1.0, SUM(total_count*ideal_cycle_sec/60.0)/SUM(run_minutes)) "
                  "* SUM(good_count)*1.0/SUM(total_count),3) oee "
                  "FROM oee_daily WHERE shift_date >= date('now','-7 days') GROUP BY asset_id ORDER BY oee"),
    VerifiedQuery("Unplanned downtime by asset (minutes, whole period)", ("downtime", "down time", "unplanned"),
                  "SELECT asset_id, COUNT(*) failures, SUM(downtime_minutes) downtime_minutes FROM failure_events "
                  "WHERE observed=1 GROUP BY asset_id ORDER BY downtime_minutes DESC"),
    VerifiedQuery("Failure modes and their frequency", ("failure mode", "common failure", "why do", "common cause", "root cause"),
                  "SELECT failure_mode, COUNT(*) n, ROUND(AVG(downtime_minutes)) avg_downtime_min FROM failure_events "
                  "WHERE observed=1 GROUP BY failure_mode ORDER BY n DESC"),
    VerifiedQuery("Open alerts", ("open alert", "show alerts", "list alerts", "alarms", "critical alerts"),
                  "SELECT alert_id, asset_id, alert_type, severity, ROUND(risk_score,2) risk, status, work_order_id, created_ts "
                  "FROM alerts WHERE status!='RESOLVED' ORDER BY severity, risk DESC"),
    VerifiedQuery("Open work orders", ("open work order", "list work orders", "show work orders", "backlog", "open wo"),
                  "SELECT wo_id, asset_id, wo_type, priority, status, created_ts, title, source FROM work_orders "
                  "WHERE status!='CLOSED' ORDER BY priority, created_ts"),
    VerifiedQuery("Maintenance cost by asset", ("cost", "spend", "expensive"),
                  "SELECT asset_id, ROUND(SUM(cost_eur)) cost_eur, COUNT(*) work_orders FROM work_orders "
                  "WHERE cost_eur IS NOT NULL GROUP BY asset_id ORDER BY cost_eur DESC"),
    VerifiedQuery("Scrap by product (last 7 days)", ("scrap", "reject", "defect"),
                  "SELECT product_code, SUM(scrap_count) scrap, SUM(total_count) total, "
                  "ROUND(SUM(scrap_count)*100.0/SUM(total_count),2) scrap_pct FROM production_orders "
                  "WHERE shift_date >= date('now','-7 days') GROUP BY product_code ORDER BY scrap_pct DESC"),
    VerifiedQuery("Latest sensor snapshot per asset", ("latest reading", "current reading", "latest sensor", "sensor snapshot",
                                                       "current vibration", "current temperature", "right now"),
                  "SELECT s.asset_id, s.ts, s.vibration_mm_s, s.temperature_c, s.rpm, s.machine_status FROM sensor_readings s "
                  "JOIN (SELECT asset_id, MAX(ts) ts FROM sensor_readings GROUP BY asset_id) m ON s.asset_id=m.asset_id AND s.ts=m.ts"),
]

FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate|grant)\b", re.I)


class UnsafeSQL(ValueError):
    pass


def guard_sql(sql: str, limit: int = 200) -> str:
    """Allow exactly one read-only SELECT/WITH statement; append a LIMIT when missing."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:
        raise UnsafeSQL("multiple statements are not allowed")
    if not re.match(r"^(select|with)\b", s, re.I):
        raise UnsafeSQL("only SELECT queries are allowed")
    if FORBIDDEN.search(s):
        raise UnsafeSQL("statement contains a forbidden keyword")
    if not re.search(r"\blimit\s+\d+", s, re.I):
        s += f" LIMIT {limit}"
    return s


def match_verified(question: str) -> VerifiedQuery | None:
    q = question.lower()
    best, best_score = None, 0
    for vq in VERIFIED_QUERIES:
        score = sum(len(k) for k in vq.keywords if k in q)
        if score > best_score:
            best, best_score = vq, score
    return best if best_score >= MIN_MATCH_SCORE else None


def _extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = m.group(1) if m else text
    m2 = re.search(r"(select|with)\b.*", sql, re.S | re.I)
    return m2.group(0).strip() if m2 else sql.strip()


def run_readonly(sql: str) -> pd.DataFrame:
    conn = sqlite3.connect(f"file:{config.DB_PATH.as_posix()}?mode=ro", uri=True)
    try:
        return pd.read_sql_query(sql, conn)
    finally:
        conn.close()


def llm_to_sql(question: str, llm: LocalLLM, error: str | None = None, prev_sql: str | None = None) -> str:
    system = ("You translate questions about a factory database into a single SQLite SELECT statement. "
              "Return only the SQL inside a ```sql code block. Never modify data.")
    user = f"SCHEMA:\n{SCHEMA_DOC}\n\nQUESTION: {question}"
    if error:
        user += f"\n\nYour previous SQL failed:\n{prev_sql}\nError: {error}\nFix it and return only the corrected SQL."
    return _extract_sql(llm.chat(user, system=system, max_tokens=300, temperature=0.0))


def answer(question: str, llm: LocalLLM | None, prefer_llm: bool = False) -> dict:
    """Returns dict(source, name, sql, df, error). `prefer_llm` skips the verified tier (for demos/evaluation)."""
    vq = None if prefer_llm else match_verified(question)
    if vq is not None:
        return dict(source="verified", name=vq.name, sql=vq.sql, df=run_readonly(vq.sql), error=None)
    if llm is None or not llm.available():
        return dict(source="none", name=None, sql=None, df=None,
                    error="No verified query matched and the local LLM is unavailable. Try asking about risk, OEE, downtime, alerts, work orders, cost or scrap.")
    sql, err = None, None
    for attempt in range(2):
        try:
            sql = guard_sql(llm_to_sql(question, llm, error=err, prev_sql=sql))
            return dict(source="granite-4.0-1b", name=f"LLM text-to-SQL (attempt {attempt + 1})", sql=sql,
                        df=run_readonly(sql), error=None)
        except UnsafeSQL as exc:
            return dict(source="granite-4.0-1b", name="blocked by guard", sql=sql, df=None, error=f"Rejected: {exc}")
        except Exception as exc:  # noqa: BLE001 - SQL errors feed the retry
            err = str(exc)
    return dict(source="granite-4.0-1b", name="failed", sql=sql, df=None, error=f"Could not produce valid SQL: {err}")
