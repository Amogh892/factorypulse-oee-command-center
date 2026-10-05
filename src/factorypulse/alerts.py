"""Alerting and work-order automation with guardrails.

Two independent alert sources feed the same queue:
  1. the ML risk score (PREDICTED_FAILURE)
  2. hard engineering limits on the raw sensors (VIBRATION_LIMIT / TEMPERATURE_LIMIT) - a rule-based
     safety net that keeps working even if the model is missing or wrong.

Guardrails: alerts are de-duplicated per asset/type, only CRITICAL predictions open a work order,
one open work order per asset at a time, and prediction alerts auto-resolve when the risk normalises.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .db import now_str, query
from .model import hours_to_failure_estimate, risk_band


def _open_alert(conn, asset_id: str, alert_type: str):
    return conn.execute("SELECT alert_id FROM alerts WHERE asset_id=? AND alert_type=? AND status IN ('OPEN','ACKNOWLEDGED')",
                        (asset_id, alert_type)).fetchone()


def _open_work_order(conn, asset_id: str):
    return conn.execute("SELECT wo_id FROM work_orders WHERE asset_id=? AND status IN ('OPEN','IN_PROGRESS')",
                        (asset_id,)).fetchone()


def next_wo_id(conn) -> str:
    row = conn.execute("SELECT MAX(CAST(SUBSTR(wo_id, 4) AS INTEGER)) FROM work_orders").fetchone()
    return f"WO-{(row[0] or 1000) + 1}"


def upsert_alert(conn, asset_id: str, alert_type: str, severity: str, risk: float | None, message: str) -> tuple[int, bool]:
    """Returns (alert_id, created)."""
    now = now_str()
    existing = _open_alert(conn, asset_id, alert_type)
    if existing:
        conn.execute("UPDATE alerts SET severity=?, risk_score=?, message=?, updated_ts=? WHERE alert_id=?",
                     (severity, risk, message, now, existing[0]))
        return int(existing[0]), False
    cur = conn.execute("INSERT INTO alerts (asset_id, alert_type, severity, risk_score, message, status, created_ts, updated_ts) "
                       "VALUES (?,?,?,?,?,'OPEN',?,?)", (asset_id, alert_type, severity, risk, message, now, now))
    return int(cur.lastrowid), True


def create_work_order(conn, asset_id: str, title: str, notes: str, priority: str = "P2",
                      wo_type: str = "PREDICTIVE", source: str = "MANUAL", alert_id: int | None = None) -> str:
    wo_id = next_wo_id(conn)
    conn.execute("INSERT INTO work_orders (wo_id, asset_id, wo_type, status, priority, created_ts, closed_ts, title, notes, "
                 "parts, cost_eur, technician, source, alert_id) VALUES (?,?,?,'OPEN',?,?,NULL,?,?,NULL,NULL,NULL,?,?)",
                 (wo_id, asset_id, wo_type, priority, now_str(), title, notes, source, alert_id))
    if alert_id is not None:
        conn.execute("UPDATE alerts SET work_order_id=?, updated_ts=? WHERE alert_id=?", (wo_id, now_str(), alert_id))
    conn.commit()
    return wo_id


def set_alert_status(conn, alert_id: int, status: str) -> None:
    assert status in {"OPEN", "ACKNOWLEDGED", "RESOLVED"}
    conn.execute("UPDATE alerts SET status=?, updated_ts=? WHERE alert_id=?", (status, now_str(), alert_id))
    conn.commit()


def set_work_order_status(conn, wo_id: str, status: str, technician: str | None = None) -> None:
    assert status in {"OPEN", "IN_PROGRESS", "CLOSED"}
    closed = now_str() if status == "CLOSED" else None
    conn.execute("UPDATE work_orders SET status=?, closed_ts=COALESCE(?, closed_ts), technician=COALESCE(?, technician) WHERE wo_id=?",
                 (status, closed, technician, wo_id))
    conn.commit()


def generate_alerts(conn, latest_risk: pd.DataFrame, auto_work_orders: bool = True) -> dict:
    """Turns the latest risk scores + raw sensor limits into alerts and (optionally) work orders."""
    stats = dict(created=0, updated=0, resolved=0, work_orders=0)
    names = dict(query(conn, "SELECT asset_id, asset_name FROM assets").values)

    for _, r in latest_risk.iterrows():
        band = risk_band(float(r.risk_score))
        if band == "HEALTHY":
            row = _open_alert(conn, r.asset_id, "PREDICTED_FAILURE")
            if row:
                set_alert_status(conn, int(row[0]), "RESOLVED")
                stats["resolved"] += 1
            continue
        drivers = json.loads(r.top_drivers) if isinstance(r.top_drivers, str) and r.top_drivers else []
        why = "; ".join(f"{d['label']} {d['value']} ({d['z']:+.1f} sigma)" for d in drivers) or "no single dominant driver"
        eta = hours_to_failure_estimate(float(r.risk_score))
        msg = (f"{names.get(r.asset_id, r.asset_id)}: {float(r.risk_score):.0%} probability of failure within "
               f"{config.PREDICTION_HORIZON_H}h (est. ~{eta:.0f}h). Drivers: {why}.")
        alert_id, created = upsert_alert(conn, r.asset_id, "PREDICTED_FAILURE", band, float(r.risk_score), msg)
        stats["created" if created else "updated"] += 1

        if auto_work_orders and band == "CRITICAL":
            linked = conn.execute("SELECT work_order_id FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()[0]
            if not linked and not _open_work_order(conn, r.asset_id):
                create_work_order(
                    conn, r.asset_id, priority="P1", wo_type="PREDICTIVE", source="AUTO_PREDICTIVE", alert_id=alert_id,
                    title=f"Predictive inspection: {names.get(r.asset_id, r.asset_id)}",
                    notes=f"Auto-created from alert #{alert_id}. {msg} Inspect bearings/coupling/cooling before next shift.")
                stats["work_orders"] += 1

    # Rule-based safety net on the last hour of raw data
    recent = query(conn, """
        SELECT s.asset_id, MAX(vibration_mm_s) AS vib_max, MAX(temperature_c) AS temp_max
        FROM sensor_readings s JOIN (SELECT asset_id, MAX(ts) AS mts FROM sensor_readings GROUP BY asset_id) m
          ON s.asset_id = m.asset_id AND s.ts >= datetime(m.mts, '-1 hour')
        WHERE s.machine_status = 'RUNNING' GROUP BY s.asset_id""")
    for _, r in recent.iterrows():
        checks = [("VIBRATION_LIMIT", r.vib_max, config.SENSOR_LIMITS["vibration_mm_s"], "mm/s"),
                  ("TEMPERATURE_LIMIT", r.temp_max, config.SENSOR_LIMITS["temperature_c"], "C")]
        for atype, value, limit, unit in checks:
            if value is not None and value > limit:
                msg = f"{names.get(r.asset_id, r.asset_id)}: {atype.replace('_', ' ').lower()} exceeded ({value:.1f} {unit} > {limit} {unit})."
                _, created = upsert_alert(conn, r.asset_id, atype, "CRITICAL", None, msg)
                stats["created" if created else "updated"] += 1
            else:
                row = _open_alert(conn, r.asset_id, atype)
                if row:
                    set_alert_status(conn, int(row[0]), "RESOLVED")
                    stats["resolved"] += 1
    conn.commit()
    return stats


def open_alerts(conn) -> pd.DataFrame:
    return query(conn, "SELECT al.*, a.asset_name, a.line_id, a.criticality FROM alerts al JOIN assets a USING (asset_id) "
                       "WHERE al.status != 'RESOLVED' ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 ELSE 1 END, risk_score DESC")
