"""Root cause investigation: assemble an evidence pack from OT + IT data, then explain it.

The LLM only ever sees the evidence pack (grounding), and the UI always shows the pack next to the
narrative so a human can check every claim. If the LLM is unavailable, a rule-based explanation
built from the same evidence is returned instead.
"""
from __future__ import annotations

import json

import pandas as pd

from . import config
from .db import query
from .llm import LocalLLM
from .model import hours_to_failure_estimate, risk_band


def build_evidence(conn, asset_id: str) -> dict:
    asset = query(conn, "SELECT * FROM assets WHERE asset_id=?", (asset_id,)).iloc[0].to_dict()
    risk = query(conn, "SELECT ts, risk_score, top_drivers FROM risk_scores WHERE asset_id=? ORDER BY ts DESC LIMIT 1",
                 (asset_id,))
    risk_row = risk.iloc[0].to_dict() if not risk.empty else dict(ts=None, risk_score=0.0, top_drivers="[]")
    drivers = json.loads(risk_row["top_drivers"] or "[]")

    recent = query(conn, """
        SELECT ROUND(AVG(vibration_mm_s),2) vib_avg, ROUND(MAX(vibration_mm_s),2) vib_max,
               ROUND(AVG(temperature_c),1) temp_avg, ROUND(MAX(temperature_c),1) temp_max,
               ROUND(AVG(rpm),0) rpm_avg, ROUND(MIN(rpm),0) rpm_min
        FROM sensor_readings WHERE asset_id=? AND machine_status='RUNNING'
          AND ts >= datetime((SELECT MAX(ts) FROM sensor_readings WHERE asset_id=?), '-24 hours')""",
                   (asset_id, asset_id)).iloc[0].to_dict()
    baseline = query(conn, """
        SELECT ROUND(AVG(vibration_mm_s),2) vib_avg, ROUND(AVG(temperature_c),1) temp_avg, ROUND(AVG(rpm),0) rpm_avg
        FROM sensor_readings WHERE asset_id=? AND machine_status='RUNNING'
          AND ts < datetime((SELECT MIN(ts) FROM sensor_readings WHERE asset_id=?), '+3 days')""",
                     (asset_id, asset_id)).iloc[0].to_dict()

    wos = query(conn, "SELECT wo_id, wo_type, status, created_ts, title, notes, parts FROM work_orders "
                      "WHERE asset_id=? ORDER BY created_ts DESC LIMIT 4", (asset_id,))
    failures = query(conn, "SELECT failure_ts, failure_mode, downtime_minutes FROM failure_events "
                           "WHERE asset_id=? AND observed=1 ORDER BY failure_ts DESC LIMIT 3", (asset_id,))
    prod = query(conn, "SELECT shift_date, shift, product_code, planned_minutes, run_minutes, total_count, good_count, scrap_count "
                       "FROM production_orders WHERE asset_id=? ORDER BY shift_date DESC, shift DESC LIMIT 3", (asset_id,))
    open_alerts = query(conn, "SELECT alert_type, severity, message FROM alerts WHERE asset_id=? AND status!='RESOLVED'",
                        (asset_id,))

    return dict(asset=asset, risk=dict(score=float(risk_row["risk_score"]), band=risk_band(float(risk_row["risk_score"])),
                                       as_of=risk_row["ts"], eta_hours=hours_to_failure_estimate(float(risk_row["risk_score"])),
                                       drivers=drivers),
                sensors_last_24h=recent, sensors_baseline=baseline, limits=config.SENSOR_LIMITS,
                recent_work_orders=wos.to_dict("records"), recent_failures=failures.to_dict("records"),
                recent_production=prod.to_dict("records"), open_alerts=open_alerts.to_dict("records"))


def evidence_to_text(ev: dict) -> str:
    a, r, s, b = ev["asset"], ev["risk"], ev["sensors_last_24h"], ev["sensors_baseline"]
    lines = [
        f"ASSET: {a['asset_id']} {a['asset_name']} ({a['asset_type']}, {a['line_id']}, criticality {a['criticality']}, "
        f"manufacturer {a['manufacturer']}, installed {a['install_date']})",
        f"PREDICTED FAILURE RISK (next {config.PREDICTION_HORIZON_H}h): {r['score']:.0%} [{r['band']}]"
        + (f", estimated time to failure ~{r['eta_hours']:.0f}h" if r["eta_hours"] else ""),
        "TOP RISK DRIVERS: " + ("; ".join(f"{d['label']} = {d['value']} ({d['z']:+.1f} sigma vs healthy)" for d in r["drivers"]) or "none"),
        f"SENSORS LAST 24H: vibration avg {s['vib_avg']} / max {s['vib_max']} mm/s (healthy baseline {b['vib_avg']}, limit {ev['limits']['vibration_mm_s']}); "
        f"temperature avg {s['temp_avg']} / max {s['temp_max']} C (baseline {b['temp_avg']}, limit {ev['limits']['temperature_c']}); "
        f"rpm avg {s['rpm_avg']} / min {s['rpm_min']} (baseline {b['rpm_avg']})",
    ]
    if ev["open_alerts"]:
        lines.append("OPEN ALERTS: " + " | ".join(f"{x['severity']} {x['alert_type']}" for x in ev["open_alerts"]))
    lines.append("MAINTENANCE HISTORY (newest first):")
    for w in ev["recent_work_orders"]:
        lines.append(f"  - {w['created_ts'][:10]} {w['wo_type']} {w['status']} '{w['title']}': {w['notes']} [parts: {w['parts']}]")
    if ev["recent_failures"]:
        lines.append("PAST FAILURES: " + "; ".join(f"{f['failure_ts'][:10]} {f['failure_mode']} ({f['downtime_minutes']} min down)"
                                                  for f in ev["recent_failures"]))
    lines.append("RECENT PRODUCTION (ERP): " + "; ".join(
        f"{p['shift_date']} {p['shift']} {p['product_code']} run {p['run_minutes']}/{p['planned_minutes']} min, "
        f"{p['good_count']}/{p['total_count']} good, {p['scrap_count']} scrap" for p in ev["recent_production"]))
    return "\n".join(lines)


def rule_based_narrative(ev: dict) -> str:
    s, b, r = ev["sensors_last_24h"], ev["sensors_baseline"], ev["risk"]
    vib_up = (s["vib_avg"] or 0) / (b["vib_avg"] or 1)
    temp_up = (s["temp_avg"] or 0) - (b["temp_avg"] or 0)
    rpm_drop = (b["rpm_avg"] or 0) - (s["rpm_min"] or 0)
    causes = []
    if vib_up > 1.6 and temp_up > 5:
        causes.append("bearing wear or lubrication breakdown (vibration and temperature rising together)")
    elif vib_up > 1.6:
        causes.append("mechanical looseness, misalignment or early bearing damage (vibration rising, temperature stable)")
    elif temp_up > 5:
        causes.append("cooling or overload problem (temperature rising without a vibration signature)")
    if rpm_drop > 0.05 * (b["rpm_avg"] or 1):
        causes.append("drive instability (RPM dips below baseline)")
    history = [w for w in ev["recent_work_orders"] if w["wo_type"] == "CORRECTIVE"]
    hist_txt = (f" The last corrective work order ({history[0]['wo_id']}) reported: \"{history[0]['notes']}\" - a recurrence is plausible."
                if history else "")
    if not causes:
        return (f"Risk is {r['score']:.0%} ({r['band']}). Sensor levels are within normal variation of the baseline "
                f"(vibration x{vib_up:.2f}, temperature {temp_up:+.1f} C). No strong root cause signal; continue monitoring.")
    return (f"Risk is {r['score']:.0%} ({r['band']}). Most likely cause: {'; or '.join(causes)}. "
            f"Vibration is x{vib_up:.2f} the healthy baseline and temperature is {temp_up:+.1f} C above it.{hist_txt} "
            "Recommended: schedule an inspection within the next shift, check lubrication and coupling alignment, "
            "and reduce load until the inspection is complete.")


def investigate(conn, asset_id: str, llm: LocalLLM | None, question: str | None = None) -> dict:
    ev = build_evidence(conn, asset_id)
    text = evidence_to_text(ev)
    ask = question or ("Explain the most likely root cause of the elevated failure risk, how confident you are, "
                       "and the top 3 recommended actions.")
    prompt = f"EVIDENCE:\n{text}\n\nQUESTION: {ask}\n\nAnswer in at most 180 words with the headings Root cause, Confidence, Actions."
    if llm is not None and llm.available():
        try:
            return dict(narrative=llm.chat(prompt), evidence=ev, evidence_text=text, source="granite-4.0-1b")
        except Exception as exc:  # noqa: BLE001
            fallback = rule_based_narrative(ev)
            return dict(narrative=f"{fallback}\n\n(LLM call failed: {exc})", evidence=ev, evidence_text=text, source="rules")
    return dict(narrative=rule_based_narrative(ev), evidence=ev, evidence_text=text, source="rules")
