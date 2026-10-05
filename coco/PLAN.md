# FactoryPulse — planning notes (CoCo lifecycle evidence)

This file records how the solution was framed before any build started, so judges can trace the
Planning -> Development -> Execution -> Testing phases.

## 1. Problem framing
Unplanned downtime is expensive because the people who see OT symptoms (vibration, temperature, RPM)
are not the people who hold IT context (ERP production schedule, CMMS maintenance history). Converging the
two lets us (a) predict failures early enough to act, (b) open the work order automatically, and
(c) express the business impact as OEE.

## 2. Data model (ontology)
```
PLANT 1─n LINE 1─n ASSET
ASSET 1─n SENSOR_READING      (OT, 5-min stream: vibration_mm_s, temperature_c, rpm, machine_status)
ASSET 1─n PRODUCTION_ORDER    (IT/ERP, per shift: planned/run minutes, total/good/scrap counts, product)
ASSET 1─n WORK_ORDER          (IT/CMMS: corrective/preventive/predictive, free-text notes, parts, cost)
ASSET 1─n FAILURE_EVENT       (ground truth for training & validation)
ASSET 1─n RISK_SCORE          (derived: hourly failure probability + top drivers)
ASSET 1─n ALERT 0..1─1 WORK_ORDER (derived + actionable)
ASSET 1─n OEE_DAILY           (derived from production orders)
```

## 3. Workflow
```
sensors ──► hourly features ──┐
ERP/CMMS ─► maintenance ctx ──┼─► failure model ─► risk ─► alerts ─► work orders
                              │                              │
production orders ─► OEE ─────┘                              ▼
                                               command center (triage, RCA chat)
```
Scheduled: rescoring every tick/15 min (local simulator / Snowflake task).

## 4. Design decisions
- **Local-first MVP**: SQLite + scikit-learn + Streamlit + a local Granite 4.0 1B GGUF, so the whole
  thing runs on a laptop with no credentials. `snowflake/` holds the 1:1 cloud twin (tables, dynamic
  tables, scheduled task, semantic model) for the CoCo / Snowsight surface.
- **Two-tier NL layer**: verified queries first (deterministic), LLM text-to-SQL second (guarded),
  which mirrors Cortex Analyst's verified-query behaviour.
- **Grounded RCA**: the LLM only sees an evidence pack built from the data; the pack is shown next to
  the answer. Rule-based fallback if the model is unavailable.
- **Guardrails**: read-only SQL guard, alert dedupe, one open WO per asset, auto-resolve, rule-based
  sensor limits as a safety net independent of the ML model.

## 5. Validation plan
pytest suite covers referential integrity of the synthetic data, OEE math, model quality floor,
alert guardrails, the SQL guard and the LLM-free fallback path. Metrics are stored in `model_metrics`
and shown in the app's System page rather than quoted by hand.
