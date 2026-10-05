# FactoryPulse — Predictive Maintenance & OEE Command Center

> Hackathon MVP: converge OT sensor streams with ERP/maintenance context to **predict failures a day ahead,
> open work orders automatically, explain root causes in natural language, and lift OEE** — all runnable on a
> laptop with a local **IBM Granite 4.0 1B** model, with a one-to-one **Snowflake** twin for the cloud.

See [RUNNING.md](RUNNING.md) for setup and run instructions. This file explains *what* the solution is and *why*.

---

## 1. The problem

Manufacturers lose value to unplanned downtime because the signals that precede a breakdown (vibration,
temperature, RPM) live in OT historians, while the context needed to act (production schedule, maintenance
history, cost, spare parts) lives in IT systems such as ERP and CMMS. Nobody sees both at once, so:

- failures are discovered when the machine stops, not when the bearing starts to wear;
- work orders are raised manually, hours later, with little diagnostic context;
- OEE is reported after the fact and never linked back to the physical cause.

## 2. What FactoryPulse does

| Capability | How |
|---|---|
| **Converge IT and OT** | One data model joins 5‑minute sensor readings with ERP production orders (per shift) and CMMS work orders (with free‑text technician notes). |
| **Predict failures in advance** | Hourly features (rolling means, trends, baseline ratios, hours since maintenance) feed a gradient‑boosting model that outputs the probability of an unplanned failure in the next 24 h, plus the top drivers behind the score. |
| **Automate work orders** | Risk ≥ 70 % raises a CRITICAL alert and auto‑creates a P1 predictive work order; 40–70 % raises a WARNING. Hard sensor limits act as a rule‑based safety net independent of the model. |
| **Root cause in natural language** | An evidence pack (sensor deltas vs baseline, risk drivers, last work orders, past failures, current production) is assembled from the data and explained by the local Granite model. The evidence is always shown next to the narrative. |
| **Ask the data** | Natural‑language questions are answered by verified queries first, then by guarded LLM text‑to‑SQL (read‑only, single statement, auto‑LIMIT, one self‑correction retry). |
| **OEE** | Availability × Performance × Quality per asset/day from the ERP orders, aggregated to line and plant. |
| **Command center** | Streamlit app with Overview, Alert triage (acknowledge / create WO / resolve / investigate), Asset explorer, Investigate (NL), Work orders and System pages. |
| **Near real time** | A stream simulator appends new readings, rescoring and re‑evaluating alerts on every tick; the app updates live. |
| **Snowflake twin** | `snowflake/` contains the tables, Dynamic Tables (features, OEE), a scheduled scoring Task with the same guardrails, and a Cortex Analyst semantic model with verified queries. |

## 3. Architecture

```
            OT                                    IT
  sensor_readings (5-min)          production_orders (ERP)   work_orders (CMMS, notes)
         │                                   │                       │
         ▼                                   ▼                       ▼
  hourly features  ◄── hours since maintenance ─────────────────────┘
  (rolling / trend / baseline)               │
         │                                   ▼
         ▼                               oee_daily  ──────────────────────────┐
  failure model (GBM, time-split) ─► risk_scores + top drivers                 │
         │                                                                    │
         ▼                                                                    ▼
  alerts (dedupe, severity, auto-resolve) ─► work_orders (auto P1)     Command center (Streamlit)
         │                                                                    ▲
         └──────────► evidence pack ─► Granite 4.0 1B (local GGUF) ─► RCA narrative / text-to-SQL
```

Local MVP stack: Python 3.13, SQLite (stand‑in for Snowflake), pandas, scikit‑learn, Streamlit, Plotly,
llama‑cpp‑python with `ibm-granite-4.0-1b-Q4_K_M.gguf` (CPU only, loads in ~2 s, answers in ~3–10 s).

## 4. Data model (ontology)

| Table | Side | Grain | Key columns |
|---|---|---|---|
| `assets` | master | asset | type, line, plant, criticality, ideal cycle time |
| `sensor_readings` | OT | asset × 5 min | vibration_mm_s, temperature_c, rpm, machine_status |
| `production_orders` | IT / ERP | asset × shift | planned/run minutes, total/good/scrap counts, product |
| `work_orders` | IT / CMMS | work order | type, status, priority, notes, parts, cost, source, alert_id |
| `failure_events` | truth | failure | mode, downtime, `observed` flag |
| `risk_scores` | derived | asset × hour | risk_score, top_drivers (JSON) |
| `alerts` | derived, actionable | alert | type, severity, status, work_order_id |
| `oee_daily` | derived | asset × day | availability, performance, quality, oee |

The synthetic generator (`src/factorypulse/synth.py`) is **referentially consistent**: every observed failure has
a corrective work order whose notes describe the real failure mode, degradation ramps lower ERP performance
and raise scrap in the affected shifts, and planned maintenance windows are excluded from planned time.
Two assets are deliberately left mid‑degradation at the end of the window so the command center opens with
live CRITICAL alerts and auto‑created work orders.

## 5. The model

- **Features (17)**: 1 h / 6 h / 24 h vibration and temperature means, peaks and variability, RPM variability,
  24 h trends, vibration ratio and temperature delta vs the asset's healthy baseline, hours since the last
  closed work order.
- **Label**: unplanned failure within the next 24 h (rows while down are excluded).
- **Model**: `GradientBoostingClassifier`, trained on the first 70 % of the timeline and evaluated on the
  last 30 % (true forward‑looking validation). Typical holdout metrics on the generated data:
  ROC‑AUC ≈ 0.97, precision ≈ 0.9, recall ≈ 0.8–0.9. The actual numbers are stored in `model_metrics`
  and shown on the app's System page.
- **Explainability**: per‑asset top drivers are the features furthest (in sigma) from the healthy training
  distribution; they appear in alerts, work‑order notes and the RCA evidence pack.

## 6. Guardrails and graceful fallback

- The natural‑language path can never write: `nlq.guard_sql` allows a single `SELECT`/`WITH`, blocks DDL/DML
  keywords, appends a `LIMIT`, and runs on a read‑only connection.
- Verified queries are tried before the LLM, so the common questions are deterministic.
- Alerts are de‑duplicated per asset and type, only CRITICAL predictions open a work order, there is at most one
  open work order per asset, and prediction alerts auto‑resolve when risk normalises.
- Rule‑based limit alerts (vibration > 7.1 mm/s, temperature > 85 °C) work even if the model is missing.
- If the GGUF model or `llama-cpp-python` is unavailable, RCA switches to a rule‑based narrative built from the
  same evidence; the UI labels the source (`granite-4.0-1b` vs `rules` vs `verified`).
- The model only ever sees the evidence pack and is told not to invent values; the pack is displayed for audit.

## 7. How CoCo (Snowflake Cortex Code) fits the lifecycle

| Phase | Evidence in this repo |
|---|---|
| Planning | `coco/PLAN.md` — problem framing, ontology, workflow, design decisions, validation plan |
| Development | `src/`, `app/`, `snowflake/` — pipeline, model, app and the Snowflake DDL / Dynamic Tables / semantic model |
| Execution | `scripts/run_pipeline.py` (end‑to‑end), `scripts/simulate_stream.py` (near real time), `snowflake/03_scoring_task.sql` (scheduled, unattended) |
| Testing & validation | `tests/test_factorypulse.py` — data integrity, OEE math, model floor, alert guardrails, SQL guard, LLM‑free fallback |
| Reusable skill | `coco/skills/factorypulse-pipeline/SKILL.md` — a documented skill any team can drop into CoCo to operate the pipeline |

Deploying the cloud twin: run `snowflake/01_schema.sql`, load `data/*.csv`, run `02_dynamic_tables.sql`,
register the trained model as `FAILURE_MODEL` in the Model Registry, run `03_scoring_task.sql`, and create the
Cortex Analyst semantic view from `semantic_model.yaml`. The Streamlit app can then be pointed at Snowflake by
replacing the SQLite calls in `db.py` with the Snowflake connector (same SQL dialect for the app queries).

## 8. Repository layout

```
app/command_center.py        Streamlit command center
src/factorypulse/            config · synth · db · features · model · oee · alerts · llm · rca · nlq
scripts/run_pipeline.py      generate → load → features → train → score → OEE → alerts/WOs
scripts/simulate_stream.py   near-real-time ticks with optional injected degradation
tests/test_factorypulse.py   validation suite (pytest)
snowflake/                   01_schema · 02_dynamic_tables · 03_scoring_task · semantic_model.yaml
coco/                        PLAN.md and the reusable factorypulse-pipeline skill
data/                        generated CSVs, SQLite DB and trained model (git-ignored)
```

## 9. Limitations of the MVP

- Synthetic data only (by design: no production data, privacy respected from the start).
- The 1B model is small: text‑to‑SQL works for simple questions and is guarded, but verified queries carry the
  demo; RCA narratives are grounded but should be read with the evidence pack.
- Time‑to‑failure is a heuristic mapping from risk, not a survival model.
- SQLite is single‑writer; the Snowflake twin is where multi‑user and scale belong.
