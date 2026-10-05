---
name: factorypulse-pipeline
description: Build, run, validate and demo the FactoryPulse predictive-maintenance + OEE pipeline (synthetic IT/OT data, failure model, alerts, work orders, Streamlit command center). Use when asked to regenerate data, retrain, rescore, simulate a live degradation, run tests, or deploy the Snowflake twin.
---

# FactoryPulse pipeline skill

Reusable skill for Snowflake CoCo (Cortex Code) or any agent runtime. It wraps the project's
scripts so an agent can operate the whole lifecycle without hand-editing code.

## When to use
- "regenerate the demo data", "retrain the failure model", "rescore the assets"
- "simulate a failure on <asset>", "show an alert being raised live"
- "run the validation suite", "why is asset X at risk" (then call the RCA module)
- "deploy this to Snowflake" (use the SQL in `snowflake/` in order)

## Commands (run from the repo root inside the project venv)
| Goal | Command |
|---|---|
| Full pipeline, fresh data | `python scripts/run_pipeline.py --regenerate` |
| Rescore only (keeps model) | `python scripts/run_pipeline.py --no-train` |
| Live degradation demo | `python scripts/simulate_stream.py --degrade CNC-003 --interval 3` |
| Validation suite | `python -m pytest -q` |
| Command center | `streamlit run app/command_center.py` |
| Root cause (CLI) | `python -c "import sys;sys.path.insert(0,'src');from factorypulse import db,rca,llm;c=db.connect(True);print(rca.investigate(c,'CNC-002',llm.get_llm())['narrative'])"` |

## Procedure for "set up the demo from scratch"
1. `python scripts/run_pipeline.py --regenerate` and confirm the printed ROC-AUC is above 0.85.
2. `python -m pytest -q` and confirm all tests pass. If a guardrail test fails, stop and report.
3. Start `streamlit run app/command_center.py`; confirm at least one CRITICAL alert and one AUTO_PREDICTIVE work order exist on the Overview page (the synthetic window ends with two assets mid-degradation by design).
4. Optionally start `simulate_stream.py --degrade <asset>` in a second terminal for a live demo.

## Snowflake deployment (cloud twin)
Run in order: `snowflake/01_schema.sql` -> load `data/*.csv` into the stage and `COPY INTO` each table ->
`snowflake/02_dynamic_tables.sql` -> register `data/models/failure_model.pkl` in the Model Registry as `FAILURE_MODEL` ->
`snowflake/03_scoring_task.sql` -> create the Cortex Analyst semantic view from `snowflake/semantic_model.yaml`.

## Guardrails the agent must keep
- Never write to the database through the natural-language path; `nlq.guard_sql` only allows single SELECT statements.
- Only CRITICAL prediction alerts create work orders, and never more than one open work order per asset.
- If the LLM is unavailable, use the rule-based narrative (`rca.rule_based_narrative`); never fabricate sensor values.
- Report metrics from `model_metrics` rather than quoting numbers from memory.
