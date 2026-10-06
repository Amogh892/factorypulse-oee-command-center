# Running FactoryPulse

Everything runs locally on CPU. Tested on Windows 11 with Python 3.13; Linux/macOS work the same with
`source .venv/bin/activate` instead of the Windows activation line.

## 1. Prerequisites

- Python **3.13** (3.11/3.12 also fine; 3.14 has no prebuilt `llama-cpp-python` wheel yet)
- The GGUF model file: `ibm-granite-4.0-1b-Q4_K_M.gguf` (≈1 GB).
  Download from Hugging Face (`ibm-granite/granite-4.0-1b-GGUF` or any Q4_K_M quantisation of Granite 4.0 1B).
- ~2 GB free RAM for the model, no GPU required.

## 2. Setup

```powershell
git clone https://github.com/Amogh892/factorypulse-oee-command-center.git
cd factorypulse-oee-command-center

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1            # Linux/macOS: source .venv/bin/activate

python -m pip install --upgrade pip
pip install streamlit pandas numpy scikit-learn plotly pytest python-dotenv
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Point the app at your model file (copy `.env.example` to `.env` and edit, or set the variable in the shell):

```
FP_LLM_MODEL_PATH=C:\Users\<you>\Downloads\Models\ibm-granite-4.0-1b-Q4_K_M.gguf
FP_LLM_THREADS=8
```

If the model file is missing the app still runs; root‑cause analysis falls back to rule‑based explanations
and natural‑language questions are answered by verified queries only.

## 3. Build the data, train and score (one command)

```powershell
python scripts/run_pipeline.py --regenerate
```

Expected output (≈30 s):

```
[OK ] synthetic_data   assets=8, sensor_readings=69120, failure_events=19, work_orders=33, production_orders=720
[OK ] features         5685 rows x 24 cols
[OK ] train_model      roc_auc=0.97.., precision_=0.8.., recall=0.9..
[OK ] score_assets     8 rows x 21 cols
[OK ] oee              248 asset-days
[OK ] alerts           created=3, updated=0, resolved=0, work_orders=2

Latest risk per asset:
  CNC-002   99.9%  CRITICAL
  PMP-001   87.9%  CRITICAL
  ...
```

Outputs: `data/factorypulse.db` (SQLite), `data/*.csv` (for Snowflake loading), `data/models/failure_model.pkl`.

Re‑score with the existing model (no retraining): `python scripts/run_pipeline.py --no-train`.

## 4. Start the command center

```powershell
streamlit run app/command_center.py
```

Open http://localhost:8501. Pages:

| Page | What to try |
|---|---|
| Overview | OEE tiles, risk by asset, OEE trend by line, alert feed |
| Alert triage | Select the CNC‑002 alert → *Investigate root cause (Granite)* → acknowledge / create WO / resolve |
| Asset explorer | Pick an asset, widen the window to 30 days to see past failures and ramps |
| Investigate (NL) | "Which assets are most likely to fail?", "Which product has the highest scrap rate?", or a free‑form question that goes to Granite text‑to‑SQL |
| Work orders | Start / close the auto‑created P1 work orders (closing resolves the linked alert) |
| System | Model metrics and feature importance, pipeline run log, *Rescore now* button, LLM status |

The first LLM call loads the model (~2 s) and is cached for the session.

## 5. Near‑real‑time demo (optional, second terminal)

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/simulate_stream.py --degrade CNC-003 --interval 3
```

Every 3 s it appends 30 simulated minutes of readings for all assets, pushes CNC‑003 along a bearing‑wear ramp,
rescores and re‑evaluates alerts. Refresh the app (or press `R`) to watch CNC‑003 climb from HEALTHY to WARNING
to CRITICAL and receive a work order. Stop with Ctrl+C. Use `--ticks 20` for a bounded run, or omit `--degrade`
for steady state.

## 6. Validate

```powershell
python -m pytest -q
```

20 tests cover synthetic‑data integrity, OEE math, a model‑quality floor, alert guardrails (dedupe, one open WO
per asset, auto‑resolve, warnings don't open WOs), the read‑only SQL guard and the LLM‑free fallback.
They run against a temporary database and never touch `data/`.

## 7. Command‑line root cause / questions (no UI)

```powershell
python -c "import sys;sys.path.insert(0,'src');from factorypulse import db,rca,llm;c=db.connect(True);r=rca.investigate(c,'CNC-002',llm.get_llm());print(r['narrative'])"
python -c "import sys;sys.path.insert(0,'src');from factorypulse import nlq,llm;print(nlq.answer('Which asset had the most unplanned downtime?',llm.get_llm())['df'])"
```

## 8. Snowflake (cloud twin)

1. `snowflake/01_schema.sql` — database, schema, tables, stage.
2. Upload `data/*.csv` to `@FACTORYPULSE_STAGE` (e.g. `snow stage copy data/ @FACTORYPULSE_STAGE`) and
   `COPY INTO <TABLE> FROM @FACTORYPULSE_STAGE/<table>.csv` for each table.
3. `snowflake/02_dynamic_tables.sql` — hourly features and daily OEE as Dynamic Tables.
4. Register `data/models/failure_model.pkl` in the Model Registry as `FAILURE_MODEL`.
5. `snowflake/03_scoring_task.sql` — scheduled scoring + alerting + auto work orders every 15 minutes.
6. Create a Cortex Analyst semantic view from `snowflake/semantic_model.yaml` and attach it to a Snowsight agent
   or the Slackbot.

## 9. Hosting a shareable version

**Demo video and deck** are attached to the GitHub release
https://github.com/Amogh892/factorypulse-oee-command-center/releases/tag/v1.0.0 and embedded on the project
page served by GitHub Pages from `docs/index.html`.

**The Streamlit app cannot run on GitHub Pages** (Pages serves static files only; Streamlit needs a Python
server). Use Streamlit Community Cloud, which is free and deploys straight from this repo:

1. Go to https://share.streamlit.io, sign in with the GitHub account that owns the repo, and click *Create app*.
2. Repository `Amogh892/factorypulse-oee-command-center`, branch `main`, main file `app/command_center.py`.
3. Deploy. On first load the app generates the synthetic data, trains and scores the model itself (~30 s),
   so no pipeline run or database upload is needed.

On the cloud the local Granite model is not available (no GGUF, no `llama-cpp-python`), so root-cause analysis
uses the rule-based narrative and questions are answered by verified queries. The source badge in the UI shows
which path answered. Everything else (prediction, alerts, work orders, OEE, triage actions) works unchanged.
Note that Community Cloud containers are ephemeral: triage actions persist until the app restarts.

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `No data yet` banner in the app | Run `python scripts/run_pipeline.py` first |
| `llama_cpp` import error | Reinstall from the CPU wheel index (command in step 2); check Python is 3.11–3.13 |
| LLM status `failed: Model file not found` | Set `FP_LLM_MODEL_PATH` in `.env` to the GGUF location |
| Slow LLM answers | Raise `FP_LLM_THREADS` to your physical core count; lower `FP_LLM_MAX_TOKENS` |
| `database is locked` | Stop the stream simulator before pressing *Rescore now*; SQLite is single‑writer |
| Want different scenarios | Edit `PENDING_FAILURES`, `ASSETS` or `FAILURE_MODES` in `src/factorypulse/synth.py` and rerun with `--regenerate` |
