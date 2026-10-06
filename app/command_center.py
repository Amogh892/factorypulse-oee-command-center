"""FactoryPulse command center (Streamlit).

Pages: Overview | Alert Triage | Asset Explorer | Investigate (natural language) | Work Orders | System
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factorypulse import alerts, config, db, features, model, nlq, oee, rca  # noqa: E402
from factorypulse.llm import LocalLLM  # noqa: E402

st.set_page_config(page_title="FactoryPulse Command Center", page_icon="🏭", layout="wide")

BAND_COLOR = {"CRITICAL": "#d62728", "WARNING": "#ff7f0e", "HEALTHY": "#2ca02c"}
SEV_ICON = {"CRITICAL": "🔴", "WARNING": "🟠"}


# ----------------------------------------------------------------------------- helpers
@st.cache_resource(show_spinner="Loading IBM Granite 4.0 1B (local GGUF)...")
def get_llm() -> LocalLLM:
    llm = LocalLLM()
    llm.available()
    return llm


def q(sql: str, params: tuple = ()) -> pd.DataFrame:
    with db.session(readonly=True) as conn:
        return db.query(conn, sql, params)


def db_ready() -> bool:
    if not config.DB_PATH.exists():
        return False
    with db.session(readonly=True) as conn:
        return db.table_exists(conn, "sensor_readings") and db.table_exists(conn, "oee_daily")


def pct(x: float) -> str:
    return f"{x:.1%}"


def has_text(x) -> bool:
    """True for a non-empty string (SQLite NULLs arrive as None or NaN through pandas)."""
    return isinstance(x, str) and x != ""


def latest_risk_df() -> pd.DataFrame:
    df = q("SELECT r.asset_id, a.asset_name, a.line_id, a.criticality, r.ts, r.risk_score, r.top_drivers "
           "FROM risk_scores r JOIN assets a USING(asset_id) "
           "JOIN (SELECT asset_id, MAX(ts) ts FROM risk_scores GROUP BY asset_id) m ON r.asset_id=m.asset_id AND r.ts=m.ts "
           "ORDER BY r.risk_score DESC")
    df["band"] = df["risk_score"].apply(model.risk_band)
    return df


def sensor_chart(asset_id: str, hours: int = 72) -> go.Figure:
    df = q("SELECT ts, vibration_mm_s, temperature_c, rpm, machine_status FROM sensor_readings WHERE asset_id=? "
           "AND ts >= datetime((SELECT MAX(ts) FROM sensor_readings WHERE asset_id=?), ?) ORDER BY ts",
           (asset_id, asset_id, f"-{hours} hours"))
    df["ts"] = pd.to_datetime(df["ts"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.ts, y=df.vibration_mm_s, name="Vibration (mm/s)", yaxis="y1", line=dict(color="#1f77b4")))
    fig.add_trace(go.Scatter(x=df.ts, y=df.temperature_c, name="Temperature (C)", yaxis="y2", line=dict(color="#d62728")))
    fig.add_trace(go.Scatter(x=df.ts, y=df.rpm, name="RPM", yaxis="y3", line=dict(color="#7f7f7f", dash="dot")))
    fig.add_hline(y=config.SENSOR_LIMITS["vibration_mm_s"], line=dict(color="#1f77b4", dash="dash"), opacity=0.4,
                  annotation_text="vib limit", annotation_position="top left")
    fails = q("SELECT failure_ts, failure_mode FROM failure_events WHERE asset_id=? AND observed=1", (asset_id,))
    for _, f in fails.iterrows():
        t = pd.Timestamp(f.failure_ts)
        if t >= df.ts.min():
            fig.add_vline(x=t, line=dict(color="black", dash="dot"))
            fig.add_annotation(x=t, y=1, yref="paper", text=f"failure: {f.failure_mode}", showarrow=False, yanchor="bottom")
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.15),
                      yaxis=dict(title="mm/s"), yaxis2=dict(title="C", overlaying="y", side="right"),
                      yaxis3=dict(overlaying="y", side="right", showticklabels=False, showgrid=False))
    return fig


def risk_history_chart(asset_id: str) -> go.Figure:
    df = q("SELECT ts, risk_score FROM risk_scores WHERE asset_id=? ORDER BY ts", (asset_id,))
    df["ts"] = pd.to_datetime(df["ts"])
    fig = px.area(df, x="ts", y="risk_score", title=None)
    fig.add_hline(y=config.RISK_CRITICAL, line=dict(color=BAND_COLOR["CRITICAL"], dash="dash"), annotation_text="critical")
    fig.add_hline(y=config.RISK_WARNING, line=dict(color=BAND_COLOR["WARNING"], dash="dash"), annotation_text="warning")
    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(range=[0, 1], title="failure risk (24h)"))
    return fig


def show_drivers(top_drivers: str | None):
    drivers = json.loads(top_drivers) if top_drivers else []
    if not drivers:
        st.caption("No feature deviates more than 1 sigma from the healthy training distribution.")
        return
    st.dataframe(pd.DataFrame(drivers)[["label", "value", "z"]].rename(columns={"label": "driver", "z": "sigma vs healthy"}),
                 hide_index=True, width="stretch")


def bootstrap_data() -> None:
    """First run on a fresh host (e.g. Streamlit Community Cloud): generate data, train and score in-process."""
    from factorypulse import synth
    with db.session() as conn:
        frames = synth.generate()
        db.write_frames(conn, frames)
        conn.execute("DELETE FROM alerts"); conn.execute("DELETE FROM risk_scores")
        feats = features.build_features(conn)
        model.train(conn, feats)
        latest = model.score(conn, feats)
        oee.compute_oee(conn)
        alerts.generate_alerts(conn, latest)
        db.log_run(conn, "bootstrap", "OK", "generated, trained and scored from the app", 0.0)


def run_rescore(auto_wo: bool = True) -> dict:
    with db.session() as conn:
        feats = features.build_features(conn)
        latest = model.score(conn, feats)
        oee.compute_oee(conn)
        return alerts.generate_alerts(conn, latest, auto_work_orders=auto_wo)


# ----------------------------------------------------------------------------- pages
def page_overview():
    st.title("🏭 FactoryPulse — Predictive Maintenance & OEE Command Center")
    risk = latest_risk_df()
    oee7 = q("SELECT * FROM oee_daily WHERE shift_date >= date('now','-7 days')")
    summary = oee.oee_summary(oee7)
    open_al = q("SELECT severity, COUNT(*) n FROM alerts WHERE status!='RESOLVED' GROUP BY severity")
    n_crit = int(open_al.loc[open_al.severity == "CRITICAL", "n"].sum()) if not open_al.empty else 0
    n_warn = int(open_al.loc[open_al.severity == "WARNING", "n"].sum()) if not open_al.empty else 0
    n_wo = int(q("SELECT COUNT(*) n FROM work_orders WHERE status!='CLOSED'").n[0])
    down7 = int(oee7["unplanned_downtime_minutes"].sum()) if not oee7.empty else 0

    c = st.columns(7)
    c[0].metric("Plant OEE (7d)", pct(summary["oee"]))
    c[1].metric("Availability", pct(summary["availability"]))
    c[2].metric("Performance", pct(summary["performance"]))
    c[3].metric("Quality", pct(summary["quality"]))
    c[4].metric("Open alerts", f"{n_crit + n_warn}", f"{n_crit} critical", delta_color="inverse")
    c[5].metric("Assets at risk", int((risk.band != "HEALTHY").sum()), f"of {len(risk)}", delta_color="off")
    c[6].metric("Unplanned downtime (7d)", f"{down7 / 60:.1f} h", f"{n_wo} open WOs", delta_color="off")

    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Failure risk by asset (next 24h)")
        fig = px.bar(risk.sort_values("risk_score"), x="risk_score", y="asset_name", orientation="h", color="band",
                     color_discrete_map=BAND_COLOR, text=risk.sort_values("risk_score")["risk_score"].map(pct),
                     hover_data=["line_id", "criticality"])
        fig.update_layout(height=340, xaxis=dict(range=[0, 1], title="probability"), yaxis_title=None,
                          margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        fig.add_vline(x=config.RISK_CRITICAL, line=dict(color=BAND_COLOR["CRITICAL"], dash="dash"))
        fig.add_vline(x=config.RISK_WARNING, line=dict(color=BAND_COLOR["WARNING"], dash="dash"))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("OEE trend by line (daily)")
        trend = q("SELECT line_id, shift_date, SUM(run_minutes) run, SUM(planned_minutes) planned, "
                  "SUM(total_count*ideal_cycle_sec/60.0) ideal, SUM(good_count) good, SUM(total_count) total "
                  "FROM oee_daily GROUP BY line_id, shift_date ORDER BY shift_date")
        trend["oee"] = (trend.run / trend.planned) * (trend.ideal / trend.run).clip(upper=1) * (trend.good / trend.total)
        fig = px.line(trend, x="shift_date", y="oee", color="line_id", markers=True)
        fig.update_layout(height=340, yaxis=dict(range=[0, 1], tickformat=".0%"), margin=dict(l=10, r=10, t=10, b=10),
                          xaxis_title=None, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Alert feed")
    feed = q("SELECT al.alert_id, al.severity, a.asset_name, al.alert_type, al.status, al.work_order_id, al.message, al.updated_ts "
             "FROM alerts al JOIN assets a USING(asset_id) WHERE al.status!='RESOLVED' "
             "ORDER BY CASE severity WHEN 'CRITICAL' THEN 0 ELSE 1 END, risk_score DESC LIMIT 8")
    if feed.empty:
        st.success("No open alerts. All assets within normal operating envelope.")
    for _, r in feed.iterrows():
        st.markdown(f"{SEV_ICON.get(r.severity, '•')} **#{r.alert_id} {r.asset_name}** · {r.alert_type} · {r.status}"
                    f"{' · ' + r.work_order_id if has_text(r.work_order_id) else ''} — {r.message}")


def page_triage():
    st.title("🚨 Alert triage")
    with db.session() as conn:
        open_df = alerts.open_alerts(conn)
    if open_df.empty:
        st.success("No open alerts.")
        return
    st.dataframe(open_df[["alert_id", "severity", "asset_name", "line_id", "alert_type", "risk_score", "status",
                          "work_order_id", "created_ts"]], hide_index=True, width="stretch")
    choice = st.selectbox("Select an alert", open_df.alert_id,
                          format_func=lambda i: f"#{i} · {open_df.set_index('alert_id').loc[i, 'asset_name']} · "
                                                f"{open_df.set_index('alert_id').loc[i, 'severity']}")
    al = open_df.set_index("alert_id").loc[choice]
    st.markdown(f"### {SEV_ICON.get(al.severity, '')} {al.asset_name} — {al.alert_type}")
    st.write(al.message)

    b = st.columns(4)
    with db.session() as conn:
        if b[0].button("✅ Acknowledge", disabled=al.status == "ACKNOWLEDGED"):
            alerts.set_alert_status(conn, int(choice), "ACKNOWLEDGED"); st.rerun()
        if b[1].button("🛠️ Create work order", disabled=has_text(al.work_order_id)):
            wo = alerts.create_work_order(conn, al.asset_id, priority="P1" if al.severity == "CRITICAL" else "P2",
                                          title=f"Inspection: {al.asset_name} ({al.alert_type.replace('_', ' ').lower()})",
                                          notes=f"Created from alert #{choice} by operator. {al.message}", alert_id=int(choice))
            st.toast(f"Work order {wo} created"); st.rerun()
        if b[2].button("🧹 Resolve"):
            alerts.set_alert_status(conn, int(choice), "RESOLVED"); st.rerun()
        investigate = b[3].button("🔎 Investigate root cause (Granite)")

    left, right = st.columns([1.3, 1])
    with left:
        st.plotly_chart(sensor_chart(al.asset_id), width="stretch")
    with right:
        st.markdown("**Top risk drivers**")
        drv = q("SELECT top_drivers FROM risk_scores WHERE asset_id=? ORDER BY ts DESC LIMIT 1", (al.asset_id,))
        show_drivers(drv.top_drivers[0] if not drv.empty else None)
        st.plotly_chart(risk_history_chart(al.asset_id), width="stretch")

    if investigate:
        with st.spinner("Assembling evidence and asking Granite..."):
            with db.session(readonly=True) as conn:
                res = rca.investigate(conn, al.asset_id, get_llm())
        st.markdown(f"**Root cause analysis** · source: `{res['source']}`")
        st.info(res["narrative"])
        with st.expander("Evidence pack given to the model (grounding)"):
            st.code(res["evidence_text"])


def page_assets():
    st.title("🔧 Asset explorer")
    assets_df = q("SELECT * FROM assets")
    asset_id = st.selectbox("Asset", assets_df.asset_id, format_func=lambda i: f"{i} · {assets_df.set_index('asset_id').loc[i, 'asset_name']}")
    a = assets_df.set_index("asset_id").loc[asset_id]
    risk = latest_risk_df().set_index("asset_id").loc[asset_id]
    last = q("SELECT * FROM sensor_readings WHERE asset_id=? ORDER BY ts DESC LIMIT 1", (asset_id,)).iloc[0]
    c = st.columns(6)
    c[0].metric("Risk (24h)", pct(risk.risk_score), risk.band, delta_color="off")
    c[1].metric("Vibration", f"{last.vibration_mm_s:.2f} mm/s")
    c[2].metric("Temperature", f"{last.temperature_c:.1f} C")
    c[3].metric("RPM", f"{last.rpm:.0f}")
    c[4].metric("Status", last.machine_status)
    c[5].metric("Line / criticality", f"{a.line_id} / {a.criticality}")
    hours = st.slider("Window (hours)", 24, 24 * 30, 72, step=24)
    st.plotly_chart(sensor_chart(asset_id, hours), width="stretch")
    l, r = st.columns(2)
    with l:
        st.markdown("**Risk history**")
        st.plotly_chart(risk_history_chart(asset_id), width="stretch")
    with r:
        st.markdown("**Top risk drivers now**")
        show_drivers(risk.top_drivers)
    st.markdown("**Maintenance history (ERP / CMMS)**")
    st.dataframe(q("SELECT wo_id, wo_type, status, priority, created_ts, closed_ts, title, notes, parts, cost_eur, source "
                   "FROM work_orders WHERE asset_id=? ORDER BY created_ts DESC", (asset_id,)), hide_index=True, width="stretch")
    st.markdown("**OEE by day (ERP production orders)**")
    st.dataframe(q("SELECT shift_date, planned_minutes, run_minutes, unplanned_downtime_minutes, total_count, good_count, scrap_count, "
                   "availability, performance, quality, oee FROM oee_daily WHERE asset_id=? ORDER BY shift_date DESC", (asset_id,)),
                 hide_index=True, width="stretch")


def page_investigate():
    st.title("💬 Investigate in natural language")
    st.caption("Ask the converged IT/OT data a question, or ask for a root cause explanation of a specific asset. "
               "Answers come from verified queries first, then from the local Granite model (text-to-SQL with a read-only guard).")
    mode = st.radio("Mode", ["Ask the data", "Root cause for an asset"], horizontal=True)
    llm = get_llm()
    st.caption(f"LLM: `{llm.describe()['model']}` · status **{llm.status}**" + (f" · {llm.error}" if llm.error else ""))

    if mode == "Ask the data":
        examples = ["Which assets are most likely to fail?", "Show OEE by asset for the last 7 days",
                    "Which asset had the most unplanned downtime?", "What are the most common failure modes?",
                    "List open work orders", "Which product has the highest scrap rate?",
                    "How many sensor readings do we have per asset?"]
        ex = st.selectbox("Examples", ["(type your own)"] + examples)
        question = st.text_input("Question", value="" if ex == "(type your own)" else ex)
        prefer_llm = st.checkbox("Force Granite text-to-SQL (skip verified queries)", value=False)
        if st.button("Ask") and question:
            with st.spinner("Answering..."):
                res = nlq.answer(question, llm, prefer_llm=prefer_llm)
            st.markdown(f"Source: `{res['source']}` · {res['name'] or ''}")
            if res["error"]:
                st.warning(res["error"])
            if res["sql"]:
                with st.expander("SQL", expanded=res["source"] != "verified"):
                    st.code(res["sql"], language="sql")
            if res["df"] is not None:
                st.dataframe(res["df"], hide_index=True, width="stretch")
                num = res["df"].select_dtypes("number")
                if len(res["df"]) > 1 and not num.empty and res["df"].columns[0] not in num.columns:
                    st.bar_chart(res["df"].set_index(res["df"].columns[0])[num.columns[-1]])
    else:
        assets_df = q("SELECT asset_id, asset_name FROM assets")
        asset_id = st.selectbox("Asset", assets_df.asset_id, format_func=lambda i: f"{i} · {assets_df.set_index('asset_id').loc[i, 'asset_name']}")
        question = st.text_input("Question (optional)", placeholder="e.g. Is this related to the last bearing replacement?")
        if st.button("Investigate"):
            with st.spinner("Assembling evidence and asking Granite..."):
                with db.session(readonly=True) as conn:
                    res = rca.investigate(conn, asset_id, llm, question or None)
            st.markdown(f"Source: `{res['source']}`")
            st.info(res["narrative"])
            with st.expander("Evidence pack (grounding)"):
                st.code(res["evidence_text"])


def page_work_orders():
    st.title("🛠️ Work orders")
    status = st.multiselect("Status", ["OPEN", "IN_PROGRESS", "CLOSED"], default=["OPEN", "IN_PROGRESS"])
    df = q(f"SELECT w.wo_id, a.asset_name, w.wo_type, w.priority, w.status, w.created_ts, w.closed_ts, w.title, w.source, "
           f"w.alert_id, w.technician, w.notes FROM work_orders w JOIN assets a USING(asset_id) "
           f"WHERE w.status IN ({','.join('?' * len(status))}) ORDER BY w.created_ts DESC", tuple(status)) if status else pd.DataFrame()
    st.dataframe(df, hide_index=True, width="stretch")
    if df.empty:
        return
    wo = st.selectbox("Work order", df.wo_id)
    c = st.columns(3)
    tech = c[0].text_input("Assign technician", value="")
    with db.session() as conn:
        if c[1].button("▶️ Start (in progress)"):
            alerts.set_work_order_status(conn, wo, "IN_PROGRESS", tech or None); st.rerun()
        if c[2].button("✔️ Close"):
            alerts.set_work_order_status(conn, wo, "CLOSED", tech or None)
            linked = conn.execute("SELECT alert_id FROM work_orders WHERE wo_id=?", (wo,)).fetchone()[0]
            if linked:
                alerts.set_alert_status(conn, int(linked), "RESOLVED")
            st.rerun()


def page_system():
    st.title("⚙️ System & pipeline")
    c1, c2, c3 = st.columns(3)
    if c1.button("🔄 Rescore now (features → model → OEE → alerts)"):
        with st.spinner("Rescoring..."):
            stats = run_rescore()
        st.success(f"Done: {stats}")
    if c2.button("🧪 Rescore without auto work orders"):
        st.success(f"Done: {run_rescore(auto_wo=False)}")
    if c3.button("🧠 Load / check local LLM"):
        st.json(get_llm().describe())

    st.subheader("Model")
    m = q("SELECT * FROM model_metrics ORDER BY trained_ts DESC LIMIT 1")
    if not m.empty:
        r = m.iloc[0]
        k = st.columns(5)
        k[0].metric("ROC-AUC (holdout)", f"{r.roc_auc:.3f}")
        k[1].metric("Precision", f"{r.precision_:.2f}")
        k[2].metric("Recall", f"{r.recall:.2f}")
        k[3].metric("Train / test rows", f"{r.n_train} / {r.n_test}")
        k[4].metric("Trained", r.trained_ts)
        fi = pd.Series(json.loads(r.feature_importance)).rename(index=features.FEATURE_LABELS).head(10)
        st.bar_chart(fi)
    st.subheader("Pipeline runs")
    st.dataframe(q("SELECT * FROM pipeline_runs ORDER BY run_ts DESC LIMIT 30"), hide_index=True, width="stretch")
    st.subheader("Data volumes")
    st.dataframe(pd.DataFrame([(t, int(q(f"SELECT COUNT(*) n FROM {t}").n[0])) for t in
                               ["assets", "sensor_readings", "production_orders", "work_orders", "failure_events",
                                "risk_scores", "alerts", "oee_daily"]], columns=["table", "rows"]), hide_index=True)


PAGES = {"Overview": page_overview, "Alert triage": page_triage, "Asset explorer": page_assets,
         "Investigate (NL)": page_investigate, "Work orders": page_work_orders, "System": page_system}


def main():
    st.sidebar.title("FactoryPulse")
    st.sidebar.caption("IT + OT converged · local Granite 4.0 1B · SQLite stand-in for Snowflake")
    if not db_ready():
        with st.spinner("First run: generating 30 days of synthetic IT/OT data, training and scoring the failure model (~30 s)..."):
            bootstrap_data()
        st.rerun()
    page = st.sidebar.radio("Navigate", list(PAGES))
    st.sidebar.divider()
    r = latest_risk_df()
    st.sidebar.markdown("**Assets at risk**")
    for _, x in r[r.band != "HEALTHY"].iterrows():
        st.sidebar.markdown(f"{SEV_ICON.get(x.band, '')} {x.asset_name} · {pct(x.risk_score)}")
    if (r.band == "HEALTHY").all():
        st.sidebar.markdown("🟢 none")
    st.sidebar.divider()
    st.sidebar.caption(f"Data as of {r.ts.max()}")
    PAGES[page]()


main()
