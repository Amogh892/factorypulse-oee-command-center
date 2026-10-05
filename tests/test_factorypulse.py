"""Validation suite: data consistency, OEE math, alert guardrails and the SQL guard.

Runs against a temporary database so the demo data is never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factorypulse import alerts, config, db, features, model, nlq, oee, rca, synth  # noqa: E402


@pytest.fixture(scope="module")
def tmp_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("fp")
    config.DATA_DIR, config.DB_PATH, config.MODEL_DIR = d, d / "t.db", d / "models"
    config.RISK_MODEL_PATH = config.MODEL_DIR / "m.pkl"
    frames = synth.generate(end=pd.Timestamp("2026-10-01 12:00"))
    with db.session() as conn:
        db.write_frames(conn, frames)
        feats = features.build_features(conn)
        metrics = model.train(conn, feats)
        latest = model.score(conn, feats)
        oee.compute_oee(conn)
        stats = alerts.generate_alerts(conn, latest)
    return dict(frames=frames, feats=feats, metrics=metrics, latest=latest, stats=stats)


# ---------------------------------------------------------------- synthetic data integrity
def test_referential_integrity(tmp_db):
    f = tmp_db["frames"]
    ids = set(f["assets"].asset_id)
    for t in ["sensor_readings", "failure_events", "work_orders", "production_orders"]:
        assert set(f[t].asset_id) <= ids, t


def test_every_observed_failure_has_corrective_work_order(tmp_db):
    f = tmp_db["frames"]
    corrective = f["work_orders"][f["work_orders"].wo_type == "CORRECTIVE"]
    observed = f["failure_events"][f["failure_events"].observed == 1]
    merged = observed.merge(corrective, left_on=["asset_id", "failure_ts"], right_on=["asset_id", "created_ts"])
    assert len(merged) == len(observed)


def test_sensor_timeline_is_regular(tmp_db):
    sr = tmp_db["frames"]["sensor_readings"]
    per_asset = sr.groupby("asset_id").ts.agg(["min", "max", "size"])
    assert per_asset["size"].nunique() == 1
    assert (sr.vibration_mm_s >= 0).all() and (sr.rpm >= 0).all()


# ---------------------------------------------------------------- OEE
def test_oee_math():
    r = oee.oee_from_counts(planned_minutes=480, run_minutes=432, total_count=500, good_count=490, ideal_cycle_sec=45)
    assert r["availability"] == pytest.approx(0.9)
    assert r["performance"] == pytest.approx(500 * 45 / 60 / 432)
    assert r["quality"] == pytest.approx(0.98)
    assert r["oee"] == pytest.approx(r["availability"] * r["performance"] * r["quality"])


def test_oee_edge_cases():
    assert oee.oee_from_counts(0, 0, 0, 0, 10)["oee"] == 0
    assert oee.oee_from_counts(480, 480, 10_000, 10_000, 45)["performance"] == 1.0  # capped


# ---------------------------------------------------------------- model
def test_model_learns_something(tmp_db):
    assert tmp_db["metrics"]["roc_auc"] > 0.8
    assert 0.02 < tmp_db["metrics"]["positive_rate"] < 0.3


def test_risk_bands():
    assert model.risk_band(0.95) == "CRITICAL"
    assert model.risk_band(0.5) == "WARNING"
    assert model.risk_band(0.1) == "HEALTHY"
    assert model.hours_to_failure_estimate(0.1) is None
    assert model.hours_to_failure_estimate(1.0) == pytest.approx(2.0)


# ---------------------------------------------------------------- alert guardrails
def test_alerts_dedupe_and_auto_work_order(tmp_db):
    with db.session() as conn:
        latest = pd.DataFrame([dict(asset_id="CNV-001", ts="2026-10-01 12:00:00", risk_score=0.9, top_drivers="[]")])
        s1 = alerts.generate_alerts(conn, latest)
        s2 = alerts.generate_alerts(conn, latest)
        n_open = conn.execute("SELECT COUNT(*) FROM alerts WHERE asset_id='CNV-001' AND alert_type='PREDICTED_FAILURE' "
                              "AND status!='RESOLVED'").fetchone()[0]
        n_wo = conn.execute("SELECT COUNT(*) FROM work_orders WHERE asset_id='CNV-001' AND status='OPEN'").fetchone()[0]
    assert s1["created"] >= 1 and s2["created"] == 0 and s2["updated"] >= 1
    assert n_open == 1, "one open prediction alert per asset"
    assert n_wo == 1, "one open work order per asset even after repeated scoring"


def test_alert_auto_resolves_when_risk_normalises(tmp_db):
    with db.session() as conn:
        alerts.generate_alerts(conn, pd.DataFrame([dict(asset_id="CNV-002", ts="x", risk_score=0.8, top_drivers="[]")]))
        alerts.generate_alerts(conn, pd.DataFrame([dict(asset_id="CNV-002", ts="x", risk_score=0.05, top_drivers="[]")]))
        status = conn.execute("SELECT status FROM alerts WHERE asset_id='CNV-002' ORDER BY alert_id DESC LIMIT 1").fetchone()[0]
    assert status == "RESOLVED"


def test_warning_does_not_create_work_order(tmp_db):
    with db.session() as conn:
        alerts.generate_alerts(conn, pd.DataFrame([dict(asset_id="PRS-002", ts="x", risk_score=0.5, top_drivers="[]")]))
        n = conn.execute("SELECT COUNT(*) FROM work_orders WHERE asset_id='PRS-002' AND source='AUTO_PREDICTIVE'").fetchone()[0]
    assert n == 0


# ---------------------------------------------------------------- natural language layer
@pytest.mark.parametrize("bad", ["DROP TABLE alerts", "DELETE FROM alerts", "SELECT 1; DROP TABLE alerts",
                                 "UPDATE alerts SET status='x'", "PRAGMA writable_schema=1", "ATTACH 'x' AS y"])
def test_sql_guard_blocks_writes(bad):
    with pytest.raises(nlq.UnsafeSQL):
        nlq.guard_sql(bad)


def test_sql_guard_adds_limit():
    assert nlq.guard_sql("select * from assets").endswith("LIMIT 200")
    assert "limit 5" in nlq.guard_sql("select * from assets limit 5").lower()
    assert "LIMIT 200" not in nlq.guard_sql("select * from assets limit 5")


def test_verified_queries_run(tmp_db):
    for vq in nlq.VERIFIED_QUERIES:
        df = nlq.run_readonly(vq.sql)
        assert isinstance(df, pd.DataFrame), vq.name


def test_verified_match():
    assert nlq.match_verified("Which assets are most likely to fail?").name.startswith("Assets ranked")
    assert nlq.match_verified("show me oee per asset").name.startswith("OEE")
    assert nlq.match_verified("tell me a joke") is None


def test_rca_rule_fallback_without_llm(tmp_db):
    with db.session(readonly=True) as conn:
        res = rca.investigate(conn, "CNC-002", llm=None)
    assert res["source"] == "rules" and "Risk is" in res["narrative"]
    assert "ASSET: CNC-002" in res["evidence_text"]
