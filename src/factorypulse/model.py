"""Failure prediction model: gradient boosting over the hourly feature table.

Time-based split (first 70 % of the window trains, last 30 % tests) so the reported metrics
reflect predicting *future* failures, not interpolating past ones.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from . import config
from .db import now_str
from .features import FEATURE_LABELS, FEATURES


def train(conn, feats: pd.DataFrame) -> dict:
    df = feats.dropna(subset=["vib_mean_1h"]).copy()
    cutoff = df["ts"].quantile(0.7)
    train_df, test_df = df[df["ts"] <= cutoff], df[df["ts"] > cutoff]
    X_tr, y_tr = train_df[FEATURES].fillna(0), train_df["label"]
    X_te, y_te = test_df[FEATURES].fillna(0), test_df["label"]

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05,
                                     subsample=0.9, random_state=config.SEED)
    clf.fit(X_tr, y_tr)
    proba = clf.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = dict(
        trained_ts=now_str(), model="GradientBoostingClassifier",
        roc_auc=float(roc_auc_score(y_te, proba)) if y_te.nunique() > 1 else float("nan"),
        precision_=float(precision_score(y_te, pred, zero_division=0)),
        recall=float(recall_score(y_te, pred, zero_division=0)),
        f1=float(f1_score(y_te, pred, zero_division=0)),
        n_train=int(len(train_df)), n_test=int(len(test_df)),
        positive_rate=float(df["label"].mean()),
        feature_importance=json.dumps({f: round(float(i), 4) for f, i in
                                       sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1])}),
    )
    conn.execute("INSERT INTO model_metrics VALUES (?,?,?,?,?,?,?,?,?,?)", tuple(metrics.values()))
    conn.commit()

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = dict(model=clf, features=FEATURES, trained_ts=metrics["trained_ts"],
                  train_mean=X_tr.mean().to_dict(), train_std=X_tr.std().replace(0, 1).to_dict())
    with open(config.RISK_MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    return metrics


def load_bundle() -> dict | None:
    if not config.RISK_MODEL_PATH.exists():
        return None
    with open(config.RISK_MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def _top_drivers(row: pd.Series, bundle: dict, k: int = 3) -> list[dict]:
    z = {f: (row[f] - bundle["train_mean"][f]) / bundle["train_std"][f] for f in FEATURES if pd.notna(row[f])}
    drivers = sorted(z.items(), key=lambda kv: -abs(kv[1]))[:k]
    return [dict(feature=f, label=FEATURE_LABELS[f], value=round(float(row[f]), 3), z=round(float(v), 2))
            for f, v in drivers if abs(v) >= 1.0]


def score(conn, feats: pd.DataFrame, bundle: dict | None = None) -> pd.DataFrame:
    """Scores every hourly row, stores the history and returns the latest risk per asset."""
    bundle = bundle or load_bundle()
    if bundle is None:
        raise RuntimeError("No trained model found. Run the pipeline with training first.")
    df = feats.dropna(subset=["vib_mean_1h"]).copy()
    df["risk_score"] = bundle["model"].predict_proba(df[FEATURES].fillna(0))[:, 1]

    latest = df.sort_values("ts").groupby("asset_id").tail(1).copy()
    latest["top_drivers"] = latest.apply(lambda r: json.dumps(_top_drivers(r, bundle)), axis=1)

    hist = df[["asset_id", "ts", "risk_score"]].copy()
    hist["ts"] = hist["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    hist["top_drivers"] = None
    hist = hist.merge(latest[["asset_id", "ts", "top_drivers"]].assign(ts=lambda d: d["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")),
                      on=["asset_id", "ts"], how="left", suffixes=("_x", ""))
    hist = hist[["asset_id", "ts", "risk_score", "top_drivers"]]
    conn.execute("DELETE FROM risk_scores")
    hist.to_sql("risk_scores", conn, if_exists="append", index=False)
    conn.commit()
    return latest[["asset_id", "ts", "risk_score", "top_drivers"] + FEATURES].reset_index(drop=True)


def latest_risk(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT r.asset_id, r.ts, r.risk_score, r.top_drivers FROM risk_scores r "
        "JOIN (SELECT asset_id, MAX(ts) AS ts FROM risk_scores GROUP BY asset_id) m "
        "ON r.asset_id = m.asset_id AND r.ts = m.ts ORDER BY r.risk_score DESC", conn)


def risk_band(risk: float) -> str:
    if risk >= config.RISK_CRITICAL:
        return "CRITICAL"
    if risk >= config.RISK_WARNING:
        return "WARNING"
    return "HEALTHY"


def hours_to_failure_estimate(risk: float) -> float | None:
    """Rough translation of risk into an expected time-to-failure inside the prediction horizon."""
    if risk < config.RISK_WARNING:
        return None
    return float(np.interp(risk, [config.RISK_WARNING, 1.0], [config.PREDICTION_HORIZON_H, 2.0]))
