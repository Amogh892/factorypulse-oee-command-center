"""End-to-end pipeline: synthetic data -> load -> features -> train -> score -> OEE -> alerts/work orders.

Usage:
    python scripts/run_pipeline.py              # full run (regenerates data if the DB is empty)
    python scripts/run_pipeline.py --regenerate # force fresh synthetic data
    python scripts/run_pipeline.py --no-train   # rescore with the existing model (used by the stream simulator)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factorypulse import alerts, config, db, features, model, oee, synth  # noqa: E402


def summarize(value) -> str:
    if hasattr(value, "shape"):
        return f"{value.shape[0]} rows x {value.shape[1]} cols"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in value.items()
                         if not isinstance(v, str) or len(v) < 40)
    return str(value)


def step(conn, name: str, fn):
    t0 = time.time()
    try:
        value = fn()
        detail = summarize(value)
        db.log_run(conn, name, "OK", detail, time.time() - t0)
        print(f"[OK ] {name:<18} {time.time() - t0:5.1f}s  {detail}")
        return value
    except Exception as exc:
        db.log_run(conn, name, "FAILED", f"{type(exc).__name__}: {exc}", time.time() - t0)
        print(f"[ERR] {name:<18} {type(exc).__name__}: {exc}")
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true", help="regenerate synthetic data even if it exists")
    ap.add_argument("--no-train", action="store_true", help="skip training and reuse the saved model")
    ap.add_argument("--no-work-orders", action="store_true", help="raise alerts but do not auto-create work orders")
    args = ap.parse_args(argv)

    with db.session() as conn:
        need_data = args.regenerate or not db.table_exists(conn, "sensor_readings")

        if need_data:
            def gen():
                frames = synth.generate()
                db.write_frames(conn, frames)
                for name, df in frames.items():
                    df.to_csv(config.DATA_DIR / f"{name}.csv", index=False)
                conn.execute("DELETE FROM alerts"); conn.execute("DELETE FROM risk_scores")
                return {k: len(v) for k, v in frames.items()}
            step(conn, "synthetic_data", gen)

        feats = step(conn, "features", lambda: features.build_features(conn))
        if not args.no_train or model.load_bundle() is None:
            m = step(conn, "train_model", lambda: model.train(conn, feats))
            print(f"      ROC-AUC {m['roc_auc']:.3f} | precision {m['precision_']:.2f} | recall {m['recall']:.2f} "
                  f"| positives {m['positive_rate']:.1%} | train/test {m['n_train']}/{m['n_test']}")
        latest = step(conn, "score_assets", lambda: model.score(conn, feats))
        step(conn, "oee", lambda: f"{len(oee.compute_oee(conn))} asset-days")
        step(conn, "alerts", lambda: alerts.generate_alerts(conn, latest, auto_work_orders=not args.no_work_orders))

        print("\nLatest risk per asset:")
        for _, r in latest.sort_values("risk_score", ascending=False).iterrows():
            print(f"  {r.asset_id:<8} {r.risk_score:6.1%}  {model.risk_band(r.risk_score)}")
        print(f"\nDatabase: {config.DB_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
