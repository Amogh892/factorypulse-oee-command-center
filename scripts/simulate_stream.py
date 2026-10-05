"""Near-real-time simulation: appends new sensor samples and rescores on every tick.

Usage:
    python scripts/simulate_stream.py                         # steady state, 1 tick / 5 s
    python scripts/simulate_stream.py --degrade CNC-002       # inject a bearing-wear ramp on CNC-002
    python scripts/simulate_stream.py --ticks 30 --interval 2 # bounded run

Each tick advances the synthetic clock by one sample (5 min) for every asset, rebuilds the hourly
features, rescores with the saved model and re-evaluates alerts/work orders. Run the Streamlit
command center at the same time to watch the risk and alerts move.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from factorypulse import alerts, db, features, model, synth  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--degrade", default=None, help="asset_id to push along a bearing-wear degradation ramp")
    ap.add_argument("--ticks", type=int, default=0, help="number of ticks (0 = run until Ctrl+C)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between ticks")
    ap.add_argument("--samples-per-tick", type=int, default=6, help="5-minute samples appended per tick (6 = 30 simulated minutes)")
    args = ap.parse_args(argv)

    bundle = model.load_bundle()
    if bundle is None:
        print("No trained model. Run scripts/run_pipeline.py first.")
        return 1

    with db.session(readonly=True) as conn:
        last = db.query(conn, """
            SELECT s.asset_id, MAX(s.ts) AS ts,
                   AVG(CASE WHEN s.machine_status='RUNNING' THEN s.vibration_mm_s END) AS vib_level,
                   AVG(CASE WHEN s.machine_status='RUNNING' THEN s.temperature_c END) AS temp_level
            FROM sensor_readings s JOIN (SELECT asset_id, MAX(ts) mts FROM sensor_readings GROUP BY asset_id) m
              ON s.asset_id = m.asset_id AND s.ts > datetime(m.mts, '-1 hour')
            GROUP BY s.asset_id""")
    sim = synth.LiveSimulator(last, args.degrade)

    step = 0
    try:
        while args.ticks == 0 or step < args.ticks:
            step += 1
            with db.session() as conn:
                for _ in range(args.samples_per_tick):
                    new = sim.next()
                    db.write_frames(conn, {"sensor_readings": new}, if_exists="append")
                feats = features.build_features(conn)
                latest = model.score(conn, feats, bundle)
                stats = alerts.generate_alerts(conn, latest)
                top = latest.sort_values("risk_score", ascending=False).iloc[0]
                print(f"tick {step:>4} | sim clock {new['ts'].max():%Y-%m-%d %H:%M} | top risk {top.asset_id} {top.risk_score:5.1%} "
                      f"| alerts +{stats['created']} ~{stats['updated']} -{stats['resolved']} | WOs +{stats['work_orders']}")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
