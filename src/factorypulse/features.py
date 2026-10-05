"""Feature engineering: hourly OT aggregates joined with IT (maintenance) context, plus the training label.

Label = "an unplanned failure happens within the next PREDICTION_HORIZON_H hours".
Rows while the machine is down are excluded (nothing to predict there).
"""
from __future__ import annotations

import pandas as pd

from . import config
from .db import query

FEATURES = [
    "vib_mean_1h", "vib_max_1h", "vib_std_1h",
    "temp_mean_1h", "temp_max_1h",
    "rpm_mean_1h", "rpm_std_1h",
    "vib_mean_6h", "temp_mean_6h", "rpm_std_6h",
    "vib_mean_24h", "temp_mean_24h",
    "vib_trend_24h", "temp_trend_24h",
    "vib_ratio_baseline", "temp_delta_baseline",
    "hours_since_maintenance",
]

FEATURE_LABELS = {
    "vib_mean_1h": "vibration (1h mean)", "vib_max_1h": "vibration (1h peak)", "vib_std_1h": "vibration variability (1h)",
    "temp_mean_1h": "temperature (1h mean)", "temp_max_1h": "temperature (1h peak)",
    "rpm_mean_1h": "RPM (1h mean)", "rpm_std_1h": "RPM variability (1h)",
    "vib_mean_6h": "vibration (6h mean)", "temp_mean_6h": "temperature (6h mean)", "rpm_std_6h": "RPM variability (6h)",
    "vib_mean_24h": "vibration (24h mean)", "temp_mean_24h": "temperature (24h mean)",
    "vib_trend_24h": "vibration trend vs 24h ago", "temp_trend_24h": "temperature trend vs 24h ago",
    "vib_ratio_baseline": "vibration vs asset baseline", "temp_delta_baseline": "temperature above baseline",
    "hours_since_maintenance": "hours since last maintenance",
}


def load_sensor_frame(conn) -> pd.DataFrame:
    sr = query(conn, "SELECT ts, asset_id, vibration_mm_s, temperature_c, rpm, machine_status "
                     "FROM sensor_readings ORDER BY asset_id, ts")
    sr["ts"] = pd.to_datetime(sr["ts"])
    return sr


def build_features(conn, sensor: pd.DataFrame | None = None) -> pd.DataFrame:
    sr = sensor if sensor is not None else load_sensor_frame(conn)
    running = sr[sr["machine_status"] == "RUNNING"]

    hourly = (running.groupby(["asset_id", pd.Grouper(key="ts", freq="1h")])
              .agg(vib_mean_1h=("vibration_mm_s", "mean"), vib_max_1h=("vibration_mm_s", "max"),
                   vib_std_1h=("vibration_mm_s", "std"),
                   temp_mean_1h=("temperature_c", "mean"), temp_max_1h=("temperature_c", "max"),
                   rpm_mean_1h=("rpm", "mean"), rpm_std_1h=("rpm", "std"), n_samples=("rpm", "size"))
              .reset_index().sort_values(["asset_id", "ts"]).reset_index(drop=True))

    g = hourly.groupby("asset_id")
    roll = lambda col, w: g[col].transform(lambda s: s.rolling(w, min_periods=1).mean())  # noqa: E731
    hourly["vib_mean_6h"] = roll("vib_mean_1h", 6)
    hourly["temp_mean_6h"] = roll("temp_mean_1h", 6)
    hourly["rpm_std_6h"] = roll("rpm_std_1h", 6)
    hourly["vib_mean_24h"] = roll("vib_mean_1h", 24)
    hourly["temp_mean_24h"] = roll("temp_mean_1h", 24)
    hourly["vib_trend_24h"] = hourly["vib_mean_6h"] - g["vib_mean_6h"].shift(24)
    hourly["temp_trend_24h"] = hourly["temp_mean_6h"] - g["temp_mean_6h"].shift(24)

    # Per-asset healthy baseline = median of the first 72 running hours
    base = g.head(72).groupby("asset_id")[["vib_mean_1h", "temp_mean_1h"]].median()
    base.columns = ["vib_base", "temp_base"]
    hourly = hourly.merge(base, left_on="asset_id", right_index=True, how="left")
    hourly["vib_ratio_baseline"] = hourly["vib_mean_1h"] / hourly["vib_base"]
    hourly["temp_delta_baseline"] = hourly["temp_mean_1h"] - hourly["temp_base"]

    # IT context: hours since the last closed work order (preventive or corrective)
    wos = query(conn, "SELECT asset_id, closed_ts FROM work_orders WHERE status='CLOSED' AND closed_ts IS NOT NULL")
    wos["closed_ts"] = pd.to_datetime(wos["closed_ts"])
    wos = wos.sort_values("closed_ts")
    hourly = pd.merge_asof(hourly.sort_values("ts"), wos, left_on="ts", right_on="closed_ts",
                           by="asset_id", direction="backward")
    hourly["hours_since_maintenance"] = ((hourly["ts"] - hourly["closed_ts"]).dt.total_seconds() / 3600).fillna(24 * 14)
    hourly = hourly.drop(columns=["closed_ts"])

    # Label from the ground-truth failure events
    fe = query(conn, "SELECT asset_id, failure_ts FROM failure_events")
    fe["failure_ts"] = pd.to_datetime(fe["failure_ts"])
    fe = fe.sort_values("failure_ts")
    hourly = pd.merge_asof(hourly.sort_values("ts"), fe, left_on="ts", right_on="failure_ts",
                           by="asset_id", direction="forward")
    hourly["hours_to_failure"] = (hourly["failure_ts"] - hourly["ts"]).dt.total_seconds() / 3600
    hourly["label"] = (hourly["hours_to_failure"].between(0, config.PREDICTION_HORIZON_H)).astype(int)
    hourly = hourly.drop(columns=["failure_ts"]).sort_values(["asset_id", "ts"]).reset_index(drop=True)
    return hourly
