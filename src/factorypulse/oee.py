"""Overall Equipment Effectiveness from ERP production orders.

OEE = Availability x Performance x Quality
  Availability = run time / planned production time
  Performance  = (ideal cycle time x total count) / run time
  Quality      = good count / total count
"""
from __future__ import annotations

import pandas as pd

from .db import query


def oee_from_counts(planned_minutes: float, run_minutes: float, total_count: int, good_count: int,
                    ideal_cycle_sec: float) -> dict:
    availability = run_minutes / planned_minutes if planned_minutes else 0.0
    performance = min((total_count * ideal_cycle_sec / 60) / run_minutes, 1.0) if run_minutes else 0.0
    quality = good_count / total_count if total_count else 0.0
    return dict(availability=availability, performance=performance, quality=quality,
                oee=availability * performance * quality)


def compute_oee(conn) -> pd.DataFrame:
    po = query(conn, "SELECT p.*, a.ideal_cycle_sec, a.line_id, a.plant_id FROM production_orders p "
                     "JOIN assets a USING (asset_id)")
    grp = (po.groupby(["plant_id", "line_id", "asset_id", "shift_date"])
           .agg(planned_minutes=("planned_minutes", "sum"), run_minutes=("run_minutes", "sum"),
                unplanned_downtime_minutes=("unplanned_downtime_minutes", "sum"),
                total_count=("total_count", "sum"), good_count=("good_count", "sum"),
                scrap_count=("scrap_count", "sum"), ideal_cycle_sec=("ideal_cycle_sec", "first"))
           .reset_index())
    parts = grp.apply(lambda r: oee_from_counts(r.planned_minutes, r.run_minutes, r.total_count, r.good_count,
                                                r.ideal_cycle_sec), axis=1, result_type="expand")
    out = pd.concat([grp, parts], axis=1).round(4)
    out.to_sql("oee_daily", conn, if_exists="replace", index=False)
    conn.commit()
    return out


def oee_summary(df: pd.DataFrame) -> dict:
    """Plant-level OEE computed from totals (not an average of ratios)."""
    if df.empty:
        return dict(availability=0, performance=0, quality=0, oee=0)
    ideal_minutes = (df["total_count"] * df["ideal_cycle_sec"] / 60).sum()
    availability = df["run_minutes"].sum() / df["planned_minutes"].sum()
    performance = min(ideal_minutes / df["run_minutes"].sum(), 1.0) if df["run_minutes"].sum() else 0
    quality = df["good_count"].sum() / df["total_count"].sum() if df["total_count"].sum() else 0
    return dict(availability=availability, performance=performance, quality=quality,
                oee=availability * performance * quality)
