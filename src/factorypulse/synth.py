"""Synthetic, referentially consistent IT + OT dataset.

OT side : 5-minute sensor readings (vibration, temperature, RPM) per asset.
IT side : ERP production orders per shift, maintenance work orders with technician notes.
Truth   : failure_events (used to label the training set and to validate predictions).

Failures are preceded by a physically plausible degradation ramp (24-48 h) so that a model
can actually learn to predict them, and every failure produces a corrective work order whose
free-text notes mention the real failure mode (the unstructured context used by root cause analysis).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

RNG = np.random.default_rng(config.SEED)


@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    name: str
    asset_type: str
    line_id: str
    vib: float   # baseline vibration mm/s
    temp: float  # baseline temperature C
    rpm: float   # baseline RPM
    ideal_cycle_sec: float
    criticality: str


ASSETS = [
    AssetSpec("CNC-001", "CNC Mill 1", "CNC_MILL", "LINE-A", 2.0, 55, 3000, 45, "HIGH"),
    AssetSpec("CNC-002", "CNC Mill 2", "CNC_MILL", "LINE-A", 2.2, 57, 3000, 45, "HIGH"),
    AssetSpec("PRS-001", "Hydraulic Press 1", "PRESS", "LINE-A", 3.0, 60, 120, 20, "MEDIUM"),
    AssetSpec("CNV-001", "Conveyor 1", "CONVEYOR", "LINE-A", 1.5, 40, 900, 6, "LOW"),
    AssetSpec("PMP-001", "Coolant Pump 1", "PUMP", "LINE-B", 2.5, 50, 1450, 10, "HIGH"),
    AssetSpec("CNC-003", "CNC Mill 3", "CNC_MILL", "LINE-B", 2.1, 56, 3000, 45, "HIGH"),
    AssetSpec("PRS-002", "Hydraulic Press 2", "PRESS", "LINE-B", 3.2, 62, 120, 20, "MEDIUM"),
    AssetSpec("CNV-002", "Conveyor 2", "CONVEYOR", "LINE-B", 1.6, 41, 900, 6, "LOW"),
]

FAILURE_MODES = {
    "BEARING_WEAR": dict(
        vib=3.5, temp=1.25, rpm=1.5,
        notes="Replaced drive-end bearing. Found spalling on inner race; vibration had been climbing for ~2 days before trip.",
        parts="Bearing 6309-2RS, grease"),
    "OVERHEATING": dict(
        vib=1.3, temp=1.8, rpm=1.2,
        notes="Motor tripped on thermal overload. Cooling fins clogged with swarf, coolant flow restricted. Cleaned and restored flow.",
        parts="Coolant filter, thermal relay"),
    "MISALIGNMENT": dict(
        vib=2.6, temp=1.15, rpm=3.0,
        notes="Coupling misalignment detected (laser alignment 0.4 mm offset). Realigned shaft, replaced worn coupling insert.",
        parts="Coupling insert, shims"),
    "LUBRICATION": dict(
        vib=2.0, temp=1.5, rpm=1.3,
        notes="Insufficient lubrication on spindle bearings, grease degraded. Flushed and re-greased; recommend shorter PM interval.",
        parts="Spindle grease cartridge"),
}

# Assets whose next failure lies just beyond the generated window: (hours after window end, failure mode)
PENDING_FAILURES = {"CNC-002": (5.0, "BEARING_WEAR"), "PMP-001": (14.0, "OVERHEATING")}

PRODUCTS = ["BRKT-7781", "HSG-2210", "SHFT-1045", "PLT-3302", "GEAR-5520"]
TECHS = ["A. Rao", "M. Fischer", "L. Chen", "S. Okafor"]
SAMPLES_PER_SHIFT = 8 * 60 // config.SAMPLE_MINUTES


def _timeline(end: pd.Timestamp) -> pd.DatetimeIndex:
    n = config.N_DAYS * 24 * 60 // config.SAMPLE_MINUTES
    return pd.date_range(end=end, periods=n, freq=f"{config.SAMPLE_MINUTES}min")


def _schedule_failures(ts: pd.DatetimeIndex) -> list[tuple[pd.Timestamp, str, int]]:
    """1-3 failures per asset, spread over the period, each followed by 2-8 h of downtime."""
    n_fail = int(RNG.integers(1, 4))
    start, end = ts[0] + pd.Timedelta(days=3), ts[-1] - pd.Timedelta(hours=12)
    picks: list[pd.Timestamp] = []
    for _ in range(50):
        if len(picks) == n_fail:
            break
        t = start + pd.Timedelta(seconds=float(RNG.uniform(0, (end - start).total_seconds())))
        t = t.floor(f"{config.SAMPLE_MINUTES}min")
        if all(abs((t - p).total_seconds()) > 5 * 86400 for p in picks):
            picks.append(t)
    out = []
    for t in sorted(picks):
        mode = str(RNG.choice(list(FAILURE_MODES)))
        downtime = int(RNG.integers(120, 480))
        out.append((t, mode, downtime))
    return out


def generate(end: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    end = (end or pd.Timestamp.now()).floor(f"{config.SAMPLE_MINUTES}min")
    ts = _timeline(end)
    n = len(ts)
    hours = ts.hour.values + ts.minute.values / 60
    diurnal = np.sin((hours - 9) / 24 * 2 * np.pi)

    assets_rows, sensor_frames, failures, wos, orders = [], [], [], [], []
    wo_seq = 1000
    pm_times = pd.date_range(start=ts[0] + pd.Timedelta(days=2), end=ts[-1], freq="14D")

    for a in ASSETS:
        assets_rows.append(dict(
            asset_id=a.asset_id, asset_name=a.name, asset_type=a.asset_type, line_id=a.line_id,
            plant_id="PLANT-01", criticality=a.criticality, ideal_cycle_sec=a.ideal_cycle_sec,
            install_date=str((ts[0] - pd.Timedelta(days=int(RNG.integers(400, 3000)))).date()),
            manufacturer=str(RNG.choice(["DMG MORI", "Schuler", "Siemens", "Grundfos", "Bosch Rexroth"]))))

        vib = a.vib * (1 + 0.06 * RNG.standard_normal(n)) + 0.1 * diurnal
        temp = a.temp + 2.5 * diurnal + 1.2 * RNG.standard_normal(n)
        rpm = a.rpm * (1 + 0.01 * RNG.standard_normal(n))
        status = np.full(n, "RUNNING", dtype=object)
        degrade = np.zeros(n)  # 0..1 severity, used by the ERP side to lower performance / raise scrap

        def apply_ramp(fi: int, g: dict) -> None:
            """Degradation ramp ending at sample index fi (fi may lie beyond the window end)."""
            ramp_n = int(RNG.integers(24, 48)) * 60 // config.SAMPLE_MINUTES
            lo, hi_r = max(0, fi - ramp_n), min(fi, n)
            frac = (np.linspace(0, 1, fi - lo) ** 2)[: hi_r - lo]
            vib[lo:hi_r] *= 1 + (g["vib"] - 1) * frac
            temp[lo:hi_r] += (g["temp"] - 1) * a.temp * frac
            rpm[lo:hi_r] += a.rpm * 0.01 * (g["rpm"] - 1) * frac * RNG.standard_normal(hi_r - lo)
            degrade[lo:hi_r] = np.maximum(degrade[lo:hi_r], frac)

        schedule = _schedule_failures(ts)
        if a.asset_id in PENDING_FAILURES:
            # A failure scheduled shortly *after* the window: its degradation is visible "now",
            # so the command center opens with something to predict. Not yet observed -> no downtime.
            hours_ahead, mode = PENDING_FAILURES[a.asset_id]
            schedule = [s for s in schedule if s[0] < ts[-1] - pd.Timedelta(days=3)]
            schedule.append((ts[-1] + pd.Timedelta(hours=hours_ahead), mode, int(RNG.integers(180, 420))))

        # Unplanned failures, each preceded by a degradation ramp
        for ft, mode, downtime in schedule:
            g = FAILURE_MODES[mode]
            observed = ft <= ts[-1]
            if not observed:
                fi = n + int((ft - ts[-1]) / pd.Timedelta(minutes=config.SAMPLE_MINUTES))
                apply_ramp(fi, g)
                failures.append(dict(asset_id=a.asset_id, failure_ts=ft, failure_mode=mode,
                                     downtime_minutes=downtime, restored_ts=pd.NaT, observed=0))
                continue
            fi = int(ts.get_indexer([ft])[0])
            apply_ramp(fi, g)
            hi = min(n, fi + downtime // config.SAMPLE_MINUTES)
            vib[fi:hi] = 0.05 * RNG.random(hi - fi)
            rpm[fi:hi] = 0
            temp[fi:hi] = a.temp - 10 + 5 * RNG.random(hi - fi)
            status[fi:hi] = "DOWN_UNPLANNED"
            restored = ts[min(hi, n - 1)]
            failures.append(dict(asset_id=a.asset_id, failure_ts=ft, failure_mode=mode,
                                 downtime_minutes=downtime, restored_ts=restored, observed=1))
            wo_seq += 1
            wos.append(dict(
                wo_id=f"WO-{wo_seq}", asset_id=a.asset_id, wo_type="CORRECTIVE", status="CLOSED", priority="P1",
                created_ts=ft, closed_ts=restored,
                title=f"Breakdown: {a.name} ({mode.replace('_', ' ').title()})",
                notes=g["notes"], parts=g["parts"], cost_eur=float(RNG.integers(800, 6000)),
                technician=str(RNG.choice(TECHS)), source="ERP_HISTORY", alert_id=None))

        # Planned maintenance windows (2 h every 14 days)
        for pt in pm_times:
            pi = int(ts.get_indexer([pt])[0])
            hi = min(n, pi + 120 // config.SAMPLE_MINUTES)
            rpm[pi:hi] = 0
            vib[pi:hi] = 0.05
            status[pi:hi] = "DOWN_PLANNED"
            wo_seq += 1
            wos.append(dict(
                wo_id=f"WO-{wo_seq}", asset_id=a.asset_id, wo_type="PREVENTIVE", status="CLOSED", priority="P3",
                created_ts=pt, closed_ts=ts[min(hi, n - 1)], title=f"PM: {a.name} 14-day inspection",
                notes="Lubrication, belt/coupling inspection and sensor check per PM plan. No abnormal findings.",
                parts="Grease", cost_eur=float(RNG.integers(100, 400)),
                technician=str(RNG.choice(TECHS)), source="ERP_HISTORY", alert_id=None))

        sensor_frames.append(pd.DataFrame(dict(
            ts=ts, asset_id=a.asset_id,
            vibration_mm_s=np.round(np.clip(vib, 0, None), 3),
            temperature_c=np.round(temp, 2),
            rpm=np.round(np.clip(rpm, 0, None), 1),
            machine_status=status)))

        # ERP production orders per 8 h shift, derived from the same timeline (referential consistency)
        sdf = pd.DataFrame(dict(ts=ts, status=status, degrade=degrade))
        shifted = sdf.ts - pd.Timedelta(hours=6)
        sdf["shift_date"] = shifted.dt.date
        sdf["shift"] = pd.cut(shifted.dt.hour, [-1, 7, 15, 23], labels=["S1", "S2", "S3"])
        for (d, s), grp in sdf.groupby(["shift_date", "shift"], observed=True):
            if len(grp) < SAMPLES_PER_SHIFT * 0.9:  # partial shift at the edges of the window
                continue
            planned = (len(grp) - int((grp.status == "DOWN_PLANNED").sum())) * config.SAMPLE_MINUTES
            unplanned = int((grp.status == "DOWN_UNPLANNED").sum()) * config.SAMPLE_MINUTES
            run = max(planned - unplanned, 0)
            perf = float(np.clip(RNG.normal(0.93, 0.03) - 0.25 * grp.degrade.mean(), 0.5, 1.0))
            total = int(run * 60 / a.ideal_cycle_sec * perf)
            scrap_rate = float(np.clip(RNG.normal(0.012, 0.004) + 0.06 * grp.degrade.mean(), 0, 0.2))
            scrap = int(total * scrap_rate)
            orders.append(dict(
                order_id=f"PO-{a.asset_id}-{d:%Y%m%d}-{s}", asset_id=a.asset_id, shift_date=str(d), shift=str(s),
                product_code=str(RNG.choice(PRODUCTS)), planned_minutes=planned,
                unplanned_downtime_minutes=unplanned, run_minutes=run,
                planned_qty=int(planned * 60 / a.ideal_cycle_sec), total_count=total,
                good_count=total - scrap, scrap_count=scrap))

    return dict(
        assets=pd.DataFrame(assets_rows),
        sensor_readings=pd.concat(sensor_frames, ignore_index=True),
        failure_events=pd.DataFrame(failures),
        work_orders=pd.DataFrame(wos),
        production_orders=pd.DataFrame(orders),
    )


class LiveSimulator:
    """Generates new 5-minute samples that continue each asset's *current* condition.

    The hidden condition (vib/temp level) starts from the last hour's running mean, so assets that are
    already degrading stay degraded, and reverts slowly (0.5 %/sample) to the healthy baseline. Noise is
    applied on top of the level, never fed back, so healthy assets do not random-walk into alerts.
    When `degrade_asset` is set, that asset follows a bearing-wear ramp that reaches full severity after
    120 samples (10 simulated hours), which lets a demo show an alert being raised in near real time.
    """

    def __init__(self, last: pd.DataFrame, degrade_asset: str | None = None):
        self.spec = {a.asset_id: a for a in ASSETS}
        self.degrade_asset = degrade_asset
        self.step = 0
        self.state: dict[str, dict] = {}
        for _, r in last.iterrows():
            a = self.spec[r.asset_id]
            self.state[r.asset_id] = dict(
                ts=pd.Timestamp(r.ts),
                vib=float(r.vib_level) if pd.notna(r.vib_level) else a.vib,
                temp=float(r.temp_level) if pd.notna(r.temp_level) else a.temp)

    def next(self) -> pd.DataFrame:
        self.step += 1
        g = FAILURE_MODES["BEARING_WEAR"]
        rows = []
        for asset_id, s in self.state.items():
            a = self.spec[asset_id]
            s["vib"] += 0.005 * (a.vib - s["vib"])
            s["temp"] += 0.005 * (a.temp - s["temp"])
            s["ts"] += pd.Timedelta(minutes=config.SAMPLE_MINUTES)
            vib, temp = s["vib"], s["temp"]
            if asset_id == self.degrade_asset:
                ramp = min(self.step / 120, 1.0) ** 2
                vib = max(vib, a.vib * (1 + (g["vib"] - 1) * ramp))
                temp = max(temp, a.temp + (g["temp"] - 1) * a.temp * ramp)
            rows.append(dict(
                ts=s["ts"], asset_id=asset_id,
                vibration_mm_s=round(max(0.0, vib * (1 + 0.06 * RNG.standard_normal())), 3),
                temperature_c=round(temp + 1.2 * RNG.standard_normal(), 2),
                rpm=round(float(a.rpm * (1 + 0.01 * RNG.standard_normal() * (1 + 2 * max(0.0, vib / a.vib - 1)))), 1),
                machine_status="RUNNING"))
        return pd.DataFrame(rows)
