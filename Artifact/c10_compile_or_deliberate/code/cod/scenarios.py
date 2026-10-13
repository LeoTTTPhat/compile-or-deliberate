"""Scenario suites and parametric fault families.

* `ALHAZMI13`: the 13-scenario benchmark of Alhazmi (2026, arXiv:2608.19819), re-encoded from
  the published scenario library (safetep-core, suite v2.0): classes A (do-no-harm),
  S (safety gap), Q (quality gap).
* `FRONTIER5`: the "F" class found in safetep-core (sensor/actuator faults that defeat both
  baselines); not evaluated in the paper.
* `FAMILIES`: parametric samplers used for falsification (synthesis families) and for the
  held-out test suite (held-out families). `sample(family, rng, onset)` returns a Scenario
  whose `params` record the sampled parameters.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .faults import ActuatorFault, IdvEvent, Scenario, SensorFault


def _s(sid, fam, desc, gt, idv=(), shifts=(), sens=(), act=(), horizon=48.0):
    return Scenario(sid, fam, desc, gt=tuple(gt), idv_events=tuple(idv), sensor_faults=tuple(sens),
                    actuator_faults=tuple(act),
                    setpoint_shifts=tuple((t, tuple(sorted(ch.items()))) for t, ch in shifts), horizon_h=horizon)


ALHAZMI13 = [
    _s("A1_nominal", "A", "no fault", ["none"]),
    _s("A2_idv04_cooling_step", "A", "reactor cooling-water inlet temperature step (IDV4)", ["reactor_cooling_fault"], idv=[IdvEvent("idv_04", 2.0, 1.0)]),
    _s("A3_idv08_feed_comp", "A", "random feed composition variation (IDV8)", ["feed_composition_upset"], idv=[IdvEvent("idv_08", 2.0, 1.0)]),
    _s("A4_rate_up_28", "A", "production target raised to 28 m3/h", ["operating_point_change"], shifts=[(2.0, {"production_rate": 28.0})]),
    _s("A5_idv06_feed_loss", "A", "loss of A feed (IDV6)", ["feed_loss_or_header_pressure"], idv=[IdvEvent("idv_06", 2.0, 1.0)]),
    _s("S1_idv01_ac_ratio", "S", "A/C feed ratio disturbance (IDV1)", ["feed_composition_upset"], idv=[IdvEvent("idv_01", 2.0, 1.0)]),
    _s("S2_idv01_07", "S", "IDV1 + C header pressure loss (IDV7)", ["feed_composition_upset", "feed_loss_or_header_pressure"],
       idv=[IdvEvent("idv_01", 2.0, 1.0), IdvEvent("idv_07", 2.0, 1.0)]),
    _s("S3_idv01_rate30", "S", "IDV1 while pushed to 30 m3/h", ["feed_composition_upset", "operating_point_change"],
       idv=[IdvEvent("idv_01", 2.0, 1.0)], shifts=[(2.0, {"production_rate": 30.0})]),
    _s("S4_idv01_mild", "S", "IDV1 at severity 0.7", ["feed_composition_upset"], idv=[IdvEvent("idv_01", 2.0, 0.7)]),
    _s("S5_idv06_maxrate", "S", "loss of A feed at 26 m3/h", ["feed_loss_or_header_pressure", "operating_point_change"],
       idv=[IdvEvent("idv_06", 2.0, 1.0)], shifts=[(0.5, {"production_rate": 26.0})]),
    _s("Q1_grade_to_40", "Q", "grade target 54->40", ["operating_point_change"], shifts=[(2.0, {"pct_g": 40.0})]),
    _s("Q2_grade_to_65", "Q", "grade target 54->65", ["operating_point_change"], shifts=[(2.0, {"pct_g": 65.0})]),
    _s("Q3_grade_40_idv08", "Q", "grade 40 during IDV8", ["operating_point_change", "feed_composition_upset"],
       shifts=[(2.0, {"pct_g": 40.0})], idv=[IdvEvent("idv_08", 2.0, 1.0)]),
]

FRONTIER5 = [
    _s("F1_press_sensor_drift", "F", "reactor pressure sensor drifting -20 kPa/h", ["sensor_fault"],
       sens=[SensorFault("reactor_pressure", "drift", 2.0, -20.0)]),
    _s("F2_temp_sensor_drift", "F", "reactor temperature sensor drifting -2 degC/h", ["sensor_fault"],
       sens=[SensorFault("reactor_temperature", "drift", 2.0, -2.0)]),
    _s("F3_reactor_cw_stuck", "F", "reactor cooling-water valve stuck from t=2 h", ["valve_stiction_or_stuck", "reactor_cooling_fault"],
       act=[ActuatorFault("reactor_cooling_water_valve", "stuck", 2.0)]),
    _s("F4_press_drift_fast", "F", "reactor pressure sensor drifting -35 kPa/h", ["sensor_fault"],
       sens=[SensorFault("reactor_pressure", "drift", 2.0, -35.0)]),
    _s("F5_compound_misleading", "F", "temperature sensor drifting -2 degC/h and pressure sensor +40 kPa false-high",
       ["compound_fault", "sensor_fault"],
       sens=[SensorFault("reactor_temperature", "drift", 2.0, -2.0), SensorFault("reactor_pressure", "bias", 2.0, 40.0)]),
]


# ---------------------------------------------------------------- parametric families
def _u(rng, lo, hi):
    return float(rng.uniform(lo, hi))


# NOTE: kernel IDV channels are binary (magnitude < 0.5 = off, >= 0.5 = fully on; verified
# empirically), so families vary onset, duration, operating point and combinations instead.
def fam_S(rng, onset):  # composition imbalance (IDV1, optionally +IDV7/IDV2), production 19-29
    dur = float(rng.choice([np.inf, np.inf, _u(rng, 1.5, 6.0)]))
    mag = 1.0
    prod = _u(rng, 19.0, 29.0)
    idv = [IdvEvent("idv_01", onset, mag, None if not np.isfinite(dur) else onset + dur)]
    gt = ["feed_composition_upset"]
    extra = rng.choice(["none", "idv_07", "idv_02"], p=[0.6, 0.25, 0.15])
    if extra != "none":
        idv.append(IdvEvent(str(extra), onset + _u(rng, 0, 2), 1.0))
        if extra == "idv_07":
            gt.append("feed_loss_or_header_pressure")
    shifts = [(0.5, {"production_rate": prod})] if abs(prod - 22.95) > 0.5 else []
    if prod > 24:
        gt.append("operating_point_change")
    return dict(idv=idv, shifts=shifts, gt=gt, params={"dur": dur, "prod": prod, "extra": str(extra)})


def fam_P(rng, onset):  # loss of A feed (IDV6) or partial A-feed valve cap at high production
    prod = _u(rng, 24.0, 30.0)
    if rng.random() < 0.6:
        idv, act, kind = [IdvEvent("idv_06", onset, 1.0)], [], "idv6"
    else:
        cap = _u(rng, 5.0, 20.0)
        idv, act, kind = [], [ActuatorFault("a_feed_valve", "cap_max", onset, cap)], f"a_cap{cap:.0f}"
    return dict(idv=idv, act=act, shifts=[(0.5, {"production_rate": prod})],
                gt=["feed_loss_or_header_pressure", "operating_point_change"], params={"prod": prod, "kind": kind})


def fam_Q(rng, onset):  # grade change, optionally with IDV8
    tgt = float(rng.choice([_u(rng, 40.0, 47.0), _u(rng, 60.0, 65.0)]))
    idv = [IdvEvent("idv_08", onset, 1.0)] if rng.random() < 0.3 else []
    gt = ["operating_point_change"] + (["feed_composition_upset"] if idv else [])
    return dict(idv=idv, shifts=[(onset, {"pct_g": tgt})], gt=gt, params={"grade": tgt, "idv8": bool(idv)})


def fam_SN(rng, onset):  # transmitter faults on regulated channels
    meas = str(rng.choice(["reactor_pressure", "reactor_temperature"], p=[0.6, 0.4]))
    kind = str(rng.choice(["drift", "bias", "freeze"], p=[0.5, 0.3, 0.2]))
    if meas == "reactor_pressure":
        mag = {"drift": _u(rng, -45.0, -12.0), "bias": _u(rng, -200.0, -80.0), "freeze": 0.0}[kind]
    else:
        mag = {"drift": _u(rng, -3.0, -1.0), "bias": _u(rng, -6.0, -3.0), "freeze": 0.0}[kind]
    return dict(sens=[SensorFault(meas, kind, onset, mag)], gt=["sensor_fault"],
                params={"meas": meas, "kind": kind, "mag": mag})


def fam_V(rng, onset):  # actuator faults
    choice = str(rng.choice(["rcw_stuck", "rcw_cap", "ccw_stuck", "purge_stuck", "d_feed_stuck", "e_feed_cap"]))
    mv, kind, mag = {
        "rcw_stuck": ("reactor_cooling_water_valve", "stuck", 0.0),
        "rcw_cap": ("reactor_cooling_water_valve", "cap_max", _u(rng, 25.0, 38.0)),
        "ccw_stuck": ("separator_cooling_water_valve", "stuck", 0.0),
        "purge_stuck": ("purge_valve", "stuck", 0.0),
        "d_feed_stuck": ("d_feed_valve", "stuck", 0.0),
        "e_feed_cap": ("e_feed_valve", "cap_max", _u(rng, 10.0, 18.0)),
    }[choice]
    prod = _u(rng, 21.0, 27.0)
    gt = ["valve_stiction_or_stuck"] + (["reactor_cooling_fault"] if "rcw" in choice else []) + \
         (["condenser_cooling_fault"] if "ccw" in choice else []) + (["feed_loss_or_header_pressure"] if "feed" in choice else [])
    shifts = [(0.5, {"production_rate": prod})] if abs(prod - 22.95) > 0.5 else []
    return dict(act=[ActuatorFault(mv, kind, onset, mag)], shifts=shifts, gt=gt,
                params={"valve": choice, "mag": mag, "prod": prod})


def fam_K(rng, onset):  # kinetics / heat-transfer / cooling degradation at elevated production
    idv = str(rng.choice(["idv_13", "idv_04", "idv_05", "idv_17", "idv_14"]))
    mag = 1.0
    prod = _u(rng, 24.0, 29.0)
    gt = {"idv_13": ["kinetics_drift"], "idv_04": ["reactor_cooling_fault"], "idv_05": ["condenser_cooling_fault"],
          "idv_17": ["heat_transfer_degradation"], "idv_14": ["valve_stiction_or_stuck", "reactor_cooling_fault"]}[idv]
    return dict(idv=[IdvEvent(idv, onset, mag)], shifts=[(0.5, {"production_rate": prod})], gt=gt + ["operating_point_change"],
                params={"idv": idv, "mag": mag, "prod": prod})


def fam_A(rng, onset):  # benign: should not need intervention
    idv = str(rng.choice(["idv_02", "idv_03", "idv_04", "idv_05", "idv_08", "idv_09", "idv_10", "idv_11", "idv_12", "idv_06"]))
    mag = 1.0
    prod = _u(rng, 19.0, 24.0)
    gt = {"idv_02": ["feed_composition_upset"], "idv_08": ["feed_composition_upset"], "idv_06": ["feed_loss_or_header_pressure"],
          "idv_04": ["reactor_cooling_fault"], "idv_11": ["reactor_cooling_fault"], "idv_05": ["condenser_cooling_fault"],
          "idv_12": ["condenser_cooling_fault"]}.get(idv, ["none"])
    return dict(idv=[IdvEvent(idv, onset, mag)], shifts=[(0.5, {"production_rate": prod})] if abs(prod - 22.95) > 0.5 else [],
                gt=gt + ["none"], params={"idv": idv, "mag": mag, "prod": prod})


def fam_C(rng, onset):  # compound: one process fault + one instrument/actuator fault (possibly misleading)
    a = fam_S(rng, onset) if rng.random() < 0.5 else fam_K(rng, onset)
    b = fam_SN(rng, onset + _u(rng, 0.0, 3.0)) if rng.random() < 0.6 else fam_V(rng, onset + _u(rng, 0.0, 3.0))
    out = {"idv": a.get("idv", []), "shifts": a.get("shifts", []), "sens": b.get("sens", []), "act": b.get("act", []),
           "gt": ["compound_fault"] + a["gt"] + b["gt"], "params": {"a": a["params"], "b": b["params"]}}
    return out


FAMILIES = {"A": fam_A, "S": fam_S, "P": fam_P, "Q": fam_Q, "SN": fam_SN, "V": fam_V, "K": fam_K, "C": fam_C}


def sample(family: str, rng: np.random.Generator, onset: float | None = None, idx: int = 0, horizon: float = 48.0) -> Scenario:
    onset = float(onset if onset is not None else rng.uniform(1.0, 6.0))
    d = FAMILIES[family](rng, onset)
    sc = _s(f"{family}{idx:04d}", family, _describe(family, d, onset), d["gt"], idv=d.get("idv", ()),
            shifts=d.get("shifts", ()), sens=d.get("sens", ()), act=d.get("act", ()), horizon=horizon)
    return dataclasses.replace(sc, params={**d.get("params", {}), "onset": onset})


def _describe(family: str, d: dict, onset: float) -> str:
    return f"family {family}, onset {onset:.2f} h, params {d.get('params')}"
