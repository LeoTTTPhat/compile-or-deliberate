"""Process-state summary: what an operator works from (measured signals only).

`features(plant)` returns a flat dict of floats/bools/strings consumed by playbooks,
GP rules, the novelty gate and the FDD classifier. `render(features, ...)` produces the
compact text card given to online LLM deliberators. Nothing here reads the true state,
the scenario, or the fault layers.
"""
from __future__ import annotations

import numpy as np

from .alarms import ALARM_NAMES
from .faults import MEAS_IDX, MV_NAMES
from .plant import Plant, soft_estimate, soft_sensor_models

# (feature key, measurement name, unit, setpoint field or None)
VARS = (
    ("P", "reactor_pressure", "kPa", "reactor_pressure"),
    ("T", "reactor_temperature", "degC", "reactor_temperature"),
    ("RL", "reactor_level", "%", "reactor_level"),
    ("SL", "separator_level", "%", "separator_level"),
    ("XL", "stripper_level", "%", "stripper_level"),
    ("G", "stripper_underflow_G_concentration", "mol%", "pct_g"),
    ("PROD", "stripper_underflow", "m3/h", "production_rate"),
    ("SP", "separator_pressure", "kPa", None),
    ("XP", "stripper_pressure", "kPa", None),
    ("ST", "separator_temperature", "degC", None),
    ("XT", "stripper_temperature", "degC", None),
    ("PURGE", "purge_flow", "kscmh", None),
    ("FA", "feed_A_flow", "kscmh", None),
    ("FD", "feed_D_flow", "kg/h", None),
    ("FE", "feed_E_flow", "kg/h", None),
    ("FAC", "feed_AC_flow", "kscmh", None),
    ("RECY", "recycle_flow", "kscmh", None),
    ("RFEED", "reactor_feed_flow", "kscmh", None),
    ("CW_R", "reactor_cooling_water_outlet_temperature_meas", "degC", None),
    ("CW_C", "condenser_cooling_water_outlet_temperature", "degC", None),
    ("WORK", "compressor_work", "kW", None),
    ("XA", "reactor_feed_A_concentration", "mol%", None),
    ("XC", "reactor_feed_C_concentration", "mol%", None),
)
LIMITS = {"P": (None, 3000.0), "T": (None, 175.0), "RL": (0.0, 100.0), "SL": (0.0, 100.0), "XL": (0.0, 100.0)}
MV_SHORT = {n: n.replace("_valve", "").replace("reactor_", "r_").replace("separator_", "s_").replace("stripper_", "x_")
            for n in MV_NAMES}


def _slope(t: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if len(t) < 5 or t[-1] - t[0] < 1e-6:
        return 0.0, 0.0
    tc = t - t.mean()
    den = float((tc * tc).sum())
    b = float((tc * (y - y.mean())).sum() / den)
    resid = y - (y.mean() + b * tc)
    se = float(np.sqrt(max((resid * resid).sum() / max(len(t) - 2, 1), 0.0) / den))
    return b, se


def features(plant: Plant, nominal: dict[str, float] | None = None) -> dict:
    """Derived quantities from the last 4 h of OBSERVED measurements."""
    T, Y, MV = plant.window(4.0)
    t = plant.t
    f: dict = {"t": t}
    sp = plant.setpoints
    nominal = nominal or NOMINAL
    m1 = T >= t - 1.0
    q = T >= t - 0.25
    for key, name, _u, spf in VARS:
        y = Y[:, MEAS_IDX[name]]
        f[key] = float(y[-1])
        f[key + "_avg"] = float(y[q].mean())
        b, se = _slope(T[m1], y[m1])
        f[key + "_slope"] = b
        f[key + "_slope_se"] = se
        f[key + "_dev_nom"] = float(y[q].mean() - nominal.get(key, y[0]))
        if spf is not None:
            f[key + "_sp"] = float(sp[spf])
            f[key + "_err"] = float(y[q].mean() - sp[spf])
    # constraint margins (observed) and signed time-to-limit
    for key, (lo, hi) in LIMITS.items():
        v, b, se = f[key + "_avg"], f[key + "_slope"], f[key + "_slope_se"]
        margins = []
        if hi is not None:
            margins.append((hi - v, b))
        if lo is not None:
            margins.append((v - lo, -b))
        mg, toward = min(margins, key=lambda z: z[0])
        f[key + "_margin"] = float(mg)
        f[key + "_ttl"] = float(mg / toward) if (toward > 0 and toward > se) else float("inf")
    # composition
    f["yA"] = 100.0 * f["XA"] / max(f["XA"] + f["XC"], 1e-9)
    f["yAC"] = f["XA"] + f["XC"]
    f["yA_sp"], f["yAC_sp"] = float(sp["ya"]), float(sp["yac"])
    f["yA_dev_nom"] = f["yA"] - nominal["yA"]
    # valves
    mv_last, mv_std = MV[-1], MV[m1].std(axis=0)
    for i, n in enumerate(MV_NAMES):
        s = MV_SHORT[n]
        f["mv_" + s] = float(mv_last[i])
        f["mv_" + s + "_std"] = float(mv_std[i])
    f["n_saturated"] = int(((mv_last <= 0.5) | (mv_last >= 99.5)).sum())
    f["saturated"] = [MV_SHORT[n] for i, n in enumerate(MV_NAMES) if mv_last[i] <= 0.5 or mv_last[i] >= 99.5]
    # consistency residuals (analytical redundancy)
    f["dP_rs"] = f["P_avg"] - f["SP_avg"] - nominal["dP_rs"]
    soft = soft_sensor_models()
    yq, mq = Y[q].mean(axis=0), MV[q].mean(axis=0)
    for key, name in (("P", "reactor_pressure"), ("T", "reactor_temperature")):
        if name in soft:
            est = soft_estimate(soft[name], yq, mq, plant.mvf)
            f[key + "_soft"] = est
            f[key + "_resid"] = f[key + "_avg"] - est
        else:
            f[key + "_soft"], f[key + "_resid"] = float("nan"), 0.0
    # alarms
    f["alarms"] = sorted(plant.monitor.active)
    f["alarm_ages"] = {a: t - t0 for a, t0 in plant.monitor.active.items()}
    for a in ALARM_NAMES:
        f["A_" + a] = a in plant.monitor.active
    f["n_alarms"] = len(plant.monitor.active)
    f["armed_composition"] = bool(plant.ctrl.enable_composition)
    f["armed_overrides"] = bool(plant.ctrl.enable_overrides)
    f["substituted"] = sorted(plant.substitutions)
    f["cmd_pct_g"], f["cmd_production"] = plant.cmd["pct_g"], plant.cmd["production_rate"]
    f["G_cmd_err"] = f["G_avg"] - plant.cmd["pct_g"]
    f["PROD_cmd_err"] = f["PROD_avg"] - plant.cmd["production_rate"]
    return f


# Mode-1 nominal values of the derived features (measured on the noiseless-ish nominal run;
# refreshed by scripts/fit_soft_sensors.py into data/nominal.json when available).
NOMINAL = {"P": 2705.0, "T": 120.4, "RL": 75.0, "SL": 50.0, "XL": 50.0, "G": 53.72, "PROD": 22.95,
           "SP": 2633.7, "XP": 3102.2, "ST": 80.1, "XT": 65.7, "PURGE": 0.337, "FA": 0.251, "FD": 3664.0,
           "FE": 4509.0, "FAC": 9.35, "RECY": 26.9, "RFEED": 42.3, "CW_R": 94.6, "CW_C": 77.3, "WORK": 341.4,
           "XA": 32.19, "XC": 26.38, "yA": 54.95, "dP_rs": 71.3}


def _load_nominal():
    import json
    import os
    from .plant import DATA_DIR
    p = os.path.join(DATA_DIR, "nominal.json")
    if os.path.exists(p):
        with open(p) as fh:
            NOMINAL.update(json.load(fh))


_load_nominal()


def _fmt(v: float, nd: int = 4) -> str:
    if v == float("inf"):
        return "stable"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    return f"{v:.{nd}g}"


def render(f: dict, recent_actions: list[str] | None = None, budget: str = "") -> str:
    """Compact, token-bounded process-state card (units carried in the header)."""
    L = [f"t = {f['t']:.2f} h. Active alarms: " + (", ".join(f"{a} ({f['alarm_ages'][a]:.2f} h)" for a in f["alarms"]) or "none") + "."]
    L.append(f"Armed components: composition_control={f['armed_composition']}, safety_overrides={f['armed_overrides']}. "
             f"Instruments out of service: {', '.join(f['substituted']) or 'none'}.")
    L.append("Constraint margins (observed; trip at P 3000 kPa, T 175 degC, levels 0/100 %): " + "; ".join(
        f"{k} margin {_fmt(f[k + '_margin'])}, time-to-limit {_fmt(f[k + '_ttl'], 3)} h" for k in LIMITS) + ".")
    L.append("Variable | value | 15-min avg | slope per h (se) | setpoint | dev from nominal")
    for key, name, unit, spf in VARS:
        spv = _fmt(f[key + "_sp"]) if spf else "-"
        L.append(f"{name} [{unit}] | {_fmt(f[key])} | {_fmt(f[key + '_avg'])} | {_fmt(f[key + '_slope'], 3)} ({_fmt(f[key + '_slope_se'], 2)}) | {spv} | {_fmt(f[key + '_dev_nom'], 3)}")
    L.append(f"Reactor feed composition: yA {f['yA']:.2f} mol% (setpoint {f['yA_sp']:.2f}, nominal dev {f['yA_dev_nom']:+.2f}); yAC {f['yAC']:.2f} (setpoint {f['yAC_sp']:.2f}).")
    L.append(f"Operator targets: product grade %G {f['cmd_pct_g']:.2f} (delivered dev {f['G_cmd_err']:+.2f}); production {f['cmd_production']:.2f} m3/h (delivered dev {f['PROD_cmd_err']:+.2f}).")
    L.append("Valve positions % (1-h std): " + ", ".join(
        f"{MV_SHORT[n]} {f['mv_' + MV_SHORT[n]]:.1f} ({f['mv_' + MV_SHORT[n] + '_std']:.2f})" for n in MV_NAMES)
             + f". Saturated: {', '.join(f['saturated']) or 'none'}.")
    L.append(f"Consistency residuals: reactor-minus-separator pressure vs nominal {f['dP_rs']:+.1f} kPa; "
             f"reactor pressure vs soft sensor {f['P_resid']:+.1f} kPa; reactor temperature vs soft sensor {f['T_resid']:+.2f} degC.")
    if recent_actions:
        L.append("Your recent actions: " + "; ".join(recent_actions[-4:]) + ".")
    if budget:
        L.append(budget)
    return "\n".join(L)
