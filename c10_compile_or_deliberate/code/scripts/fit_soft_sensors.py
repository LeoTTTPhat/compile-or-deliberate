"""Fit linear soft sensors (reactor temperature: cooling-water outlet temperature + low-pass-filtered
cooling-water valve; chosen among 5 designs by closed-loop stability, see scratch/tsoft.py)

Fit linear soft sensors (analytical redundancy) on fault-free / benign-disturbance runs.

Writes data/soft_sensors.json and data/nominal.json. Sensor-fault-free data only: soft sensors
must be built from what a plant historian would hold for normal operation.
"""
import json, os, sys, itertools
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cod.faults import Scenario, IdvEvent, MEAS_IDX as _MI, MV_IDX
MEAS_IDX = dict(_MI); MEAS_IDX.update({"mv:" + k: 41 + v for k, v in MV_IDX.items()}); MEAS_IDX.update({"mvf:" + k: 53 + v for k, v in MV_IDX.items()})
from cod.plant import Plant, DATA_DIR

TARGETS = {
    "reactor_pressure": ["separator_pressure", "stripper_pressure", "compressor_work", "recycle_flow"],
    "reactor_temperature": ["reactor_cooling_water_outlet_temperature_meas", "mvf:reactor_cooling_water_valve"],
    "separator_temperature": ["condenser_cooling_water_outlet_temperature", "reactor_temperature", "separator_pressure"],
    "stripper_pressure": ["separator_pressure", "reactor_pressure", "compressor_work"],
}

def gen(args):
    idv, mag, prod, base, seed = args[:5]
    sp_steps = args[5] if len(args) > 5 else ()
    shifts = ((1.0, (("production_rate", prod),)),) if prod else ()
    shifts = shifts + tuple(sp_steps)
    ev = (IdvEvent(idv, 2.0, mag),) if idv else ()
    sc = Scenario(f"ss_{idv}_{mag}_{prod}_{base}", "train", "", idv_events=ev, setpoint_shifts=shifts, horizon_h=24.0)
    p = Plant(sc, seed, baseline=base)
    rows = []
    while not p.done:
        p.step()
        if p.k % 10 == 0:
            rows.append(np.concatenate([p.obs, p.mv, p.mvf]))
    return np.array(rows), p.tripped

if __name__ == "__main__":
    idvs = [None, "idv_01", "idv_02", "idv_03", "idv_04", "idv_05", "idv_07", "idv_08", "idv_09", "idv_10", "idv_11", "idv_12", "idv_13", "idv_16", "idv_17", "idv_18", "idv_20"]
    jobs = []
    for i, idv in enumerate(idvs):
        for mag in ([1.0] if idv is None else [0.4, 1.0]):
            for prod in [None, 19.0, 25.0]:
                for base in ["strong", "weak"]:
                    jobs.append((idv, mag, prod, base, 100 + len(jobs)))
    # identification step tests (setpoint excitation), as a plant would run for soft-sensor design
    for k, (fld, vals) in enumerate([("reactor_temperature", [116.0, 118.0, 123.0, 126.0, 128.0]),
                                      ("reactor_pressure", [2550.0, 2620.0, 2780.0, 2850.0])]):
        for v in vals:
            for base in ["strong", "weak"]:
                for idv in [None, "idv_04", "idv_13"]:
                    steps = ((3.0, ((fld, v),)), (12.0, ((fld, (v + (120.4 if fld == "reactor_temperature" else 2705.0)) / 2),)))
                    jobs.append((idv, 0.6, None, base, 500 + len(jobs), steps))
    par = int(os.environ.get("MEMGUARD_PARALLEL", "16"))
    with Pool(par) as pool:
        out = pool.map(gen, jobs)
    X = [(j, r) for j, (r, tr) in zip(jobs, out)]
    # split by job index (held-out runs) for honest R2
    train = np.vstack([r[20:] for k, (j, r) in enumerate(X) if k % 4 != 0])
    test = np.vstack([r[20:] for k, (j, r) in enumerate(X) if k % 4 == 0])
    models = {}
    for tgt, ins in TARGETS.items():
        A = np.column_stack([train[:, MEAS_IDX[n]] for n in ins] + [np.ones(len(train))])
        y = train[:, MEAS_IDX[tgt]]
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        At = np.column_stack([test[:, MEAS_IDX[n]] for n in ins] + [np.ones(len(test))])
        yt = test[:, MEAS_IDX[tgt]]
        pred = At @ coef
        rmse = float(np.sqrt(np.mean((pred - yt) ** 2))); r2 = 1 - np.var(pred - yt) / np.var(yt)
        models[tgt] = {"inputs": ins, "coef": [float(c) for c in coef[:-1]], "intercept": float(coef[-1]),
                       "test_rmse": rmse, "test_r2": float(r2), "n_train": int(len(train)), "n_test": int(len(test))}
        print(f"{tgt}: R2={r2:.4f} RMSE={rmse:.3f}")
    os.makedirs(DATA_DIR, exist_ok=True)
    json.dump(models, open(os.path.join(DATA_DIR, "soft_sensors.json"), "w"), indent=1)
    # nominal features: weak baseline, no disturbance, 2-6 h average
    nom = [r for (j, r) in X if j[0] is None and j[2] is None and j[3] == "weak" and len(j) == 5][0]
    seg = nom[20:60].mean(axis=0)
    g = lambda n: float(seg[MEAS_IDX[n]])
    nominal = {"P": g("reactor_pressure"), "T": g("reactor_temperature"), "RL": g("reactor_level"), "SL": g("separator_level"),
               "XL": g("stripper_level"), "G": g("stripper_underflow_G_concentration"), "PROD": g("stripper_underflow"),
               "SP": g("separator_pressure"), "XP": g("stripper_pressure"), "ST": g("separator_temperature"), "XT": g("stripper_temperature"),
               "PURGE": g("purge_flow"), "FA": g("feed_A_flow"), "FD": g("feed_D_flow"), "FE": g("feed_E_flow"), "FAC": g("feed_AC_flow"),
               "RECY": g("recycle_flow"), "RFEED": g("reactor_feed_flow"), "CW_R": g("reactor_cooling_water_outlet_temperature_meas"),
               "CW_C": g("condenser_cooling_water_outlet_temperature"), "WORK": g("compressor_work"),
               "XA": g("reactor_feed_A_concentration"), "XC": g("reactor_feed_C_concentration")}
    nominal["yA"] = 100 * nominal["XA"] / (nominal["XA"] + nominal["XC"])
    nominal["dP_rs"] = nominal["P"] - nominal["SP"]
    json.dump(nominal, open(os.path.join(DATA_DIR, "nominal.json"), "w"), indent=1)
    print(json.dumps(nominal))
    print("trips in training runs:", sum(tr for _, tr in out), "/", len(out))
