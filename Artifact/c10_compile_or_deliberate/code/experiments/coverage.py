"""RQ4: empirical scenario-coverage bound of a playbook and its validity under shift.

For a playbook P:
  (1) in-distribution bound: evaluate P on M fresh scenarios sampled from the SYNTHESIS
      families (A, S, P, Q, SN) with the synthesis sampler (base seed 303, never used before);
      exclude scenarios the oracle cannot rescue; report k failures / n and the one-sided
      95 % Clopper-Pearson upper bound on the failure probability under that distribution.
  (2) independent adversary: same families, a DIFFERENT sampler (wider onset 0.5-10 h,
      production 18-30 m3/h, harsher sensor faults) + a local search that perturbs the
      parameters of failures found -> empirical failure rate (is it inside the bound?).
  (3) family shift: failure rate on held-out families (V, C, K) - expected to exceed the bound.
Results: results/coverage/<name>.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.faults import ActuatorFault, IdvEvent, Scenario, SensorFault  # noqa: E402
from cod.scenarios import sample  # noqa: E402
from cod.synth.evaluate import evaluate, label_scenarios  # noqa: E402

SYN = ("A", "S", "P", "Q", "SN")
OUT = os.path.join(os.path.dirname(ROOT), "results", "coverage")


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import beta
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def shifted_sample(fam: str, rng, idx: int) -> Scenario:
    """Independent adversary distribution: wider/harsher than the synthesis sampler."""
    onset = float(rng.uniform(0.5, 10.0))
    sc = sample(fam, rng, onset=onset, idx=idx)
    prod = float(rng.uniform(18.0, 30.0))
    shifts = tuple((t, tuple((k, prod if k == "production_rate" else v) for k, v in ch)) for t, ch in sc.setpoint_shifts)
    if fam in ("S", "SN", "A") and not any(k == "production_rate" for _, ch in shifts for k, _ in ch):
        shifts = shifts + ((0.5, (("production_rate", prod),)),)
    sens = tuple(dataclasses.replace(f, magnitude=f.magnitude * float(rng.uniform(1.0, 1.6))) for f in sc.sensor_faults)
    return dataclasses.replace(sc, id=f"ADV{fam}{idx:04d}", setpoint_shifts=shifts, sensor_faults=sens,
                               params={**sc.params, "adv_prod": prod})


def cached_labels(batch: list[dict], key: str, workers: int) -> list[str]:
    """Headroom labels depend only on (scenario, seed): cache them across playbooks."""
    path = os.path.join(OUT, f"labels_{key}.json")
    if os.path.exists(path):
        lab = json.load(open(path))
        if len(lab) == len(batch):
            return lab
    lab = label_scenarios(batch, workers)
    json.dump(lab, open(path, "w"))
    return lab


def run(src: str, name: str, m: int = 400, workers: int = 16, adv_m: int = 300, local_rounds: int = 2):
    os.makedirs(OUT, exist_ok=True)
    res = {"name": name}
    # (1) in-distribution
    rng = np.random.default_rng(303)
    batch = [{"scenario": sample(SYN[i % 5], rng, idx=90_000 + i).as_dict(), "seed": 30_000 + i} for i in range(m)]
    labels = cached_labels(batch, f"indist_{m}", workers)
    for b, lab in zip(batch, labels):
        b["label"] = lab
    keep = [b for b in batch if b["label"] in ("benign", "quality_gap", "rescuable")]
    ev = evaluate(src, keep, workers=workers)
    k = sum(not e["success"] for e in ev.episodes)
    res["in_dist"] = {"n": len(keep), "fail": k, "rate": k / max(len(keep), 1), "cp_upper95": clopper_pearson_upper(k, len(keep)),
                      "by_family": ev.by_family, "labels": {lab: labels.count(lab) for lab in set(labels)}}
    # (2) independent adversary: shifted sampler + local perturbation of failures
    rng2 = np.random.default_rng(404)
    adv = [{"scenario": shifted_sample(SYN[i % 5], rng2, 95_000 + i).as_dict(), "seed": 40_000 + i} for i in range(adv_m)]
    for r in range(local_rounds + 1):
        labs = cached_labels(adv, f"adv_{adv_m}", workers) if r == 0 else label_scenarios(adv, workers)
        for b, lab in zip(adv, labs):
            b["label"] = lab
        kept = [b for b in adv if b["label"] in ("benign", "quality_gap", "rescuable")]
        eva = evaluate(src, kept, workers=workers)
        fails = [e for e in eva.episodes if not e["success"]]
        res[f"adversary_round{r}"] = {"n": len(kept), "fail": len(fails), "rate": len(fails) / max(len(kept), 1)}
        if r == local_rounds or not fails:
            break
        # local search: perturb onset / production / magnitudes of each failure (4 neighbours each)
        new = []
        for j, e in enumerate(fails[:60]):
            sc = Scenario.from_dict(e["scenario"])
            for q in range(4):
                g = np.random.default_rng([505, r, j, q])
                d_on = float(g.normal(0, 0.5))
                new_sc = dataclasses.replace(
                    sc, id=f"{sc.id}_r{r}q{q}",
                    idv_events=tuple(dataclasses.replace(x, start_h=max(0.3, x.start_h + d_on)) for x in sc.idv_events),
                    sensor_faults=tuple(dataclasses.replace(x, start_h=max(0.3, x.start_h + d_on), magnitude=x.magnitude * float(g.uniform(0.8, 1.25))) for x in sc.sensor_faults),
                    actuator_faults=tuple(dataclasses.replace(x, start_h=max(0.3, x.start_h + d_on)) for x in sc.actuator_faults))
                new.append({"scenario": new_sc.as_dict(), "seed": 41_000 + 10 * j + q})
        adv = new
    # (3) family shift
    rng3 = np.random.default_rng(606)
    fs = [{"scenario": sample(("V", "C", "K")[i % 3], rng3, idx=97_000 + i).as_dict(), "seed": 42_000 + i} for i in range(150)]
    labs = cached_labels(fs, "famshift_150", workers)
    for b, lab in zip(fs, labs):
        b["label"] = lab
    kept = [b for b in fs if b["label"] in ("benign", "quality_gap", "rescuable")]
    evs = evaluate(src, kept, workers=workers)
    kf = sum(not e["success"] for e in evs.episodes)
    res["family_shift"] = {"n": len(kept), "fail": kf, "rate": kf / max(len(kept), 1), "by_family": evs.by_family}
    json.dump(res, open(os.path.join(OUT, f"{name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "name"}, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("playbook")
    ap.add_argument("--name", required=True)
    ap.add_argument("--m", type=int, default=300)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "16")))
    a = ap.parse_args()
    run(open(a.playbook).read(), a.name, a.m, a.workers)
