"""Feature dataset for the FDD classifier (B3) and the novelty gate.

Weak-baseline episodes with a recording supervisor that always holds and stores the full numeric
feature vector (`summary.features`, plus `fdd.augment`) at every invocation of the default runner
Policy, so rows are sampled exactly when a deployed supervisor would be asked.

A second, "active" pass over the fault families (S, P, Q, SN) replays the same scenario and seed but
lets the recorder take the lookup recovery of the TRUE class at the first invocation after onset +
settle time; it keeps only the rows after that action was accepted (earlier rows duplicate the
passive run). These cover the post-recovery regime a deployed FDD supervisor sees after acting
(armed trims, substituted transmitter), so it does not misread its own recovery as a new fault.

  synth   : synthesis families A,S,P,Q,SN sampled with `scenarios.sample` under a fresh base seed
            (202; the screen uses 7 and the test split 101) plus no-fault runs (family N, some with a
            production move) -> data/fdd_features.npz + data/fdd_features.json (feature names).
  heldout : a few V / C episodes from results/screen (seed-7 split; never screen_101), only for
            the novelty sanity check -> data/fdd_heldout_VC.npz.
Rows are labelled with `fdd.row_label` and carry time-since-onset. One JSON per episode under
data/fdd_rows/ makes runs resumable; a cached episode recorded under a different
`fdd.features_version()` is recomputed, and `assemble` refuses to mix versions.

Usage (from the repo root):
  .venv/bin/python tools/memguard.py --limit-gb 200 --parallel 6 --tag fdd_data -- \
      .venv/bin/python c10_compile_or_deliberate/code/experiments/fdd_data.py synth
  .venv/bin/python c10_compile_or_deliberate/code/experiments/fdd_data.py heldout --n 15
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.faults import Scenario  # noqa: E402
from cod.actions import NO_ACTION  # noqa: E402
from cod.fdd import (DATA_DIR, FEATURES_NPZ, augment, fault_class, fault_onset, features_version,  # noqa: E402
                     numeric_features, row_label)
from cod.runner import Decision, Policy, run_episode  # noqa: E402
from cod.scenarios import sample  # noqa: E402
from cod.supervisors.fdd import LOOKUP, done  # noqa: E402

ROWS_DIR = os.path.join(DATA_DIR, "fdd_rows")
SCREEN = os.path.join(os.path.dirname(ROOT), "results", "screen")
BASE_SEED = 202
COUNTS = {"A": 60, "S": 60, "P": 60, "Q": 60, "SN": 90, "N": 30}
ACTIVE_FAMILIES = ("S", "P", "Q", "SN")
SETTLE_H = 0.25


class Recorder:
    """Records the numeric feature vector at every invocation; holds, or (active) applies the
    true-class lookup recovery once the fault is on and keeps only the post-recovery rows."""
    name = "fdd_recorder"

    def __init__(self, active: bool = False):
        self.active = active

    def episode_start(self, plant, scenario, seed):
        self.sc = scenario
        self.rows: list[dict] = []
        self.meta: list[tuple[float, str]] = []

    def decide(self, plant, feats, trigger, history):
        if not self.active or any(h["accepted"] for h in history):
            self.rows.append(numeric_features(augment(feats, plant)))
            self.meta.append((plant.t, trigger))
        if self.active:
            lab = row_label(self.sc, plant.t)
            if lab != "none" and plant.t >= fault_onset(self.sc) + SETTLE_H and not done(LOOKUP[lab], feats):
                return Decision(LOOKUP[lab], lab, source="recorder:oracle_lookup")
        return Decision(NO_ACTION, "none", source="recorder")

    def revise(self, plant, feats, rejected, reasons, history):
        return None

    def episode_end(self, plant, outcome):
        pass


def no_fault(rng: np.random.Generator, idx: int) -> Scenario:
    """Fault-free run; half of them with an operator production move at 0.5 h (no onset)."""
    prod = float(rng.uniform(19.0, 26.0))
    shifts = ((0.5, (("production_rate", prod),)),) if idx % 2 else ()
    return Scenario(f"N{idx:04d}", "N", f"no fault, production {'nominal' if not shifts else f'{prod:.1f}'}",
                    gt=("none",), setpoint_shifts=shifts, params={"prod": prod if shifts else 22.95})


def record(sc: Scenario, seed: int, active: bool = False) -> dict:
    """One weak-baseline episode; held-out families (no FDD class) are labelled novel after onset."""
    rec = Recorder(active)
    out = run_episode(sc, seed, rec, policy=Policy())
    cls = fault_class(sc)
    on = float("inf") if sc.family == "N" else fault_onset(sc)
    lab = [row_label(sc, t) if cls is not None else ("novel" if t >= on else "none") for t, _ in rec.meta]
    return {"scenario": sc.as_dict(), "seed": seed, "family": sc.family, "ep_class": cls or "novel", "onset": on,
            "stabilized": out["stabilized"], "trip_time": out["trip_time"], "t": [m[0] for m in rec.meta],
            "trigger": [m[1] for m in rec.meta], "label": lab, "rows": rec.rows, "active": active,
            "n_accepted": out["n_accepted"], "fver": features_version()}


def _dump(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as fh:
        json.dump(rec, fh, default=lambda v: str(v))
    os.replace(path + ".tmp", path)


def _fresh(path: str) -> bool:
    """Cached episode exists and was recorded with the current feature extractor."""
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        return json.load(fh).get("fver") == features_version()


def synth_job(arg) -> str:
    fam, idx, base, active = arg
    path = os.path.join(ROWS_DIR, f"b{base}" + ("a" if active else ""), fam, f"{idx:04d}.json")
    if not _fresh(path):
        rng = np.random.default_rng([base, zlib.crc32(fam.encode()), idx])
        sc = no_fault(rng, idx) if fam == "N" else sample(fam, rng, idx=idx)
        _dump(path, record(sc, 20000 + 1000 * list(COUNTS).index(fam) + idx, active))
    return path


def heldout_job(src: str) -> str:
    fam = os.path.basename(os.path.dirname(src))
    path = os.path.join(ROWS_DIR, "screen7", fam, os.path.basename(src))
    if not _fresh(path):
        r = json.load(open(src))
        _dump(path, record(Scenario.from_dict(r["scenario"]), int(r["seed"])) | {"screen_label": r["label"]})
    return path


def assemble(paths: list[str], out: str) -> dict:
    """Stack per-episode JSONs into one npz (+ feature-name json). Raw values: inf is kept (see fdd.clean)."""
    recs = [json.load(open(p)) for p in sorted(paths)]
    fvers = sorted({str(r.get("fver")) for r in recs})
    if len(fvers) != 1:
        raise ValueError(f"episodes recorded with different feature versions {fvers}; re-record the stale ones")
    names: list[str] = []
    seen = set()
    for r in recs:
        for row in r["rows"]:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    names.append(k)
    keys = ("y", "episode", "scenario_id", "family", "ep_class", "t", "tso", "trigger", "seed", "stabilized", "active")
    X, cols = [], {k: [] for k in keys}
    for e, r in enumerate(recs):
        on = float(r["onset"])
        for row, t, trig, lab in zip(r["rows"], r["t"], r["trigger"], r["label"]):
            X.append([float(row.get(n, np.nan)) for n in names])
            for k, v in zip(keys, (lab, e, r["scenario"]["id"], r["family"], r["ep_class"], t,
                                   t - on if np.isfinite(on) else np.nan, trig, r["seed"], r["stabilized"],
                                   bool(r.get("active", False)))):
                cols[k].append(v)
    arrs = {k: np.array(v) for k, v in cols.items()}
    arrs["group"] = np.unique(arrs["scenario_id"], return_inverse=True)[1]  # scenario-level CV groups
    np.savez_compressed(out, X=np.array(X, dtype=float), **arrs)
    with open(out.replace(".npz", ".json"), "w") as fh:
        json.dump({"names": names, "fver": recs[0].get("fver"), "n_rows": len(X), "n_episodes": len(recs),
                   "families": sorted(set(arrs["family"].tolist()))}, fh, indent=1)
    return {"rows": len(X), "episodes": len(recs), "features": len(names),
            "classes": {c: int((arrs["y"] == c).sum()) for c in sorted(set(arrs["y"].tolist()))}}


def run(jobs, fn, workers: int) -> list[str]:
    t0 = time.time()
    paths = []
    with Pool(workers, maxtasksperchild=40) as pool:
        for k, p in enumerate(pool.imap_unordered(fn, jobs)):
            paths.append(p)
            if k % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["synth", "heldout"])
    ap.add_argument("--base-seed", type=int, default=BASE_SEED)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies the per-family episode counts")
    ap.add_argument("--families", default="V,C", help="heldout: families from results/screen")
    ap.add_argument("--n", type=int, default=15, help="heldout: episodes per family")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    assert a.base_seed not in (7, 101), "seed 7 / 101 are the screen and test splits"
    workers = min(a.workers, int(os.environ.get("MEMGUARD_PARALLEL", a.workers)), 6)
    if a.what == "synth":
        jobs = [(f, i, a.base_seed, act) for act in (False, True) for f, n in COUNTS.items()
                for i in range(max(1, int(round(n * a.scale)))) if not act or f in ACTIVE_FAMILIES]
        paths = run(jobs, synth_job, workers)
        out = a.out or FEATURES_NPZ
    else:
        fams = a.families.split(",")
        assert not set(fams) & {"A", "S", "P", "Q", "SN"}, "held-out families only"
        srcs = [p for f in fams for p in sorted(glob.glob(os.path.join(SCREEN, f, "*.json")))[: a.n]]
        paths = run(srcs, heldout_job, workers)
        out = a.out or os.path.join(DATA_DIR, "fdd_heldout_" + "".join(fams) + ".npz")
    print(json.dumps(assemble([p for p in paths if p], out), indent=1), "->", out)


if __name__ == "__main__":
    main()
