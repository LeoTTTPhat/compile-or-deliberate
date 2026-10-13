"""Run supervisor arms on scenario suites; one JSON per (arm, scenario, seed) -> resumable.

Usage:
  python experiments/run_arms.py --arms weak,strong,B2a,B2b --suite alhazmi,frontier --seeds 1,2,3
  python experiments/run_arms.py --arms "online:qwen2.5:7b" --suite alhazmi --seeds 1 --workers 8
  python experiments/run_arms.py --arms "onstrong:B2b" --suite test_id   # supervisor over the strong configuration
Arm grammar:
  weak | strong | hold | B2a | B2b
  online:<model>[@<per_episode_tokens>]          online LLM (f=0), optional token bucket per episode
  onlineg:<model>[@<per_episode_tokens>]         gated online LLM (called only when abnormal)
  playbook:<path-to-.py>[@name]                   zero-token playbook
  hybrid:<playbook.py>:<model>@<per_episode_tokens>[:opts]   (see cod/supervisors/hybrid.py)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
from cod.faults import Scenario  # noqa: E402
from cod.runner import Hold, Policy, run_episode  # noqa: E402
from cod.scenarios import ALHAZMI13, FRONTIER5  # noqa: E402

RESULTS = os.path.join(os.path.dirname(ROOT), "results", "arms")


def build_arm(spec: str):
    """Return (supervisor or None, baseline)."""
    if spec in ("weak", "strong"):
        return None, spec
    if spec.startswith("onstrong:"):  # the same supervisor on top of the strong configuration m = (1, 1)
        sup, _ = build_arm(spec[len("onstrong:"):])
        return sup, "strong"
    if spec == "hold":
        return Hold(), "weak"
    if spec in ("B2a", "B2b"):
        from cod.supervisors.rules import AlarmResponseTable, ExpertTable
        return (AlarmResponseTable() if spec == "B2a" else ExpertTable()), "weak"
    if spec.startswith(("online:", "onlineg:")):
        from cod.supervisors.llm_online import OnlineLLMSupervisor
        gate = spec.startswith("onlineg:")
        body = spec.split(":", 1)[1]
        model, _, budget = body.partition("@")
        return OnlineLLMSupervisor(model, per_episode_tokens=float(budget) if budget else None, name=spec, gate=gate), "weak"
    if spec.startswith("playbook:"):
        from cod.playbook import PlaybookSupervisor
        body = spec[len("playbook:"):]
        path, _, name = body.partition("@")
        return PlaybookSupervisor(open(path).read(), name=name or spec), "weak"
    if spec.startswith("hybrid:"):
        from cod.supervisors.hybrid import build_hybrid
        return build_hybrid(spec), "weak"
    if spec.startswith("search:"):
        from cod.supervisors.search import VerifierSearch
        return VerifierSearch(spec), "weak"
    if spec.startswith("fdd:"):
        from cod.supervisors.fdd import FDDLookup
        return FDDLookup(spec), "weak"
    raise ValueError(spec)


def load_suite(name: str) -> list[tuple[Scenario, list[int] | None]]:
    """Returns (scenario, fixed seeds or None)."""
    if name == "alhazmi":
        return [(s, None) for s in ALHAZMI13]
    if name == "frontier":
        return [(s, None) for s in FRONTIER5]
    path = os.path.join(os.path.dirname(ROOT), "data", "suites", f"{name}.json")
    with open(path) as f:
        d = json.load(f)
    return [(Scenario.from_dict(x["scenario"]), x.get("seeds")) for x in d]


def arm_dir(spec: str) -> str:
    safe = spec.replace("/", "_").replace(":", "~").replace("@", "_at_")
    if len(safe) > 120:
        import hashlib
        safe = safe[:80] + "_" + hashlib.sha1(spec.encode()).hexdigest()[:10]
    return os.path.join(RESULTS, safe)


def job(args):
    spec, sc, seed, suite, policy_kw = args
    d = os.path.join(arm_dir(spec), suite)
    path = os.path.join(d, f"{sc.id}__s{seed}.json")
    if os.path.exists(path):
        return path, None
    sup, base = build_arm(spec)
    t0 = time.time()
    try:
        out = run_episode(sc, seed, sup, baseline=base, policy=Policy(**policy_kw), keep_trace=True)
    except Exception as e:  # noqa: BLE001
        import traceback
        return path, f"{type(e).__name__}: {e}\n{traceback.format_exc()[-800:]}"
    out["arm"] = spec
    out["suite"] = suite
    out["scenario_def"] = sc.as_dict()
    out["policy"] = policy_kw
    out["elapsed"] = time.time() - t0
    os.makedirs(d, exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(out, f, default=str)
    os.replace(path + ".tmp", path)
    return path, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "16")))
    ap.add_argument("--policy", default="{}", help="JSON overrides for runner.Policy")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    pol = json.loads(a.policy)
    jobs = []
    for suite in a.suite.split(","):
        for sc, fixed in load_suite(suite):
            for seed in (fixed or seeds):
                for spec in arms:
                    jobs.append((spec, sc, seed, suite, pol))
    if a.limit:
        jobs = jobs[: a.limit]
    t = time.time()
    errs = 0
    workers = min(a.workers, int(os.environ.get("MEMGUARD_PARALLEL", a.workers)))
    with Pool(workers, maxtasksperchild=50) as pool:
        for k, (path, err) in enumerate(pool.imap_unordered(job, jobs)):
            if err:
                errs += 1
                print("ERROR", path, err, flush=True)
            if k % 20 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t:.0f}s errors={errs}", flush=True)
    print(f"done {len(jobs)} jobs in {time.time() - t:.0f}s, errors={errs}", flush=True)


if __name__ == "__main__":
    main()
