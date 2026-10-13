"""Headroom screen: sample scenarios per fault family and label them (CPU only, no LLM).

Labels: benign (weak baseline stabilises and stays on spec), quality_gap (weak stabilises but
off-spec while the strong reference holds spec), rescuable (weak trips; an oracle single/two-step
plan on the perfect twin survives), unrescuable. One JSON per scenario -> resumable.
"""
import json, os, sys, time, zlib
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cod.scenarios import sample, FAMILIES
from cod.runner import run_episode
from cod.oracle import oracle_rescue

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "screen")

def out_dir(base_seed):
    return OUT if base_seed == 7 else OUT + f"_{base_seed}"

def job(arg):
    fam, idx, base_seed = arg
    path = os.path.join(out_dir(base_seed), fam, f"{idx:04d}.json")
    if os.path.exists(path):
        return path
    rng = np.random.default_rng([base_seed, zlib.crc32(fam.encode()), idx])
    sc = sample(fam, rng, idx=idx)
    seed = 1000 + idx
    t0 = time.time()
    weak = run_episode(sc, seed, None, baseline="weak")
    strong = run_episode(sc, seed, None, baseline="strong")
    on = sc.first_onset_h
    lab, orc = None, None
    if weak["stabilized"]:
        lab = "quality_gap" if (weak["g_iae"] > 2.5 * max(strong["g_iae"], 30.0) and strong["stabilized"]) else "benign"
    else:
        orc = oracle_rescue(sc, seed, [on + 0.5, on + 1.5, on + 3.0], two_step=True)
        lab = "rescuable" if orc["rescued"] else "unrescuable"
    keep = lambda o: {k: o[k] for k in ("stabilized", "trip_time", "trip_msg", "cost", "production", "g_iae", "offspec_h", "min_margin")}
    rec = {"scenario": sc.as_dict(), "seed": seed, "label": lab, "weak": keep(weak), "strong": keep(strong),
           "oracle": orc, "secs": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(rec, f, default=str)
    os.replace(path + ".tmp", path)
    return path

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    fams = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FAMILIES)
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    jobs = [(f, i, base_seed) for i in range(n) for f in fams]
    par = int(os.environ.get("MEMGUARD_PARALLEL", "16"))
    t = time.time()
    with Pool(par, maxtasksperchild=20) as pool:
        for k, _ in enumerate(pool.imap_unordered(job, jobs)):
            if k % 40 == 0:
                print(f"{k}/{len(jobs)} {time.time()-t:.0f}s", flush=True)
    print("done", time.time() - t)
