"""Run B5: genetic programming over the playbook DSL with the LLM synthesiser's evaluation and falsifier.

  python experiments/gp_run.py --seed 1 --evals 20000 --workers 6
  python experiments/gp_run.py --seed 1 --evals 20000 --no-falsify --tag gp_nofals
The budget counts playbook episode evaluations (training batch, falsification and re-scoring); checkpoint
E<n> is the incumbent found with at most n of them. Restarting the same command resumes from state.pkl.
Output: results/synth/<tag>_s<seed>/playbook_<checkpoint>.py + meta_*.json + log.txt + train_batch.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.synth.gp import GPSynthesiser  # noqa: E402
from experiments.synth_run import SYNTH_FAMILIES, training_batch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--evals", type=int, default=20_000, help="budget in playbook episode evaluations")
    ap.add_argument("--checkpoints", default="2500,5000,10000,20000")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--max-generations", type=int, default=None)
    ap.add_argument("--falsify-every", type=int, default=1, help="generations between falsification rounds")
    ap.add_argument("--falsify-n", type=int, default=48)
    ap.add_argument("--no-falsify", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tag", default="gp")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "6")))
    a = ap.parse_args()
    workers = max(1, min(a.workers, 12))  # shared machine: at most 12 simulation workers
    out = os.path.join(os.path.dirname(ROOT), "results", "synth", f"{a.tag}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    batch = training_batch()
    json.dump(batch, open(os.path.join(out, "train_batch.json"), "w"))
    log(f"training batch: {len(batch)} scenarios; GP seed {a.seed}; budget {a.evals} episode evaluations; "
        f"pop {a.pop}; workers {workers}")
    gp = GPSynthesiser(out, batch, budget_evals=a.evals, checkpoints=tuple(int(x) for x in a.checkpoints.split(",") if x),
                       pop_size=a.pop, elite=a.elite, max_generations=a.max_generations, falsify=not a.no_falsify,
                       falsify_every=a.falsify_every, falsify_n=a.falsify_n, falsify_families=SYNTH_FAMILIES,
                       seed=a.seed, workers=workers, resume=not a.no_resume, log=log)
    best = gp.run()
    log(f"DONE best gen={best.gen} sr={best.sr:.3f} score={best.score:.3f} rules={len(best.genome)} "
        f"evals={gp.evals} label_evals={gp.label_evals} generations={gp.gen}")


if __name__ == "__main__":
    main()
