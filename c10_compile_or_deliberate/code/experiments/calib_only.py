"""Zero-token ablation: the coder's initial playbook (v0) hardened by the CPU part of the synthesis loop only.

Starting from playbook_v0.py of a main synthesis run, the loop replays that run's falsification rounds (same fresh
scenarios and seeds, same oracle labelling, same batch growth rule) and CPU calibration after every round, but makes
no language-model call: there is no repair. It isolates how much of a playbook's success comes from the language
model's iterative repairs rather than from its first draft plus simulation-based calibration.
  python experiments/calib_only.py --seed 1 --rounds 12 --workers 6
Output: results/synth/calibonly_qwen3-coder-30b_s<seed>/playbook_{v0,final}.py + meta_*.json + log.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.synth.llm_synth import Synthesiser  # noqa: E402
from experiments.synth_run import training_batch  # noqa: E402

SYN = os.path.join(os.path.dirname(ROOT), "results", "synth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-coder:30b", help="coder whose v0 is calibrated (no call is made)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rounds", type=int, default=12, help="falsification rounds (main runs: s1 12, s2 8, s3 6)")
    ap.add_argument("--tune-budget", type=int, default=24)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "6")))
    a = ap.parse_args()
    src_dir = os.path.join(SYN, f"main_{a.model.replace(':', '-')}_s{a.seed}")
    out = os.path.join(SYN, f"calibonly_{a.model.replace(':', '-')}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    v0_meta = json.load(open(os.path.join(src_dir, "meta_v0.json")))
    v0 = open(os.path.join(src_dir, "playbook_v0.py")).read().split("\n", 2)[2]  # drop the two header lines
    batch = training_batch()
    log(f"calibration-only from {os.path.basename(src_dir)}/playbook_v0.py; batch {len(batch)}; rounds {a.rounds}")
    syn = Synthesiser(out, batch, model=a.model, checkpoints=(), max_tokens_total=0, seed=a.seed, workers=a.workers,
                      tune_budget=a.tune_budget, log=log)
    syn.tokens, syn.calls = v0_meta["tokens"], 0  # v0 was written once by the coder; no further LLM call is made
    c = syn._eval(v0, 1)
    syn._insert(c)
    log(f"[v0] sr={c.sr:.3f} score={c.score:.3f} {c.table}")
    syn._save("v0")
    for r in range(1, a.rounds + 1):
        syn._falsification_round()  # fresh scenarios, oracle labels, batch growth, then CPU calibration
        syn._save(f"R{r}")  # checkpoint after every round (R1 is comparable with the main run's B25000)
    syn._save("final")
    log(f"DONE sr={syn.pop[0].sr:.3f} episodes={syn.episodes} (no LLM calls after v0)")


if __name__ == "__main__":
    main()
