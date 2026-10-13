"""Run one offline playbook synthesis (LLM or ablation) with token checkpoints.

  python experiments/synth_run.py --model qwen3-coder:30b --seed 1 --budget 320000 --tag main
  python experiments/synth_run.py --model qwen3-coder:30b --seed 1 --one-shot --tag oneshot
  python experiments/synth_run.py --model qwen3-coder:30b --seed 1 --no-falsify --tag nofals
  python experiments/synth_run.py --model qwen3-coder:30b --seed 1 --no-minimize --tag nomin
Output: results/synth/<tag>_<model>_s<seed>/playbook_<checkpoint>.py + meta_*.json + log.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.synth.evaluate import load_screen  # noqa: E402
from cod.synth.llm_synth import Synthesiser  # noqa: E402

SYNTH_FAMILIES = ("A", "S", "P", "Q", "SN")
# stratified training batch drawn from the seed-7 screen (never from the seed-101 test screen)
STRATA = {"A": {"benign": 12}, "S": {"rescuable": 16, "benign": 2}, "P": {"rescuable": 12, "benign": 4},
          "Q": {"quality_gap": 12}, "SN": {"rescuable": 14, "benign": 6}}


def training_batch() -> list[dict]:
    out = []
    for fam, want in STRATA.items():
        pool = load_screen([fam])
        for lab, n in want.items():
            out.extend([b for b in pool if b["label"] == lab][:n])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-coder:30b")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--budget", type=int, default=320_000)
    ap.add_argument("--checkpoints", default="40000,80000,160000,320000")
    ap.add_argument("--tag", default="main")
    ap.add_argument("--one-shot", action="store_true")
    ap.add_argument("--no-falsify", action="store_true")
    ap.add_argument("--no-minimize", action="store_true")
    ap.add_argument("--no-remedy", action="store_true", help="counterexample cards without counterfactual remedies")
    ap.add_argument("--tune-budget", type=int, default=24, help="CPU threshold/priority tuning per falsification round (0=off)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "16")))
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(ROOT), "results", "synth", f"{a.tag}_{a.model.replace(':', '-')}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "log.txt"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    batch = training_batch()
    json.dump(batch, open(os.path.join(out, "train_batch.json"), "w"))
    log(f"training batch: {len(batch)} scenarios; model {a.model}; seed {a.seed}; budget {a.budget}")
    syn = Synthesiser(out, batch, model=a.model, checkpoints=tuple(int(x) for x in a.checkpoints.split(",")),
                      max_tokens_total=a.budget, falsify=not a.no_falsify, minimize=not a.no_minimize,
                      seed=a.seed, workers=a.workers, one_shot=a.one_shot, remedy=not a.no_remedy,
                      tune_budget=a.tune_budget, log=log)
    best = syn.run()
    log(f"DONE best it={best.it} sr={best.sr:.3f} tokens={syn.tokens} calls={syn.calls}")


if __name__ == "__main__":
    main()
