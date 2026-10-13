"""RQ1 driver: build the arm list for the offline x online token grid and run it on the test suites.

Offline axis: playbook checkpoints of a synthesis run (v0, B25000 ... B400000, final).
Online axis: per-episode token bucket for the gated online deliberator and for the hybrid.
  playbook only      (f = 1)       : playbook:<ckpt>
  online only        (f = 0)       : onlineg:<model>@<tokens>
  hybrid             (0 < f < 1)   : hybrid:<ckpt>:<model>@<tokens>
Allocation f for a deployment of N episodes follows in analysis from the measured tokens:
  f = C_off / (C_off + sum_e C_on(e)).

  python experiments/grid.py --synth results/synth/main_qwen3-coder-30b_s1 --model qwen2.5:14b-instruct-q4_K_M \
         --ckpts v0,B25000,B100000,final --tokens 8000,16000,inf --suites test_id,test_ood --print
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(os.path.dirname(ROOT))


def arms(synth_dir: str, ckpts: list[str], model: str, tokens: list[str], playbook_only: bool = True,
         online_only: bool = True, hybrid: bool = True) -> list[str]:
    out = []
    paths = {c: os.path.join(synth_dir, f"playbook_{c}.py") for c in ckpts}
    missing = [p for p in paths.values() if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing checkpoints: {missing}")
    if playbook_only:
        out += [f"playbook:{paths[c]}" for c in ckpts]
    if online_only:
        out += [f"onlineg:{model}" + ("" if t == "inf" else f"@{t}") for t in tokens]
    if hybrid:
        out += [f"hybrid:{paths[c]}:{model}@{t}" for c in ckpts for t in tokens]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", required=True)
    ap.add_argument("--ckpts", default="v0,B25000,B100000,final")
    ap.add_argument("--model", default="qwen2.5:14b-instruct-q4_K_M")
    ap.add_argument("--tokens", default="8000,16000,inf")
    ap.add_argument("--suites", default="test_id,test_ood")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", default="", help="playbook|online|hybrid (comma list) to restrict")
    ap.add_argument("--print", action="store_true")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else {"playbook", "online", "hybrid"}
    synth = a.synth if os.path.isabs(a.synth) else os.path.join(os.path.dirname(ROOT), a.synth)
    lst = arms(synth, a.ckpts.split(","), a.model, a.tokens.split(","), "playbook" in only, "online" in only,
               "hybrid" in only)
    if a.print:
        print("\n".join(lst))
        return
    cmd = [sys.executable, os.path.join(ROOT, "experiments", "run_arms.py"), "--arms", ",".join(lst),
           "--suite", a.suites, "--seeds", "1", "--workers", str(a.workers)]
    print(" ".join(cmd[:4]), f"... ({len(lst)} arms)", flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ, PYTHONPATH=ROOT))


if __name__ == "__main__":
    main()
