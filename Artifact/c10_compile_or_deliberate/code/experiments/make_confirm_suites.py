"""Fresh suites for seed selection and for confirming the held-out results of the frozen (LLM-free at run time) arms.

  confirm_id   the 300 anticipated-family scenarios of the coverage experiment (sampler seed 303; oracle-unrescued
               scenarios excluded), three noise seeds each
  confirm_ood  the novel-family scenarios of the coverage experiment (sampler seed 606; V, C, K; oracle-unrescued
               excluded), three noise seeds each
  select_id    150 further anticipated-family scenarios from a new sampler seed (707), one noise seed each, used only
               to choose among synthesis runs (disjoint from every suite above and from the test suites)
These scenarios were drawn after the method was frozen and were never used to develop it. The coverage experiment
evaluated four final playbooks on the first noise seed of confirm_id/confirm_ood; nothing was tuned on them.
  python experiments/make_confirm_suites.py
Writes data/suites/{confirm_id,confirm_ood,select_id}.json + .sha (content hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.scenarios import sample  # noqa: E402

SYN = ("A", "S", "P", "Q", "SN")
SUITES = os.path.join(os.path.dirname(ROOT), "data", "suites")
COV = os.path.join(os.path.dirname(ROOT), "results", "coverage")
KEEP = ("benign", "quality_gap", "rescuable")


def write(name: str, items: list[dict]):
    path = os.path.join(SUITES, f"{name}.json")
    blob = json.dumps(items, sort_keys=True).encode()
    with open(path, "w") as f:
        json.dump(items, f)
    with open(path + ".sha", "w") as f:
        f.write(hashlib.sha256(blob).hexdigest() + "\n")
    print(f"{name}: {len(items)} scenarios, seeds per scenario {len(items[0]['seeds'])}")


def main():
    # anticipated families: identical draws to experiments/coverage.py (in-distribution part, m = 300)
    rng = np.random.default_rng(303)
    batch = [(sample(SYN[i % 5], rng, idx=90_000 + i), 30_000 + i) for i in range(300)]
    labels = json.load(open(os.path.join(COV, "labels_indist_300.json")))
    write("confirm_id", [{"scenario": sc.as_dict(), "seeds": [s, s + 100_000, s + 200_000], "label": lab}
                         for (sc, s), lab in zip(batch, labels) if lab in KEEP])
    # novel families: identical draws to experiments/coverage.py (family-shift part)
    rng3 = np.random.default_rng(606)
    fs = [(sample(("V", "C", "K")[i % 3], rng3, idx=97_000 + i), 42_000 + i) for i in range(150)]
    labs = json.load(open(os.path.join(COV, "labels_famshift_150.json")))
    write("confirm_ood", [{"scenario": sc.as_dict(), "seeds": [s, s + 100_000, s + 200_000], "label": lab}
                          for (sc, s), lab in zip(fs, labs) if lab in KEEP])
    # selection sample: new sampler seed, unlabelled (unrescuable scenarios count as failures for every candidate)
    rng7 = np.random.default_rng(707)
    write("select_id", [{"scenario": sample(SYN[i % 5], rng7, idx=99_000 + i).as_dict(), "seeds": [70_000 + i],
                         "label": None} for i in range(150)])


if __name__ == "__main__":
    main()
