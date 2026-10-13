"""Tests for B5, genetic programming over the playbook DSL (LLM-free, CPU only).

Run:
  cd <workspace> && PYTHONPATH=c10_compile_or_deliberate/code \
      .venv/bin/python -m pytest -s c10_compile_or_deliberate/code/tests/test_gp.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cod.actions import BOUNDS, ActionLog, parse_action, static_check  # noqa: E402
from cod.playbook import PlaybookSupervisor, check_source, compile_playbook  # noqa: E402
from cod.plant import Plant  # noqa: E402
from cod.scenarios import ALHAZMI13, FRONTIER5  # noqa: E402
from cod.summary import features  # noqa: E402
from cod.synth.evaluate import evaluate, load_screen  # noqa: E402
from cod.synth.gp import (ACTIONS, DEFAULT_QUANTILES, NUM_FEATURES, Atom, GPSynthesiser, Rule, Variation,  # noqa: E402
                          compile_genome, genome_from_json, genome_json)

SC = {s.id: s for s in ALHAZMI13 + FRONTIER5}


@pytest.fixture(scope="module")
def feats():
    """Real feature dicts: a quiet plant and a pressure-sensor drift after onset."""
    out = []
    for sid, t in (("A1_nominal", 1.0), ("F1_press_sensor_drift", 5.0)):
        p = Plant(SC[sid], 1)
        p.run_until(t)
        out.append((p, features(p)))
    return out


def test_quantile_table_covers_features(feats):
    for k in NUM_FEATURES:
        if k in ("n_saturated", "n_alarms"):
            continue
        v = DEFAULT_QUANTILES[k]
        assert len(v) >= 3 and list(v) == sorted(v) and all(math.isfinite(x) for x in v), k
    assert set(Variation(random.Random(0)).features) == set(NUM_FEATURES)
    # every curated feature is a number at run time (time-to-limit may be +inf, never NaN)
    for _, f in feats:
        for k in NUM_FEATURES:
            assert not math.isnan(float(f[k])), k


def test_random_genomes_compile_and_act_admissibly(feats):
    var = Variation(random.Random(1))
    genomes = [var.genome() for _ in range(150)]
    for _ in range(300):
        a, b = var.rng.sample(genomes, 2)
        genomes.append(var.mutate(var.crossover(a, b)))
    n_act = 0
    for g in genomes:
        assert 1 <= len(g) <= 8 and all(1 <= len(r.atoms) <= 2 for r in g)
        assert genome_from_json(json.loads(json.dumps(genome_json(g)))) == g
        src = compile_genome(g, "test")
        check_source(src)
        fn = compile_playbook(src)
        for plant, f in feats:
            mem: dict = {}
            out = fn(dict(f, trigger="heartbeat"), mem)
            if out is None:
                continue
            n_act += 1
            act, err = parse_action(out["action"])
            assert act is not None, err
            # guards make every catalogue action statically admissible on a fresh episode
            assert static_check(act, plant, ActionLog()) == [], (out, src)
            for k, v in act.changes:
                assert BOUNDS[k].lo <= v <= BOUNDS[k].hi
            # the rule that fired is released at the revision call and its action is not re-proposed there
            again = fn(dict(f, trigger="revision"), mem)
            assert again is None or (again["rule"] != out["rule"] and again["action"] != out["action"])
            assert mem.get(out["rule"]) == 0
    assert n_act > 20


def test_rule_fires_once_per_episode(feats):
    g = (Rule((Atom("num", "P_avg", ">", 0.0),), "production_rate-2"),
         Rule((Atom("alarm", "PRESSURE_HIHI", "not"),), "arm_composition"))
    sup = PlaybookSupervisor(compile_genome(g))
    plant, f = feats[0]
    sup.episode_start(plant, None, 1)
    d1 = sup.decide(plant, f, "heartbeat", [])
    d2 = sup.decide(plant, f, "heartbeat", [])
    d3 = sup.decide(plant, f, "heartbeat", [])
    assert d1.source == "rule:R1" and d1.action.changes == (("production_rate", f["PROD_sp"] - 2.0),)
    assert d2.source == "rule:R2" and d2.action.components == ("composition_control",)
    assert d3.action.type == "no_action"
    # at the revision call, rules re-proposing the vetoed action are skipped (they stay available)
    g2 = (Rule((Atom("num", "P_avg", ">", 0.0),), "rate_cut_25"), Rule((Atom("num", "T_avg", ">", 0.0),), "rate_cut_25"),
          Rule((Atom("num", "RL_avg", ">", 0.0),), "arm_overrides"))
    fn = compile_playbook(compile_genome(g2))
    mem: dict = {}
    assert fn(dict(f, trigger="heartbeat"), mem)["rule"] == "R1"
    assert fn(dict(f, trigger="revision"), mem)["rule"] == "R3"
    assert fn(dict(f, trigger="heartbeat"), mem)["rule"] == "R1" and not mem.get("R2")
    assert set(ACTIONS) >= {"arm_composition", "subst_P", "rate_cut_25", "reactor_pressure-100"}


def test_tiny_gp_run(tmp_path):
    """pop 6, 2 generations, 8 training scenarios, 4 workers: produces a valid, evaluable playbook."""
    batch = (load_screen(["A"], ("benign",), 2) + load_screen(["S"], ("rescuable",), 2)
             + load_screen(["Q"], ("quality_gap",), 2) + load_screen(["SN"], ("rescuable",), 2))
    assert len(batch) == 8
    out = str(tmp_path / "gp_s0")
    gp = GPSynthesiser(out, batch, budget_evals=400, checkpoints=(60, 400), pop_size=6, max_generations=2,
                       falsify_every=1, falsify_n=4, seed=0, workers=4, resume=False)
    best = gp.run()
    for h in gp.history:
        print(json.dumps(h))
    print(best.src)
    assert gp.gen == 2 and gp.evals <= 400 and len(gp.pop) == 6
    files = sorted(os.listdir(out))
    print(files)
    for name in ("playbook_g0.py", "playbook_E60.py", "playbook_final.py", "meta_final.json", "state.pkl"):
        assert name in files
    src = open(os.path.join(out, "playbook_final.py")).read()
    check_source(src)
    meta = json.load(open(os.path.join(out, "meta_final.json")))
    assert meta["evals"] == gp.evals and meta["n_rules"] == len(best.genome) and meta["batch_size"] == len(gp.batch)
    assert json.load(open(os.path.join(out, "meta_E60.json")))["evals"] <= 60
    # the cached-episode score equals a fresh evaluate() of the saved playbook on the final batch
    ev = evaluate(src, gp.batch, workers=4)
    print(f"evaluate(): score {ev.score:.4f} sr {ev.success_rate:.3f} ({ev.table()}); GP: {best.score:.4f}")
    assert ev.score == pytest.approx(best.score, abs=1e-12)
    # resuming from state.pkl continues where the run stopped
    gp2 = GPSynthesiser(out, batch, budget_evals=400, pop_size=6, max_generations=2, seed=0, workers=4,
                        log=lambda *_: None)
    assert gp2.gen == 2 and gp2.evals == gp.evals and gp2.pop[0].src == best.src
    # a budget that runs out mid-generation: evaluations never exceed it, remaining marks get the incumbent
    out3 = str(tmp_path / "gp_budget")
    gp3 = GPSynthesiser(out3, batch, budget_evals=70, checkpoints=(70,), pop_size=6, falsify=False, seed=0,
                        workers=4, resume=False)
    gp3.run()
    meta3 = json.load(open(os.path.join(out3, "meta_E70.json")))
    print(f"budget run: evals {gp3.evals}, generations {gp3.gen}, pop {len(gp3.pop)}, E70 evals {meta3['evals']}")
    assert gp3.exhausted and gp3.evals <= 70 and meta3["evals"] <= 70 and len(gp3.pop) == 6
