"""Tests for the B4/B11 verifier-only search supervisor (LLM-free, CPU only).

Run (prints episode outcomes):
  cd <workspace> && PYTHONPATH=c10_compile_or_deliberate/code \
      .venv/bin/python -m pytest -s c10_compile_or_deliberate/code/tests/test_search.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import get_context

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cod.actions import NO_ACTION  # noqa: E402
from cod.plant import Plant  # noqa: E402
from cod.runner import run_episode  # noqa: E402
from cod.scenarios import ALHAZMI13, FRONTIER5  # noqa: E402
from cod.summary import features  # noqa: E402
from cod.supervisors.search import DEFAULTS, VerifierSearch, log_from_history, parse_spec  # noqa: E402

SC = {s.id: s for s in ALHAZMI13 + FRONTIER5}
EPISODES = [(sid, spec) for sid in ("S1_idv01_ac_ratio", "Q2_grade_to_65", "F1_press_sensor_drift")
            for spec in ("search:nominal", "search:perfect")] + [("F1_press_sensor_drift", "search:perfect@gate=0")]


def test_parse_spec():
    assert parse_spec("search") == ("nominal", DEFAULTS)
    fid, p = parse_spec("search:perfect@h=4,thr=30")
    assert fid == "perfect" and p["h"] == 4.0 and p["thr"] == 30.0 and p["surv"] == DEFAULTS["surv"]
    for bad in ("search:hypothesis", "search@foo=1", "fdd:x", "search@h", "search@h=0"):
        with pytest.raises(ValueError):
            parse_spec(bad)
    assert VerifierSearch("search:perfect").name.startswith("B11_")
    # run_arms splits --arms on commas, so '+' and ';' also separate options
    assert parse_spec("search:nominal@h=4+thr=30") == parse_spec("search:nominal@h=4;thr=30") == ("nominal", p)
    assert parse_spec("search@thr=1e+2")[1]["thr"] == 100.0


def test_exhausted_budget_holds_without_rollouts():
    sup = VerifierSearch("search@gate=0")
    p = Plant(SC["A1_nominal"], 1)
    sup.episode_start(p, SC["A1_nominal"], 1)
    p.run_until(1.0)
    hist = [{"t": 0.1 * i, "accepted": True, "action": {"type": "reconfigure", "components": ["safety_overrides"]}}
            for i in range(12)]
    assert len(log_from_history(hist).accepted) == 12 and log_from_history(hist).accepted[0][0] == pytest.approx(0.05)
    d = sup.decide(p, features(p), "heartbeat", hist)
    assert d.action == NO_ACTION and d.meta["rollouts"] == 0 and d.source == "search:none"


def test_quiet_plant_holds_without_rollouts():
    sup = VerifierSearch("search")
    p = Plant(SC["A1_nominal"], 1)
    sup.episode_start(p, SC["A1_nominal"], 1)
    p.run_until(1.0)
    d = sup.decide(p, features(p), "heartbeat", [])
    assert d.action == NO_ACTION and d.meta["rollouts"] == 0 and d.diagnosis == "unknown"


def test_decide_and_revise_on_perfect_twin():
    sc = SC["S1_idv01_ac_ratio"]
    sup = VerifierSearch("search:perfect")
    p = Plant(sc, 1)
    sup.episode_start(p, sc, 1)
    p.run_until(4.0)
    d = sup.decide(p, features(p), "new_alarm", [])
    assert d.action.type != "no_action" and d.meta["rollouts"] > 10 and d.meta["cpu_s"] > 0
    assert d.meta["criterion"] == "survival"
    d2 = sup.revise(p, features(p), d, ["shadow: test veto"], [])
    assert d2 is not None and d2.action.key() != d.action.key()
    assert d2.meta["cpu_s"] == d.meta["cpu_s"] and d2.meta["rank"] == 1
    print(f"\n  t=4.0 S1 perfect: first={d.action.as_dict()} ({d.meta['rollouts']} rollouts, "
          f"{d.meta['cpu_s']:.2f} cpu-s); revision={d2.action.as_dict()}")


def _episode(arg):
    sid, spec = arg
    sup = VerifierSearch(spec)
    out = run_episode(SC[sid], 1, sup)
    acts = [(round(r["t"], 2), r["action"], r["source"]) for r in out["decisions"] if r["accepted"]]
    searched = [r["meta"]["rollouts"] for r in out["decisions"] if r["meta"].get("rollouts")]
    return {"sid": sid, "spec": spec, "stabilized": out["stabilized"], "trip_time": out["trip_time"],
            "g_iae": round(out["g_iae"], 1), "calls": out["calls"], "n_accepted": out["n_accepted"],
            "cpu_s": round(sum(r["meta"].get("cpu_s", 0.0) for r in out["decisions"]), 1),
            "rollouts": sum(searched), "searched_calls": len(searched), "accepted": acts}


def test_episodes_both_fidelities():
    with get_context("spawn").Pool(min(6, len(EPISODES))) as pool:
        res = pool.map(_episode, EPISODES)
    print()
    for r in res:
        print(f"  {r['sid']:22s} {r['spec']:22s} stabilized={r['stabilized']!s:5s} trip={r['trip_time']} "
              f"g_iae={r['g_iae']} calls={r['calls']} searched={r['searched_calls']} rollouts={r['rollouts']} "
              f"cpu_s={r['cpu_s']} accepted={r['accepted']}")
    by = {(r["sid"], r["spec"]): r for r in res}
    assert by[("S1_idv01_ac_ratio", "search:perfect")]["stabilized"]
    assert all(r["searched_calls"] > 0 and r["cpu_s"] > 0 for r in res if r["sid"] != "F1_press_sensor_drift")
    ungated = by[("F1_press_sensor_drift", "search:perfect@gate=0")]
    assert ungated["searched_calls"] == ungated["calls"]
