"""Evaluate a playbook on a scenario batch (CPU, parallel) and build counterexample cards.

Episode success (the score used everywhere offline):
  * trip or orderly shutdown -> 0;
  * quality-gap scenarios additionally require the commanded grade to be delivered
    (%G IAE <= QUALITY_IAE_MAX over 48 h);
  * otherwise 1.
`score` = mean success - ACTION_PENALTY x mean accepted actions on scenarios labelled benign
(do-no-harm), so equal-success playbooks prefer restraint.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from multiprocessing import Pool

from ..faults import Scenario
from ..playbook import PlaybookError, PlaybookSupervisor, check_source
from ..runner import Policy, run_episode

QUALITY_IAE_MAX = 150.0  # mol%*h over 48 h (strong reference achieves ~70-100 on grade changes)
ACTION_PENALTY = 0.02


def success(out: dict, label: str | None) -> bool:
    if not out["stabilized"]:
        return False
    if label == "quality_gap" and out["g_iae"] > QUALITY_IAE_MAX:
        return False
    return True


_POOL = None


def _pool(workers: int):
    global _POOL
    if _POOL is None:
        _POOL = Pool(workers, maxtasksperchild=200)
    return _POOL


def _job(args):
    src, sc_d, seed, label, pol = args
    sc = Scenario.from_dict(sc_d)
    sup = PlaybookSupervisor(src)
    out = run_episode(sc, seed, sup, policy=Policy(**pol))
    return {
        "id": sc.id, "family": sc.family, "seed": seed, "label": label, "success": success(out, label),
        "stabilized": out["stabilized"], "trip_time": out["trip_time"], "trip_msg": out["trip_msg"],
        "shutdown": out["shutdown"], "g_iae": out["g_iae"], "cost": out["cost"], "n_accepted": out["n_accepted"],
        "errors": sup.errors, "error_msgs": sup.error_msgs, "decisions": out["decisions"], "scenario": sc_d,
    }


@dataclass
class EvalResult:
    score: float
    success_rate: float
    by_family: dict
    episodes: list
    n_errors: int

    def failures(self) -> list[dict]:
        return [e for e in self.episodes if not e["success"]]

    def error_summary(self, top: int = 5) -> str:
        agg: dict[str, list[int]] = {}
        for e in self.episodes:
            for m, c in e.get("error_msgs", {}).items():
                a = agg.setdefault(m, [0, 0])
                a[0] += c
                a[1] += 1
        if not agg:
            return ""
        items = sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]
        return "; ".join(f"{m} ({c} times in {n} episodes)" for m, (c, n) in items)

    def table(self) -> str:
        return ", ".join(f"{k}: {v['ok']}/{v['n']}" for k, v in sorted(self.by_family.items()))


def evaluate(src: str, batch: list[dict], workers: int = 16, policy: dict | None = None) -> EvalResult:
    """batch: list of {"scenario": dict, "seed": int, "label": str}."""
    check_source(src)
    pol = policy or {}
    jobs = [(src, b["scenario"], b["seed"], b.get("label"), pol) for b in batch]
    eps = _pool(workers).map(_job, jobs, chunksize=1)
    fam: dict = {}
    for e in eps:
        d = fam.setdefault(e["family"], {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += int(e["success"])
    sr = sum(e["success"] for e in eps) / max(len(eps), 1)
    benign_acts = [e["n_accepted"] for e in eps if e["label"] == "benign"]
    pen = ACTION_PENALTY * (sum(benign_acts) / len(benign_acts)) if benign_acts else 0.0
    return EvalResult(sr - pen, sr, fam, eps, sum(e["errors"] for e in eps))


def _idv_desc() -> dict:
    import tep_studio
    return {name: desc for name, desc in tep_studio.list_disturbances()}


_IDV_DESC = None


def describe_faults(sc: dict) -> str:
    global _IDV_DESC
    if _IDV_DESC is None:
        _IDV_DESC = _idv_desc()
    parts = []
    for e in sc.get("idv_events", []):
        end = f" until t={e['end_h']:.1f} h" if e.get("end_h") else ""
        parts.append(f"process disturbance {e['idv'].upper().replace('_0', '').replace('_', '')} "
                     f"({_IDV_DESC.get(e['idv'], '?')}) from t={e['start_h']:.2f} h{end}")
    for f in sc.get("sensor_faults", []):
        parts.append(f"{f['measurement']} transmitter {f['kind']} ({f['magnitude']:+.3g}{'/h' if f['kind'] == 'drift' else ''}) from t={f['start_h']:.2f} h")
    for f in sc.get("actuator_faults", []):
        kind = {"stuck": "stuck at its current position", "stuck_at": f"stuck at {f['magnitude']:.3g} %",
                "cap_max": f"cannot open beyond {f['magnitude']:.3g} %", "cap_min": f"cannot close below {f['magnitude']:.3g} %",
                "offset": f"offset {f['magnitude']:+.3g} %"}.get(f["kind"], f["kind"])
        parts.append(f"{f['mv'].replace('_', ' ')} {kind} from t={f['start_h']:.2f} h")
    for t, ch in sc.get("setpoint_shifts", []):
        parts.append("operator sets " + ", ".join(f"{k}={v:.3g}" for k, v in ch) + f" at t={t:.2f} h")
    return "; ".join(parts) or "no fault"


def counterexample_card(e: dict, max_snaps: int = 4, remedy: dict | None = None, use_remedy: bool = False) -> str:
    """Compact (<~300 tokens) description of a failed episode for LLM repair."""
    if e["shutdown"]:
        outcome = "ORDERLY SHUTDOWN (counts as failure)"
    elif not e["stabilized"]:
        outcome = f"TRIP at t={e['trip_time']:.2f} h ({e['trip_msg'][:40].strip()})"
    else:
        outcome = f"stayed up but product grade not delivered (%G IAE {e['g_iae']:.0f} > {QUALITY_IAE_MAX:.0f})"
    acts = [d for d in e["decisions"] if d["action"].get("type") != "no_action"]
    act_s = "; ".join(f"t={d['t']:.2f} {d['source']} {json.dumps(d['action'])} {'accepted' if d['accepted'] else 'REJECTED: ' + (d['rejected_reasons'] or [''])[0][:60]}"
                      for d in acts[:5]) or "none (playbook never acted)"
    snaps = [d for d in e["decisions"] if "fx" in d]
    if len(snaps) > max_snaps:
        idx = sorted({0, len(snaps) // 3, 2 * len(snaps) // 3, len(snaps) - 1})
        snaps = [snaps[i] for i in idx]
    snap_s = "\n".join(f"  t={d['t']:.2f} trigger={d['trigger']} alarms={d['alarms']} " + json.dumps(d["fx"]) for d in snaps)
    rem = ""
    if use_remedy:
        rem = ("\nCounterfactual what-if on the twin: " + (f"applying {json.dumps(remedy['action'])} at t={remedy['t']:.2f} h "
               "(and nothing else) would have stabilised the plant." if remedy else
               "no single supervisory action at onset+0.5 h or onset+1.5 h stabilises the plant."))
    return (f"[{e['id']} | family {e['family']} | label {e['label']}] Injected (simulation): {describe_faults(e['scenario'])}.\n"
            f"Outcome: {outcome}. Playbook actions: {act_s}.\nFeature snapshots at invocations:\n{snap_s}{rem}")


def load_screen(families: list[str], labels: tuple[str, ...] | None = None, limit_per_family: int | None = None,
                root: str | None = None) -> list[dict]:
    import glob
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
                                "results", "screen")
    out = []
    for fam in families:
        files = sorted(glob.glob(os.path.join(root, fam, "*.json")))
        k = 0
        for p in files:
            r = json.load(open(p))
            if labels and r["label"] not in labels:
                continue
            out.append({"scenario": r["scenario"], "seed": r["seed"], "label": r["label"]})
            k += 1
            if limit_per_family and k >= limit_per_family:
                break
    return out


def _label_job(args):
    sc_d, seed = args
    from ..oracle import oracle_rescue
    sc = Scenario.from_dict(sc_d)
    w = run_episode(sc, seed, None)
    if w["stabilized"]:
        if w["g_iae"] > QUALITY_IAE_MAX:
            s = run_episode(sc, seed, None, baseline="strong")
            return "quality_gap" if (s["stabilized"] and s["g_iae"] <= QUALITY_IAE_MAX) else "benign_offspec"
        return "benign"
    on = sc.first_onset_h
    r = oracle_rescue(sc, seed, [on + 0.5, on + 1.5], two_step=False)
    return "rescuable" if r["rescued"] else "unrescuable"


def label_scenarios(batch: list[dict], workers: int = 16) -> list[str]:
    """Headroom labels (benign / quality_gap / rescuable / unrescuable) via weak run + cheap oracle."""
    return _pool(workers).map(_label_job, [(b["scenario"], b["seed"]) for b in batch], chunksize=1)


def _remedy_job(args):
    sc_d, seed = args
    from ..oracle import oracle_rescue
    sc = Scenario.from_dict(sc_d)
    on = sc.first_onset_h
    r = oracle_rescue(sc, seed, [on + 0.5, on + 1.5], two_step=False)
    if r["rescued"] and r["plan"] and r["plan"] != ["hold"]:
        return {"action": r["plan"][0], "t": r["t"]}
    return None


def counterfactual_remedies(eps: list[dict], workers: int = 16) -> list[dict | None]:
    """Offline what-if analysis on the twin: which single supervisory action, applied 0.5 h or 1.5 h after
    fault onset, would have stabilised each failed episode (None if no single action does)."""
    return _pool(workers).map(_remedy_job, [(e["scenario"], e["seed"]) for e in eps], chunksize=1)
