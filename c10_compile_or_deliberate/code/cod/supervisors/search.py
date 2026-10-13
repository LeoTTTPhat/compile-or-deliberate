"""B4 verifier-only search (and B11 oracle): zero-token model-predictive supervision.

At each invocation where something needs attention (an active alarm, a constraint whose
observed time-to-limit is below `ttl` hours, or the delivered grade more than `gerr` mol%
off the commanded one), the supervisor enumerates the action grammar instantiated at the
current setpoints (`oracle.candidate_actions`, filtered by the static validator), rolls
each candidate and holding out on a forked twin for `h` hours and ranks them with
`RolloutResult.score`. The best candidate is taken only if it beats holding by a margin:

  * survival: predicted time to trip extended by at least `surv` hours; or
  * score: score gain above `thr` without shortening predicted survival; or
  * quality: mean |%G - commanded grade| reduced by at least `q` mol% without shortening
    survival or lowering the score (`q=0` disables this criterion).

Otherwise it holds. It forms no diagnosis. With the `nominal` twin (B4) the search is blind
to every fault from now on, as an operator's simulator would be; the `perfect` twin (B11)
sees the true future faults and is an upper bound only. The gate reads OBSERVED signals, so
a lying transmitter can hide the need to search; `gate=0` searches at every invocation.

Spec grammar: ``search`` | ``search:nominal`` | ``search:perfect``, optionally followed by
``@key=value,...`` with keys h, thr, surv, q, ttl, gerr, subst, gate
(e.g. ``search:nominal@h=6,thr=50``, ``search:perfect@gate=0``). Options may also be separated
by ``+`` or ``;`` because run_arms splits ``--arms`` on commas (``search:nominal@h=6+thr=50``).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from ..actions import NO_ACTION, Action, ActionLog, parse_action, static_check
from ..oracle import candidate_actions
from ..runner import Decision
from ..twin import RolloutResult, rollout

FIDELITIES = ("nominal", "perfect")
DEFAULTS = {"h": 6.0, "thr": 50.0, "surv": 0.25, "q": 1.0, "ttl": 6.0, "gerr": 2.0, "subst": 1.0, "gate": 1.0}
TTL_KEYS = ("P", "T", "RL", "SL", "XL")


def parse_spec(spec: str) -> tuple[str, dict[str, float]]:
    """'search[:fidelity][@k=v,...]' -> (fidelity, params)."""
    head, _, opts = spec.strip().partition("@")
    kind, _, fid = head.partition(":")
    if kind != "search":
        raise ValueError(f"not a search spec: {spec!r}")
    fid = fid or "nominal"
    if fid not in FIDELITIES:
        raise ValueError(f"search fidelity must be one of {FIDELITIES}, got {fid!r}")
    params = dict(DEFAULTS)
    for kv in filter(None, (s.strip() for s in re.split(r"[,;+](?=\s*[A-Za-z_])", opts))):
        k, sep, v = kv.partition("=")
        if not sep or k not in DEFAULTS:
            raise ValueError(f"bad search option {kv!r}; keys {sorted(DEFAULTS)}")
        params[k] = float(v)
    if params["h"] <= 0:
        raise ValueError(f"search horizon h must be positive, got {params['h']:g}")
    return fid, params


def log_from_history(history: list[dict], apply_delay_h: float = 0.05) -> ActionLog:
    """Rebuild the runner's accepted-action log from the decision records (for static pre-filtering)."""
    log = ActionLog()
    for rec in history:
        if rec.get("accepted"):
            a, _ = parse_action(rec["action"])
            if a is not None:
                log.accepted.append((rec["t"] + apply_delay_h, a))
    return log


@dataclass
class Scored:
    action: Action
    result: RolloutResult
    score: float
    surv: float


def _surv(r: RolloutResult) -> float:
    return r.trip_time if r.tripped else r.end_t


class VerifierSearch:
    """Enumerate-and-roll-out supervisor; see the module docstring for the method."""

    def __init__(self, spec: str = "search", max_actions: int = 12, apply_delay_h: float = 0.05):
        self.spec = spec
        self.fidelity, self.p = parse_spec(spec)
        self.name = ("B11_" if self.fidelity == "perfect" else "B4_") + spec
        self.max_actions = max_actions
        self.apply_delay_h = apply_delay_h
        self.cpu_s = 0.0
        self.n_rollouts = 0

    def episode_start(self, plant, scenario, seed):
        self.ranked: list[Scored] = []
        self.hold: Scored | None = None
        self.last_meta: dict = {}
        self.cpu_s = 0.0
        self.n_rollouts = 0

    def gate(self, f: dict) -> str:
        """Reason to search now, or '' when the plant is quiet."""
        if not self.p["gate"]:
            return "ungated"
        if f["alarms"]:
            return "alarms " + ",".join(f["alarms"])
        near = [k for k in TTL_KEYS if f[k + "_ttl"] < self.p["ttl"]]
        if near:
            return "time-to-limit " + ",".join(f"{k} {f[k + '_ttl']:.1f} h" for k in near)
        if abs(f["G_cmd_err"]) > self.p["gerr"]:
            return f"grade off target {f['G_cmd_err']:+.1f}"
        return ""

    def qualifies(self, c: Scored, hold: Scored) -> str:
        """Criterion by which a candidate beats holding, or ''."""
        p = self.p
        if c.surv - hold.surv >= p["surv"]:
            return "survival"
        if c.surv < hold.surv - 1e-6:
            return ""
        if c.score - hold.score > p["thr"]:
            return "score"
        if p["q"] > 0 and hold.result.g_dev - c.result.g_dev >= p["q"] and c.score >= hold.score:
            return "quality"
        return ""

    def decide(self, plant, f, trigger, history):
        self.ranked, self.hold = [], None
        why = self.gate(f)
        if not why:
            self.last_meta = {"rollouts": 0, "cpu_s": 0.0}
            return Decision(NO_ACTION, "unknown", "quiet: no search", source="search:quiet", meta=dict(self.last_meta))
        log = log_from_history(history, self.apply_delay_h)
        cands = [a for a in candidate_actions(plant, include_subst=bool(self.p["subst"]))
                 if not static_check(a, plant, log, self.max_actions)]
        if not cands:
            self.last_meta = {"rollouts": 0, "cpu_s": 0.0, "gate": why}
            return Decision(NO_ACTION, "unknown", "no admissible candidate (budget or cooldowns)",
                            source="search:none", meta=dict(self.last_meta))
        h, fid = self.p["h"], self.fidelity
        c0, w0 = time.process_time(), time.time()
        r = rollout(plant, None, h, fid)
        hold = Scored(NO_ACTION, r, r.score(h), _surv(r))
        scored = []
        for a in cands:
            r = rollout(plant, a, h, fid)
            scored.append(Scored(a, r, r.score(h), _surv(r)))
        cpu, wall = time.process_time() - c0, time.time() - w0
        n = len(cands) + 1
        self.cpu_s += cpu
        self.n_rollouts += n
        scored.sort(key=lambda s: -s.score)
        self.hold = hold
        self.ranked = [s for s in scored if self.qualifies(s, hold)]
        self.last_meta = {
            "rollouts": n, "cpu_s": round(cpu, 3), "wall_s": round(wall, 3), "fidelity": fid, "gate": why,
            "hold_score": round(hold.score, 1), "hold_trip": hold.result.trip_time, "n_qualified": len(self.ranked),
            "top": [[s.action.as_dict(), round(s.score - hold.score, 1), s.result.trip_time] for s in scored[:3]],
        }
        return self._decision(0)

    def _decision(self, i: int, extra: dict | None = None) -> Decision:
        meta = dict(self.last_meta, **(extra or {}))
        hold = self.hold
        if i >= len(self.ranked):
            return Decision(NO_ACTION, "unknown", f"{self.fidelity} twin {self.p['h']:g} h: no candidate beats holding",
                            source="search:hold", meta=meta)
        c = self.ranked[i]
        crit = self.qualifies(c, hold)
        trip = (f"hold trips at +{hold.result.trip_time:.2f} h" if hold.result.tripped else "hold survives")
        why = (f"{self.fidelity} twin {self.p['h']:g} h: {crit}; score {c.score - hold.score:+.0f} over hold "
               f"({trip}; action {'trips at +%.2f h' % c.result.trip_time if c.result.tripped else 'survives'}; "
               f"%G dev {c.result.g_dev:.2f} vs {hold.result.g_dev:.2f})")
        meta.update(rank=i, criterion=crit, gain=round(c.score - hold.score, 1))
        return Decision(c.action, "unknown", why, source=f"search:{crit}", meta=meta)

    def revise(self, plant, feats, rejected, reasons, history):
        """Next-best qualifying candidate (no new rollouts); None when exhausted."""
        keys = [s.action.key() for s in self.ranked]
        k = rejected.action.key()
        if k not in keys or keys.index(k) + 1 >= len(self.ranked):
            return None
        return self._decision(keys.index(k) + 1, {"revision_of": rejected.action.as_dict()})

    def episode_end(self, plant, outcome):
        pass
