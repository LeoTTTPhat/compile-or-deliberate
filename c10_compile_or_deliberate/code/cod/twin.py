"""Forked-twin rollouts: shadow validation and candidate scoring.

Fidelity:
  * "nominal"    - the twin starts from the current plant state but is blind to every fault
                   from now on (operator's simulator; Alhazmi's stated validator).
  * "hypothesis" - the twin injects a hypothesised fault scenario (e.g. a classifier's top-k).
  * "perfect"    - the twin keeps the true future faults (upper bound; oracle arms only).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .actions import Action, apply_now
from .faults import FaultLayers, Scenario
from .plant import EpisodeStats, Plant

MARGIN_FLOORS = {"reactor_pressure_high": 25.0, "reactor_temperature_high": 3.0}


@dataclass
class RolloutResult:
    tripped: bool
    trip_time: float | None
    min_margin: dict[str, float]
    cost_rate: float
    g_dev: float
    prod: float
    end_margin_P: float
    end_t: float

    def score(self, horizon: float) -> float:
        """Scalar for ranking candidates: survival time dominates, then safety margin, then economics."""
        surv = (self.trip_time if self.tripped else self.end_t)
        mp = min(self.min_margin.get("reactor_pressure_high", 0.0), 200.0)
        ml = min(min(self.min_margin.get(k, 100.0) for k in self.min_margin if "level" in k), 30.0)
        return 1000.0 * surv + 2.0 * mp + 3.0 * ml - 0.01 * self.cost_rate - 2.0 * self.g_dev


def make_twin(plant: Plant, fidelity: str = "nominal", hypothesis: Scenario | None = None, sim=None) -> Plant:
    if fidelity == "perfect":
        twin = plant.fork(faults_active=True, sim=sim)
    elif fidelity == "nominal":
        twin = plant.fork(faults_active=False, sim=sim)
    elif fidelity == "hypothesis":
        twin = plant.fork(faults_active=False, sim=sim)
        if hypothesis is not None:
            twin.layers = FaultLayers(hypothesis, active=True)
    else:
        raise ValueError(fidelity)
    twin.stats = EpisodeStats(min_margin=dict(plant.stats.min_margin), peak=dict(plant.stats.peak))
    twin.stats.min_margin = {k: np.inf for k in plant.stats.min_margin}
    twin.monitor = type(plant.monitor)()
    return twin


def rollout(plant: Plant, action: Action | None, horizon_h: float, fidelity: str = "nominal",
            hypothesis: Scenario | None = None, sim=None, then: Action | None = None) -> RolloutResult:
    twin = make_twin(plant, fidelity, hypothesis, sim)
    t0 = twin.t
    if action is not None and action.type not in ("no_action",):
        if action.type == "shutdown":
            return RolloutResult(False, None, {"reactor_pressure_high": 0.0}, 0.0, 0.0, 0.0, 0.0, t0)
        apply_now(twin, action)
    if then is not None:
        apply_now(twin, then)
    c0, p0, g0 = twin.stats.cost, twin.stats.production, twin.stats.g_iae
    twin.run_until(t0 + horizon_h)
    dur = max(twin.t - t0, 1e-6)
    mm = {k: (v if np.isfinite(v) else 1e9) for k, v in twin.stats.min_margin.items()}
    return RolloutResult(
        tripped=twin.tripped, trip_time=(twin.t - t0) if twin.tripped else None, min_margin=mm,
        cost_rate=(twin.stats.cost - c0) / dur, g_dev=(twin.stats.g_iae - g0) / dur,
        prod=(twin.stats.production - p0) / dur,
        end_margin_P=3000.0 - float(twin.true[6]), end_t=twin.t - t0,
    )


def shadow_check(plant: Plant, action: Action, horizon_h: float = 6.0, fidelity: str = "nominal",
                 hypotheses: list[Scenario] | None = None, trip_grace_h: float = 0.1,
                 margin_tol: float = 1.0) -> tuple[bool, str | None, dict]:
    """Alhazmi-style veto: reject if the action trips where holding does not, trips >0.1 h earlier,
    or pushes a fast safety margin below its floor and below holding. With hypotheses, the check must
    pass under EVERY hypothesis (trajectory-level, multi-hypothesis verification)."""
    if action.type in ("no_action", "shutdown"):
        return True, None, {}
    hyps = hypotheses if (fidelity == "hypothesis" and hypotheses) else [None]
    info = {}
    for i, h in enumerate(hyps):
        base = rollout(plant, None, horizon_h, fidelity, h)
        cand = rollout(plant, action, horizon_h, fidelity, h)
        info[f"h{i}"] = {"base_trip": base.trip_time, "cand_trip": cand.trip_time,
                         "base_mP": base.min_margin.get("reactor_pressure_high"),
                         "cand_mP": cand.min_margin.get("reactor_pressure_high")}
        if cand.tripped and not base.tripped:
            return False, f"shadow: predicted trip at +{cand.trip_time:.2f} h under the action (none without it)", info
        if cand.tripped and base.tripped and cand.trip_time < base.trip_time - trip_grace_h:
            return False, (f"shadow: predicted trip earlier with the action (+{cand.trip_time:.2f} h vs "
                           f"+{base.trip_time:.2f} h)"), info
        for name, floor in MARGIN_FLOORS.items():
            b, c = base.min_margin.get(name, np.inf), cand.min_margin.get(name, np.inf)
            if c < floor and c < b - margin_tol:
                return False, f"shadow: predicted {name} margin {c:.1f} below floor {floor:g} and below holding ({b:.1f})", info
    return True, None, info
