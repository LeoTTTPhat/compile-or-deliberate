"""Candidate action catalogue and oracle (perfect-twin) search.

`candidate_actions(plant)` enumerates the admissible single actions a supervisor could take
(the grammar instantiated at the current setpoints). `oracle_rescue` runs the closed loop with
NO supervisor up to a decision time, then tries every candidate (and short two-step plans) on
the PERFECT twin to the end of the horizon; it labels scenarios for the headroom screen and is
the B11 upper bound. Never used inside a deployable arm.
"""
from __future__ import annotations

from .actions import BOUNDS, PROCEDURES, Action
from .plant import Plant, soft_sensor_models
from .twin import rollout


def candidate_actions(plant: Plant, include_subst: bool = True) -> list[Action]:
    sp = plant.setpoints
    C: list[Action] = []
    armed = plant.armed
    if not armed["composition_control"]:
        C.append(Action("reconfigure", components=("composition_control",)))
    if not armed["safety_overrides"]:
        C.append(Action("reconfigure", components=("safety_overrides",)))
    if not armed["composition_control"] and not armed["safety_overrides"]:
        C.append(Action("reconfigure", components=("composition_control", "safety_overrides")))
    for name in sorted(PROCEDURES):
        C.append(Action("procedure", name=name))
    for fld, deltas in (("reactor_pressure", (-100.0, -50.0, 50.0)), ("reactor_temperature", (-3.0, 3.0)),
                        ("reactor_level", (-10.0, 10.0)), ("separator_level", (-10.0, 10.0)),
                        ("stripper_level", (-10.0, 10.0)), ("production_rate", (-4.0, -2.0, 2.0)),
                        ("pct_g", (-5.0, 5.0))):
        b = BOUNDS[fld]
        for d in deltas:
            v = min(max(sp[fld] + d, b.lo), b.hi)
            if abs(v - sp[fld]) > 1e-6:
                C.append(Action("adjust_setpoints", changes=((fld, round(v, 3)),)))
    if include_subst:
        for m in ("reactor_pressure", "reactor_temperature"):
            if m in soft_sensor_models() and m not in plant.substitutions:
                C.append(Action("substitute_instrument", measurement=m, method="soft"))
    return C


def oracle_rescue(scenario, seed: int, decision_times: list[float], two_step: bool = True,
                  baseline: str = "weak", top_k: int = 8) -> dict:
    """For each decision time, try every candidate on the perfect twin to the end of the horizon; if none
    rescues, try ordered pairs (applied at the same instant) among the top_k singles by survival time.
    Returns the earliest rescuing plan (by decision time) and statistics."""
    p = Plant(scenario, seed, baseline=baseline, history=False)
    res = {"rescued": False, "plan": None, "t": None, "n_tried": 0, "hold_trip": None, "best_single": None}
    hold = rollout(p, None, scenario.horizon_h, "perfect")
    res["hold_trip"] = hold.trip_time
    if not hold.tripped:
        res.update(rescued=True, plan=["hold"], t=0.0)
        return res
    for td in sorted(decision_times):
        p.run_until(td)
        if p.done:
            break
        rem = scenario.horizon_h - p.t
        scored = []
        for a in candidate_actions(p):
            r = rollout(p, a, rem, "perfect")
            res["n_tried"] += 1
            if not r.tripped:
                res.update(rescued=True, plan=[a.as_dict()], t=p.t)
                return res
            scored.append((r.trip_time or 0.0, a))
        scored.sort(key=lambda z: -z[0])
        if res["best_single"] is None or scored[0][0] + p.t > res["best_single"][0]:
            res["best_single"] = (scored[0][0] + p.t, scored[0][1].as_dict())
        if two_step:
            top = [a for _, a in scored[:top_k]]
            for a in top:
                for b in top:
                    if a is b or a.type == b.type == "adjust_setpoints" and dict(a.changes).keys() == dict(b.changes).keys():
                        continue
                    r = rollout(p, a, rem, "perfect", then=b)
                    res["n_tried"] += 1
                    if not r.tripped:
                        res.update(rescued=True, plan=[a.as_dict(), b.as_dict()], t=p.t)
                        return res
    return res
