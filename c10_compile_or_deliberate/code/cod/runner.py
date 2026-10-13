"""Supervised episode runner: plant + regulatory layer + (optional) ASM supervisor.

Invocation policy follows Alhazmi (2026): on a new alarm, while alarms persist, and on a
periodic heartbeat, subject to a minimum inter-invocation interval and a per-episode call
budget. Every proposal goes through static checks and (optionally) a shadow rollout; a
rejected proposal gets one revision. Accepted actions land after a modelled latency.
All supervisors (rule tables, playbooks, LLMs, search) get identical invocation
opportunities, so arms differ only in decision content and cost.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .actions import NO_ACTION, Action, ActionLog, schedule_action, static_check
from .faults import Scenario
from .plant import Plant
from .summary import features
from .twin import shadow_check


@dataclass
class Policy:
    min_interval_h: float = 0.25  # after a NEW alarm
    hold_interval_h: float = 1.0  # re-invocation while alarms persist (Alhazmi: onset + periodic)
    periodic_h: float | None = 2.0
    max_calls: int = 12
    max_actions: int = 12
    apply_delay_h: float = 0.05
    shadow: bool = True
    shadow_fidelity: str = "nominal"
    shadow_horizon_h: float = 6.0
    allow_revision: bool = True


@dataclass
class Decision:
    action: Action
    diagnosis: str = "unknown"
    rationale: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_calls: int = 0
    gpu_s: float = 0.0  # summed LLM latency (wall) for this decision
    source: str = ""  # e.g. "playbook:rule_3", "llm", "fallback"
    meta: dict = field(default_factory=dict)


class Supervisor(Protocol):
    name: str

    def episode_start(self, plant: Plant, scenario: Scenario, seed: int) -> None: ...

    def decide(self, plant: Plant, feats: dict, trigger: str, history: list[dict]) -> Decision: ...

    def revise(self, plant: Plant, feats: dict, rejected: Decision, reasons: list[str],
               history: list[dict]) -> Decision | None: ...

    def episode_end(self, plant: Plant, outcome: dict) -> None: ...


def run_episode(scenario: Scenario, seed: int, supervisor: Supervisor | None = None, *, baseline: str = "weak",
                policy: Policy | None = None, keep_trace: bool = False, trace_every: int = 10) -> dict[str, Any]:
    policy = policy or Policy()
    wall0 = time.time()
    plant = Plant(scenario, seed, baseline=baseline)
    log = ActionLog()
    history: list[dict] = []
    calls = 0
    last_call = 0.0
    new_alarm = False
    trace = []
    if supervisor is not None:
        supervisor.episode_start(plant, scenario, seed)
    while not plant.done:
        plant.step()
        if keep_trace and plant.k % trace_every == 0:
            trace.append([plant.t] + [float(x) for x in plant.true[[6, 7, 8, 11, 14, 16, 39]]]
                         + [float(x) for x in plant.obs[[6, 8]]])
        if plant.new_alarms:
            new_alarm = True
        if supervisor is None or plant.done or calls >= policy.max_calls:
            continue
        since = plant.t - last_call
        if since < policy.min_interval_h - 1e-9:
            continue
        trigger = None
        if new_alarm:
            trigger = "new_alarm"
        elif plant.monitor.active and since >= policy.hold_interval_h - 1e-9:
            trigger = "alarm_hold"
        elif policy.periodic_h is not None and since >= policy.periodic_h - 1e-9:
            trigger = "heartbeat"
        if trigger is None:
            continue
        calls += 1
        last_call = plant.t
        new_alarm = False
        feats = features(plant)
        rec = _invoke(plant, supervisor, feats, trigger, history, log, policy)
        history.append(rec)
    out = plant.outcome()
    if supervisor is not None:
        supervisor.episode_end(plant, out)
    out.update(
        scenario=scenario.id, family=scenario.family, seed=seed, baseline=baseline,
        supervisor=getattr(supervisor, "name", "none"), calls=calls,
        n_accepted=len(log.accepted), decisions=history,
        prompt_tokens=sum(r["prompt_tokens"] for r in history),
        completion_tokens=sum(r["completion_tokens"] for r in history),
        llm_calls=sum(r["llm_calls"] for r in history),
        gpu_s=sum(r["gpu_s"] for r in history),
        wall_s=time.time() - wall0,
        diag_correct_any=any(r["diagnosis"] in scenario.gt for r in history) if history else None,
        diag_final=history[-1]["diagnosis"] if history else None,
        gt=list(scenario.gt),
    )
    if keep_trace:
        out["trace"] = trace
    return out


def _invoke(plant: Plant, sup: Supervisor, feats: dict, trigger: str, history: list[dict], log: ActionLog,
            policy: Policy) -> dict:
    t = plant.t
    d = sup.decide(plant, feats, trigger, history)
    ptok, ctok, calls, gpu = d.prompt_tokens, d.completion_tokens, d.llm_calls, d.gpu_s
    reasons = _check(plant, d.action, log, policy)
    revised = False
    if reasons and policy.allow_revision:
        d2 = sup.revise(plant, feats, d, reasons, history)
        if d2 is not None:
            revised = True
            ptok, ctok, calls, gpu = ptok + d2.prompt_tokens, ctok + d2.completion_tokens, calls + d2.llm_calls, gpu + d2.gpu_s
            d = d2
            reasons = _check(plant, d.action, log, policy)
    accepted = not reasons and d.action.type != "no_action"
    if accepted:
        log.accepted.append((t + policy.apply_delay_h, d.action))
        schedule_action(plant, d.action, policy.apply_delay_h, source=len(history))
    return {
        "t": t, "trigger": trigger, "alarms": list(feats["alarms"]), "action": d.action.as_dict(),
        "fx": compact(feats),
        "diagnosis": d.diagnosis, "rationale": d.rationale[:400], "accepted": accepted, "rejected_reasons": reasons,
        "revised": revised, "source": d.source, "prompt_tokens": ptok, "completion_tokens": ctok,
        "llm_calls": calls, "gpu_s": gpu, "meta": d.meta,
    }


COMPACT_KEYS = ("P", "P_slope", "P_ttl", "T", "T_slope", "RL", "SL", "XL", "XL_slope", "G_cmd_err", "PROD", "PROD_cmd_err",
                "yA_dev_nom", "P_resid", "T_resid", "dP_rs", "mv_r_cooling_water", "mv_r_cooling_water_std", "mv_purge",
                "mv_a_feed", "n_saturated")


def compact(f: dict) -> dict:
    out = {}
    for k in COMPACT_KEYS:
        v = f.get(k)
        if isinstance(v, float):
            v = round(v, 3) if abs(v) < 1e6 else ("inf" if v > 0 else "-inf")
        out[k] = v
    return out


def _check(plant: Plant, a: Action, log: ActionLog, policy: Policy) -> list[str]:
    if a.type == "no_action":
        return []
    reasons = static_check(a, plant, log, policy.max_actions)
    if not reasons and policy.shadow and a.type not in ("shutdown",):
        ok, why, _ = shadow_check(plant, a, policy.shadow_horizon_h, policy.shadow_fidelity)
        if not ok:
            reasons = [why]
    return reasons


class Hold:
    """No-op supervisor (sanity check: must equal the bare baseline)."""
    name = "hold"

    def episode_start(self, plant, scenario, seed):
        pass

    def decide(self, plant, feats, trigger, history):
        return Decision(NO_ACTION, "none", source="hold")

    def revise(self, plant, feats, rejected, reasons, history):
        return None

    def episode_end(self, plant, outcome):
        pass
