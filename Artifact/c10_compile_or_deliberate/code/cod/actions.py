"""Bounded supervisory action interface, static validator and action application.

The interface follows Alhazmi (2026): the supervisor can never write a manipulated
variable. It can move regulatory setpoints within validated bounds, arm pre-engineered
regulatory components (one-way), run pre-audited procedures, take a transmitter out of
service in favour of a pre-engineered soft sensor (our extension), do nothing, or order
an orderly shutdown (a failure on the primary endpoint).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plant import SETPOINT_FIELDS, Pending, Plant, soft_sensor_models

DIAGNOSES = (
    "none", "feed_composition_upset", "feed_loss_or_header_pressure", "reactor_cooling_fault",
    "condenser_cooling_fault", "kinetics_drift", "heat_transfer_degradation", "valve_stiction_or_stuck",
    "sensor_fault", "operating_point_change", "compound_fault", "unknown",
)
ACTION_TYPES = ("adjust_setpoints", "reconfigure", "procedure", "substitute_instrument", "no_action", "shutdown")
COMPONENTS = ("composition_control", "safety_overrides")


@dataclass(frozen=True)
class Bound:
    lo: float
    hi: float
    max_delta: float
    cooldown_h: float


BOUNDS: dict[str, Bound] = {
    "reactor_pressure": Bound(2500.0, 2895.0, 100.0, 0.5),
    "reactor_temperature": Bound(115.0, 128.0, 3.0, 0.5),
    "reactor_level": Bound(50.0, 85.0, 10.0, 0.5),
    "separator_level": Bound(30.0, 70.0, 10.0, 0.5),
    "stripper_level": Bound(30.0, 70.0, 10.0, 0.5),
    "production_rate": Bound(11.0, 26.0, 4.0, 1.0),
    "pct_g": Bound(40.0, 65.0, 5.0, 2.0),
    "ya": Bound(50.0, 60.0, 2.0, 1.0),
    "yac": Bound(54.0, 62.0, 2.0, 1.0),
}
assert set(BOUNDS) <= set(SETPOINT_FIELDS)

# name -> list of (offset_h, {field: fn(setpoints_at_start) | value})
PROCEDURES: dict[str, list[tuple[float, dict[str, Any]]]] = {
    "rate_cut_25": [(0.0, {"production_rate": lambda sp: sp["production_rate"] * 0.75})],
    "rate_cut_50": [(0.0, {"production_rate": lambda sp: sp["production_rate"] * 0.50})],
    "depressurize_step": [(0.0, {"reactor_pressure": lambda sp: sp["reactor_pressure"] - 75.0})],
    "min_stable_park": [(0.0, {"production_rate": lambda sp: sp["production_rate"] * 0.50}),
                        (1.0, {"reactor_pressure": 2600.0})],
}
SUBSTITUTABLE = ("reactor_pressure", "reactor_temperature", "separator_temperature", "stripper_pressure")
MAX_SUBSTITUTED = 2
SUBSTITUTE_COOLDOWN_H = 0.5


@dataclass(frozen=True)
class Action:
    type: str
    changes: tuple[tuple[str, float], ...] = ()
    components: tuple[str, ...] = ()
    name: str = ""
    measurement: str = ""
    method: str = "soft"  # soft | estimate
    estimate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.type == "adjust_setpoints":
            d["changes"] = dict(self.changes)
        elif self.type == "reconfigure":
            d["components"] = list(self.components)
        elif self.type == "procedure":
            d["name"] = self.name
        elif self.type == "substitute_instrument":
            d.update(measurement=self.measurement, method=self.method)
            if self.estimate is not None:
                d["estimate"] = self.estimate
        return d

    def key(self) -> str:
        import json
        return json.dumps(self.as_dict(), sort_keys=True)


NO_ACTION = Action("no_action")


def parse_action(d: Any) -> tuple[Action | None, str | None]:
    """Parse a JSON-like dict into an Action; returns (action, error)."""
    if not isinstance(d, dict):
        return None, "action must be an object"
    t = d.get("type")
    if t not in ACTION_TYPES:
        return None, f"unknown action type {t!r}; allowed {list(ACTION_TYPES)}"
    try:
        if t == "adjust_setpoints":
            ch = d.get("changes") or {}
            if not isinstance(ch, dict) or not ch:
                return None, "adjust_setpoints needs a non-empty 'changes' object"
            return Action(t, changes=tuple(sorted((str(k), float(v)) for k, v in ch.items()))), None
        if t == "reconfigure":
            comps = d.get("components") or ([d["component"]] if "component" in d else [])
            if isinstance(comps, str):
                comps = [comps]
            return Action(t, components=tuple(dict.fromkeys(str(c) for c in comps))), None
        if t == "procedure":
            return Action(t, name=str(d.get("name", ""))), None
        if t == "substitute_instrument":
            est = d.get("estimate")
            return Action(t, measurement=str(d.get("measurement", "")), method=str(d.get("method", "soft")),
                          estimate=None if est is None else float(est)), None
        return Action(t), None
    except (TypeError, ValueError) as e:
        return None, f"malformed action: {e}"


def resolve_procedure(name: str, sp: dict[str, float]) -> list[tuple[float, dict[str, float]]]:
    out = []
    for off, ch in PROCEDURES[name]:
        res = {k: float(v(sp)) if callable(v) else float(v) for k, v in ch.items()}
        out.append((off, clamp(res)))
    return out


def clamp(changes: dict[str, float]) -> dict[str, float]:
    return {k: (min(max(v, BOUNDS[k].lo), BOUNDS[k].hi) if k in BOUNDS else v) for k, v in changes.items()}


@dataclass
class ActionLog:
    """Per-episode memory the validator needs (accepted actions and their times)."""
    accepted: list[tuple[float, Action]] = field(default_factory=list)

    def last_change(self, fieldname: str) -> float | None:
        ts = [t for t, a in self.accepted if (a.type == "adjust_setpoints" and fieldname in dict(a.changes))
              or (a.type == "procedure" and any(fieldname in ch for _, ch in PROCEDURES.get(a.name, [])))]
        return max(ts) if ts else None

    def armed(self) -> set[str]:
        return {c for _, a in self.accepted if a.type == "reconfigure" for c in a.components}

    def substituted(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for t, a in self.accepted:
            if a.type == "substitute_instrument":
                out[a.measurement] = t
        return out


def static_check(a: Action, plant: Plant, log: ActionLog, max_actions: int = 12) -> list[str]:
    """Machine-readable rejection reasons (empty list = admissible)."""
    if a.type in ("no_action", "shutdown"):
        return []
    r: list[str] = []
    if len(log.accepted) >= max_actions:
        r.append(f"budget: action budget exhausted ({len(log.accepted)}/{max_actions})")
    t = plant.t
    if a.type == "adjust_setpoints":
        sp = plant.setpoints
        for k, v in a.changes:
            b = BOUNDS.get(k)
            if b is None:
                r.append(f"field: {k!r} is not supervisor-adjustable; allowed {sorted(BOUNDS)}")
                continue
            if not (b.lo - 1e-6 <= v <= b.hi + 1e-6):
                r.append(f"bounds: {k} target {v:g} outside [{b.lo:g}, {b.hi:g}]")
            if abs(v - sp[k]) > b.max_delta + 1e-6:
                r.append(f"delta: {k} move {abs(v - sp[k]):.3g} exceeds max {b.max_delta:g} (current {sp[k]:.4g})")
            last = log.last_change(k)
            if last is not None and t - last < b.cooldown_h:
                r.append(f"cooldown: {k} changed at t={last:.2f} h; cooldown {b.cooldown_h:g} h")
    elif a.type == "reconfigure":
        if not a.components:
            r.append("reconfigure: no components given")
        bad = [c for c in a.components if c not in COMPONENTS]
        if bad:
            r.append(f"reconfigure: unknown components {bad}; allowed {list(COMPONENTS)}")
        already = [c for c in a.components if plant.armed.get(c)]
        if already and len(already) == len(a.components):
            r.append(f"reconfigure: {already} already armed")
    elif a.type == "procedure":
        if a.name not in PROCEDURES:
            r.append(f"procedure: unknown {a.name!r}; allowed {sorted(PROCEDURES)}")
        else:
            for _, ch in PROCEDURES[a.name]:
                for k in ch:
                    last = log.last_change(k)
                    if last is not None and t - last < BOUNDS[k].cooldown_h:
                        r.append(f"cooldown: {k} changed at t={last:.2f} h; cooldown {BOUNDS[k].cooldown_h:g} h")
    elif a.type == "substitute_instrument":
        if a.measurement not in SUBSTITUTABLE:
            r.append(f"substitute: {a.measurement!r} not substitutable; allowed {list(SUBSTITUTABLE)}")
        elif a.method == "soft" and a.measurement not in soft_sensor_models():
            r.append(f"substitute: no soft sensor for {a.measurement}")
        elif a.method == "estimate" and a.estimate is None:
            r.append("substitute: method 'estimate' needs a numeric 'estimate'")
        elif a.method not in ("soft", "estimate"):
            r.append("substitute: method must be 'soft' or 'estimate'")
        subs = log.substituted()
        if a.measurement in subs and t - subs[a.measurement] < SUBSTITUTE_COOLDOWN_H:
            r.append(f"cooldown: {a.measurement} substituted at t={subs[a.measurement]:.2f} h")
        if a.measurement not in subs and len(subs) >= MAX_SUBSTITUTED:
            r.append(f"substitute: already {len(subs)} instruments out of service (max {MAX_SUBSTITUTED})")
    return r


def schedule_action(plant: Plant, a: Action, delay_h: float, source: int = -1) -> None:
    """Queue an ACCEPTED action to land after the modelled decision latency."""
    due = plant.t + delay_h
    if a.type == "adjust_setpoints":
        plant.schedule(Pending(due, "setpoints", dict(a.changes), source))
    elif a.type == "reconfigure":
        comps = tuple(c for c in a.components if not plant.armed.get(c))
        plant.schedule(Pending(due, "reconfigure", comps, source))
    elif a.type == "procedure":
        for off, ch in resolve_procedure(a.name, plant.setpoints):
            plant.schedule(Pending(due + off, "setpoints", ch, source))
    elif a.type == "substitute_instrument":
        val = a.measurement if a.method == "soft" else a.estimate
        plant.schedule(Pending(due, "substitute", (a.measurement, a.method, val), source))
    elif a.type == "shutdown":
        plant.schedule(Pending(due, "shutdown", None, source))


def apply_now(plant: Plant, a: Action) -> None:
    """Apply an action immediately (twin rollouts)."""
    schedule_action(plant, a, 0.0)
    plant._apply_pending()


# JSON schema for constrained LLM decoding (Ollama `format`).
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {"type": "string", "enum": list(DIAGNOSES)},
        "rationale": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": list(ACTION_TYPES)},
                "changes": {"type": "object", "properties": {k: {"type": "number"} for k in BOUNDS}},
                "components": {"type": "array", "items": {"type": "string", "enum": list(COMPONENTS)}},
                "name": {"type": "string", "enum": sorted(PROCEDURES)},
                "measurement": {"type": "string", "enum": list(SUBSTITUTABLE)},
                "method": {"type": "string", "enum": ["soft", "estimate"]},
                "estimate": {"type": "number"},
            },
            "required": ["type"],
        },
    },
    "required": ["diagnosis", "rationale", "action"],
}
