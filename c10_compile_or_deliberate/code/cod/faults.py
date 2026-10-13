"""Fault specifications and the scenario dataclass.

Faults enter at three boundaries (as in Alhazmi, 2026, and safetep-core):
  * kernel disturbances (IDV channels, latched from onset, magnitude in [0, 1]);
  * the sensor boundary: what the controller, alarms and supervisor OBSERVE (the true
    measurements still drive trips and metrics);
  * the actuator boundary: what reaches the plant from the controller.
Operator-scheduled setpoint shifts (production pushes, grade changes) are part of the
scenario too.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tep_studio import TEP_SCHEMA

MEAS_NAMES = TEP_SCHEMA.names("measurements")
MV_NAMES = TEP_SCHEMA.names("manipulated_variables")
IDV_NAMES = TEP_SCHEMA.names("disturbances")
MEAS_IDX = {n: i for i, n in enumerate(MEAS_NAMES)}
MV_IDX = {n: i for i, n in enumerate(MV_NAMES)}
IDV_IDX = {n: i for i, n in enumerate(IDV_NAMES)}


@dataclass(frozen=True)
class IdvEvent:
    idv: str  # "idv_01".."idv_28"
    start_h: float
    magnitude: float = 1.0
    end_h: float | None = None


@dataclass(frozen=True)
class SensorFault:
    measurement: str
    kind: str  # "bias" | "drift" | "freeze" | "scale"
    start_h: float
    magnitude: float = 0.0  # bias: units; drift: units/h; scale: multiplicative factor
    end_h: float | None = None


@dataclass(frozen=True)
class ActuatorFault:
    mv: str
    kind: str  # "stuck" (hold position at onset) | "stuck_at" (magnitude=%) | "cap_max" | "cap_min" | "offset"
    start_h: float
    magnitude: float = 0.0
    end_h: float | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    family: str  # A S P Q SN V K C ...
    description: str
    gt: tuple[str, ...] = ("none",)  # diagnosis vocabulary tags
    idv_events: tuple[IdvEvent, ...] = ()
    sensor_faults: tuple[SensorFault, ...] = ()
    actuator_faults: tuple[ActuatorFault, ...] = ()
    setpoint_shifts: tuple[tuple[float, tuple[tuple[str, float], ...]], ...] = ()
    horizon_h: float = 48.0
    params: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def content_hash(self) -> str:
        d = self.as_dict()
        d.pop("params", None)
        return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scenario":
        return Scenario(
            id=d["id"], family=d["family"], description=d.get("description", ""),
            gt=tuple(d.get("gt", ("none",))),
            idv_events=tuple(IdvEvent(**e) for e in d.get("idv_events", ())),
            sensor_faults=tuple(SensorFault(**e) for e in d.get("sensor_faults", ())),
            actuator_faults=tuple(ActuatorFault(**e) for e in d.get("actuator_faults", ())),
            setpoint_shifts=tuple((float(t), tuple((str(k), float(v)) for k, v in ch))
                                  for t, ch in d.get("setpoint_shifts", ())),
            horizon_h=float(d.get("horizon_h", 48.0)),
            params=dict(d.get("params", {})),
        )

    @property
    def first_onset_h(self) -> float:
        ts = [e.start_h for e in self.idv_events] + [f.start_h for f in self.sensor_faults] + \
             [f.start_h for f in self.actuator_faults] + [t for t, _ in self.setpoint_shifts]
        return min(ts) if ts else float("inf")


class FaultLayers:
    """Stateful application of a scenario's faults (deep-copyable for forks)."""

    def __init__(self, scenario: Scenario, active: bool = True):
        self.scenario = scenario
        self.active = active  # a nominal twin sets active=False (blind to faults)
        self._freeze: dict[int, float] = {}
        self._stuck: dict[int, float] = {}
        self._idv = np.zeros(28)

    # kernel disturbances
    def idv(self, t: float) -> np.ndarray:
        v = self._idv
        v[:] = 0.0
        if not self.active:
            return v
        for e in self.scenario.idv_events:
            if t >= e.start_h - 1e-9 and (e.end_h is None or t < e.end_h):
                v[IDV_IDX[e.idv]] = max(v[IDV_IDX[e.idv]], float(np.clip(e.magnitude, 0.0, 1.0)))
        return v

    # sensor boundary
    def observe(self, meas: np.ndarray, t: float) -> np.ndarray:
        if not self.active or not self.scenario.sensor_faults:
            return meas
        out = meas.copy()
        for f in self.scenario.sensor_faults:
            if t < f.start_h - 1e-9 or (f.end_h is not None and t >= f.end_h):
                continue
            j = MEAS_IDX[f.measurement]
            if f.kind == "bias":
                out[j] += f.magnitude
            elif f.kind == "drift":
                out[j] += f.magnitude * (t - f.start_h)
            elif f.kind == "freeze":
                if j not in self._freeze:
                    self._freeze[j] = float(meas[j])
                out[j] = self._freeze[j]
            elif f.kind == "scale":
                out[j] *= f.magnitude
        return out

    # actuator boundary
    def actuate(self, action: np.ndarray, t: float) -> np.ndarray:
        if not self.active or not self.scenario.actuator_faults:
            return action
        out = action.copy()
        for f in self.scenario.actuator_faults:
            if t < f.start_h - 1e-9 or (f.end_h is not None and t >= f.end_h):
                continue
            k = MV_IDX[f.mv]
            if f.kind == "stuck":
                if k not in self._stuck:
                    self._stuck[k] = float(action[k])
                out[k] = self._stuck[k]
            elif f.kind == "stuck_at":
                out[k] = f.magnitude
            elif f.kind == "cap_max":
                out[k] = min(out[k], f.magnitude)
            elif f.kind == "cap_min":
                out[k] = max(out[k], f.magnitude)
            elif f.kind == "offset":
                out[k] = out[k] + f.magnitude
        return np.clip(out, 0.0, 100.0)
