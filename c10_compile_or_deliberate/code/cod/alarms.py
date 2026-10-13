"""ISA-18.2-style alarm system on OBSERVED measurements.

Absolute alarms sit strictly inside the kernel trip limits, with deadband (hysteresis)
and an on-delay. Deviation alarms compare a measurement with its CURRENT regulatory
setpoint (quality and throughput). Alarm names are stable identifiers used by
playbooks, the LLM summary and the novelty gate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .faults import MEAS_IDX


@dataclass(frozen=True)
class AlarmSpec:
    name: str
    measurement: str
    kind: str  # "HI" | "LO" | "DEV" (|meas - setpoint| > limit)
    limit: float
    deadband: float
    on_delay_h: float
    priority: int  # 1 = highest
    setpoint: str | None = None  # for DEV alarms


# Limits were rationalised for time-to-respond (ISA-18.2): PRESSURE_HI at 2760 kPa annunciates
# ~0.9 h before the latest effective intervention on a reactant-composition upset.
# Nominal mode-1: P 2705 kPa, T 120.4 C, reactor level 75 %, separator/stripper 50 %,
# %G 53.7 mol %, production 22.95 m3/h. Trips: P 3000 kPa, T 175 C, levels at 0/100 % indicated.
ALARMS: tuple[AlarmSpec, ...] = (
    AlarmSpec("PRESSURE_HI", "reactor_pressure", "HI", 2760.0, 15.0, 0.02, 2),
    AlarmSpec("PRESSURE_HIHI", "reactor_pressure", "HI", 2900.0, 10.0, 0.0, 1),
    AlarmSpec("PRESSURE_LO", "reactor_pressure", "LO", 2550.0, 15.0, 0.05, 3),
    AlarmSpec("TEMP_HI", "reactor_temperature", "HI", 127.0, 0.5, 0.02, 2),
    AlarmSpec("TEMP_HIHI", "reactor_temperature", "HI", 140.0, 0.5, 0.0, 1),
    AlarmSpec("TEMP_LO", "reactor_temperature", "LO", 114.0, 0.5, 0.05, 3),
    AlarmSpec("RLEVEL_HI", "reactor_level", "HI", 92.0, 1.0, 0.02, 2),
    AlarmSpec("RLEVEL_LO", "reactor_level", "LO", 45.0, 1.0, 0.02, 2),
    AlarmSpec("SLEVEL_HI", "separator_level", "HI", 85.0, 1.0, 0.02, 2),
    AlarmSpec("SLEVEL_LO", "separator_level", "LO", 15.0, 1.0, 0.02, 2),
    AlarmSpec("XLEVEL_HI", "stripper_level", "HI", 85.0, 1.0, 0.02, 2),
    AlarmSpec("XLEVEL_LO", "stripper_level", "LO", 15.0, 1.0, 0.02, 2),
    AlarmSpec("GRADE_DEV", "stripper_underflow_G_concentration", "DEV", 3.0, 0.5, 0.5, 3, "pct_g"),
    AlarmSpec("RATE_DEV", "stripper_underflow", "DEV", 2.0, 0.3, 1.0, 3, "production_rate"),
)
ALARM_NAMES = tuple(a.name for a in ALARMS)


class AlarmMonitor:
    def __init__(self, specs: tuple[AlarmSpec, ...] = ALARMS):
        self.specs = specs
        self.active: dict[str, float] = {}  # name -> annunciation time
        self._pending: dict[str, float] = {}  # name -> first-exceed time (on-delay)
        self.events: list[tuple[float, str, str]] = []  # (t, name, "on"|"off")

    def _exceeds(self, a: AlarmSpec, v: float, sp: dict[str, float] | None, active: bool) -> bool:
        db = a.deadband if active else 0.0
        if a.kind == "HI":
            return v > a.limit - db
        if a.kind == "LO":
            return v < a.limit + db
        ref = (sp or {}).get(a.setpoint, None)
        if ref is None:
            return False
        return abs(v - ref) > a.limit - db

    def update(self, observed: np.ndarray, t: float, setpoints: dict[str, float] | None = None) -> list[str]:
        new: list[str] = []
        for a in self.specs:
            v = float(observed[MEAS_IDX[a.measurement]])
            is_active = a.name in self.active
            exc = self._exceeds(a, v, setpoints, is_active)
            if is_active:
                if not exc:
                    del self.active[a.name]
                    self.events.append((t, a.name, "off"))
            else:
                if exc:
                    t0 = self._pending.setdefault(a.name, t)
                    if t - t0 >= a.on_delay_h - 1e-9:
                        self.active[a.name] = t
                        self._pending.pop(a.name, None)
                        self.events.append((t, a.name, "on"))
                        new.append(a.name)
                else:
                    self._pending.pop(a.name, None)
        return new
