"""Steppable closed-loop TEP plant: kernel + Ricker regulatory layer + fault layers.

`Plant.step()` advances one regulatory scan (default 0.01 h = 36 s). The supervisory
interface (setpoints, reconfiguration, procedures, instrument substitution, shutdown)
is exposed through `apply_*` methods that the runner calls when a validated action lands.
`Plant.fork()` produces an exact, independent copy (kernel snapshot incl. noise RNG +
deep-copied controller/fault state) for twin rollouts and falsification.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tep_studio import RickerMultiLoopController, TennesseeEastmanProcess
from tep_studio.control.controller import ControllerSetpoints

from .faults import MEAS_IDX, MV_IDX, FaultLayers, Scenario

SETPOINT_FIELDS = tuple(f.name for f in dataclasses.fields(ControllerSetpoints))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")

# Trip limits evaluated by the kernel on the TRUE state (for reference / margins):
TRIP = {"reactor_pressure": 3000.0, "reactor_temperature": 175.0}
# Product-quality and production metrics are referenced to the OPERATOR'S commanded targets
# (initial nominal values, updated only by scenario-scheduled operator shifts), not to
# whatever setpoint a supervisor chose.


class AWRickerController(RickerMultiLoopController):
    """Ricker (1996) decentralized controller with conditional-integration anti-windup on the
    yA trim (as in Alhazmi's safetep-core fork; without it the strong strategy winds up
    and trips on loss of A feed)."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._a_hi = False
        self._a_lo = False

    def reset(self, measurements, *, time: float = 0.0) -> None:
        super().reset(measurements, time=time)
        self._a_hi = False
        self._a_lo = False

    def _update_velocity(self, loop, setpoint, measurement):
        delta = super()._update_velocity(loop, setpoint, measurement)
        if loop.name == self.registry.ya.name and ((self._a_hi and delta > 0.0) or (self._a_lo and delta < 0.0)):
            return 0.0
        return delta

    def _update_ratio(self, loop, setpoint, meas):
        value = super()._update_ratio(loop, setpoint, meas)
        if loop.ratio_key == "r1":
            self._a_hi = value >= loop.hi - 1e-6
            self._a_lo = value <= loop.lo + 1e-6
        return value


def load_soft_sensors(path: str | None = None) -> dict[str, dict]:
    path = path or os.path.join(DATA_DIR, "soft_sensors.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


_SOFT = None


def soft_sensor_models() -> dict[str, dict]:
    global _SOFT
    if _SOFT is None:
        _SOFT = load_soft_sensors()
    return _SOFT


def soft_estimate(model: dict, observed: np.ndarray, mv: np.ndarray | None = None,
                  mvf: np.ndarray | None = None) -> float:
    """Linear soft sensor; inputs are measurement names, 'mv:<valve>' (valve position feedback) or
    'mvf:<valve>' (valve position low-pass filtered, tau = 0.25 h, which breaks the algebraic loop
    when the soft sensor feeds the loop that moves that valve)."""
    def val(n):
        if n.startswith("mvf:"):
            return mvf[MV_IDX[n[4:]]] if mvf is not None else np.nan
        if n.startswith("mv:"):
            return mv[MV_IDX[n[3:]]] if mv is not None else np.nan
        return observed[MEAS_IDX[n]]
    x = np.array([val(n) for n in model["inputs"]])
    return float(model["intercept"] + np.dot(model["coef"], x))


@dataclass
class Pending:
    due_h: float
    kind: str  # "setpoints" | "reconfigure" | "substitute" | "shutdown"
    payload: Any
    source: int = -1


@dataclass
class EpisodeStats:
    cost: float = 0.0  # integral of operating cost $/h * h
    production: float = 0.0  # integral of stripper underflow m3/h * h
    g_iae: float = 0.0  # integral |%G - commanded grade|
    prod_iae: float = 0.0  # integral |production - commanded rate|
    offspec_h: float = 0.0  # hours with |%G - grade| > 1.0 mol%
    min_margin: dict[str, float] = field(default_factory=dict)
    peak: dict[str, float] = field(default_factory=dict)
    alarm_hours: float = 0.0  # integral of (#active alarms) dt


class Plant:
    def __init__(self, scenario: Scenario, seed: int, *, baseline: str = "weak", ci: float = 0.01,
                 faults_active: bool = True, history: bool = True, sim: TennesseeEastmanProcess | None = None):
        self.scenario = scenario
        self.seed = int(seed)
        self.ci = float(ci)
        self.horizon = float(scenario.horizon_h)
        strong = baseline == "strong"
        self.baseline = baseline
        self.sim = sim or TennesseeEastmanProcess()
        self.ctrl = AWRickerController(enable_overrides=strong, enable_composition=strong)
        self.layers = FaultLayers(scenario, active=faults_active)
        self.keep_history = history
        from .alarms import AlarmMonitor  # local import to avoid cycle
        self.monitor = AlarmMonitor()
        self.substitutions: dict[str, tuple[str, float | str]] = {}
        self.pending: list[Pending] = []
        self.shifts = sorted([(t, dict(ch)) for t, ch in scenario.setpoint_shifts], key=lambda s: s[0])
        self.applied: list[dict] = []
        self.tripped = False
        self.trip_msg = ""
        self.shutdown_ordered = False
        self.t = 0.0
        self.k = 0
        self.stats = EpisodeStats()
        self.new_alarms: list[str] = []
        self._reset()

    # ------------------------------------------------------------------ setup
    def _reset(self):
        meas, info = self.sim.reset(mode="mode1", seed=float(self.seed))
        self.true = meas.copy()
        self.obs = self.layers.observe(meas, 0.0)
        self.ctrl.reset(self._controller_view(self.obs), time=0.0)
        self.sp0 = dataclasses.asdict(self.ctrl.setpoints)
        self.cmd = {"pct_g": self.sp0["pct_g"], "production_rate": self.sp0["production_rate"]}
        self.mv = np.array(self.sim.kernel.xmv, dtype=float) if hasattr(self.sim.kernel, "xmv") else np.zeros(12)
        self.mvf = self.mv.copy()
        n = int(round(self.horizon / self.ci)) + 2
        if self.keep_history:
            self.H_t = np.zeros(n)
            self.H_obs = np.zeros((n, 41))
            self.H_mv = np.zeros((n, 12))
            self.H_t[0], self.H_obs[0], self.H_mv[0] = 0.0, self.obs, self.mv
        self.n_hist = 1
        m = info["constraint_margins"]
        self.stats.min_margin = dict(m)
        self.stats.peak = {"reactor_pressure_max": float(meas[6]), "reactor_temperature_max": float(meas[8])}

    # --------------------------------------------------------------- views
    def _controller_view(self, observed: np.ndarray) -> np.ndarray:
        if not self.substitutions:
            return observed
        view = observed.copy()
        for name, (kind, val) in self.substitutions.items():
            j = MEAS_IDX[name]
            if kind == "offset":
                view[j] = observed[j] + float(val)
            elif kind == "soft":
                view[j] = soft_estimate(soft_sensor_models()[str(val)], observed, self.mv, self.mvf)
            elif kind == "value":
                view[j] = float(val)
        return view

    @property
    def setpoints(self) -> dict[str, float]:
        return dataclasses.asdict(self.ctrl.setpoints)

    @property
    def armed(self) -> dict[str, bool]:
        return {"composition_control": bool(self.ctrl.enable_composition),
                "safety_overrides": bool(self.ctrl.enable_overrides)}

    @property
    def done(self) -> bool:
        return self.tripped or self.shutdown_ordered or self.t >= self.horizon - 1e-9

    # ------------------------------------------------------- supervisory acts
    def set_setpoints(self, changes: dict[str, float], who: str = "supervisor"):
        unknown = set(changes) - set(SETPOINT_FIELDS)
        if unknown:
            raise ValueError(f"unknown setpoints {unknown}")
        old = self.ctrl.setpoints
        self.ctrl.setpoints = dataclasses.replace(old, **{k: float(v) for k, v in changes.items()})
        for k, v in changes.items():
            self.applied.append({"t": self.t, "who": who, "field": k, "old": float(getattr(old, k)), "new": float(v)})

    def reconfigure(self, component: str):
        if component == "composition_control":
            self.ctrl.enable_composition = True
        elif component == "safety_overrides":
            self.ctrl.enable_overrides = True
        else:
            raise ValueError(component)
        self.applied.append({"t": self.t, "who": "supervisor", "field": f"reconfigure.{component}"})

    def substitute(self, measurement: str, method: str, value: float | str | None = None):
        """Take a transmitter out of service for the regulatory layer.
        method="soft": use the pre-engineered soft sensor `value` (model name, default = measurement);
        method="estimate": static operator estimate `value` (offset re-anchored now, Alhazmi-style)."""
        if method == "soft":
            name = str(value or measurement)
            if name not in soft_sensor_models():
                raise ValueError(f"no soft sensor {name}")
            self.substitutions[measurement] = ("soft", name)
        elif method == "estimate":
            offset = float(value) - float(self.obs[MEAS_IDX[measurement]])
            self.substitutions[measurement] = ("offset", offset)
        elif method == "restore":
            self.substitutions.pop(measurement, None)
        else:
            raise ValueError(method)
        self.applied.append({"t": self.t, "who": "supervisor", "field": f"substitute.{measurement}", "new": method})

    def order_shutdown(self):
        self.shutdown_ordered = True
        self.applied.append({"t": self.t, "who": "supervisor", "field": "shutdown"})

    def schedule(self, p: Pending):
        self.pending.append(p)

    def _apply_pending(self):
        if not self.pending:
            return
        due = [p for p in self.pending if p.due_h <= self.t + 1e-9]
        if not due:
            return
        self.pending = [p for p in self.pending if p.due_h > self.t + 1e-9]
        for p in sorted(due, key=lambda p: p.due_h):
            if p.kind == "setpoints":
                self.set_setpoints(p.payload)
            elif p.kind == "reconfigure":
                for c in p.payload:
                    self.reconfigure(c)
            elif p.kind == "substitute":
                self.substitute(*p.payload)
            elif p.kind == "shutdown":
                self.order_shutdown()

    # ------------------------------------------------------------------ step
    def step(self) -> None:
        if self.done:
            return
        t = self.t
        while self.shifts and self.shifts[0][0] <= t + 1e-9:
            _, ch = self.shifts.pop(0)
            self.set_setpoints(ch, who="operator")
            for k in ("pct_g", "production_rate"):
                if k in ch:
                    self.cmd[k] = float(ch[k])
        self._apply_pending()
        if self.shutdown_ordered:
            return
        action, _diag = self.ctrl.compute_action(self._controller_view(self.obs), time=t)
        implemented = self.layers.actuate(action, t)
        try:
            res = self.sim.advance(implemented, control_interval=self.ci, disturbances=self.layers.idv(t).copy())
        except RuntimeError as e:  # stiff model left its numerical envelope: physically an extreme state
            self.tripped = True
            self.trip_msg = f"integration diverged (treated as trip): {str(e)[:60]}"
            self.t = t + self.ci
            return
        self.t = res.time
        self.k += 1
        self.true = res.measurements
        self.mv = res.implemented_action
        self.mvf = self.mvf + (self.ci / 0.25) * (self.mv - self.mvf)
        self.obs = self.layers.observe(self.true, self.t)
        # metrics on TRUE values
        st, dt, m = self.stats, self.ci, self.true
        st.cost += res.objective_terms.get("operating_cost_internal_per_hour", 0.0) * dt
        st.production += float(m[16]) * dt
        dg = abs(float(m[39]) - self.cmd["pct_g"])
        st.g_iae += dg * dt
        st.offspec_h += dt if dg > 1.0 else 0.0
        st.prod_iae += abs(float(m[16]) - self.cmd["production_rate"]) * dt
        for kname, v in res.constraint_margins.items():
            if v < st.min_margin.get(kname, np.inf):
                st.min_margin[kname] = v
        st.peak["reactor_pressure_max"] = max(st.peak["reactor_pressure_max"], float(m[6]))
        st.peak["reactor_temperature_max"] = max(st.peak["reactor_temperature_max"], float(m[8]))
        # alarms on OBSERVED values
        self.new_alarms = self.monitor.update(self.obs, self.t, self.setpoints)
        st.alarm_hours += len(self.monitor.active) * dt
        if self.keep_history:
            i = self.n_hist
            self.H_t[i], self.H_obs[i], self.H_mv[i] = self.t, self.obs, self.mv
        self.n_hist += 1
        if res.shutdown_status["terminated"]:
            self.tripped = True
            self.trip_msg = res.shutdown_status["message"]

    def run_until(self, t_end: float) -> None:
        t_end = min(t_end, self.horizon)
        while not self.done and self.t < t_end - 1e-9:
            self.step()

    # ------------------------------------------------------------------ fork
    def fork(self, *, faults_active: bool | None = None, sim: TennesseeEastmanProcess | None = None,
             history: bool = False) -> "Plant":
        """Exact independent copy. faults_active=False gives a NOMINAL twin that is blind to all
        faults from now on (the current state still carries their effect so far)."""
        new = Plant.__new__(Plant)
        skip = {"sim", "H_t", "H_obs", "H_mv"}
        for k, v in self.__dict__.items():
            if k not in skip:
                new.__dict__[k] = copy.deepcopy(v) if k in ("ctrl", "layers", "monitor", "stats", "pending",
                                                              "substitutions", "shifts", "cmd") else v
        new.applied = list(self.applied)
        new.sim = sim or TennesseeEastmanProcess()
        new.sim.restore(self.sim.snapshot())
        if faults_active is not None:
            new.layers.active = faults_active
        new.keep_history = history and self.keep_history
        if new.keep_history:
            new.H_t, new.H_obs, new.H_mv = self.H_t.copy(), self.H_obs.copy(), self.H_mv.copy()
        return new

    # ------------------------------------------------------------- history
    def window(self, hours: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.keep_history:
            raise RuntimeError("history disabled")
        n = self.n_hist
        k0 = max(0, n - int(round(hours / self.ci)) - 1)
        return self.H_t[k0:n], self.H_obs[k0:n], self.H_mv[k0:n]

    def outcome(self) -> dict[str, Any]:
        st = self.stats
        stabilized = (not self.tripped) and (not self.shutdown_ordered) and self.t >= self.horizon - 1e-6
        return {
            "stabilized": bool(stabilized),
            "tripped": bool(self.tripped),
            "trip_msg": self.trip_msg,
            "trip_time": float(self.t) if self.tripped else None,
            "shutdown": bool(self.shutdown_ordered),
            "shutdown_time": float(self.t) if self.shutdown_ordered else None,
            "final_time": float(self.t),
            "cost": st.cost, "production": st.production, "g_iae": st.g_iae, "offspec_h": st.offspec_h,
            "prod_iae": st.prod_iae, "min_margin": dict(st.min_margin), "peak": dict(st.peak),
            "alarm_hours": st.alarm_hours, "armed": self.armed, "applied": self.applied,
        }
