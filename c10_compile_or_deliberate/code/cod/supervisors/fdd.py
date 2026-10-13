"""B3: FDD classifier + recovery lookup table (zero-token baseline).

At every invocation the gradient-boosted classifier (cod.fdd.FDDModel) scores the feature dict;
if the top class is a fault with probability >= thr, the supervisor takes that class's
pre-engineered recovery action (one-way, so it acts at most once per class):

  composition, grade -> reconfigure composition_control
  feed_loss          -> reconfigure safety_overrides
  sensor_P / sensor_T -> substitute reactor_pressure / reactor_temperature (soft sensor)

If the recovery is already armed/substituted it holds. It never moves setpoints and never
forms a diagnosis outside the classifier's classes.

Spec: "fdd:[<model.pkl>][@thr=0.6][@persist=1]" (persist = consecutive invocations the same
class must win before acting).
"""
from __future__ import annotations

import os

from ..actions import NO_ACTION, Action
from ..fdd import CLASS_DIAGNOSIS, MODEL_PATH, FDDModel, augment, check_version
from ..runner import Decision

LOOKUP: dict[str, Action] = {
    "composition": Action("reconfigure", components=("composition_control",)),
    "grade": Action("reconfigure", components=("composition_control",)),
    "feed_loss": Action("reconfigure", components=("safety_overrides",)),
    "sensor_P": Action("substitute_instrument", measurement="reactor_pressure", method="soft"),
    "sensor_T": Action("substitute_instrument", measurement="reactor_temperature", method="soft"),
}
_MODELS: dict[str, FDDModel] = {}
OPTIONS = ("thr", "persist")


def parse_spec(spec: str) -> tuple[str, dict[str, float]]:
    body = spec.split(":", 1)[1] if ":" in spec else ""
    path, *opts = body.split("@")
    kw = {k: float(v) for k, v in (o.split("=", 1) for o in opts if o)}
    bad = set(kw) - set(OPTIONS)
    if bad:
        raise ValueError(f"fdd spec {spec!r}: unknown options {sorted(bad)}; allowed {list(OPTIONS)}")
    return (path if path and path != "default" else MODEL_PATH), kw


def load_model(path: str) -> FDDModel:
    """Cached per process; warns if the model predates the current feature extractor."""
    path = os.path.abspath(path)
    if path not in _MODELS:
        _MODELS[path] = FDDModel.load(path)
        check_version(_MODELS[path].meta.get("features_version"), path)
    return _MODELS[path]


def single_thread() -> None:
    """Single-row prediction with one OpenMP thread: the default (all cores) costs ~0.5 s per call on a
    loaded host. Must run after sklearn's libomp is loaded (i.e. after unpickling the model)."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1, user_api="openmp")


def done(action: Action, f: dict) -> bool:
    if action.type == "reconfigure":
        return all(f["armed_composition" if c == "composition_control" else "armed_overrides"] for c in action.components)
    return action.measurement in f["substituted"]


class FDDLookup:
    def __init__(self, spec: str = "fdd:", model: FDDModel | None = None):
        path, kw = parse_spec(spec)
        self.name = spec
        self.thr = float(kw.get("thr", 0.6))
        self.persist = int(kw.get("persist", 1))
        self.model = model or load_model(path)
        single_thread()

    def episode_start(self, plant, scenario, seed):
        self.streak: tuple[str, int] = ("none", 0)

    def decide(self, plant, f, trigger, history):
        p = self.model.predict_proba(augment(f, plant))
        cls = max(p, key=p.get)
        prev, n = self.streak
        self.streak = (cls, n + 1 if cls == prev else 1)
        meta = {"proba": {k: round(v, 3) for k, v in p.items()}}
        why = f"FDD top class {cls} (p={p[cls]:.2f})"
        if cls == "none" or p[cls] < self.thr:
            return Decision(NO_ACTION, "none", why + " below threshold" if cls != "none" else why, source="fdd:none", meta=meta)
        diag = CLASS_DIAGNOSIS[cls]
        action = LOOKUP[cls]
        if self.streak[1] < self.persist:
            return Decision(NO_ACTION, diag, why + f"; awaiting persistence {self.streak[1]}/{self.persist}",
                            source=f"fdd:{cls}:wait", meta=meta)
        if done(action, f):
            return Decision(NO_ACTION, diag, why + "; recovery already in place", source=f"fdd:{cls}:done", meta=meta)
        return Decision(action, diag, why + "; lookup recovery", source=f"fdd:{cls}", meta=meta)

    def revise(self, plant, f, rejected, reasons, history):
        return None

    def episode_end(self, plant, outcome):
        pass
