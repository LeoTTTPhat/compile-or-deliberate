"""Fault detection and diagnosis (B3) and the conformal novelty gate (used by the hybrid agent).

* Labels: every recorded invocation of a synthesis-family episode gets a fault CLASS derived
  from the scenario (`fault_class`): none (family A, no-fault runs, or before onset),
  composition (S), feed_loss (P), grade (Q), sensor_P / sensor_T (SN, by transmitter).
* `FDDModel`: gradient boosting (sklearn HistGradientBoostingClassifier) on the numeric
  features of `summary.features` (lists/strings dropped, alarm ages flattened, times-to-limit
  capped at 48 h), plus Hotelling T^2 and SPE of a PCA fitted on the none-class rows, the
  classical TEP monitoring statistics. Trained on passive (hold) rows and post-recovery rows
  (see experiments/fdd_data.py); held-out accuracy is estimated by scenario-grouped CV.
* `NoveltyDetector`: mean distance to the k nearest reference rows in a robustly standardised
  space of 36 process variables (15-min means vs nominal, valve positions, consistency
  residuals). A split-conformal threshold on calibration SCENARIOS disjoint from the reference
  set gives P(score > threshold) <= alpha for a new in-distribution row.
* `features_version()`: fingerprint of the feature extractor; cached rows, the dataset and the
  trained models record it so rows from different extractor versions are never mixed.

Entry point: `python -m cod.fdd train` (reads data/fdd_features.npz from experiments/fdd_data.py).
"""
from __future__ import annotations

import functools
import hashlib
import inspect
import json
import math
import os
import pickle
import warnings
from typing import Iterable

import numpy as np

from .alarms import ALARM_NAMES
from .plant import DATA_DIR, Plant, soft_estimate, soft_sensor_models

CLASSES = ("none", "composition", "feed_loss", "grade", "sensor_P", "sensor_T")
CLASS_DIAGNOSIS = {"none": "none", "composition": "feed_composition_upset", "feed_loss": "feed_loss_or_header_pressure",
                   "grade": "operating_point_change", "sensor_P": "sensor_fault", "sensor_T": "sensor_fault"}
FAMILY_CLASS = {"A": "none", "N": "none", "S": "composition", "P": "feed_loss", "Q": "grade"}
SYNTH_FAMILIES = ("A", "S", "P", "Q", "SN")
TTL_CAP = 48.0
BIG = 1e6
EXCLUDE = ("t",)  # episode clock: onset times are a sampling artefact
FEATURES_NPZ = os.path.join(DATA_DIR, "fdd_features.npz")
MODEL_PATH = os.path.join(DATA_DIR, "fdd_model.pkl")
NOVELTY_PATH = os.path.join(DATA_DIR, "novelty.pkl")

# Classical MSPC variable set (decided a priori, not from held-out families): 15-min means of the
# measured process variables relative to nominal, all valve positions except the fixed agitator,
# and the analytical-redundancy residuals.
_PV = ("P", "T", "RL", "SL", "XL", "G", "PROD", "SP", "XP", "ST", "XT", "PURGE", "FA", "FD", "FE", "FAC", "RECY",
       "RFEED", "CW_R", "CW_C", "WORK")
_MV = ("d_feed", "e_feed", "a_feed", "ac_feed", "compressor_recycle", "purge", "s_underflow", "x_underflow",
       "x_steam", "r_cooling_water", "s_cooling_water")
NOVELTY_FEATURES = (tuple(k + "_dev_nom" for k in _PV) + tuple("mv_" + m for m in _MV)
                    + ("yA_dev_nom", "dP_rs", "P_resid", "T_resid"))


def max_threads() -> int:
    """OpenMP threads for batch fitting/prediction (resource rule: at most 6)."""
    return max(1, min(6, int(os.environ.get("MEMGUARD_PARALLEL", "6"))))


# ------------------------------------------------------------------------------- labels
def fault_class(scenario) -> str | None:
    """Fault class of a synthesis-family scenario after onset; None for families outside the FDD design."""
    fam = scenario.family
    if fam == "SN":
        return "sensor_P" if scenario.sensor_faults[0].measurement == "reactor_pressure" else "sensor_T"
    return FAMILY_CLASS.get(fam)


def fault_onset(scenario) -> float:
    """Onset of the FAULT (the sampled onset), not of earlier operator production moves."""
    on = scenario.params.get("onset") if scenario.params else None
    return float(on) if on is not None else float(scenario.first_onset_h)


def row_label(scenario, t: float) -> str | None:
    cls = fault_class(scenario)
    return "none" if cls is None or cls == "none" or t < fault_onset(scenario) else cls


# ------------------------------------------------------------------------- vectorising
def augment(f: dict, plant: Plant | None) -> dict:
    """Adds `T_resid_mvf`: temperature soft-sensor residual evaluated with the filtered valve position
    (introduced while summary.features omitted mvf; it now duplicates `T_resid`, which is the fallback
    when no plant history is available)."""
    g = dict(f)
    model = soft_sensor_models().get("reactor_temperature")
    val = f.get("T_resid", float("nan"))
    if plant is not None and model is not None and getattr(plant, "keep_history", False):
        _, Y, MV = plant.window(0.25)
        val = f["T_avg"] - soft_estimate(model, Y.mean(axis=0), MV.mean(axis=0), plant.mvf)
    g["T_resid_mvf"] = float(val)
    return g


@functools.lru_cache(maxsize=1)
def features_version() -> str:
    """Fingerprint of everything that shapes a feature row: summary/alarms source, the soft-sensor and
    nominal data files, and the vectorisers here."""
    from . import alarms, summary
    h = hashlib.sha1()
    for part in (summary, alarms, soft_estimate, augment, numeric_features):
        h.update(inspect.getsource(part).encode())
    for fn in ("soft_sensors.json", "nominal.json"):
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            with open(p, "rb") as fh:
                h.update(fh.read())
    return h.hexdigest()[:12]


def numeric_features(f: dict) -> dict[str, float]:
    """Flat numeric view of a feature dict: floats/ints/bools kept, alarm ages flattened, lists/strings dropped."""
    out: dict[str, float] = {}
    for k, v in f.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = float(v)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out[k] = float(v)
    ages = f.get("alarm_ages") or {}
    for a in ALARM_NAMES:
        out["age_" + a] = float(ages.get(a, 0.0))
    return out


def rows_to_matrix(rows: Iterable[dict], names: list[str]) -> np.ndarray:
    return np.array([[r.get(n, np.nan) for n in names] for r in rows], dtype=float)


def clean(M: np.ndarray, names: list[str]) -> np.ndarray:
    """Cap times-to-limit at 48 h and clip remaining infinities (NaN is kept: the booster treats it as missing)."""
    M = np.array(M, dtype=float, copy=True)
    for j, n in enumerate(names):
        if n.endswith("_ttl"):
            M[:, j] = np.minimum(M[:, j], TTL_CAP)
    return np.clip(M, -BIG, BIG)


class _RobustScale:
    """Median / IQR standardisation with a floor on the scale (near-constant columns)."""

    def fit(self, M: np.ndarray) -> "_RobustScale":
        self.center = np.nanmedian(M, axis=0)
        q75, q25 = np.nanpercentile(M, [75, 25], axis=0)
        sd = np.nanstd(M, axis=0)
        self.scale = np.maximum.reduce([(q75 - q25) / 1.349, 0.1 * sd, np.full(M.shape[1], 1e-6)])
        return self

    def __call__(self, M: np.ndarray, zclip: float = 1e3) -> np.ndarray:
        Z = (M - self.center) / self.scale
        return np.clip(np.nan_to_num(Z, nan=0.0), -zclip, zclip)


# ----------------------------------------------------------------------------- classifier
class FDDModel:
    """Gradient-boosted fault classifier with PCA T^2 / SPE monitoring statistics as extra inputs."""

    def __init__(self, max_iter: int = 300, learning_rate: float = 0.08, settle_h: float = 0.25, seed: int = 0):
        self.max_iter, self.learning_rate, self.settle_h, self.seed = max_iter, learning_rate, settle_h, seed
        self.classes = list(CLASSES)
        self.meta: dict = {}

    def _pca_fit(self, M: np.ndarray, names: list[str], frac: float = 0.9) -> None:
        self.pca_cols = [names.index(n) for n in NOVELTY_FEATURES if n in names]
        X = M[:, self.pca_cols]
        self.pca_scale = _RobustScale().fit(X)
        Z = self.pca_scale(X)
        mu = Z.mean(axis=0)
        U, s, Vt = np.linalg.svd(Z - mu, full_matrices=False)
        ev = s ** 2 / max(len(Z) - 1, 1)
        r = int(np.searchsorted(np.cumsum(ev) / ev.sum(), frac) + 1)
        self.pca_mu, self.pca_P, self.pca_ev = mu, Vt[:r].T, ev[:r]

    def _pca_stats(self, M: np.ndarray) -> np.ndarray:
        Z = self.pca_scale(M[:, self.pca_cols]) - self.pca_mu
        S = Z @ self.pca_P
        t2 = (S * S / self.pca_ev).sum(axis=1)
        spe = ((Z - S @ self.pca_P.T) ** 2).sum(axis=1)
        return np.column_stack([np.log1p(t2), np.log1p(spe)])

    def design(self, M: np.ndarray) -> np.ndarray:
        """Full matrix (columns = self.full_names) -> booster input."""
        M = clean(M, self.full_names)
        return np.hstack([M[:, self.cols], self._pca_stats(M)])

    def fit(self, M: np.ndarray, names: list[str], y: np.ndarray, tso: np.ndarray | None = None) -> "FDDModel":
        """Rows within `settle_h` after onset are ambiguous (fault not yet visible) and are not trained on."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
        self.full_names = list(names)
        Mc = clean(M, self.full_names)
        self.cols = [j for j, n in enumerate(names) if n not in EXCLUDE and not np.isnan(Mc[:, j]).all()]
        self.input_names = [names[j] for j in self.cols] + ["pca_T2", "pca_SPE"]
        keep = np.ones(len(y), bool)
        if tso is not None:
            keep &= ~((tso >= 0) & (tso < self.settle_h) & (y != "none"))
        self._pca_fit(Mc[keep & (y == "none")], self.full_names)
        yi = np.array([self.classes.index(c) for c in y])
        self.clf = HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=self.learning_rate,
                                                  l2_regularization=1.0, class_weight="balanced",
                                                  early_stopping=False, random_state=self.seed)
        with threadpool_limits(limits=max_threads(), user_api="openmp"):
            self.clf.fit(self.design(M[keep]), yi[keep])
        return self

    def proba_matrix(self, M: np.ndarray) -> np.ndarray:
        P = np.zeros((len(M), len(self.classes)))
        P[:, self.clf.classes_] = self.clf.predict_proba(self.design(M))
        return P

    def predict_proba(self, f: dict) -> dict[str, float]:
        """Class probabilities for one feature dict (from summary.features, optionally `augment`ed)."""
        x = numeric_features(f)
        p = self.proba_matrix(rows_to_matrix([x], self.full_names))[0]
        return dict(zip(self.classes, map(float, p)))

    def save(self, path: str = MODEL_PATH) -> None:
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(self, fh)
        os.replace(path + ".tmp", path)

    @staticmethod
    def load(path: str = MODEL_PATH) -> "FDDModel":
        with open(path, "rb") as fh:
            return pickle.load(fh)


def grouped_cv(M, names, y, groups, tso, n_splits: int = 5, **kw) -> tuple[dict, np.ndarray]:
    """Scenario-grouped K-fold (passive and post-recovery runs of a scenario share a fold):
    out-of-fold class probabilities and summary metrics."""
    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import GroupKFold
    P = np.zeros((len(y), len(CLASSES)))
    for tr, te in GroupKFold(n_splits).split(M, y, groups):
        m = FDDModel(**kw).fit(M[tr], names, y[tr], tso[tr])
        P[te] = m.proba_matrix(M[te])
    pred = np.array(CLASSES)[P.argmax(1)]
    vis = ~((tso >= 0) & (tso < kw.get("settle_h", 0.25)) & (y != "none"))
    rep = {
        "n_rows": int(len(y)), "n_scenarios": int(len(np.unique(groups))),
        "accuracy": float((pred == y).mean()), "accuracy_visible": float((pred[vis] == y[vis]).mean()),
        "macro_f1": float(f1_score(y, pred, labels=list(CLASSES), average="macro", zero_division=0)),
        "recall": {c: float((pred[y == c] == c).mean()) if (y == c).any() else None for c in CLASSES},
        "confusion": {"labels": list(CLASSES), "matrix": confusion_matrix(y, pred, labels=list(CLASSES)).tolist()},
    }
    return rep, P


def episode_metrics(P: np.ndarray, y: np.ndarray, episodes: np.ndarray, t: np.ndarray, ep_class: np.ndarray,
                    thr: float = 0.6) -> dict:
    """Lookup-level behaviour on passive episodes: first invocation whose top non-none class reaches `thr`."""
    out = {"thr": thr, "fault_eps": 0, "fault_first_correct": 0, "fault_never": 0, "none_eps": 0, "none_false_act": 0}
    for g in np.unique(episodes):
        i = np.where(episodes == g)[0]
        i = i[np.argsort(t[i])]
        top = P[i].argmax(1)
        fire = [k for k in range(len(i)) if top[k] != 0 and P[i[k], top[k]] >= thr]
        cls = ep_class[i[0]]
        if cls == "none":
            out["none_eps"] += 1
            out["none_false_act"] += int(bool(fire))
        else:
            out["fault_eps"] += 1
            if not fire:
                out["fault_never"] += 1
            elif CLASSES[top[fire[0]]] == cls and y[i[fire[0]]] != "none":
                out["fault_first_correct"] += 1
    return out


# ------------------------------------------------------------------------- novelty gate
class NoveltyDetector:
    """Standardised k-NN distance score with a split-conformal threshold (level alpha)."""

    def __init__(self, features: tuple[str, ...] = NOVELTY_FEATURES, k: int = 5, alpha: float = 0.05,
                 calib_frac: float = 0.3, seed: int = 0):
        self.features, self.k, self.alpha, self.calib_frac, self.seed = tuple(features), k, alpha, calib_frac, seed

    def _matrix(self, X, names: list[str] | None) -> np.ndarray:
        if len(X) and isinstance(X[0], dict):
            return rows_to_matrix([numeric_features(x) for x in X], list(self.features))
        X = np.asarray(X, dtype=float)
        if names is None:
            return X
        return X[:, [list(names).index(n) for n in self.features]]

    def fit(self, X, names: list[str] | None = None, groups: np.ndarray | None = None) -> "NoveltyDetector":
        """X: list of feature dicts, or a matrix (columns `names`, or exactly `self.features`).
        Calibration rows come from groups (episodes) disjoint from the reference set."""
        M = self._matrix(X, names)
        rng = np.random.default_rng(self.seed)
        groups = np.arange(len(M)) if groups is None else np.asarray(groups)
        ug = rng.permutation(np.unique(groups))
        cal_g = set(ug[: max(1, int(round(self.calib_frac * len(ug))))].tolist())
        cal = np.array([g in cal_g for g in groups])
        self.scaler = _RobustScale().fit(M[~cal])
        self.ref = self.scaler(M[~cal])
        self.cal_scores = np.sort(self._scores_z(self.scaler(M[cal])))
        self.set_alpha(self.alpha)
        return self

    def set_alpha(self, alpha: float) -> None:
        """Conformal quantile: the ceil((n+1)(1-alpha))-th smallest calibration score."""
        n = len(self.cal_scores)
        r = math.ceil((n + 1) * (1.0 - alpha))
        self.alpha = alpha
        self.threshold = float(self.cal_scores[r - 1]) if r <= n else float("inf")

    def _scores_z(self, Z: np.ndarray) -> np.ndarray:
        out = np.empty(len(Z))
        for i, z in enumerate(Z):
            d = np.sqrt(((self.ref - z) ** 2).sum(axis=1))
            out[i] = np.partition(d, self.k - 1)[: self.k].mean()
        return out

    def scores(self, X, names: list[str] | None = None) -> np.ndarray:
        return self._scores_z(self.scaler(self._matrix(X, names)))

    def score(self, x: dict) -> float:
        return float(self.scores([x])[0])

    def is_novel(self, x: dict) -> bool:
        return self.score(x) > self.threshold

    def p_value(self, x: dict) -> float:
        """Conformal p-value of a row (small = unusual)."""
        s = self.score(x)
        return float((1 + (self.cal_scores >= s).sum()) / (len(self.cal_scores) + 1))

    def save(self, path: str = NOVELTY_PATH) -> None:
        with open(path + ".tmp", "wb") as fh:
            pickle.dump(self, fh)
        os.replace(path + ".tmp", path)

    @staticmethod
    def load(path: str = NOVELTY_PATH) -> "NoveltyDetector":
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ------------------------------------------------------------------------------ training
def load_features(path: str = FEATURES_NPZ) -> dict:
    d = np.load(path, allow_pickle=False)
    with open(path.replace(".npz", ".json")) as fh:
        meta = json.load(fh)
    return {k: d[k] for k in d.files} | {"names": meta["names"], "fver": meta.get("fver")}


def check_version(fver: str | None, what: str) -> None:
    """Warn when `what` was built with a different feature extractor than the current code."""
    if fver != features_version():
        warnings.warn(f"{what} was built with features version {fver}, current is {features_version()}: "
                      "regenerate it (experiments/fdd_data.py, python -m cod.fdd train)", stacklevel=2)


def novelty_report(det: NoveltyDetector, M, names, episodes, tso) -> dict:
    s = det.scores(M, names)
    flag = s > det.threshold
    post = np.isfinite(tso) & (tso >= 0)
    eps = np.unique(episodes)
    any_flag = [flag[episodes == g].any() for g in eps]
    any_post = [flag[(episodes == g) & post].any() for g in eps if ((episodes == g) & post).any()]
    return {"rows": int(len(s)), "episodes": int(len(eps)), "row_rate": float(flag.mean()),
            "row_rate_pre_onset": float(flag[~post].mean()) if (~post).any() else None,
            "row_rate_post_onset": float(flag[post].mean()) if post.any() else None,
            "episode_any_flag": float(np.mean(any_flag)),
            "episode_any_flag_post_onset": float(np.mean(any_post)) if any_post else None}


def train(features_path: str = FEATURES_NPZ, heldout_path: str | None = None, seed: int = 0,
          test_frac: float = 0.2) -> dict:
    """Grouped CV report, final FDD model on all rows, novelty detector fit on (1 - test_frac) of the scenarios."""
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=max_threads(), user_api="openmp"):
        return _train(features_path, heldout_path, seed, test_frac)


def _train(features_path: str, heldout_path: str | None, seed: int, test_frac: float) -> dict:
    D = load_features(features_path)
    check_version(D["fver"], features_path)
    names, M, y, groups, tso, t = D["names"], D["X"], D["y"].astype(str), D["group"], D["tso"], D["t"]
    ep_class, episodes, active = D["ep_class"].astype(str), D["episode"], D["active"].astype(bool)
    rep, P = grouped_cv(M, names, y, groups, tso)
    pred = np.array(CLASSES)[P.argmax(1)]
    rep["accuracy_passive"] = float((pred[~active] == y[~active]).mean())
    rep["accuracy_post_recovery"] = float((pred[active] == y[active]).mean()) if active.any() else None
    pa = ~active
    rep["episode_lookup"] = {f"thr={thr}": episode_metrics(P[pa], y[pa], episodes[pa], t[pa], ep_class[pa], thr)
                             for thr in (0.5, 0.6, 0.8)}
    model = FDDModel(seed=seed).fit(M, names, y, tso)
    model.meta = {"cv": rep, "features_path": features_path, "features_version": D["fver"]}
    model.save()
    rng = np.random.default_rng(seed)
    ug = rng.permutation(np.unique(groups))
    test_g = set(ug[: int(round(test_frac * len(ug)))].tolist())
    te = np.array([g in test_g for g in groups])
    det = NoveltyDetector(seed=seed).fit(M[~te], names, groups[~te])
    nov = {"threshold": det.threshold, "alpha": det.alpha, "n_ref": int(len(det.ref)), "n_cal": int(len(det.cal_scores)),
           "heldout_synthesis": novelty_report(det, M[te], names, episodes[te], tso[te]),
           "heldout_synthesis_passive": novelty_report(det, M[te & pa], names, episodes[te & pa], tso[te & pa])}
    if heldout_path and os.path.exists(heldout_path):
        H = load_features(heldout_path)
        check_version(H["fver"], heldout_path)
        fam = H["family"].astype(str)
        for f in sorted(set(fam)):
            m = fam == f
            nov[f"heldout_family_{f}"] = novelty_report(det, H["X"][m], H["names"], H["episode"][m], H["tso"][m])
    det.meta = nov | {"features_version": D["fver"]}
    det.save()
    return {"fdd": rep, "novelty": nov}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"])
    ap.add_argument("--features", default=FEATURES_NPZ)
    ap.add_argument("--heldout", default=os.path.join(DATA_DIR, "fdd_heldout_VC.npz"))
    a = ap.parse_args()
    from cod import fdd as _fdd  # pickle the classes under cod.fdd, not __main__
    r = _fdd.train(a.features, a.heldout)
    with open(os.path.join(DATA_DIR, "fdd_report.json"), "w") as fh:
        json.dump(r, fh, indent=1)
    print(json.dumps(r, indent=1))
