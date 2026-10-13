"""FDD dataset labelling, B3 FDDLookup supervisor and the conformal novelty detector.

Run: cd <repo> && PYTHONPATH=c10_compile_or_deliberate/code .venv/bin/python -m pytest c10_compile_or_deliberate/code/tests/test_fdd.py
"""
from __future__ import annotations

import dataclasses
import os
import sys

import numpy as np
import pytest

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "experiments"))

import fdd_data  # noqa: E402
from cod.actions import DIAGNOSES  # noqa: E402
from cod.fdd import (CLASS_DIAGNOSIS, CLASSES, MODEL_PATH, NOVELTY_FEATURES, FDDModel, NoveltyDetector,  # noqa: E402
                     fault_class, features_version, load_features, row_label)
from cod.runner import run_episode  # noqa: E402
from cod.scenarios import ALHAZMI13, FRONTIER5, sample  # noqa: E402
from cod.supervisors.fdd import LOOKUP, FDDLookup, parse_spec  # noqa: E402


def short(sc, h=12.0):
    return dataclasses.replace(sc, horizon_h=h)


def test_labels_and_vocabulary():
    rng = np.random.default_rng(0)
    for fam, ok in (("A", {"none"}), ("S", {"composition"}), ("P", {"feed_loss"}), ("Q", {"grade"}),
                    ("SN", {"sensor_P", "sensor_T"})):
        sc = sample(fam, rng, onset=3.0)
        assert fault_class(sc) in ok
        assert row_label(sc, 2.9) == "none"
        assert row_label(sc, 3.1) == fault_class(sc)
    assert fault_class(sample("V", rng)) is None
    assert set(CLASS_DIAGNOSIS.values()) <= set(DIAGNOSES)
    assert set(LOOKUP) == set(CLASSES) - {"none"}
    assert parse_spec("fdd:@thr=0.7@persist=2") == (MODEL_PATH, {"thr": 0.7, "persist": 2.0})
    for bad in ("fdd:@threshold=0.7", "fdd:@thr", "fdd:@thr=x"):
        with pytest.raises(ValueError):
            parse_spec(bad)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """Small dataset: short episodes of every synthesis family (passive + one active pass) and no-fault runs."""
    d = tmp_path_factory.mktemp("fdd")
    rng = np.random.default_rng(5)
    paths = []
    for fam in ("A", "S", "P", "Q", "SN", "SN", "SN"):
        for i in range(2):
            sc = sample(fam, rng, onset=2.5, idx=i, horizon=10.0)
            for active in ((False, True) if fam != "A" and i == 0 else (False,)):
                p = str(d / f"{fam}_{i}_{len(paths)}_{int(active)}.json")
                fdd_data._dump(p, fdd_data.record(sc, 300 + len(paths), active))
                paths.append(p)
    for i in range(2):
        p = str(d / f"N_{i}.json")
        fdd_data._dump(p, fdd_data.record(short(fdd_data.no_fault(rng, i), 10.0), 400 + i))
        paths.append(p)
    out = str(d / "tiny.npz")
    summary = fdd_data.assemble(paths, out)
    return d, summary, load_features(out)


def test_dataset_and_classifier(tiny):
    d, summary, D = tiny
    assert summary["rows"] == len(D["y"]) > 40 and summary["features"] > 150
    assert D["fver"] == features_version()
    for k in NOVELTY_FEATURES + ("T_soft", "T_resid_mvf"):  # finite on every row (no extractor gaps)
        assert np.isfinite(D["X"][:, D["names"].index(k)]).all(), k
    assert {"none", "composition", "feed_loss", "grade"} <= set(D["y"].tolist())
    assert D["active"].any() and (~D["active"]).any()
    pre = np.isfinite(D["tso"]) & (D["tso"] < 0)
    assert (D["y"][pre] == "none").all()
    m = FDDModel(max_iter=30).fit(D["X"], D["names"], D["y"].astype(str), D["tso"])
    P = m.proba_matrix(D["X"])
    assert P.shape == (len(D["y"]), len(CLASSES)) and np.allclose(P.sum(1), 1.0)
    assert (np.array(CLASSES)[P.argmax(1)] == D["y"]).mean() > 0.8  # training fit
    m.save(str(d / "m.pkl"))
    m2 = FDDModel.load(str(d / "m.pkl"))
    assert np.allclose(m2.proba_matrix(D["X"][:5]), P[:5])


def test_novelty_conformal_gaussian():
    rng = np.random.default_rng(1)
    names = [f"x{i}" for i in range(8)]
    X = rng.normal(size=(3000, 8))
    det = NoveltyDetector(features=tuple(names), k=5, alpha=0.05).fit(X, names, groups=np.arange(3000) // 10)
    fresh = det.scores(rng.normal(size=(4000, 8)), names) > det.threshold
    shifted = det.scores(rng.normal(size=(500, 8)) + 3.0, names) > det.threshold
    assert abs(fresh.mean() - 0.05) < 0.03
    assert shifted.mean() > 0.9
    assert det.is_novel({n: 5.0 for n in names}) and not det.is_novel({n: 0.0 for n in names})


def test_novelty_on_features(tiny):
    d, _, D = tiny
    det = NoveltyDetector().fit(D["X"], D["names"], D["group"])
    assert np.isfinite(det.threshold) and 0 < det.threshold
    row = dict(zip(D["names"], D["X"][0]))
    assert isinstance(det.score(row), float) and isinstance(det.is_novel(row), bool)
    far = dict(row, P_dev_nom=row["P_dev_nom"] + 5000.0)
    assert det.is_novel(far)
    det.save(str(d / "nov.pkl"))
    assert NoveltyDetector.load(str(d / "nov.pkl")).score(row) == pytest.approx(det.score(row))


def _check_episode(out):
    assert out["decisions"]
    for r in out["decisions"]:
        assert r["source"].startswith("fdd:") and r["diagnosis"] in DIAGNOSES
        assert sum(r["meta"]["proba"].values()) == pytest.approx(1.0, abs=0.01)
        if r["accepted"]:
            assert r["action"] in [a.as_dict() for a in LOOKUP.values()]


def test_fdd_lookup_runs_with_tiny_model(tiny):
    d, _, D = tiny
    m = FDDModel(max_iter=30).fit(D["X"], D["names"], D["y"].astype(str), D["tso"])
    for sc in (short(FRONTIER5[0]), short(ALHAZMI13[5])):
        _check_episode(run_episode(sc, 1, FDDLookup("fdd:@thr=0.6", model=m)))


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="run `python -m cod.fdd train` first")
def test_fdd_lookup_trained_model():
    """F1 (pressure transmitter drift) -> substitute reactor_pressure; S1 (IDV1) -> arm composition trims."""
    sys.path.insert(0, os.path.join(CODE, "experiments"))
    from run_arms import build_arm
    sup, base = build_arm("fdd:@thr=0.6")
    assert base == "weak" and isinstance(sup, FDDLookup) and sup.thr == 0.6
    for sc, action, diag in ((FRONTIER5[0], LOOKUP["sensor_P"], "sensor_fault"),
                             (ALHAZMI13[5], LOOKUP["composition"], "feed_composition_upset")):
        out = run_episode(short(sc), 1, sup)
        _check_episode(out)
        acc = [r for r in out["decisions"] if r["accepted"]]
        assert acc and acc[0]["action"] == action.as_dict() and acc[0]["diagnosis"] == diag
        assert out["stabilized"]
