"""Tests for cod.analysis: known values (statsmodels/scipy, Newcombe 1998 Table III) and properties."""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from statsmodels.stats.contingency_tables import cochrans_q as sm_cochrans_q
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportion_confint

from cod.analysis import load as L
from cod.analysis.stats import (clopper_pearson, cluster_bootstrap_ci, cochran_q, continuous_vs, holm, mcnemar_exact,
                                newcombe_paired_counts, newcombe_paired_diff_ci, paired_wide, pareto_front, success_vs,
                                summary_table, wilcoxon_paired, wilson_ci)

KN = [(0, 10), (1, 10), (5, 10), (10, 10), (3, 37), (36, 37), (0, 1), (1, 1), (17, 250)]


@pytest.mark.parametrize("k,n", KN)
@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_wilson_and_clopper_pearson_match_statsmodels(k, n, alpha):
    assert np.allclose(wilson_ci(k, n, alpha), proportion_confint(k, n, alpha, method="wilson"), atol=1e-12)
    assert np.allclose(clopper_pearson(k, n, alpha), proportion_confint(k, n, alpha, method="beta"), atol=1e-12)


def test_clopper_pearson_contains_wilson_point():
    for k, n in KN:
        lo, hi = clopper_pearson(k, n)
        assert 0 <= lo <= k / n <= hi <= 1


def _pairs(rng, n, p1, p2, rho):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return z[:, 0] < sps.norm.ppf(p1), z[:, 1] < sps.norm.ppf(p2)


def test_mcnemar_matches_statsmodels_exact():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(5, 80))
        a, b = _pairs(rng, n, rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(-0.5, 0.9))
        bb, c, p = mcnemar_exact(a, b)
        tab = [[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]]
        ref = sm_mcnemar(tab, exact=True)
        assert (bb, c) == (tab[0][1], tab[1][0])
        assert p == pytest.approx(ref.pvalue, abs=1e-12)
    assert mcnemar_exact([1, 1, 0], [1, 1, 0]) == (0, 0, 1.0)
    assert mcnemar_exact([1] * 6, [0] * 6)[2] == pytest.approx(2 * 0.5 ** 6)


# Newcombe (1998) Stat Med 17:2635, Table III: (e, f, g, h) -> method 8 and method 10 limits (4 dp)
TABLE_III = [
    ((36, 12, 2, 0), (0.0569, 0.3404), (0.0569, 0.3404)),
    ((20, 12, 2, 16), (0.0618, 0.3242), (0.0562, 0.3292)),
    ((18, 12, 2, 18), (0.0618, 0.3239), (0.0562, 0.3290)),
    ((36, 14, 0, 0), (0.1528, 0.4167), (0.1528, 0.4167)),
    ((35, 14, 0, 1), (0.1573, 0.4149), (0.1461, 0.4175)),
    ((18, 14, 0, 18), (0.1504, 0.3910), (0.1441, 0.3963)),
    ((2, 97, 1, 0), (0.8721, 0.9854), (0.8721, 0.9854)),
    # the paper prints 0.8736 for method 10 here, but eh < fg so method 10 equals method 8 (0.8737)
    ((1, 97, 1, 1), (0.8737, 0.9850), (0.8737, 0.9850)),
    ((0, 29, 1, 0), (0.6666, 0.9882), (0.6666, 0.9882)),
    ((2, 98, 0, 0), (0.9178, 0.9945), (0.9178, 0.9945)),
    ((1, 98, 0, 1), (0.9174, 0.9916), (0.9171, 0.9916)),
    ((0, 30, 0, 0), (0.8395, 1.0), (0.8395, 1.0)),
    ((54, 0, 0, 0), (-0.0664, 0.0664), (-0.0664, 0.0664)),
    ((53, 0, 0, 1), (-0.0640, 0.0640), (-0.0729, 0.0729)),
    ((30, 0, 0, 24), (-0.0074, 0.0074), (-0.0358, 0.0358)),
    ((29, 0, 0, 25), (-0.0049, 0.0049), (-0.0354, 0.0354)),
    ((28, 0, 0, 26), (-0.0025, 0.0025), (-0.0352, 0.0352)),
    ((27, 0, 0, 27), (0.0, 0.0), (-0.0351, 0.0351)),
]


@pytest.mark.parametrize("cells,m8,m10", TABLE_III)
def test_newcombe_paper_table_iii(cells, m8, m10):
    e, f, g, h = cells
    d, lo, hi = newcombe_paired_counts(e, f, g, h, phi_cc=True)
    assert d == pytest.approx((f - g) / (e + f + g + h))
    assert (lo, hi) == pytest.approx(m10, abs=6e-5)
    _, lo8, hi8 = newcombe_paired_counts(e, f, g, h, phi_cc=False)
    assert (lo8, hi8) == pytest.approx(m8, abs=6e-5)


def test_newcombe_phi_zero_reduces_to_unpaired_newcombe():
    # eh == fg -> phi = 0 -> the hybrid score interval for two independent samples of size n
    for e, f, g, h in [(4, 2, 2, 1), (6, 3, 2, 1), (10, 5, 4, 2), (9, 3, 6, 2)]:
        n = e + f + g + h
        _, lo, hi = newcombe_paired_counts(e, f, g, h)
        ref = confint_proportions_2indep(e + f, n, e + g, n, method="newcomb", compare="diff")
        assert (lo, hi) == pytest.approx(ref, abs=1e-10)


def test_newcombe_arrays_properties():
    rng = np.random.default_rng(2)
    for _ in range(200):
        n = int(rng.integers(1, 60))
        a, b = _pairs(rng, n, rng.uniform(0.05, 0.95), rng.uniform(0.05, 0.95), rng.uniform(-0.3, 0.95))
        d, lo, hi = newcombe_paired_diff_ci(a, b)
        assert d == pytest.approx(a.mean() - b.mean())
        assert -1 <= lo <= d <= hi <= 1
        d2, lo2, hi2 = newcombe_paired_diff_ci(b, a)
        assert (d2, lo2, hi2) == pytest.approx((-d, -hi, -lo), abs=1e-12)


def test_newcombe_coverage_monte_carlo():
    rng = np.random.default_rng(3)
    p1, p2, rho, n, reps = 0.7, 0.5, 0.4, 40, 1500
    truth = p1 - p2  # exact: the latent-normal construction preserves the marginals
    hits = 0
    for _ in range(reps):
        a, b = _pairs(rng, n, p1, p2, rho)
        _, lo, hi = newcombe_paired_diff_ci(a, b)
        hits += lo <= truth <= hi
    assert 0.93 <= hits / reps <= 0.99


def test_cochran_q_matches_statsmodels_and_mcnemar():
    rng = np.random.default_rng(4)
    for k in (2, 3, 5):
        x = (rng.random((60, k)) < rng.uniform(0.2, 0.8, size=k)).astype(int)
        q, df, p = cochran_q(x)
        ref = sm_cochrans_q(x)
        assert q == pytest.approx(ref.statistic) and df == ref.df and p == pytest.approx(ref.pvalue)
        if k == 2:
            b, c, _ = mcnemar_exact(x[:, 0], x[:, 1])
            assert q == pytest.approx((b - c) ** 2 / (b + c))
    assert cochran_q(np.ones((10, 3))) == (0.0, 2, 1.0)


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(5)
    for m in (1, 2, 5, 13):
        p = rng.random(m) ** 3
        assert np.allclose(holm(p), multipletests(p, method="holm")[1])
    p = np.array([0.01, np.nan, 0.04, 0.03])
    out = holm(p)
    assert np.isnan(out[1])
    assert np.allclose(out[[0, 2, 3]], multipletests(p[[0, 2, 3]], method="holm")[1])


def test_wilcoxon_matches_scipy_and_drops_nan():
    rng = np.random.default_rng(6)
    x = rng.normal(size=30)
    y = x + rng.normal(0.3, 1.0, size=30)
    y[:4] = x[:4]  # zero differences
    st, p = wilcoxon_paired(x, y)
    ref = sps.wilcoxon(x, y, zero_method="zsplit")
    assert (st, p) == pytest.approx((ref.statistic, ref.pvalue))
    x2, y2 = np.r_[x, np.nan], np.r_[y, 1.0]
    assert wilcoxon_paired(x2, y2) == pytest.approx((st, p))


def _paired_frame(rng, n_sc=40, n_seed=3, arms=("A", "B"), sd_sc=1.0):
    rows = []
    for s in range(n_sc):
        u = rng.normal(0, sd_sc)
        for seed in range(n_seed):
            e = rng.normal()
            for j, a in enumerate(arms):
                rows.append({"suite": "x", "scenario": f"s{s}", "seed": seed, "arm": a, "v": u + e + 0.1 * j,
                             "success": bool(u + e + 0.3 * j > 0)})
    return pd.DataFrame(rows)


def _diff_by_pivot(d):
    w = d.pivot(index=["scenario", "seed"], columns="arm", values="v")
    return float((w["B"] - w["A"]).mean())


def test_cluster_bootstrap_properties():
    rng = np.random.default_rng(7)
    df = _paired_frame(rng)
    mean_v = lambda d: float(d["v"].mean())
    r1 = cluster_bootstrap_ci(df, mean_v, B=400, seed=1)
    assert r1 == cluster_bootstrap_ci(df, mean_v, B=400, seed=1)
    assert r1[1] < r1[0] < r1[2]
    # paired contrast where both arms differ by a constant: the pairing is kept, so the CI is degenerate
    est, lo, hi = cluster_bootstrap_ci(df, _diff_by_pivot, B=300, seed=2)
    assert est == pytest.approx(0.1) and lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
    # one-stage cluster bootstrap also works
    assert cluster_bootstrap_ci(df, mean_v, inner=None, B=200)[1] < r1[0]


def test_cluster_bootstrap_width_matches_cluster_se():
    rng = np.random.default_rng(8)
    df = _paired_frame(rng, n_sc=150, n_seed=3, arms=("A",), sd_sc=2.0)
    est, lo, hi = cluster_bootstrap_ci(df, lambda d: float(d["v"].mean()), B=2000, seed=3)
    se = df.groupby("scenario")["v"].mean().std(ddof=1) / np.sqrt(150)
    assert (hi - lo) == pytest.approx(2 * 1.96 * se, rel=0.2)
    # an i.i.d. bootstrap ignoring the clustering would be clearly narrower
    iid_se = df["v"].std(ddof=1) / np.sqrt(len(df))
    assert (hi - lo) > 1.3 * 2 * 1.96 * iid_se


def test_pareto_front():
    pts = [(0, 0.5), (1, 0.7), (1, 0.6), (2, 0.7), (3, 0.9), (0.5, 0.4), (3, 0.9)]
    assert pareto_front(pts) == [0, 1, 4, 6]
    assert pareto_front([]) == []
    assert pareto_front([(1, 1)]) == [0]


def test_tables_on_tidy_frame():
    rng = np.random.default_rng(9)
    df = _paired_frame(rng, arms=("A", "B", "C"))
    df = df.drop(df.index[(df.arm == "C") & (df.scenario == "s0")])  # C misses one scenario
    t = summary_table(df, ["A", "B", "C"])
    assert list(t["n"]) == [117, 117, 117]
    a = df[(df.arm == "A") & (df.scenario != "s0")]["success"]
    assert t.loc[0, "rate"] == pytest.approx(a.mean())
    assert (t.loc[0, "lo"], t.loc[0, "hi"]) == pytest.approx(wilson_ci(int(a.sum()), len(a)))
    assert list(summary_table(df, ["A", "C"], paired=False)["n"]) == [120, 117]
    s = success_vs(df, "A", ["B", "C"])
    assert list(s["n"]) == [120, 117]
    w = paired_wide(df, ["B", "A"])
    assert tuple(s.loc[0, ["b", "c", "p"]]) == mcnemar_exact(w["B"], w["A"])
    assert tuple(s.loc[0, ["diff", "lo", "hi"]]) == pytest.approx(newcombe_paired_diff_ci(w["B"], w["A"]))
    assert np.allclose(s["p_holm"], holm(s["p"]))
    c = continuous_vs(df, "A", ["B"], "v")
    assert c.loc[0, "median_diff"] == pytest.approx(0.1)
    assert summary_table(df, ["A"], metric="v").loc[0, "n"] == 120
    with pytest.raises(ValueError):
        paired_wide(pd.concat([df, df.iloc[:1]]), ["A", "B"])


# ---------------------------------------------------------------- loader
RES = L.RESULTS


def _first(split, fam):
    p = os.path.join(RES, split, fam, "0000.json")
    if not os.path.exists(p):
        pytest.skip(f"no screen record {p}")
    return p


def _episode(sc: dict, seed: int, arm: str, suite: str, stabilized=True, g_iae=50.0) -> dict:
    return {"stabilized": stabilized, "tripped": not stabilized, "trip_msg": "", "trip_time": None if stabilized else 7.5,
            "shutdown": False, "cost": 1000.0, "production": 900.0, "g_iae": g_iae, "offspec_h": 0.0,
            "scenario": sc["id"], "family": sc["family"], "seed": seed, "calls": 3, "n_accepted": 1,
            "prompt_tokens": 100, "completion_tokens": 20, "llm_calls": 1, "gpu_s": 0.5, "wall_s": 2.0,
            "diag_correct_any": None, "arm": arm, "suite": suite, "scenario_def": sc}


def _write(root, arm, suite, rec):
    d = os.path.join(root, "arms", L.arm_dir_name(arm), suite)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{rec['scenario']}__s{rec['seed']}.json"), "w") as f:
        json.dump(rec, f)


def test_load_episodes_label_join(tmp_path):
    root = str(tmp_path)
    for split in ("screen", "screen_101"):
        for fam in ("Q", "S"):
            src = _first(split, fam)
            os.makedirs(os.path.join(root, split, fam), exist_ok=True)
            shutil.copy(src, os.path.join(root, split, fam, "0000.json"))
    test_q = json.load(open(os.path.join(root, "screen_101", "Q", "0000.json")))
    train_q = json.load(open(os.path.join(root, "screen", "Q", "0000.json")))
    train_s = json.load(open(os.path.join(root, "screen", "S", "0000.json")))
    assert test_q["scenario"]["id"] == train_q["scenario"]["id"]  # ids collide across splits
    from cod.scenarios import ALHAZMI13
    q1 = next(s for s in ALHAZMI13 if s.id.startswith("Q1")).as_dict()
    arm = "online:qwen2.5:7b@3000"
    _write(root, arm, "test101", _episode(test_q["scenario"], test_q["seed"], arm, "test101", g_iae=400.0))
    _write(root, arm, "test101", _episode(test_q["scenario"], 5, arm, "test101", g_iae=100.0))
    _write(root, arm, "train", _episode(train_s["scenario"], train_s["seed"], arm, "train", stabilized=False))
    _write(root, arm, "alhazmi", _episode(json.loads(json.dumps(q1)), 1, arm, "alhazmi", g_iae=151.0))
    _write(root, "weak", "alhazmi", _episode(json.loads(json.dumps(q1)), 1, "weak", "alhazmi", g_iae=20.0))
    df = L.load_episodes(root=root, suites_dir=str(tmp_path / "none"))
    assert list(df.columns) == L.COLUMNS and len(df) == 5
    r = df.set_index(["arm", "suite", "seed"])
    t = r.loc[(arm, "test101", test_q["seed"])]
    assert (t["label"], t["label_src"], bool(t["label_seed_match"]), bool(t["success"])) == \
        ("quality_gap", "screen_101", True, False)
    t5 = r.loc[(arm, "test101", 5)]
    assert (t5["label"], t5["label_src"], bool(t5["label_seed_match"]), bool(t5["success"])) == \
        ("quality_gap", "screen_101", False, True)
    tr = r.loc[(arm, "train", train_s["seed"])]
    assert tr["label_src"] == "screen" and tr["label"] == train_s["label"] and not tr["success"]
    assert tr["trip_time"] == 7.5 and tr["fail_time"] == 7.5
    assert t["fail_time"] == test_q["scenario"].get("horizon_h", 48.0) and np.isnan(t["trip_time"])
    al = r.loc[(arm, "alhazmi", 1)]
    assert al["label"] == "quality_gap" and not al["success"] and al["tokens"] == 120
    assert bool(r.loc[("weak", "alhazmi", 1), "success"])
    assert df["diag_correct_any"].isna().all()
    assert len(L.load_episodes([arm], ["alhazmi"], root=root)) == 1
    assert len(L.load_episodes(["weak", "nonexistent"], root=root)) == 1
    assert len(L.load_episodes("weak", "alhazmi", root=root)) == 1  # plain strings, not character sets
    # explicit suite-file labels take precedence
    sdir = tmp_path / "suites"
    sdir.mkdir()
    (sdir / "test101.json").write_text(json.dumps([{"scenario": test_q["scenario"], "seeds": [5], "label": "benign"}]))
    L.suite_labels.cache_clear()
    df2 = L.load_episodes(root=root, suites_dir=str(sdir), suites=["test101"])
    assert set(df2["label"]) == {"benign"} and set(df2["label_src"]) == {"suite_file"} and df2["success"].all()
    assert dict(zip(df2["seed"], df2["label_seed_match"])) == {5: True, test_q["seed"]: False}


def test_arm_dir_name_matches_run_arms():
    from experiments.run_arms import arm_dir
    for spec in ("weak", "online:qwen2.5:7b@3000", "playbook:/a/b/pb.py@x", "hybrid:" + "p" * 200 + ":m@1"):
        assert L.arm_dir_name(spec) == os.path.basename(arm_dir(spec))


def test_load_real_alhazmi_results_if_present():
    if not os.path.isdir(os.path.join(RES, "arms", "weak", "alhazmi")):
        pytest.skip("no example run")
    df = L.load_episodes(["weak", "strong"], ["alhazmi"])
    assert len(df) and set(df["label_src"]) == {"alhazmi"}
    q = df[df.family == "Q"]
    assert (q["success"] == (q["stabilized"] & (q["g_iae"] <= 150.0))).all()
    nq = df[df.family != "Q"]
    assert (nq["success"] == nq["stabilized"]).all()
    w = paired_wide(df, ["weak", "strong"])
    assert 0 < len(w) <= min((df.arm == "weak").sum(), (df.arm == "strong").sum())
