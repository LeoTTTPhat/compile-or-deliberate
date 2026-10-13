"""Paired statistics for arm comparisons (DESIGN.md section 6).

Every comparison is paired by (suite, scenario, seed): all arms see the same scenario and the
same measurement-noise seed. Binary endpoints use the exact McNemar test with Newcombe's
score-based interval for the paired difference; k arms at once use Cochran's Q; continuous
endpoints use the Wilcoxon signed-rank test; families of tests are Holm-adjusted. Uncertainty
that must respect the scenario/seed hierarchy uses a two-stage cluster bootstrap.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ("suite", "scenario", "seed")


def _z(alpha: float) -> float:
    return float(stats.norm.ppf(1.0 - alpha / 2.0))


def _binary(x) -> np.ndarray:
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    a = a.astype(float)
    if not np.all((a == 0) | (a == 1)):
        raise ValueError("expected 0/1 values")
    return a.astype(bool)


# ---------------------------------------------------------------- single proportions
def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval (no continuity correction); (nan, nan) when n = 0."""
    if n <= 0:
        return float("nan"), float("nan")
    z = _z(alpha)
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """'Exact' Clopper-Pearson interval from beta quantiles (used for the coverage bounds)."""
    if n <= 0:
        return float("nan"), float("nan")
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------- paired binary
def mcnemar_exact(a_success, b_success) -> tuple[int, int, float]:
    """Exact two-sided McNemar test on paired 0/1 outcomes.

    b = #(a succeeds, b fails), c = #(a fails, b succeeds); p = min(1, 2 P[Bin(b+c, 1/2) <= min(b, c)]).
    """
    a, bb = _binary(a_success), _binary(b_success)
    if a.shape != bb.shape:
        raise ValueError("paired arrays differ in length")
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(min(1.0, 2.0 * stats.binom.cdf(min(b, c), b + c, 0.5)))


def newcombe_paired_counts(e: int, f: int, g: int, h: int, alpha: float = 0.05,
                           phi_cc: bool = True) -> tuple[float, float, float]:
    """Newcombe (1998) score-based interval for a paired difference of proportions.

    Cells: e = (+,+), f = (+,-), g = (-,+), h = (-,-); theta = (f - g)/n = p1 - p2. Wilson
    intervals (l2, u2) for p1 = (e+f)/n and (l3, u3) for p2 = (e+g)/n are combined as
      L = theta - sqrt(dl2^2 - 2 phi dl2 du3 + du3^2),  U = theta + sqrt(du2^2 - 2 phi du2 dl3 + dl3^2),
    with phi = (eh - fg)/sqrt((e+f)(g+h)(e+g)(f+h)) (0 if the denominator is 0). phi_cc=True is
    method 10 of the paper (numerator of phi replaced by max(eh - fg - n/2, 0) when eh > fg);
    phi_cc=False is method 8.
    """
    n = e + f + g + h
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p1, p2 = (e + f) / n, (e + g) / n
    l2, u2 = wilson_ci(e + f, n, alpha)
    l3, u3 = wilson_ci(e + g, n, alpha)
    num = float(e * h - f * g)
    if phi_cc and num > 0:
        num = max(num - n / 2.0, 0.0)
    den = float(e + f) * (g + h) * (e + g) * (f + h)
    phi = num / np.sqrt(den) if den > 0 else 0.0
    dl2, du2, dl3, du3 = p1 - l2, u2 - p1, p2 - l3, u3 - p2
    delta = np.sqrt(max(dl2 ** 2 - 2 * phi * dl2 * du3 + du3 ** 2, 0.0))
    eps = np.sqrt(max(du2 ** 2 - 2 * phi * du2 * dl3 + dl3 ** 2, 0.0))
    d = p1 - p2
    return float(d), float(max(-1.0, d - delta)), float(min(1.0, d + eps))


def newcombe_paired_diff_ci(a, b, alpha: float = 0.05, phi_cc: bool = True) -> tuple[float, float, float]:
    """(diff, lo, hi) for mean(a) - mean(b) on paired 0/1 outcomes (Newcombe 1998, method 10)."""
    a, b = _binary(a), _binary(b)
    if a.shape != b.shape:
        raise ValueError("paired arrays differ in length")
    return newcombe_paired_counts(int(np.sum(a & b)), int(np.sum(a & ~b)), int(np.sum(~a & b)),
                                  int(np.sum(~a & ~b)), alpha, phi_cc)


def cochran_q(x) -> tuple[float, int, float]:
    """Cochran's Q for k related 0/1 samples (rows = episodes, cols = arms); chi-square(k-1) p-value."""
    m = np.asarray(x, dtype=float)
    if m.ndim != 2 or m.shape[1] < 2:
        raise ValueError("need an (n, k>=2) matrix")
    _binary(m.ravel())
    k = m.shape[1]
    col, row = m.sum(axis=0), m.sum(axis=1)
    tot = row.sum()
    den = k * tot - np.sum(row ** 2)
    if den <= 0:
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * np.sum(col ** 2) - tot ** 2) / den
    return float(q), k - 1, float(stats.chi2.sf(q, k - 1))


# ---------------------------------------------------------------- multiplicity, continuous
def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (same order as input; NaNs are left out of the family)."""
    p = np.asarray(pvals, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="mergesort")]
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out[order] = adj
    return out


def wilcoxon_paired(x, y) -> tuple[float, float]:
    """Wilcoxon signed-rank test on paired values (zero differences split, Pratt-like 'zsplit').

    Pairs with a non-finite member are dropped; (nan, nan) if fewer than one pair remains.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 1:
        return float("nan"), float("nan")
    r = stats.wilcoxon(x[ok], y[ok], zero_method="zsplit")
    return float(r.statistic), float(r.pvalue)


# ---------------------------------------------------------------- bootstrap
def cluster_bootstrap_ci(df: pd.DataFrame, value_fn: Callable[[pd.DataFrame], float], cluster: str = "scenario",
                         inner: str | None = "seed", B: int = 10000, alpha: float = 0.05,
                         seed: int = 0) -> tuple[float, float, float]:
    """Two-stage percentile bootstrap: (value_fn(df), lo, hi).

    Each replicate draws clusters with replacement, then, inside each drawn cluster, its inner
    units with replacement. All rows of a drawn unit are kept together (e.g. every arm's
    episode of one (scenario, seed)), so paired contrasts stay paired. In a replicate the
    `cluster` and `inner` columns are relabelled with integer draw ids, so repeated draws stay
    distinct units for value_fn (e.g. when it pivots on them). inner=None gives a one-stage
    cluster bootstrap.
    """
    base = df.reset_index(drop=True)
    est = float(value_fn(base))
    if base.empty:
        return est, float("nan"), float("nan")
    cl = base.groupby(cluster, sort=False, dropna=False).ngroup().to_numpy()
    un = base.groupby([cluster, inner], sort=False, dropna=False).ngroup().to_numpy() if inner else cl
    order = np.lexsort((un, cl))
    rows = order.astype(np.int64)
    u_sorted = un[order]
    u_start = np.flatnonzero(np.r_[True, u_sorted[1:] != u_sorted[:-1]])
    u_len = np.diff(np.r_[u_start, len(rows)])
    u_cl = cl[order][u_start]
    nc = int(cl.max()) + 1
    c_first = np.searchsorted(u_cl, np.arange(nc))
    c_nu = np.bincount(u_cl, minlength=nc)
    rng = np.random.default_rng(seed)
    vals = np.empty(B)
    for i in range(B):
        cs = rng.integers(0, nc, nc)
        nu = c_nu[cs]
        units = c_first[np.repeat(cs, nu)] + (rng.random(nu.sum()) * np.repeat(nu, nu)).astype(np.int64)
        lens = u_len[units]
        tot = lens.sum()
        offs = np.repeat(u_start[units] - (np.cumsum(lens) - lens), lens) + np.arange(tot)
        sub = base.take(rows[offs])
        relabel = {cluster: np.repeat(np.repeat(np.arange(nc), nu), lens)}
        if inner:
            relabel[inner] = np.repeat(np.arange(len(units)), lens)
        vals[i] = value_fn(sub.assign(**relabel))
    lo, hi = np.nanquantile(vals, [alpha / 2, 1 - alpha / 2])
    return est, float(lo), float(hi)


# ---------------------------------------------------------------- Pareto
def pareto_front(points: Sequence[tuple[float, float]]) -> list[int]:
    """Indices of non-dominated (cost, benefit) points (lower cost, higher benefit), sorted by cost."""
    pts = [(float(c), float(b)) for c, b in points]
    keep = []
    for i, (ci, bi) in enumerate(pts):
        dominated = any(cj <= ci and bj >= bi and (cj < ci or bj > bi) for j, (cj, bj) in enumerate(pts) if j != i)
        if not dominated:
            keep.append(i)
    return sorted(keep, key=lambda i: (pts[i][0], -pts[i][1]))


# ---------------------------------------------------------------- tables over the tidy frame
def paired_wide(df: pd.DataFrame, arms: Sequence[str], metric: str = "success",
                keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Episodes x arms matrix of `metric`, restricted to (keys) observed in every arm.

    Rows where any arm's metric is NaN are dropped too, so a metric that is undefined by design
    (trip_time of an episode that did not trip) would condition on the outcome: use fail_time.
    """
    d = df[df["arm"].isin(arms)]
    if d.duplicated([*keys, "arm"]).any():
        raise ValueError("duplicate episodes for some (keys, arm)")
    w = d.pivot(index=list(keys), columns="arm", values=metric)
    missing = [a for a in arms if a not in w.columns]
    if missing:
        raise KeyError(f"arms without episodes: {missing}")
    return w[list(arms)].dropna()


def _is_binary(s: pd.Series) -> bool:
    v = pd.unique(s.dropna().astype(float))
    return bool(len(v)) and set(v.tolist()) <= {0.0, 1.0}


def summary_table(df: pd.DataFrame, arms: Sequence[str], metric: str = "success", alpha: float = 0.05,
                  paired: bool = True, keys: Sequence[str] = KEYS, binary: bool | None = None) -> pd.DataFrame:
    """Per-arm summary. Binary metric: n, k, rate with Wilson CI; else n, mean, sd, median, IQR.

    paired=True keeps only episodes present in every arm (the paired analysis set); binary=None
    infers the metric type from its values.
    """
    if paired:
        w = paired_wide(df, arms, metric, keys)
        cols = {a: w[a] for a in arms}
    else:
        cols = {a: df.loc[df["arm"] == a, metric].dropna() for a in arms}
    if binary is None:
        binary = all(_is_binary(s) for s in cols.values() if len(s))
    rows = []
    for a, s in cols.items():
        x = s.astype(float).to_numpy()
        if binary:
            k, n = int(x.sum()), len(x)
            lo, hi = wilson_ci(k, n, alpha)
            rows.append({"arm": a, "n": n, "k": k, "rate": k / n if n else np.nan, "lo": lo, "hi": hi})
        else:
            q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75]) if len(x) else (np.nan,) * 3
            rows.append({"arm": a, "n": len(x), "mean": x.mean() if len(x) else np.nan,
                         "sd": x.std(ddof=1) if len(x) > 1 else np.nan, "median": med, "q1": q1, "q3": q3})
    return pd.DataFrame(rows)


def success_vs(df: pd.DataFrame, ref: str, arms: Sequence[str], metric: str = "success", alpha: float = 0.05,
               keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Each arm vs `ref` on a binary endpoint: exact McNemar (Holm over arms) + Newcombe CI of arm - ref.

    Each comparison uses the episodes shared by that arm and ref.
    """
    rows = []
    for a in arms:
        w = paired_wide(df, [a, ref], metric, keys)
        x, r = w[a].to_numpy(), w[ref].to_numpy()
        b, c, p = mcnemar_exact(x, r)
        d, lo, hi = newcombe_paired_diff_ci(x, r, alpha)
        rows.append({"arm": a, "ref": ref, "n": len(w), "rate": x.mean() if len(w) else np.nan,
                     "rate_ref": r.mean() if len(w) else np.nan, "b": b, "c": c, "p": p,
                     "diff": d, "lo": lo, "hi": hi})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy()) if len(out) else []
    return out


def continuous_vs(df: pd.DataFrame, ref: str, arms: Sequence[str], metric: str,
                  keys: Sequence[str] = KEYS) -> pd.DataFrame:
    """Each arm vs `ref` on a continuous endpoint: Wilcoxon signed-rank (zsplit), Holm over arms."""
    rows = []
    for a in arms:
        w = paired_wide(df, [a, ref], metric, keys)
        x, r = w[a].astype(float).to_numpy(), w[ref].astype(float).to_numpy()
        st, p = wilcoxon_paired(x, r) if len(w) else (np.nan, np.nan)
        rows.append({"arm": a, "ref": ref, "n": len(w), "median": np.median(x) if len(w) else np.nan,
                     "median_ref": np.median(r) if len(w) else np.nan,
                     "median_diff": np.median(x - r) if len(w) else np.nan, "stat": st, "p": p})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"].to_numpy()) if len(out) else []
    return out
