"""Load arm episodes (experiments/run_arms.py output) into a tidy DataFrame, one row per episode.

Layout: results/arms/<arm_dir>/<suite>/<scenario>__s<seed>.json. Success is
cod.synth.evaluate.success, which needs the scenario's headroom label (quality-gap scenarios
must also deliver the commanded grade). Labels are joined by scenario *content*
(Scenario.content_hash), not by id, because the training screen (results/screen) and the
held-out test screen (results/screen_101) reuse ids such as "S0000". Precedence:
  1. an explicit "label" on the entry in data/suites/<suite>.json;
  2. the screen record of the identical scenario, held-out split first, same seed preferred;
  3. Alhazmi classes: Q -> quality_gap, A -> benign, S -> safety_gap;
  4. otherwise None (success = stabilised).
`label_src` records which source applied, so a test-suite analysis can assert it never drew
on the training split.
"""
from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
from typing import Iterable

import pandas as pd

from ..faults import Scenario
from ..synth.evaluate import success

PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS = os.path.join(PROJECT, "results")
SUITES = os.path.join(PROJECT, "data", "suites")
SCREEN_SPLITS = ("screen_101", "screen")
ALHAZMI_LABELS = {"Q": "quality_gap", "A": "benign", "S": "safety_gap"}

COLUMNS = ["arm", "suite", "scenario", "family", "seed", "label", "stabilized", "tripped", "shutdown", "trip_time",
           "fail_time", "success", "cost", "production", "g_iae", "offspec_h", "n_accepted", "calls", "prompt_tokens",
           "completion_tokens", "tokens", "llm_calls", "gpu_s", "wall_s", "diag_correct_any", "label_src",
           "label_seed_match", "scenario_hash", "arm_dir", "path"]


WORKSPACE = os.path.dirname(PROJECT)
_PNAME = os.path.basename(PROJECT)


def local_spec(spec: str) -> str:
    """Arm spec with workspace-relative paths (as in the anonymised release) made absolute for this checkout.
    A no-op for specs that already carry absolute paths, so the original results load unchanged."""
    return spec.replace(":" + _PNAME + os.sep, ":" + PROJECT + os.sep)


def release_spec(spec: str) -> str:
    """Arm spec with this checkout's absolute paths made workspace-relative (the form stored in the release)."""
    return spec.replace(":" + WORKSPACE + os.sep, ":")


def arm_dir_name(spec: str) -> str:
    """Directory name run_arms.arm_dir uses for an arm spec."""
    safe = spec.replace("/", "_").replace(":", "~").replace("@", "_at_")
    if len(safe) > 120:
        safe = safe[:80] + "_" + hashlib.sha1(spec.encode()).hexdigest()[:10]
    return safe


def scenario_hash(sc: dict) -> str:
    return Scenario.from_dict(sc).content_hash()


@functools.lru_cache(maxsize=8)
def label_index(root: str = RESULTS, splits: tuple[str, ...] = SCREEN_SPLITS) -> dict:
    """{(hash, seed) or hash: (label, split)} over the screen JSONs; earlier splits win."""
    idx: dict = {}
    for split in reversed(splits):
        for p in glob.glob(os.path.join(root, split, "*", "*.json")):
            with open(p) as f:
                r = json.load(f)
            h = scenario_hash(r["scenario"])
            idx[(h, int(r["seed"]))] = (r["label"], split)
            idx[h] = (r["label"], split)
    return idx


@functools.lru_cache(maxsize=32)
def suite_labels(suite: str, suites_dir: str = SUITES) -> dict:
    """{(hash, seed) or hash: (label, seed_matched)} for entries of data/suites/<suite>.json with a label.

    A label belongs to the entry's listed seeds; an entry without seeds labels every seed.
    """
    path = os.path.join(suites_dir, f"{suite}.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out: dict = {}
    for x in d:
        if not (isinstance(x, dict) and x.get("label")):
            continue
        h, seeds = scenario_hash(x["scenario"]), x.get("seeds")
        for s in seeds or ():
            out[(h, int(s))] = (x["label"], True)
        if not seeds or h not in out:
            out[h] = (x["label"], not seeds)
    return out


def episode_label(rec: dict, suite: str, h: str | None, root: str = RESULTS,
                  splits: tuple[str, ...] = SCREEN_SPLITS, suites_dir: str = SUITES) -> tuple[str | None, str | None, bool]:
    """(label, source, seed_matched) for one episode record."""
    if h is not None:
        lab = suite_labels(suite, suites_dir)
        hit = lab.get((h, int(rec["seed"]))) or lab.get(h)
        if hit:
            return hit[0], "suite_file", hit[1]
        idx = label_index(root, tuple(splits))
        hit = idx.get((h, int(rec["seed"])))
        if hit:
            return hit[0], hit[1], True
        hit = idx.get(h)
        if hit:
            return hit[0], hit[1], False
    if suite == "alhazmi":
        return ALHAZMI_LABELS.get(rec["family"]), "alhazmi", True
    return None, None, False


def _row(rec: dict, arm_d: str, suite: str, path: str, root: str, splits: tuple[str, ...], suites_dir: str) -> dict:
    sd = rec.get("scenario_def")
    h = scenario_hash(sd) if sd else None
    suite = rec.get("suite", suite)
    label, src, seed_match = episode_label(rec, suite, h, root, splits, suites_dir)
    ptok, ctok = int(rec.get("prompt_tokens", 0)), int(rec.get("completion_tokens", 0))
    tt = rec.get("trip_time")
    if rec["stabilized"]:
        fail = float((sd or {}).get("horizon_h", 48.0))
    else:
        fail = next((float(v) for v in (tt, rec.get("shutdown_time"), rec.get("final_time")) if v is not None),
                    float("nan"))
    return {
        "arm": local_spec(rec.get("arm", arm_d)), "suite": suite, "scenario": rec["scenario"], "family": rec["family"],
        "seed": int(rec["seed"]), "label": label, "stabilized": bool(rec["stabilized"]),
        "tripped": bool(rec["tripped"]), "shutdown": bool(rec["shutdown"]),
        "trip_time": float(tt) if tt is not None else float("nan"), "fail_time": fail, "success": bool(success(rec, label)),
        "cost": float(rec["cost"]), "production": float(rec["production"]), "g_iae": float(rec["g_iae"]),
        "offspec_h": float(rec["offspec_h"]), "n_accepted": int(rec.get("n_accepted", 0)),
        "calls": int(rec.get("calls", 0)), "prompt_tokens": ptok, "completion_tokens": ctok, "tokens": ptok + ctok,
        "llm_calls": int(rec.get("llm_calls", 0)), "gpu_s": float(rec.get("gpu_s", 0.0)),
        "wall_s": float(rec.get("wall_s", float("nan"))), "diag_correct_any": rec.get("diag_correct_any"),
        "label_src": src, "label_seed_match": seed_match, "scenario_hash": h, "arm_dir": arm_d, "path": path,
    }


def load_episodes(arms: Iterable[str] | None = None, suites: Iterable[str] | None = None, root: str = RESULTS,
                  splits: Iterable[str] = SCREEN_SPLITS, suites_dir: str = SUITES) -> pd.DataFrame:
    """All episodes under <root>/arms as a tidy frame (columns COLUMNS), sorted by arm, suite, scenario, seed.

    trip_time is NaN when the episode did not trip; fail_time (hours to a trip or orderly shutdown,
    the horizon if neither) is the censored version to use for paired time-to-failure comparisons.

    arms: arm specs (as passed to run_arms) or their directory names; None = every arm.
    suites: suite names; None = every suite found.
    """
    splits = tuple(splits)
    arms = [arms] if isinstance(arms, str) else arms
    suites = [suites] if isinstance(suites, str) else suites
    base = os.path.join(root, "arms")
    if arms is None:
        dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))) if os.path.isdir(base) else []
    else:
        # an arm's directory is named after its spec as run: absolute paths in the original checkout,
        # workspace-relative ones in the anonymised release; use whichever exists
        cands = [(arm_dir_name(a), arm_dir_name(release_spec(a))) for a in arms]
        dirs = list(dict.fromkeys(next((d for d in c if os.path.isdir(os.path.join(base, d))), c[0]) for c in cands))
    want = set(suites) if suites is not None else None
    rows = []
    for d in dirs:
        for sdir in sorted(glob.glob(os.path.join(base, d, "*"))):
            suite = os.path.basename(sdir)
            if not os.path.isdir(sdir) or (want is not None and suite not in want):
                continue
            for p in sorted(glob.glob(os.path.join(sdir, "*.json"))):
                with open(p) as f:
                    rec = json.load(f)
                rows.append(_row(rec, d, suite, p, root, splits, suites_dir))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["diag_correct_any"] = df["diag_correct_any"].astype("boolean")
    return df.sort_values(["arm", "suite", "scenario", "seed"], kind="mergesort").reset_index(drop=True)
