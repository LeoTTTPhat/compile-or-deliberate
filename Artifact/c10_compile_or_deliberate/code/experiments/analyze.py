"""Analysis pipeline: ONE command regenerates every number, table and figure of the results section.

    cd <workspace>
    PYTHONPATH=c10_compile_or_deliberate/code .venv/bin/python c10_compile_or_deliberate/code/experiments/analyze.py
        [--workers 4] [--no-figs] [--no-pending] [--case SN0003]

Inputs (read-only): results/arms/<arm_dir>/<suite>/*.json (one per episode), results/synth/*/ and
results/synth_archive/*/ (meta_*.json, log.txt), results/coverage/*.json, results/stream/<name>/block*.json.

Outputs (paper/generated/, paper/figures/):
  numbers.tex          \\newcommand macros for every quantitative claim (naming scheme documented in its header)
  results_main.tex     main results (arms grouped by method family; Wilson 95 % CIs on the held-out suites)
  results_all.tex      every arm with data (appendix), same layout
  results_family.tex   per-family success on the held-out suites
  stats_vs_ref.tex     paired exact McNemar / Newcombe CI vs references, Holm-adjusted
  cost_online.tex      online and hybrid cost: LLM calls, tokens, escalations by trigger, decision latency
  coverage.tex         empirical scenario-coverage bounds (results/coverage/*.json)
  stream.tex           compile-back deployment stream (results/stream/*/block*.json)
  synthesis.tex        per synthesis run: tokens, iterations, lint fixes, CPU episodes, tuning, training success
  analysis.json        machine-readable version of everything above
  preview.tex          standalone document that typesets all generated tables and figures (compile check)
  ../figures/fig_family.pdf, fig_pareto.pdf, fig_synthesis.pdf, fig_case.pdf, fig_escalation.pdf
                       (drawn by figures_extra.py)

Arms that the experiment conductor (code/scripts/pipeline.sh) will produce but that have no
episodes yet are listed in EXPECTED: their macros render as \\textbf{[pending]} and their table rows
say "pending". Suites with fewer episodes than planned are marked partial.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import functools
import glob
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../code
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cod.analysis.stats import holm, mcnemar_exact, newcombe_paired_diff_ci, wilson_ci  # noqa: E402

PROJ = os.path.dirname(ROOT)
RES = os.environ.get("COD_RESULTS", os.path.join(PROJ, "results"))  # overridable for tests
GEN = os.environ.get("COD_GEN", os.path.join(PROJ, "paper", "generated"))
FIG = os.environ.get("COD_FIG", os.path.join(PROJ, "paper", "figures"))
SYN = os.path.join(RES, "synth")
SYN_ARCH = os.path.join(RES, "synth_archive")
COV = os.path.join(RES, "coverage")
STREAM = os.path.join(RES, "stream")
SUITES_DIR = os.path.join(PROJ, "data", "suites")

# ======================================================================================= constants
SUITES = ("alhazmi", "frontier", "test_id", "test_ood", "test_unrescuable")
HELD = ("test_id", "test_ood")
SUITE_KEY = {"alhazmi": "Anchor", "frontier": "Frontier", "test_id": "Id", "test_ood": "Ood",
             "test_unrescuable": "Unres", "held": "Held"}
SUITE_NAME = {"alhazmi": "Anchor-13", "frontier": "Frontier-5", "test_id": "held-out anticipated",
              "test_ood": "held-out novel", "test_unrescuable": "oracle-unrescued", "held": "held-out (anticipated + novel)"}
FAMILIES_ID = ("A", "S", "P", "Q", "SN")
FAMILIES_OOD = ("V", "C", "K")
GROUP_ORDER = ["ref", "rule", "fdd", "search", "gp", "online", "playbook", "hybrid"]
GROUP_LABEL = {"ref": "references", "rule": "rule tables", "fdd": "FDD + lookup", "search": "verifier search",
               "gp": "GP playbook", "playbook": "LLM playbook (offline)", "online": "online LLM", "hybrid": "hybrid"}
TABLE_SECTIONS = [("References", ("ref",)), ("LLM-free supervisors", ("rule", "fdd", "search", "gp")),
                  ("Online LLM deliberation ($f=0$)", ("online",)), ("Offline LLM playbook ($f=1$)", ("playbook",)),
                  ("Hybrid: playbook + gated escalation ($0<f<1$)", ("hybrid",))]
# fixed categorical palette per method family (light mode) and secondary marker encoding for print
COLOR = {"ref": "#8a8984", "rule": "#4a3aa7", "fdd": "#008300", "gp": "#eda100", "search": "#e87ba4",
         "playbook": "#2a78d6", "online": "#eb6834", "hybrid": "#1baf7a"}
MARKER = {"ref": "s", "rule": "D", "fdd": "v", "gp": "P", "search": "X", "playbook": "o", "online": "^", "hybrid": "*"}
TRIGGERS = (("a", "unexplained_abnormality", "unexplained abnormality"),
            ("b", "playbook_action_fails_forward_check", "forward-check failure"),
            ("c", "out_of_distribution", "conformal novelty"))
N_LARGE = 1000  # second deployment horizon for amortisation / allocation views
MAIN_MODEL = "qwen3-coder:30b"
REF_ARMS = (("B2b", "BTwob"), ("strong", "Strong"), ("fdd:default", "FDD"), (f"onlineg:{MAIN_MODEL}", "OnlGated"),
            ("weak", "Weak"))
EXCLUDE_MAIN = ("/preview/", "/synth_archive/", "/pilots/")  # arms shown only in results_all.tex


def _suite_sizes() -> dict:
    n = {"alhazmi": 39, "frontier": 15}
    for s in ("test_id", "test_ood", "test_unrescuable"):
        p = os.path.join(SUITES_DIR, f"{s}.json")
        if os.path.exists(p):
            n[s] = sum(len(x.get("seeds") or [1]) for x in json.load(open(p)))
    return n


SUITE_N = _suite_sizes()


def _family_sizes() -> dict:
    out: dict = {}
    for s in ("test_id", "test_ood"):
        p = os.path.join(SUITES_DIR, f"{s}.json")
        if os.path.exists(p):
            for x in json.load(open(p)):
                f = x["scenario"]["family"]
                out[f] = out.get(f, 0) + len(x.get("seeds") or [1])
    return out


FAMILY_N = _family_sizes()


# ======================================================================================= expected arms
def _pb(seed: int, ck: str, tag: str = "main", model: str = "qwen3-coder-30b") -> str:
    return os.path.join(SYN, f"{tag}_{model}_s{seed}", f"playbook_{ck}.py")


def _gp(seed: int, ck: str) -> str:
    return os.path.join(SYN, f"gp_s{seed}", f"playbook_{ck}.py")


def expected_arms() -> list[tuple[str, tuple[str, ...], str]]:
    """(arm spec, planned suites, stage) for every arm the conductor (code/scripts/pipeline.sh) produces."""
    ALL, M = SUITES, MAIN_MODEL
    ex = [(a, ALL, "CPU baselines") for a in ("weak", "strong", "search:perfect@gate=0", "B2a", "B2b", "fdd:default",
                                              "search:nominal")]
    ex += [(f"playbook:{_gp(1, 'E5000')}", ALL, "GP"), (f"playbook:{_gp(1, 'E20000')}", ALL, "GP")]
    ex += [(f"onlineg:{M}", ("test_id", "test_ood", "alhazmi", "frontier"), "online main")]
    ex += [(f"playbook:{_pb(s, c)}", ALL, "A1") for s, c in ((1, "v0"), (1, "B25000"), (1, "B100000"), (2, "B100000"),
                                                             (3, "B100000"))]
    ex += [(f"hybrid:{_pb(s, c)}:{M}@{t}", HELD, "A2") for s, c, t in
           ((1, "B25000", "16000"), (1, "B100000", "8000"), (1, "B100000", "16000"), (1, "B100000", "inf"),
            (2, "B100000", "16000"), (3, "B100000", "16000"))]
    ex += [(f"onlineg:{M}@{t}", HELD, "B") for t in (4000, 8000, 16000, 32000)]
    ex += [(f"playbook:{_pb(s, c)}", ALL, "C1") for s in (1, 2, 3) for c in ("B200000", "B400000", "final")]
    ex += [(f"hybrid:{_pb(s, 'final')}:{M}@{t}", HELD if (s, t) != (1, "16000") else ALL, "C3")
           for s, t in ((1, "8000"), (1, "16000"), (1, "inf"), (2, "16000"), (3, "16000"))]
    ex += [(f"playbook:{_pb(1, 'final', tag)}", ALL, "D2") for tag in ("norem", "notune")]
    ex += [(f"playbook:{_gp(s, c)}", ALL, "D2") for s in (2, 3) for c in ("final", "E5000")]
    ex += [(f"playbook:{_gp(1, 'final')}", ALL, "D2")]
    ex += [("onlineg:qwen2.5:14b-instruct-q4_K_M", HELD, "D3")]
    return ex


EXPECTED_RUNS = {  # synthesis runs: dir -> expected checkpoints
    "main_qwen3-coder-30b_s1": ("v0", "B25000", "B50000", "B100000", "B200000", "B400000", "final"),
    "main_qwen3-coder-30b_s2": ("v0", "B25000", "B50000", "B100000", "B200000", "B400000", "final"),
    "main_qwen3-coder-30b_s3": ("v0", "B25000", "B50000", "B100000", "B200000", "B400000", "final"),
    "main_qwen2.5-coder-7b_s1": ("v0", "B25000", "B50000", "B100000"),
    "norem_qwen3-coder-30b_s1": ("v0", "B25000", "B50000", "B100000", "B200000", "final"),
    "notune_qwen3-coder-30b_s1": ("v0", "B25000", "B50000", "B100000", "B200000", "final"),
    "gp_s1": ("g0", "E2500", "E5000", "E10000", "E20000", "final"),
    "gp_s2": ("g0", "E2500", "E5000", "E10000", "E20000", "final"),
    "gp_s3": ("g0", "E2500", "E5000", "E10000", "E20000", "final"),
}
EXPECTED_COVERAGE = ("llm_final_s1", "llm_final_s2", "llm_final_s3", "gp_final_s1")
EXPECTED_STREAMS = ("cb_s1_final_16k", "nocb_s1_final_16k")


# ======================================================================================= arm naming
_ONES = ("Zero One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen Fourteen Fifteen Sixteen "
         "Seventeen Eighteen Nineteen").split()
_TENS = {2: "Twenty", 3: "Thirty", 4: "Forty", 5: "Fifty", 6: "Sixty", 7: "Seventy", 8: "Eighty", 9: "Ninety"}


def words(n: int) -> str:
    """Integer -> letters-only CamelCase words (LaTeX macro names cannot contain digits). 25000 -> TwentyFiveK."""
    n = int(n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + "Hundred" + ("" if n % 100 == 0 else words(n % 100))
    if n < 1_000_000:
        return words(n // 1000) + ("K" if n % 1000 == 0 else "Thousand" + words(n % 1000))
    return words(n // 1_000_000) + "M" + ("" if n % 1_000_000 == 0 else words(n % 1_000_000))


def letters_key(s: str) -> str:
    """Arbitrary string -> letters-only CamelCase (digit runs spelled out)."""
    out = []
    for tok in re.findall(r"[A-Za-z]+|\d+", s):
        out.append(words(int(tok)) if tok.isdigit() else tok[:1].upper() + tok[1:])
    return "".join(out)


MODEL_SHORT = {"qwen3-coder:30b": "Coder-30B", "qwen3-coder-30b": "Coder-30B", "qwen2.5-coder:7b": "Coder-7B",
               "qwen2.5-coder-7b": "Coder-7B", "qwen2.5:7b": "Qwen2.5-7B", "qwen2.5:14b-instruct-q4_K_M": "Qwen2.5-14B",
               "qwen2.5:14b": "Qwen2.5-14B", "mistral-small3.1:24b": "Mistral-24B", "gemma3:27b": "Gemma3-27B",
               "llama3.1:8b": "Llama3.1-8B", "llama3.2:3b": "Llama3.2-3B"}
MODEL_KEY = {"Coder-30B": "CoderThirty", "Coder-7B": "CoderSeven", "Qwen2.5-7B": "QwenSeven",
             "Qwen2.5-14B": "QwenFourteen", "Mistral-24B": "MistralTwentyFour", "Gemma3-27B": "GemmaTwentySeven",
             "Llama3.1-8B": "LlamaEight", "Llama3.2-3B": "LlamaThree"}
TAG_SHORT = {"main": "", "norem": "no remedies", "notune": "no tuning", "oneshot": "one-shot", "nofals": "no falsifier",
             "nomin": "no minimisation", "calibonly": "calibration only, no repair"}


def model_short(m: str | None) -> str:
    return MODEL_SHORT.get(m or "", m or "")


def model_key(m: str | None) -> str:
    s = model_short(m)
    return MODEL_KEY.get(s, letters_key(s))


def ckpt_rank(c: str | None) -> float:
    if c in (None, ""):
        return 1e11
    if c in ("v0", "g0"):
        return 0.0
    if c == "final":
        return 1e12
    m = re.match(r"[BE](\d+)$", c)
    return float(m.group(1)) if m else 1e11


def ckpt_label(c: str | None, gp: bool = False) -> str:
    if c is None:
        return "?"
    if c in ("v0", "final"):
        return c
    if c == "g0":
        return "gen 0"
    m = re.match(r"([BE])(\d+)$", c)
    if m:
        n = int(m.group(2))
        k = f"{n / 1000:g}k"
        return f"{k} evals" if m.group(1) == "E" else k
    return c


def ckpt_key(c: str | None) -> str:
    if c is None:
        return "X"
    if c in ("v0", "g0"):
        return {"v0": "Vzero", "g0": "Gzero"}[c]
    if c == "final":
        return "Final"
    m = re.match(r"([BE])(\d+)$", c)
    if m:
        return ("Ev" if m.group(1) == "E" else "") + words(int(m.group(2)))
    return letters_key(c)


def seed_key(s) -> str:
    return "" if s is None else "S" + words(int(s))


def budget_label(b) -> str:
    if b is None:
        return "unbounded"
    return f"{b / 1000:g}k/ep."


def budget_key(b) -> str:
    return "Inf" if b is None else words(int(b))


def parse_pb_path(path: str) -> dict:
    """Attributes of a playbook file from its location (synthesis run, archive, preview, stream version)."""
    d = {"path": path, "group": "playbook", "tag": "main", "model": "qwen3-coder-30b", "seed": None, "ckpt": None,
         "origin": "other"}
    m = re.search(r"/synth/gp_s(\d+)/playbook_(\w+)\.py$", path)
    if m:
        return {**d, "group": "gp", "tag": "gp", "model": None, "seed": int(m.group(1)), "ckpt": m.group(2), "origin": "synth"}
    m = re.search(r"/(synth|synth_archive)/([a-z0-9]+)_(.+?)_s(\d+)/playbook_(\w+)\.py$", path)
    if m:
        return {**d, "tag": m.group(2), "model": m.group(3), "seed": int(m.group(4)), "ckpt": m.group(5),
                "origin": "archive" if m.group(1) == "synth_archive" else "synth"}
    m = re.search(r"/preview/tuned_s(\d+)_(\w+)\.py$", path)
    if m:
        return {**d, "tag": "main", "seed": int(m.group(1)), "ckpt": m.group(2), "origin": "preview"}
    m = re.search(r"/stream/([^/]+)/playbook_v(\d+)\.py$", path)
    if m:
        return {**d, "tag": "stream", "seed": None, "ckpt": "v" + m.group(2), "origin": "stream", "stream": m.group(1)}
    return {**d, "ckpt": os.path.basename(path).replace(".py", "")}


def parse_arm(spec: str) -> dict:
    """Structured attributes of an arm spec (see run_arms.build_arm for the grammar)."""
    a = {"spec": spec, "kind": spec, "group": "ref", "tag": "", "model": None, "seed": None, "ckpt": None,
         "budget": None, "opts": "", "path": None, "origin": None}
    if spec in ("weak", "strong", "hold"):
        return a
    if spec.startswith("search:perfect"):
        return {**a, "kind": "oracle"}
    if spec in ("B2a", "B2b"):
        return {**a, "group": "rule"}
    if spec.startswith("fdd:"):
        return {**a, "kind": "fdd", "group": "fdd", "opts": spec[4:]}
    if spec.startswith("search:"):
        return {**a, "kind": "search", "group": "search", "opts": spec[7:]}
    if spec.startswith(("online:", "onlineg:")):
        body = spec.split(":", 1)[1]
        model, _, b = body.partition("@")
        return {**a, "kind": "onlineg" if spec.startswith("onlineg:") else "online", "group": "online", "model": model,
                "budget": None if b in ("", "inf") else float(b)}
    if spec.startswith("playbook:"):
        path, _, name = spec[len("playbook:"):].partition("@")
        p = parse_pb_path(path)
        return {**a, **p, "kind": "playbook", "opts": name}
    if spec.startswith("hybrid:"):
        body = spec[len("hybrid:"):]
        if ".py:" not in body:
            return {**a, "kind": "hybrid", "group": "hybrid"}
        path, rest = body.split(".py:", 1)
        path += ".py"
        opts = ""
        if "@" in rest:
            model, tail = rest.split("@", 1)
            tok, _, opts = tail.partition(":")
        else:
            model, tok = rest, "inf"
        p = parse_pb_path(path)
        return {**a, "kind": "hybrid", "group": "hybrid", "model": model, "budget": None if tok in ("", "inf") else float(tok),
                "opts": opts, "path": path, "pb_tag": p["tag"], "pb_model": p["model"], "seed": p["seed"], "ckpt": p["ckpt"],
                "origin": p["origin"]}
    return a


def group(arm: str) -> str:
    return parse_arm(arm)["group"]


@functools.lru_cache(maxsize=None)
def _code_equals_final(path: str) -> bool:
    """A checkpoint file whose code (below the two header lines) equals the run's playbook_final.py."""
    fin = os.path.join(os.path.dirname(path), "playbook_final.py")
    if os.path.abspath(path) == os.path.abspath(fin):
        return True
    try:
        return open(path).read().split("\n", 2)[-1] == open(fin).read().split("\n", 2)[-1]
    except OSError:
        return False


def pretty(arm: str) -> str:
    """Human-readable arm name (plain text; safe in LaTeX and matplotlib)."""
    a = parse_arm(arm)
    fixed = {"weak": "Weak baseline", "strong": "Strong reference", "hold": "Hold (no-op supervisor)",
             "B2a": "Alarm table (B2a)", "B2b": "Expert table (B2b)", "oracle": "Perfect-twin search",
             "fdd": "FDD + lookup (B3)", "search": "Verifier search (B4)"}
    if a["kind"] in fixed and a["kind"] not in ("fdd", "search"):
        return fixed[a["kind"]]
    if a["kind"] == "fdd":
        return fixed["fdd"] + ("" if a["opts"] in ("default", "") else f" [{a['opts']}]")
    if a["kind"] == "search":
        return fixed["search"] + ("" if a["opts"] in ("nominal", "") else f" [{a['opts']}]")
    if a["kind"] in ("online", "onlineg"):
        s = "Online, gated" if a["kind"] == "onlineg" else "Online, every call"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            s += f" ({model_short(a['model'])})"
        return s + ("" if a["budget"] is None else f", {budget_label(a['budget'])}")
    if a["kind"] == "playbook":
        if a["group"] == "gp":
            final = a["ckpt"] == "final" or _code_equals_final(arm[len("playbook:"):].partition("@")[0])
            return f"GP s{a['seed']} " + ("final" if final else "@" + ckpt_label(a["ckpt"], True))
        if a["origin"] == "stream":
            return f"Compile-back playbook {a['ckpt']} ({a.get('stream', '')})"
        if a["seed"] is None:
            return f"Playbook {a['ckpt']}"
        s = "Playbook"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            s += f" ({model_short(a['model'])})"
        ck = a["ckpt"] if a["ckpt"] in ("v0", "final") else "@" + ckpt_label(a["ckpt"])
        s += f" s{a['seed']} {ck}"
        extra = [TAG_SHORT.get(a["tag"], a["tag"])] if a["tag"] != "main" else []
        if a["origin"] == "archive":
            extra.append("archived pre-tuning run")
        if a["origin"] == "preview":
            extra.append("preview")
        return s + "".join(f", {x}" for x in extra if x)
    if a["kind"] == "hybrid":
        ck = a["ckpt"] if a["ckpt"] in ("v0", "final") else "@" + ckpt_label(a["ckpt"])
        s = f"Hybrid s{a['seed']} {ck}, {budget_label(a['budget'])}"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            s += f" ({model_short(a['model'])})"
        if a.get("pb_tag") not in (None, "main"):
            s += f" [{TAG_SHORT.get(a['pb_tag'], a['pb_tag'])}]"
        if a.get("origin") == "preview":
            s += " [preview playbook]"
        if a["opts"]:
            s += f" [{a['opts']}]"
        return s
    return arm


def arm_key(arm: str) -> str:
    """Letters-only key used in macro names (see the scheme in numbers.tex)."""
    a = parse_arm(arm)
    fixed = {"weak": "Weak", "strong": "Strong", "hold": "Hold", "oracle": "Oracle", "B2a": "BTwoa", "B2b": "BTwob"}
    if a["kind"] in fixed:
        return fixed[a["kind"]]
    if a["kind"] == "fdd":
        return "FDD" + ("" if a["opts"] in ("default", "") else letters_key(a["opts"]))
    if a["kind"] == "search":
        return "Search" + ("" if a["opts"] in ("nominal", "") else letters_key(a["opts"]))
    if a["kind"] in ("online", "onlineg"):
        k = "OnlGated" if a["kind"] == "onlineg" else "OnlAll"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            k += model_key(a["model"])
        return k + ("" if a["budget"] is None else "Tok" + budget_key(a["budget"]))
    if a["kind"] == "playbook":
        if a["group"] == "gp":
            return "GP" + ckpt_key(a["ckpt"]) + seed_key(a["seed"])
        if a["origin"] == "stream":
            return "Stream" + letters_key(a.get("stream", "")) + letters_key(a["ckpt"] or "")
        k = "Pb"
        if a["origin"] == "archive":
            k += "Arch"
        if a["origin"] == "preview":
            k += "Preview"
        if a["tag"] not in ("main", ""):
            k += letters_key(a["tag"])
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            k += model_key(a["model"])
        if a["seed"] is None:
            return k + letters_key(a["ckpt"] or "")
        return k + ckpt_key(a["ckpt"]) + seed_key(a["seed"])
    if a["kind"] == "hybrid":
        k = "Hyb"
        if a.get("pb_tag") not in (None, "main"):
            k += letters_key(a["pb_tag"])
        if a.get("origin") == "preview":
            k += "Preview"
        k += ckpt_key(a["ckpt"]) + seed_key(a["seed"])
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            k += model_key(a["model"])
        k += "Tok" + budget_key(a["budget"])
        if a["opts"]:
            k += letters_key(a["opts"])
        return k
    return letters_key(arm)


def sort_key(arm: str):
    a = parse_arm(arm)
    g = GROUP_ORDER.index(a["group"]) if a["group"] in GROUP_ORDER else 99
    sub = {"weak": 0, "strong": 1, "hold": 2, "oracle": 3, "B2a": 0, "B2b": 1}.get(a["kind"], 5)
    origin = {"synth": 0, "archive": 2, "preview": 3, "stream": 4}.get(a.get("origin") or "", 1)
    tag = 0 if a["tag"] in ("main", "", "gp") else 1
    model = 0 if model_short(a["model"]) in ("", model_short(MAIN_MODEL)) else 1
    budget = 1e12 if a["budget"] is None else a["budget"]
    if a["group"] == "online":
        return (g, model, 0 if a["kind"] == "onlineg" else 1, budget, arm)
    return (g, sub, origin, tag, model, a["seed"] or 0, ckpt_rank(a["ckpt"]), budget, arm)


PROMOTED: set = set()  # preview arms shown in the main tables/figures as stand-ins (see promote_previews)


def is_main_arm(arm: str) -> bool:
    return arm in PROMOTED or not any(x in arm for x in EXCLUDE_MAIN)


def promote_previews(present: set) -> set:
    """A preview playbook (results/preview/tuned_s<seed>_<ckpt>.py) is a code-identical copy of a main-run checkpoint
    (only the header comment differs), so its episodes are those of that checkpoint. Until the main checkpoint itself
    has been evaluated, the preview stands in for it in the main tables and figures (marked "preview"), so that the
    LLM playbook the text discusses appears there; it drops out automatically once the main arm has episodes."""
    def body(path: str) -> str | None:
        if not os.path.exists(path):
            return None
        lines = open(path).read().splitlines()
        while lines and lines[0].lstrip().startswith("#"):
            lines.pop(0)
        return "\n".join(lines).strip()

    out = set()
    for arm in present:
        if arm.startswith("playbook:") and "/preview/" in arm:
            a = parse_arm(arm)
            if a["seed"] is None or not a["ckpt"]:
                continue
            main_path = _pb(a["seed"], a["ckpt"])
            src = body(a["path"])
            if f"playbook:{main_path}" not in present and src is not None and src == body(main_path):
                out.add(arm)
    return out


# ======================================================================================= offline costs
def _meta_for(path: str) -> dict | None:
    m = path.replace("playbook_", "meta_").replace(".py", ".json")
    if os.path.exists(m):
        try:
            return json.load(open(m))
        except json.JSONDecodeError:
            return None
    return None


def offline_cost(arm: str) -> dict:
    """{'tokens': LLM tokens spent to synthesise the arm's playbook, 'cpu_episodes': simulated episodes (if known)}."""
    a = parse_arm(arm)
    if a["kind"] not in ("playbook", "hybrid") or not a["path"]:
        return {"tokens": 0.0, "cpu_episodes": float("nan"), "source": "none"}
    path = a["path"]
    if a["group"] == "gp":
        meta = _meta_for(path) or {}
        return {"tokens": 0.0, "cpu_episodes": float(meta.get("evals", float("nan"))), "source": "gp meta"}
    meta = _meta_for(path)
    if meta is not None:
        return {"tokens": float(meta["tokens"]), "cpu_episodes": float(meta.get("episodes", float("nan"))), "source": "meta"}
    if a["origin"] == "preview":  # preview = an archived checkpoint + CPU tuning; cost = that checkpoint's tokens
        cand = glob.glob(os.path.join(SYN_ARCH, f"*_s{a['seed']}", f"meta_{a['ckpt']}.json"))
        if cand:
            meta = json.load(open(sorted(cand)[0]))
            return {"tokens": float(meta["tokens"]), "cpu_episodes": float("nan"), "source": "archive meta"}
    if os.path.exists(path):
        head = open(path).readline()
        m = re.search(r"tokens (\d+)", head)
        if m:
            return {"tokens": float(m.group(1)), "cpu_episodes": float("nan"), "source": "header"}
    return {"tokens": float("nan"), "cpu_episodes": float("nan"), "source": "unknown"}


# ======================================================================================= loading
def _extras(rec: dict) -> dict:
    """Per-episode fields not in load.py: escalations by trigger, suppressed escalations, LLM decision latencies,
    instrument substitutions."""
    esc = rec.get("escalations") or []
    out = {"n_esc": len(esc), "n_supp": 0, "lat": [], "n_llm_dec": 0, "n_cached_dec": 0, "n_exhausted": 0,
           "n_subst": 0, "n_subst_rep": 0, "gpu_s_ep": float(rec.get("gpu_s", 0.0) or 0.0)}
    for code, key, _ in TRIGGERS:
        out["esc_" + code] = sum(key in (e.get("why") or []) for e in esc)
        out["escf_" + code] = sum((key in (e.get("why") or [])) / max(len(e.get("why") or []), 1) for e in esc)
    subst: set = set()
    for d in rec.get("decisions") or []:
        src = d.get("source", "")
        meta = d.get("meta") or {}
        act = d.get("action") or {}
        if act.get("type") == "substitute_instrument":
            m = act.get("measurement")
            if m in subst:
                out["n_subst_rep"] += 1  # proposal to substitute a transmitter that is already out of service
            elif d.get("accepted"):
                subst.add(m)
                out["n_subst"] += 1
        if meta.get("triggers") and not src.startswith("escalation"):
            out["n_supp"] += 1
        if src == "budget_exhausted":
            out["n_exhausted"] += 1
        if int(d.get("llm_calls", 0)) > 0:
            out["n_llm_dec"] += 1
            if meta.get("cached") or float(d.get("gpu_s", 0.0)) <= 0:
                out["n_cached_dec"] += 1
            else:
                out["lat"].append(float(d["gpu_s"]))
    return out


def _load_dir(dirname: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One arm directory -> (episode rows as in cod.analysis.load, extras); every JSON is read once."""
    from cod.analysis import load as L
    rows, ex = [], []
    for sdir in sorted(glob.glob(os.path.join(RES, "arms", dirname, "*"))):
        if not os.path.isdir(sdir):
            continue
        suite = os.path.basename(sdir)
        for p in sorted(glob.glob(os.path.join(sdir, "*.json"))):
            try:
                with open(p) as f:
                    rec = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue  # file being written by a running job
            row = L._row(rec, dirname, suite, p, RES, L.SCREEN_SPLITS, L.SUITES)
            rows.append(row)
            e = _extras(rec)
            e.update(arm=row["arm"], suite=row["suite"], scenario=row["scenario"], seed=row["seed"])
            ex.append(e)
    df = pd.DataFrame(rows, columns=L.COLUMNS)
    df["diag_correct_any"] = df["diag_correct_any"].astype("boolean")
    return df, pd.DataFrame(ex)


def load_all(workers: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = os.path.join(RES, "arms")
    # "onstrong:" arms (a compiled supervisor on top of the strong configuration) are robustness checks reported by
    # final_stats.py (supplementary table r3checks.tex); they are kept out of the arm tables and figures here
    dirs = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and not d.startswith("onstrong~")) \
        if os.path.isdir(base) else []
    workers = max(1, min(4, workers))
    if workers == 1 or len(dirs) <= 1:
        parts = [_load_dir(d) for d in dirs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_load_dir, dirs))
    dfs = [p[0] for p in parts if len(p[0])]
    exs = [p[1] for p in parts if len(p[1])]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if len(df):
        df = df.sort_values(["arm", "suite", "scenario", "seed"], kind="mergesort").reset_index(drop=True)
    ext = pd.concat(exs, ignore_index=True) if exs else pd.DataFrame()
    return df, ext


# ======================================================================================= formatting
def fmt_pct(x: float, nd: int = 1) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{nd}f}"


def fmt_signed(x: float, nd: int = 1, plus: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    v = round(x, nd)
    if v == 0:
        return f"{0:.{nd}f}"
    if v > 0:
        return ("$+$" if plus else "") + f"{v:.{nd}f}"
    return "$-$" + f"{abs(v):.{nd}f}"


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    if p < 0.001:
        return "$<$0.001"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def fmt_tok(x: float) -> str:
    """Token count for tables: 0, 850, 3.4k, 104k, 1.2M."""
    if x is None or not np.isfinite(x):
        return "--"
    if x < 1:
        return "0"
    if x < 1000:
        return f"{x:.0f}"
    if x < 10_000:
        return f"{x / 1000:.1f}k"
    if x < 1_000_000:
        return f"{x / 1000:.0f}k"
    return f"{x / 1e6:.2f}M"


def fmt_num(x: float, nd: int = 0) -> str:
    """Number for running text via siunitx (group separators)."""
    if x is None or not np.isfinite(x):
        return "--"
    return f"\\num{{{x:.{nd}f}}}"


def tex(s: str) -> str:
    return (s.replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#").replace("$", "\\$").replace("~", "\\textasciitilde{}"))


PENDING = "\\codpending"


def partial(s: str, is_partial: bool) -> str:
    return f"\\codpartial{{{s}}}" if is_partial else s


# ======================================================================================= macros
class Macros:
    NAME = re.compile(r"^[A-Za-z]+$")

    def __init__(self):
        self.items: list[tuple[str, str, str, str]] = []
        self.names: set[str] = set()
        self.section = ""

    def sec(self, title: str):
        self.section = title

    def add(self, name: str, value: str, comment: str = ""):
        if not self.NAME.match(name):
            raise ValueError(f"invalid macro name {name!r}")
        if name in self.names:
            return
        self.names.add(name)
        self.items.append((self.section, name, value, comment.replace("\n", " ")))

    def write(self, path: str, header: str):
        lines = [header, "\\providecommand{\\codpending}{\\textbf{[pending]}}",
                 "\\providecommand{\\codpartial}[1]{#1\\textsuperscript{\\textbf{[partial]}}}", ""]
        cur = None
        for sec, name, value, comment in self.items:
            if sec != cur:
                lines += ["", f"% ---------------------------------------------------------------- {anon(sec)}"]
                cur = sec
            lines.append(f"\\newcommand{{\\{name}}}{{{value}}}" + (f" % {anon(comment)}" if comment else ""))
        open(path, "w").write("\n".join(lines) + "\n")


def anon(s: str) -> str:
    """Strip absolute local paths (they contain the user name) from text that ships with the manuscript source
    (double-anonymised review)."""
    for p, rep in ((RES, "<results>"), (PROJ, "<project>"), (os.path.dirname(PROJ), "<root>")):
        s = s.replace(p, rep).replace(p.replace("/", "_"), rep.replace("<", "_").replace(">", "_"))  # arm dir names
    return s


NUMBERS_HEADER = r"""%% numbers.tex -- GENERATED by code/experiments/analyze.py on {date}. DO NOT EDIT BY HAND.
%% Regenerate (from the repository root): PYTHONPATH=c10_compile_or_deliberate/code \
%%             .venv/bin/python c10_compile_or_deliberate/code/experiments/analyze.py
%% Load once in the preamble: \input{{generated/numbers}} (needs siunitx for \num, which the manuscript loads).
%%
%% NAMING SCHEME  \<metric><Suite><Arm>  (letters only; digits are spelled out, 25000 -> TwentyFiveK)
%%   Suite : Anchor (Anchor-13, 3 seeds = 39 ep.), Frontier (Frontier-5 x 3 = 15), Id (held-out anticipated, 98),
%%           Ood (held-out novel, 64), Unres (oracle-unrescued, 16), Held (Id + Ood pooled, 162)
%%   Arm   : Weak, Strong, Oracle (perfect-twin search), BTwoa (B2a), BTwob (B2b, privileged expert table),
%%           FDD (B3), Search (B4, nominal-twin verifier search),
%%           GP<ckpt>S<seed>        GP over the rule DSL, e.g. GPEvFiveKSOne (5k evaluations, seed 1), GPFinalSTwo
%%           Pb<ckpt>S<seed>        LLM playbook of the main Coder-30B run, e.g. PbVzeroSOne, PbOneHundredKSOne, PbFinalSThree
%%           Pb<Tag><ckpt>S<seed>   ablation runs: PbNoremFinalSOne (no counterfactual remedies), PbNotuneFinalSOne (no CPU tuning)
%%           PbCoderSeven<ckpt>S<seed> Coder-7B synthesis run; PbArch... archived pre-tuning runs; PbPreview... preview playbook
%%           OnlGated[Model][Tok<budget>]  gated online deliberation (main model Coder-30B unless a model key is
%%                                  present, e.g. OnlGatedQwenFourteen); Tok<budget> = per-episode token bucket, e.g. OnlGatedTokEightK
%%           OnlAll...              online deliberation at every invocation
%%           Hyb<ckpt>S<seed>Tok<budget>  hybrid = that playbook checkpoint + gated escalation, e.g. HybFinalSOneTokSixteenK,
%%                                  HybOneHundredKSOneTokInf (unbounded bucket)
%%   ckpt  : Vzero (v0), TwentyFiveK, FiftyK, OneHundredK, TwoHundredK, FourHundredK (synthesis-token checkpoints),
%%           Final, EvFiveK / EvTwentyK (GP evaluation checkpoints), Gzero
%%
%% METRICS (unit included in the value where noted)
%%   succ<Suite><Arm>    successes k/n                 rate<Suite><Arm>  success rate, "xx.x\,\%"
%%   ci<Suite><Arm>      Wilson 95 % CI "[lo, hi]\,\%"  nEp<Suite><Arm>   number of episodes analysed
%%   fam<F><Arm> / famRate<F><Arm>   per-family k/n and %, F in A S P Q SN V C K (held-out suites)
%%   tokEp, ptokEp, ctokEp <Arm>  mean LLM tokens (total / prompt / completion) per held-out episode (\num)
%%   callsEp<Arm> LLM calls/ep.; actEp<Arm> accepted actions/ep.; actBenign<Arm> actions/ep. on benign scenarios
%%   escEp<Arm> escalations/ep. (hybrids); escAEp/escBEp/escCEp<Arm> per trigger (a) unexplained abnormality,
%%   (b) forward-check failure, (c) conformal novelty (an escalation with two triggers counts in both);
%%   escTot<Arm> total escalations; escShare<Arm> share of episodes with >=1 escalation; suppEp<Arm> triggers that
%%   fired but were not escalated (cooldown, cap or empty bucket) per episode; exhaustEp<Arm> invocations refused
%%   because the token bucket was empty, per episode
%%   latMed<Arm>, latIQR<Arm>  median [IQR] wall-clock latency of an LLM decision in s (uncached calls only; the
%%   machine was shared, so latencies are pessimistic); latN<Arm> number of timed decisions
%%   offTok<Arm> offline synthesis tokens of the arm's playbook (\num); amortTok<Arm> offline/N + online tokens per
%%   episode at N = \amortN; budgetB<Arm> total tokens over a deployment of N = \amortN episodes;
%%   allocF<Arm> allocation f = offline / total at N = \amortN; allocFLarge<Arm> the same at N = \amortNLarge
%%   status<Arm> complete | partial (k of n planned episodes) | pending
%%   wallEp<Arm> mean wall-clock s per held-out episode (includes the plant simulation, so it grows with survival time);
%%   latDec<Arm> median added wall-clock s per supervisor invocation vs the weak baseline and wallAdd<Arm> median added
%%   wall-clock s per episode, both paired on held-out episodes in which arm AND weak baseline ran the full 48 h (features +
%%   decision + validator shadow rollout; an upper bound on the decision itself); latEp<Arm> mean LLM s per episode
%%   (online/hybrid, uncached episodes); prod<Arm> / opCost<Arm> mean production (m3, integral of stripper underflow) /
%%   operating cost per held-out episode (a trip ends the integral early); prodStab<Arm> mean production over the held-out
%%   episodes the arm stabilised to 48 h (aliases prodGp / prodExpert point here);
%%   substEp<Arm> accepted instrument substitutions per episode; repSubstEp<Arm> proposals to substitute an already
%%   substituted transmitter per episode; substSN<Arm> SN-family episodes with >= 1 accepted substitution (k/n);
%%   breakEven<Arm> episodes after which the arm (offline + online tokens) has used fewer tokens than online gated
%%   deliberation; diagAny<Arm> share of episodes with a correct diagnosis at some decision
%% ALIASES (end of file) for names used in sections/results.tex: prodExpert, prodGp, latEpOnline, latEpSearch (->
%%   wallAddSearch), pGpEarlyVsGpOod, subOnlineSN, cbProposed, cbRejected, wallSynthMain, breakEven, latPlaybook (-> latDec;
%%   includes the validator's shadow rollout, so it bounds the playbook decision time from above; the last two follow
%%   \mainPbArm = latest evaluated seed-1 main checkpoint, the preview playbook until main checkpoints are evaluated)
%% PAIRED STATISTICS vs a reference R in {{BTwob, Strong, FDD, OnlGated, Weak}} (paired by suite/scenario/seed):
%%   dVs<R><Suite><Arm>   success difference arm - R in percentage points, "$+$12.3\,pp"
%%   dciVs<R><Suite><Arm> Newcombe (method 10) 95 % CI "[$+$5.1, $+$19.4]\,pp"
%%   pVs<R><Suite><Arm>   exact McNemar p, Holm-adjusted over all candidate arms compared with R on that suite
%%                        (references weak/strong/oracle are not candidates; the family grows as arms are added)
%%   prawVs<R><Suite><Arm> unadjusted exact McNemar p;  bcVs<R><Suite><Arm> discordant pairs b/c
%%                        (b = arm succeeds and R fails, c = the reverse);  nVs<R><Suite><Arm> paired episodes
%%   hybGain*/hybGainCi*/hybGainP*<Suite><Arm>: hybrid arm vs its own playbook checkpoint (unadjusted p)
%% NAMED CONTRASTS  cDiff/cCi/cP/cBc/cN<Name><Suite> (A minus B, paired, exact McNemar unadjusted), Name in
%%   TuneAbl (s1 final - no-tuning s1 final), RemAbl (s1 final - no-remedy s1 final), PbVsGp (s1 final - GP s1 final),
%%   PbVsGpFiveK (s1 final - GP s1 @5k evals), PbFinalVsHundredK, PbHundredKVsVzero, HybVsOnl (hybrid s1 final 16k/ep. -
%%   online gated), PbVsOnl (s1 final - online gated), GpEarlyVsGp (GP s1 @5k evals - GP s1 @20k evals)
%% SYNTHESIS  syn<Metric><Run>[<ckpt>]: Run in MainSOne..MainSThree, MainCoderSevenSOne, NoremSOne, NotuneSOne,
%%   ArchNotuneSOne.., GPSOne..GPSThree. synTok<Run><ckpt> tokens actually spent when the checkpoint was saved (\num),
%%   synSr<Run><ckpt> incumbent training-batch success at that checkpoint (%), synEvals<Run><ckpt> (GP) evaluations;
%%   synTokTotal, synCalls, synIters, synLint, synFals, synEp (CPU episodes), synTuneN (tuning passes that improved the score), synTuneGainN (... that raised success),
%%   synTuneBest ("a $\to$ b"), synTuneMaxGain (pp), synTuneFirst / synTuneFirstEvals (first improving pass: "a $\to$ b",
%%   batch evaluations), synTuneEvals (batch evaluations summed over ALL improving passes), synSrNow (current incumbent training success, %), synStatus,
%%   synTokToNinety (tokens until training success >= 90 %), synLatMed (median s per uncached coder call),
%%   synWall (wall-clock hours since the run (re)started; [partial] while running; alias wallSynthMain stays pending
%%   until the seed-1 main run is done), synEp<Run><ckpt> CPU episodes at a checkpoint
%% COVERAGE  cov<Metric><Name> (Name: LlmFinalSOne.., GpFinalSOne): covN, covFail (k/n), covRate, covUpper (one-sided
%%   95 % Clopper-Pearson upper bound on the failure probability, %), covAdvRate (independent adversary, first round),
%%   covAdvFinal (failure rate among the perturbed neighbours of earlier adversary failures in the last local-search
%%   round; conditional on failure, NOT a distribution-level rate), covShiftRate (novel families V/C/K), covAdvWithin
%%   (yes/no: first-round adversary failure rate <= bound)
%% STREAM  str<Metric><Name> (Name: CbSOneFinalSixteenK = with compile-back, NocbSOneFinalSixteenK = without):
%%   strSucc (k/n), strRate, strBlocks, strRules (admitted rules), strEscEp, strEscFirst/strEscLast (escalations/ep. in
%%   the first/last third of the stream), strTok (online tokens), strCbTok (compile-back tokens), strCbProposed /
%%   strCbRejected (compile-back rules proposed / rejected); strDiff, strDiffCi, strDiffP (with minus without, paired)
%% CASE STUDY  case*: scenario, description and per-arm outcome of the trajectory figure (fig_case.pdf)
%% \codpending renders \textbf{{[pending]}} (arm or run not available yet); \codpartial{{x}} marks values computed on an
%% incomplete set of episodes (run in progress). Values regenerate from data; never edit numbers by hand."""


# ======================================================================================= per-arm summaries
def summarise(df: pd.DataFrame, ext: pd.DataFrame, expected: list) -> dict:
    exp_suites = {a: s for a, s, _ in expected}
    exp_stage = {a: st for a, _, st in expected}
    arms = sorted(set(df["arm"]) | set(exp_suites), key=sort_key)
    ex_by_arm = {a: g for a, g in ext.groupby("arm")} if len(ext) else {}
    out = {}
    for arm in arms:
        g = df[df["arm"] == arm]
        planned = exp_suites.get(arm, tuple(sorted(set(g["suite"]))))
        r = {"arm": arm, "name": pretty(arm), "key": arm_key(arm), "group": group(arm), "stage": exp_stage.get(arm, ""),
             "planned": list(planned), "suites": {}, "has_data": len(g) > 0, "main": is_main_arm(arm)}
        for s in SUITES:
            x = g[g["suite"] == s]
            if len(x):
                k, n = int(x["success"].sum()), len(x)
                lo, hi = wilson_ci(k, n)
                r["suites"][s] = {"k": k, "n": n, "lo": lo, "hi": hi, "rate": k / n, "partial": n < SUITE_N.get(s, n)}
        held = g[g["suite"].isin(HELD)]
        if len(held):
            k, n = int(held["success"].sum()), len(held)
            lo, hi = wilson_ci(k, n)
            r["suites"]["held"] = {"k": k, "n": n, "lo": lo, "hi": hi, "rate": k / n,
                                   "partial": n < sum(SUITE_N[s] for s in HELD)}
        n_plan = sum(SUITE_N.get(s, 0) for s in planned)
        n_have = int(g["suite"].isin(planned).sum()) if len(g) else 0
        r["status"] = "pending" if not len(g) else ("complete" if n_have >= n_plan else f"partial ({n_have} of {n_plan})")
        # cost / workload on the held-out suites (fall back to all suites if no held-out episodes)
        base = held if len(held) else g
        r["n_cost"] = len(base)
        for col, nm in (("tokens", "tok_ep"), ("prompt_tokens", "ptok_ep"), ("completion_tokens", "ctok_ep"),
                        ("llm_calls", "calls_ep"), ("n_accepted", "act_ep"), ("calls", "inv_ep")):
            r[nm] = float(base[col].mean()) if len(base) else float("nan")
        ben = base[base["label"] == "benign"] if len(base) else base
        r["act_benign"] = float(ben["n_accepted"].mean()) if len(ben) else float("nan")
        r["diag_any"] = float(base["diag_correct_any"].dropna().astype(float).mean()) if len(base) and base["diag_correct_any"].notna().any() else float("nan")
        r["wall_ep"] = float(base["wall_s"].mean()) if len(base) else float("nan")
        r["dec_overhead"], r["ep_overhead"], r["overhead_n"] = float("nan"), float("nan"), 0
        if arm != "weak" and len(base) and base["calls"].sum() > 0 and (df["arm"] == "weak").any():
            W = df[df["arm"] == "weak"].set_index(["suite", "scenario", "seed"])
            X = base.set_index(["suite", "scenario", "seed"])
            idx = X.index.intersection(W.index)
            # only pairs in which BOTH runs simulated the full horizon: otherwise the wall-clock difference is
            # dominated by the extra hours the supervised run survived, not by the supervisor's own computation
            idx = idx[(X.loc[idx, "stabilized"].values & W.loc[idx, "stabilized"].values).astype(bool)]
            if len(idx):
                inv = X.loc[idx, "calls"].clip(lower=1).astype(float)
                add = X.loc[idx, "wall_s"] - W.loc[idx, "wall_s"]
                r["dec_overhead"] = float(np.median(add / inv))
                r["ep_overhead"] = float(np.median(add))
                r["overhead_n"] = int(len(idx))
        r["prod_mean"] = float(base["production"].mean()) if len(base) else float("nan")
        st_ = base[base["stabilized"]] if len(base) else base
        r["prod_stab"] = float(st_["production"].mean()) if len(st_) else float("nan")
        r["n_stab"] = int(len(st_))
        r["cost_mean"] = float(base["cost"].mean()) if len(base) else float("nan")
        e = ex_by_arm.get(arm)
        if e is not None and len(e):
            eh = e[e["suite"].isin(HELD)] if e["suite"].isin(HELD).any() else e
            r["esc_ep"] = float(eh["n_esc"].mean())
            r["esc_tot"] = int(eh["n_esc"].sum())
            r["esc_share"] = float((eh["n_esc"] > 0).mean())
            for code, _, _ in TRIGGERS:
                r[f"esc_{code}_ep"] = float(eh["esc_" + code].mean())
                r[f"esc_{code}_tot"] = int(eh["esc_" + code].sum())
            r["supp_ep"] = float(eh["n_supp"].mean())
            r["exhaust_ep"] = float(eh["n_exhausted"].mean())
            lat = [v for L in e["lat"] for v in L]  # all suites: latency is a property of the call, not the suite
            r["lat_n"] = len(lat)
            r["lat_med"] = float(np.median(lat)) if lat else float("nan")
            r["lat_q1"], r["lat_q3"] = (float(np.quantile(lat, 0.25)), float(np.quantile(lat, 0.75))) if lat else (float("nan"),) * 2
            r["llm_dec"] = int(e["n_llm_dec"].sum())
            r["cached_dec"] = int(e["n_cached_dec"].sum())
            r["subst_ep"] = float(eh["n_subst"].mean())
            r["rep_subst_ep"] = float(eh["n_subst_rep"].mean())
            un = eh[(eh["n_cached_dec"] == 0)]
            r["lat_ep"] = float(un["gpu_s_ep"].mean()) if len(un) and r["group"] in ("online", "hybrid") else float("nan")
            r["lat_ep_n"] = int(len(un))
            sn = eh.merge(df[df["arm"] == arm][["suite", "scenario", "seed", "family"]], on=["suite", "scenario", "seed"])
            sn = sn[sn["family"] == "SN"]
            r["subst_sn"] = (int((sn["n_subst"] > 0).sum()), int(len(sn)))
        oc = offline_cost(arm)
        r["off_tok"], r["off_cpu"], r["off_src"] = oc["tokens"], oc["cpu_episodes"], oc["source"]
        on = r["tok_ep"] if np.isfinite(r["tok_ep"]) else 0.0
        off = r["off_tok"] if np.isfinite(r["off_tok"]) else 0.0
        for nm, N in (("", NH()), ("_large", N_LARGE)):
            tot = off + N * on
            r["budget" + nm] = tot
            r["amort" + nm] = tot / N
            r["f" + nm] = off / tot if tot > 0 else float("nan")
        out[arm] = r
    return out


def NH() -> int:
    return sum(SUITE_N[s] for s in HELD)


# ======================================================================================= paired statistics
def paired_stats(df: pd.DataFrame, arms: list[str]) -> list[dict]:
    rows = []
    idx_cols = ["suite", "scenario", "seed"]
    present = set(df["arm"])
    series = {a: df[df["arm"] == a].set_index(idx_cols)["success"] for a in present}
    for ref, rkey in REF_ARMS:
        if ref not in present:
            continue
        R = series[ref]
        for suite in ("held", "test_id", "test_ood"):
            fam = []
            for arm in arms:
                if arm == ref or arm not in present or group(arm) == "ref":
                    continue  # references (weak, strong, oracle) are not candidates
                A = series[arm]
                sel = HELD if suite == "held" else (suite,)
                Ai = A[A.index.get_level_values(0).isin(sel)]
                idx = R.index.intersection(Ai.index)
                if len(idx) < 10:
                    continue
                a = Ai.loc[idx].values.astype(bool)
                b = R.loc[idx].values.astype(bool)
                bb, cc, p = mcnemar_exact(a, b)
                d, lo, hi = newcombe_paired_diff_ci(a, b)
                fam.append({"ref": ref, "ref_key": rkey, "arm": arm, "suite": suite, "n": len(idx), "diff": d, "lo": lo,
                            "hi": hi, "b": bb, "c": cc, "p": p, "rate": float(a.mean()), "rate_ref": float(b.mean()),
                            "partial": len(idx) < (NH() if suite == "held" else SUITE_N[suite])})
            if fam:
                adj = holm([r["p"] for r in fam])
                for r, q in zip(fam, adj):
                    r["p_holm"] = float(q)
                    r["holm_m"] = len(fam)
            rows += fam
    return rows


def named_contrasts() -> list[tuple[str, str, str, str]]:
    """(macro name, arm A, arm B, description) for paired contrasts quoted in the text (A minus B)."""
    pb1 = lambda c, tag="main": f"playbook:{_pb(1, c, tag)}"  # noqa: E731
    return [("TuneAbl", pb1("final"), pb1("final", "notune"), "CPU tuning ablation: main s1 final minus no-tuning s1 final"),
            ("RemAbl", pb1("final"), pb1("final", "norem"), "remedy ablation: main s1 final minus no-remedy s1 final"),
            ("PbVsGp", pb1("final"), f"playbook:{_gp(1, 'final')}", "LLM playbook s1 final minus GP s1 final"),
            ("PbVsGpFiveK", pb1("final"), f"playbook:{_gp(1, 'E5000')}", "LLM playbook s1 final minus GP s1 @5k evals"),
            ("GpEarlyVsGp", f"playbook:{_gp(1, 'E5000')}", f"playbook:{_gp(1, 'E20000')}",
             "GP s1 @5k evals minus GP s1 @20k evals (= final)"),
            ("PbFinalVsHundredK", pb1("final"), pb1("B100000"), "more synthesis: s1 final minus s1 @100k"),
            ("PbHundredKVsVzero", pb1("B100000"), pb1("v0"), "repair + falsification: s1 @100k minus s1 v0"),
            ("HybVsOnl", f"hybrid:{_pb(1, 'final')}:{MAIN_MODEL}@16000", f"onlineg:{MAIN_MODEL}",
             "hybrid s1 final 16k/ep. minus online gated (unbounded)"),
            ("PbVsOnl", pb1("final"), f"onlineg:{MAIN_MODEL}", "playbook s1 final minus online gated (unbounded)")]


def contrast_stats(df: pd.DataFrame) -> list[dict]:
    out = []
    present = set(df["arm"])
    for name, a_arm, b_arm, desc in named_contrasts():
        for suite in ("held", "test_id", "test_ood"):
            r = {"name": name, "a": a_arm, "b": b_arm, "desc": desc, "suite": suite, "available": False}
            if a_arm in present and b_arm in present:
                sel = HELD if suite == "held" else (suite,)
                X = df[(df["arm"] == a_arm) & df["suite"].isin(sel)].set_index(["suite", "scenario", "seed"])["success"]
                Y = df[(df["arm"] == b_arm) & df["suite"].isin(sel)].set_index(["suite", "scenario", "seed"])["success"]
                idx = X.index.intersection(Y.index)
                if len(idx) >= 10:
                    x, y = X.loc[idx].values.astype(bool), Y.loc[idx].values.astype(bool)
                    b, c, p = mcnemar_exact(x, y)
                    d, lo, hi = newcombe_paired_diff_ci(x, y)
                    full = NH() if suite == "held" else SUITE_N[suite]
                    r.update(available=True, n=len(idx), diff=d, lo=lo, hi=hi, b=b, c=c, p=p, partial=len(idx) < full)
            out.append(r)
    return out


def hybrid_gain(df: pd.DataFrame, arms: list[str]) -> list[dict]:
    """Each hybrid arm vs the playbook it escalates from (same checkpoint), paired."""
    out = []
    present = set(df["arm"])
    for arm in arms:
        a = parse_arm(arm)
        if a["kind"] != "hybrid" or arm not in present:
            continue
        pb = f"playbook:{a['path']}"
        if pb not in present:
            continue
        for suite in ("held", "test_id", "test_ood"):
            sel = HELD if suite == "held" else (suite,)
            H = df[(df["arm"] == arm) & df["suite"].isin(sel)].set_index(["suite", "scenario", "seed"])["success"]
            P = df[(df["arm"] == pb) & df["suite"].isin(sel)].set_index(["suite", "scenario", "seed"])["success"]
            idx = H.index.intersection(P.index)
            if len(idx) < 10:
                continue
            x, y = H.loc[idx].values.astype(bool), P.loc[idx].values.astype(bool)
            b, c, p = mcnemar_exact(x, y)
            d, lo, hi = newcombe_paired_diff_ci(x, y)
            out.append({"arm": arm, "playbook": pb, "suite": suite, "n": len(idx), "diff": d, "lo": lo, "hi": hi, "b": b,
                        "c": c, "p": p})
    return out


# ======================================================================================= synthesis runs
def parse_synth_log(path: str) -> dict:
    """Events of a synthesis log: incumbent curve, falsification rounds, tuning events, invalid candidates."""
    curve, tune, fals, ckpts = [], [], [], []
    iters, invalid, last_tok, done, gp, wall = 0, 0, 0, False, False, 0.0
    if not os.path.exists(path):
        return {"curve": curve, "tune": tune, "fals": fals, "ckpts": ckpts, "iters": 0, "invalid": 0, "done": False,
                "gp": False, "last_x": 0}
    for ln in open(path):
        mw = re.search(r"\((\d+)s\)\s*$", ln)
        if mw:
            wall = max(wall, float(mw.group(1)))
        m = re.search(r"\[it (\d+)\] v0 sr=([0-9.]+).*tokens=(\d+)", ln)
        if m:
            iters = max(iters, int(m.group(1)))
            last_tok = int(m.group(3))
            curve.append((last_tok, float(m.group(2)), "v0"))
            continue
        m = re.search(r"\[it (\d+)\] child sr=([0-9.]+).*\| best ([0-9.]+) tokens=(\d+)", ln)
        if m:
            iters = max(iters, int(m.group(1)))
            last_tok = int(m.group(4))
            curve.append((last_tok, float(m.group(3)), "it"))
            continue
        m = re.search(r"\[gen (\d+)\] best sr=([0-9.]+).*evals=(\d+)", ln)
        if m:
            gp = True
            iters = max(iters, int(m.group(1)) + 1)
            last_tok = int(m.group(3))
            curve.append((last_tok, float(m.group(2)), "gen"))
            continue
        if re.search(r"\[it (\d+)\] invalid playbook", ln):
            invalid += 1
            m = re.search(r"\[it (\d+)\]", ln)
            iters = max(iters, int(m.group(1)))
            continue
        m = re.search(r"falsification round (\d+): incumbent failed (\d+)/(\d+).*batch -> (\d+)", ln)
        if m:
            fals.append({"x": last_tok, "round": int(m.group(1)), "failed": int(m.group(2)), "fresh": int(m.group(3)),
                         "batch": int(m.group(4))})
            continue
        m = re.search(r"tuned incumbent: ([0-9.]+) -> ([0-9.]+) \((\d+) batch evals", ln)
        if m:
            a, b = float(m.group(1)), float(m.group(2))
            tune.append({"x": last_tok, "from": a, "to": b, "evals": int(m.group(3))})
            curve.append((last_tok, a, "pre-tune"))
            curve.append((last_tok, b, "tune"))
            continue
        m = re.search(r"checkpoint (\w+): (?:tokens|evals)=(\d+) best (?:it|gen)=\d+ sr=([0-9.]+)", ln)
        if m:
            x = int(m.group(2))
            ckpts.append({"name": m.group(1), "x": x, "sr": float(m.group(3))})
            curve.append((x, float(m.group(3)), "ckpt"))
            last_tok = max(last_tok, x)
            continue
        if ln.startswith("DONE"):
            done = True
    return {"curve": curve, "tune": tune, "fals": fals, "ckpts": ckpts, "iters": iters, "invalid": invalid,
            "done": done, "gp": gp, "last_x": last_tok, "wall_s": wall}


def run_key(name: str, archived: bool = False) -> str:
    m = re.match(r"gp_s(\d+)$", name)
    if m:
        return "GP" + seed_key(m.group(1))
    m = re.match(r"([a-z0-9]+)_(.+?)_s(\d+)$", name)
    if not m:
        return letters_key(name)
    tag, model, seed = m.groups()
    k = ("Arch" if archived else "") + tag[:1].upper() + tag[1:]
    if model_short(model) != model_short(MAIN_MODEL):
        k += model_key(model)
    return k + seed_key(seed)


def run_label(name: str, archived: bool = False) -> str:
    m = re.match(r"gp_s(\d+)$", name)
    if m:
        return f"GP s{m.group(1)}"
    m = re.match(r"([a-z0-9]+)_(.+?)_s(\d+)$", name)
    if not m:
        return name
    tag, model, seed = m.groups()
    s = f"{model_short(model)} s{seed}"
    if tag != "main":
        s += f", {TAG_SHORT.get(tag, tag)}"
    return s


def _running_cmdlines() -> list[str]:
    try:
        import psutil
        out = []
        for p in psutil.process_iter(["cmdline"]):
            c = " ".join(p.info.get("cmdline") or [])
            if "synth_run.py" in c or "gp_run.py" in c:
                out.append(c)
        return out
    except Exception:  # noqa: BLE001 - status is cosmetic
        return []


def _is_running(name: str, procs: list[str]) -> bool:
    m = re.match(r"gp_s(\d+)$", name)
    if m:
        return any("gp_run.py" in c and re.search(rf"--seed {m.group(1)}\b", c) for c in procs)
    m = re.match(r"([a-z0-9]+)_(.+?)_s(\d+)$", name)
    if not m:
        return False
    tag, model, seed = m.groups()
    for c in procs:
        mm = re.search(r"--model (\S+)", c)
        if ("synth_run.py" in c and re.search(rf"--seed {seed}\b", c) and re.search(rf"--tag {tag}\b", c)
                and mm and mm.group(1).replace(":", "-") == model):
            return True
    return False


def synthesis_runs() -> list[dict]:
    runs = []
    dirs = [(d, False) for d in sorted(glob.glob(os.path.join(SYN, "*_s*")))]
    dirs += [(d, True) for d in sorted(glob.glob(os.path.join(SYN_ARCH, "*_s*")))]
    seen = set()
    procs = _running_cmdlines()
    for d, arch in dirs:
        name = os.path.basename(d)
        if not arch:
            seen.add(name)
        metas = []
        for p in glob.glob(os.path.join(d, "meta_*.json")):
            try:
                metas.append(json.load(open(p)))
            except json.JSONDecodeError:
                continue
        log = parse_synth_log(os.path.join(d, "log.txt"))
        is_gp = name.startswith("gp_")
        xkey = "evals" if is_gp else "tokens"
        metas.sort(key=lambda m: (m.get(xkey, 0), ckpt_rank(m.get("name"))))
        last = metas[-1] if metas else {}
        hist = [h for m in metas[-1:] for h in m.get("history", [])]
        lat = [h["latency"] for h in hist if not h.get("cached") and h.get("latency")]
        cur = [c for c in log["curve"]]
        sr_now = cur[-1][1] if cur else last.get("best_sr", float("nan"))
        tok_total = max([log["last_x"]] + [m.get(xkey, 0) for m in metas]) if (metas or cur) else float("nan")
        reach90 = next((x for x, y, _ in cur if y >= 0.9), None)
        r = {"name": name, "key": run_key(name, arch), "label": run_label(name, arch), "archived": arch, "gp": is_gp,
             "model": None if is_gp else (metas[0].get("model") if metas else None),
             "seed": int(re.search(r"_s(\d+)$", name).group(1)),
             "status": "done" if log["done"] else ("archived" if arch else
                                                   ("running" if _is_running(name, procs) else "stopped")),
             "x_total": tok_total, "calls": last.get("calls"), "iters": log["iters"] if log["iters"] else last.get("generation"),
             "lint": last.get("lint_fixes"), "invalid": log["invalid"], "fals": len(log["fals"]),
             "fals_failed": sum(f["failed"] for f in log["fals"]), "tune": log["tune"],
             "cpu_ep": last.get("episodes") if not is_gp else last.get("evals"),
             "sr_now": sr_now, "batch": last.get("batch_size"), "rules": len(last.get("rules", []) or []),
             "loc": last.get("loc"), "reach90": reach90, "lat_med": float(np.median(lat)) if lat else float("nan"),
             "ckpts": {m["name"]: {"x": m.get(xkey), "sr": m.get("best_sr"), "episodes": m.get("episodes")} for m in metas},
             "curve": cur, "fals_events": log["fals"], "remedy": last.get("remedy"), "tune_budget": last.get("tune_budget"),
             "gp_history": last.get("history") if is_gp else None,
             "wall_h": (log["wall_s"] / 3600.0) if log["wall_s"] else (last.get("wall_s", 0.0) or 0.0) / 3600.0}
        runs.append(r)
    for name in EXPECTED_RUNS:
        if name not in seen:
            runs.append({"name": name, "key": run_key(name), "label": run_label(name), "archived": False,
                         "gp": name.startswith("gp_"), "status": "pending", "ckpts": {}, "curve": [], "tune": [],
                         "seed": int(re.search(r"_s(\d+)$", name).group(1))})
    order = {"main": 0, "norem": 2, "notune": 3}
    runs.sort(key=lambda r: (r["archived"], r["gp"], order.get(r["name"].split("_")[0], 4),
                             0 if "30b" in r["name"] or r["gp"] else 1, r["seed"]))
    return runs


# ======================================================================================= coverage + stream
def coverage_results() -> list[dict]:
    out = []
    for p in sorted(glob.glob(os.path.join(COV, "*.json"))):
        if os.path.basename(p).startswith("labels_"):
            continue
        try:
            r = json.load(open(p))
        except json.JSONDecodeError:
            continue
        if "in_dist" not in r:
            continue
        r["name"] = r.get("name") or os.path.basename(p)[:-5]
        advs = sorted(k for k in r if k.startswith("adversary_round"))
        r["adv_first"] = r[advs[0]] if advs else None
        r["adv_last"] = r[advs[-1]] if advs else None
        out.append(r)
    return out


def coverage_label(name: str) -> str:
    m = re.match(r"(llm|gp)_(\w+?)_s(\d+)$", name)
    if m:
        return f"{'LLM playbook' if m.group(1) == 'llm' else 'GP'} s{m.group(3)} {m.group(2)}"
    return name.replace("_", " ")


def stream_results() -> list[dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(STREAM, "*"))):
        if not os.path.isdir(d):
            continue
        blocks = []
        for p in sorted(glob.glob(os.path.join(d, "block*.json"))):
            try:
                b = json.load(open(p))
            except json.JSONDecodeError:
                continue
            eps = b.get("episodes", [])
            last_log = b.get("log") or []
            blocks.append({"block": int(re.search(r"block(\d+)", p).group(1)), "n": len(eps),
                           "k": int(sum(bool(e.get("success")) for e in eps)),
                           "esc": int(sum(len(e.get("escalations") or []) for e in eps)),
                           "tok": int(sum(int(e.get("prompt_tokens", 0)) + int(e.get("completion_tokens", 0)) for e in eps)),
                           "cb_tok": int(b.get("compile_tokens", 0)), "admitted": b.get("admitted") is not None,
                           "version": int(b.get("version_after", 0)),
                           "outcomes": {(e.get("scenario"), int(e.get("seed", 0))): bool(e.get("success")) for e in eps}})
        if blocks:
            blocks.sort(key=lambda b: b["block"])
            out.append({"name": os.path.basename(d), "blocks": blocks, "log": last_log})
    return out


# ======================================================================================= LaTeX tables
def short_name(arm: str) -> str:
    """Name relative to its table section (the section header carries the method family)."""
    a = parse_arm(arm)
    if a["kind"] in ("online", "onlineg"):
        s = "gated" if a["kind"] == "onlineg" else "every call"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            s += f", {model_short(a['model'])}"
        return s[:1].upper() + s[1:] + ("" if a["budget"] is None else f", {budget_label(a['budget'])}")
    if a["kind"] == "playbook" and a["group"] == "playbook" and a["seed"] is not None:
        ck = a["ckpt"] if a["ckpt"] in ("v0", "final") else "@" + ckpt_label(a["ckpt"])
        s = (f"{model_short(a['model'])} " if model_short(a["model"]) != model_short(MAIN_MODEL) else "") + f"s{a['seed']} {ck}"
        if a["tag"] != "main":
            s += f", {TAG_SHORT.get(a['tag'], a['tag'])}"
        if a["origin"] == "archive":
            s += " (archived)"
        if a["origin"] == "preview":
            s += " (preview)"
        return s
    if a["kind"] == "hybrid":
        ck = a["ckpt"] if a["ckpt"] in ("v0", "final") else "@" + ckpt_label(a["ckpt"])
        s = f"s{a['seed']} {ck} + {budget_label(a['budget'])}"
        if model_short(a["model"]) != model_short(MAIN_MODEL):
            s += f", {model_short(a['model'])}"
        if a.get("pb_tag") not in (None, "main"):
            s += f", {TAG_SHORT.get(a['pb_tag'], a['pb_tag'])}"
        if a["opts"]:
            s += f" [{a['opts']}]"
        return s
    return pretty(arm)


def stats_keep(arm: str, present: set) -> bool:
    """Rows of the paired-statistics table (all comparisons are in the macros): candidate arms only, main budget
    variants (online unbounded; hybrid 16k/ep. and unbounded), playbook checkpoints @100k and final."""
    a = parse_arm(arm)
    if a["group"] == "ref" or not main_table_keep(arm, present):
        return False
    if a["group"] == "online":
        return a["budget"] is None
    if a["group"] == "playbook":
        return a["ckpt"] in ("B100000", "final") or arm in PROMOTED
    if a["group"] == "hybrid":
        return a["budget"] in (None, 16000.0)
    return True


def action_phrase(a: dict) -> str:
    """Noun phrase for an action record, e.g. 'substitution of the reactor temperature transmitter by its soft sensor'."""
    t = a.get("type", "")
    if t == "substitute_instrument":
        return f"substitution of the {a.get('measurement', '').replace('_', ' ')} transmitter by its soft sensor"
    if t == "adjust_setpoints":
        sp = {"pct_g": ("product-grade setpoint", "mol%"), "reactor_pressure": ("reactor pressure", "kPa"),
              "reactor_temperature": ("reactor temperature", "degC"), "production_rate": ("production rate", "m3/h"),
              "reactor_level": ("reactor level", "%"), "separator_level": ("separator level", "%"),
              "stripper_level": ("stripper level", "%"), "ya": ("A in reactor feed", "mol%"),
              "yac": ("A+C in reactor feed", "mol%")}
        ch = ", ".join(f"{sp.get(k, (k.replace('_', ' '), ''))[0]} to {v:g} {sp.get(k, ('', ''))[1]}".rstrip()
                       for k, v in (a.get("changes") or {}).items())
        return f"a setpoint change ({ch})"
    if t == "reconfigure":
        names = {"composition_control": "composition trims", "safety_overrides": "safety overrides"}
        return "arming of the " + " and ".join(names.get(c, c.replace("_", " ")) for c in a.get("components", []))
    if t == "procedure":
        return f"the {a.get('name', '').replace('_', ' ')} procedure"
    if t == "shutdown":
        return "an orderly shutdown"
    return t.replace("_", " ")


def same_as_final(arm: str, present: set) -> bool:
    """True for a synthesis checkpoint whose code equals its run's final playbook when that final arm was also
    evaluated (the last token checkpoint usually is the final incumbent); such rows are duplicates."""
    a = parse_arm(arm)
    if a["kind"] != "playbook" or a["ckpt"] in ("final", None):
        return False
    path = arm[len("playbook:"):].partition("@")[0]
    fin = os.path.join(os.path.dirname(path), "playbook_final.py")
    if os.path.abspath(path) == os.path.abspath(fin) or f"playbook:{fin}" not in present:
        return False
    try:
        body = [open(q).read().split("\n", 2)[-1] for q in (path, fin)]
    except OSError:
        return False
    return body[0] == body[1]


def main_table_keep(arm: str, present: set) -> bool:
    """Rows of the main results table (results_all.tex lists every arm): drop intermediate synthesis checkpoints
    (200k/400k of the main runs), GP @20k evals when that seed's final checkpoint exists, and GP @5k of seeds 2-3."""
    a = parse_arm(arm)
    if not is_main_arm(arm):
        return False
    if a["kind"] == "playbook" and a["group"] == "playbook" and a["ckpt"] in ("B200000", "B400000", "B50000", "B100000"):
        return False  # intermediate checkpoints: Table S1 (the 25k checkpoint of seed 1 stays)
    if a["kind"] == "playbook" and a["group"] == "playbook" and (a["tag"] != "main" or a["ckpt"] in ("v0", "B25000")):
        return False  # ablations, calibration-only runs and early checkpoints: Table 4 (ablations) and Table S1
    if a["kind"] == "hybrid" and (a["ckpt"] != "final" or (a["budget"] != 16000.0 and not (a["seed"] == 1 and a["budget"] == 8000.0))):
        return False  # hybrids on the final playbooks with a 16k bucket (and 8k for seed 1); the rest in Table S1
    if a["group"] == "gp":
        if a["ckpt"] == "E20000" and f"playbook:{_gp(a['seed'], 'final')}" in present:
            return False
        if a["ckpt"] == "E5000" and a["seed"] != 1:
            return False
    return True


def ptab(s: str, is_partial: bool) -> str:
    return s + "\\textsuperscript{\\dag}" if is_partial else s


PEND_CELL = "\\textcolor{gray}{\\textit{pend.}}"
# Footnote fragments that only apply while runs are incomplete; dropped when no cell of the table is partial or pending.
_STALE_NOTES = (" \\dag\\ run in progress (fewer episodes than planned); pend. = planned, not yet run.",
                " pend. = planned, not yet available; \\dag\\ run in progress.",
                " \\dag\\ run in progress (paired on the available episodes).",
                " \\dag\\ stream in progress.", " \\dag\\ run in progress.")


def _write_table(path: str, L: list[str]):
    body = "\n".join(x for x in L if not x.startswith("\\item "))
    if "\\dag}" not in body and "pend." not in body and "[pending]" not in body:
        L = [x if not x.startswith("\\item ") else _strip_stale(x) for x in L]
    open(path, "w").write("\n".join(L) + "\n")


def _strip_stale(note: str) -> str:
    for frag in _STALE_NOTES:
        note = note.replace(frag, "")
    return note


def _open(ncol_spec: str, sep: str = "2.5pt") -> list[str]:
    return ["% GENERATED by code/experiments/analyze.py -- do not edit. Wrap in a table environment with a caption.",
            "\\begin{threeparttable}", f"\\linespread{{1}}\\scriptsize\\selectfont\\setlength{{\\tabcolsep}}{{{sep}}}",
            f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}{ncol_spec}@{{}}}}", "\\toprule"]


def _close(notes: str) -> list[str]:
    return ["\\bottomrule", "\\end{tabular*}", "\\begin{tablenotes}[flushleft]\\scriptsize", "\\item " + notes,
            "\\end{tablenotes}", "\\end{threeparttable}"]


def _section_row(title: str, ncol: int) -> str:
    return f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{title}}}}}\\\\"


def _pending_row(names: list[str], ncol: int) -> str:
    txt = "; ".join(tex(n) for n in names)
    return (f"\\multicolumn{{{ncol}}}{{@{{}}l@{{}}}}{{\\parbox[t]{{0.97\\linewidth}}{{\\raggedright\\leftskip=0.6em "
            f"\\textcolor{{gray}}{{\\textit{{pending:}} {txt}}}\\strut}}}}\\\\")


def _cell_kn(r: dict, s: str) -> str:
    v = r["suites"].get(s)
    if v is None:
        return PEND_CELL if s in r["planned"] else "--"
    return ptab(f"{v['k']}/{v['n']}", v["partial"])


def _cell_ci(r: dict, s: str) -> str:
    v = r["suites"].get(s)
    if v is None:
        return ""
    return f"{100 * v['lo']:.0f}--{100 * v['hi']:.0f}"


SECTION_TITLES = {"ref": "References", "llmfree": "LLM-free supervisors",
                  "online": f"Online LLM deliberation, {model_short(MAIN_MODEL)} unless stated ($f=0$)",
                  "playbook": f"Offline LLM playbook, {model_short(MAIN_MODEL)} coder unless stated ($f=1$)",
                  "hybrid": "Hybrid: playbook checkpoint + gated escalation, per-episode token bucket ($0<f<1$)"}


def _sections(rows: list[dict]):
    spec = [("ref", ("ref",)), ("llmfree", ("rule", "fdd", "search", "gp")), ("online", ("online",)),
            ("playbook", ("playbook",)), ("hybrid", ("hybrid",))]
    for key, groups in spec:
        sub = [r for r in rows if r["group"] in groups]
        if sub:
            yield SECTION_TITLES[key], sub


def table_results(rows: list[dict], path: str, note_extra: str = "", sep: str = "2pt"):
    ncol = 11
    L = _open("l" + "r" * (ncol - 1), sep)
    L += [" & \\multicolumn{2}{c}{Anchors} & \\multicolumn{5}{c}{Held-out test suites} & "
          "\\multicolumn{2}{c}{Per episode} & Once\\\\",
          "\\cmidrule(lr){2-3}\\cmidrule(lr){4-8}\\cmidrule(lr){9-10}\\cmidrule(l){11-11}",
          " & A-13 & F-5 & \\multicolumn{2}{c}{Anticipated} & \\multicolumn{2}{c}{Novel} & Unresc. & Actions & "
          "Online & Offline\\\\",
          "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
          "Arm & $k/n$ & $k/n$ & $k/n$ & CI (\\%) & $k/n$ & CI (\\%) & $k/n$ & & tokens & tokens\\\\", "\\midrule"]
    first = True
    for title, sub in _sections(rows):
        if not first:
            L.append("\\addlinespace")
        first = False
        L.append(_section_row(title, ncol))
        for r in sub:
            if not r["has_data"]:
                continue
            tok = fmt_tok(r["tok_ep"]) if r["group"] in ("online", "hybrid") else "0"
            off = fmt_tok(r["off_tok"]) if r["group"] in ("playbook", "hybrid") else "0"
            act = "--" if not np.isfinite(r["act_ep"]) else f"{r['act_ep']:.1f}"
            L.append(f"\\hspace{{0.6em}}{tex(short_name(r['arm']))} & " + " & ".join([
                _cell_kn(r, "alhazmi"), _cell_kn(r, "frontier"), _cell_kn(r, "test_id"), _cell_ci(r, "test_id"),
                _cell_kn(r, "test_ood"), _cell_ci(r, "test_ood"), _cell_kn(r, "test_unrescuable"), act, tok, off]) + "\\\\")
        pend = [short_name(r["arm"]) for r in sub if not r["has_data"]]
        if pend:
            L.append(_pending_row(pend, ncol))
    L += _close("Success = stabilised within all hard constraints at 48\\,h (and on grade for quality-gap scenarios); "
                "$k/n$ = successes/episodes; CI = Wilson 95\\,\\% score interval. A-13 and F-5 = the 13 published "
                f"scenarios and 5 frontier scenarios of the prior harness, three noise seeds each. Held-out suites: "
                f"{SUITE_N.get('test_id', '?')} anticipated-family, {SUITE_N.get('test_ood', '?')} novel-family and "
                f"{SUITE_N.get('test_unrescuable', '?')} oracle-unrescued scenarios, one noise seed each. Actions = "
                "accepted supervisory actions; online tokens = LLM prompt + completion tokens per held-out episode; offline "
                "tokens = LLM tokens spent once to synthesise the playbook (not per episode). sN = synthesis seed; @25k = "
                "first checkpoint saved after 25k synthesis tokens (actual tokens in the offline column); 16k/ep. = "
                "per-episode token bucket. The expert table (B2b) was written by the authors "
                "with knowledge of the fault mechanisms (privileged reference); the GP rule language is expert-curated. "
                "\\dag\\ run in progress (fewer episodes than planned); pend. = planned, not yet run." + note_extra)
    _write_table(path, L)


def table_family(rows: list[dict], df: pd.DataFrame, path: str):
    fams = [f for f in FAMILIES_ID + FAMILIES_OOD if f in FAMILY_N]
    held = df[df["suite"].isin(HELD)]
    ncol = len(fams) + 2
    nid = len([f for f in FAMILIES_ID if f in FAMILY_N])
    L = _open("l" + "r" * (ncol - 1), "3pt")
    L += [f" & \\multicolumn{{{nid}}}{{c}}{{Anticipated families}} & "
          f"\\multicolumn{{{len(fams) - nid}}}{{c}}{{Novel families}} & \\\\",
          f"\\cmidrule(lr){{2-{1 + nid}}}\\cmidrule(lr){{{2 + nid}-{1 + len(fams)}}}",
          "Arm & " + " & ".join(fams) + " & All\\\\",
          "\\quad episodes & " + " & ".join(str(FAMILY_N[f]) for f in fams) + f" & {NH()}\\\\", "\\midrule"]
    first = True
    for title, sub in _sections([r for r in rows if r["has_data"]]):
        if not first:
            L.append("\\addlinespace")
        first = False
        L.append(_section_row(title, ncol))
        for r in sub:
            g = held[held["arm"] == r["arm"]]
            cells = []
            for f in fams:
                x = g[g["family"] == f]
                cells.append("--" if not len(x) else ptab(f"{100 * x['success'].mean():.0f}", len(x) < FAMILY_N[f]))
            h = r["suites"].get("held")
            cells.append("--" if h is None else ptab(f"{100 * h['rate']:.0f}", h["partial"]))
            L.append(f"\\hspace{{0.6em}}{tex(short_name(r['arm']))} & " + " & ".join(cells) + "\\\\")
    L += _close("Success rate (\\%) per fault family on the held-out suites. Anticipated: A benign disturbances, S "
                "reactant-composition upset, P A-feed loss, Q grade change, SN reactor pressure/temperature transmitter "
                "fault. Novel families, not used in synthesis or training: V actuator fault, C compound fault, K kinetics, "
                "cooling or heat-transfer disturbance at high production. \\dag\\ run in progress.")
    _write_table(path, L)


def table_stats(rows: list[dict], stats: list[dict], path: str):
    want = ("strong", "B2b", f"onlineg:{MAIN_MODEL}")
    refs = [(r, None) for r in want if any(s["ref"] == r for s in stats)]
    if not refs:
        open(path, "w").write("% GENERATED: no reference arm with data yet\n\\textbf{[pending]}\n")
        return
    S = {(s["ref"], s["suite"], s["arm"]): s for s in stats}
    ncol = 1 + 2 * len(refs)
    L = _open("l" + "rr" * len(refs), "1.5pt")
    L += [" & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{vs {tex(pretty(r))}}}" for r, _ in refs) + "\\\\",
          "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(refs))),
          "Arm & " + " & ".join("$\\Delta$ [95\\,\\% CI] & $p_{\\mathrm{Holm}}$" for _ in refs) + "\\\\", "\\midrule"]
    for si, (suite, title) in enumerate((("test_id", f"Held-out anticipated ({SUITE_N['test_id']} episodes)"),
                                         ("test_ood", f"Held-out novel ({SUITE_N['test_ood']} episodes)"))):
        if si:
            L.append("\\addlinespace")
        L.append(_section_row(title, ncol))
        for r in rows:
            cells, anyv = [], False
            for ref, _ in refs:
                s = S.get((ref, suite, r["arm"]))
                if s is None:
                    cells += ["", ""]
                    continue
                anyv = True
                cells += [ptab(f"{fmt_signed(100 * s['diff'], 0)} [{fmt_signed(100 * s['lo'], 0, False)}, "
                               f"{fmt_signed(100 * s['hi'], 0, False)}]", s["partial"]), fmt_p(s["p_holm"])]
            if anyv:
                L.append(f"\\hspace{{0.6em}}{tex(pretty(r['arm']))} & " + " & ".join(cells) + "\\\\")
    L += _close("$\\Delta$ = paired difference in success rate, arm minus reference, in percentage points (rounded; "
                "one-decimal values are in the generated macros), with the Newcombe (method 10) score interval; $p_{\\mathrm{Holm}}$ = exact two-sided McNemar test, Holm-adjusted "
                "over all candidate arms compared with the same reference on the same suite (the family grows as arms "
                "are added). Episodes are paired by (scenario, noise seed). The expert table is a privileged reference "
                "written with knowledge of the fault mechanisms. Comparisons with the FDD and weak baselines are in the "
                "generated macros. \\dag\\ run in progress (paired on the available episodes).")
    _write_table(path, L)


def table_cost(rows: list[dict], path: str):
    sub = [r for r in rows if r["group"] in ("online", "hybrid")]
    ncol = 11
    L = _open("l" + "r" * (ncol - 1), "2.5pt")
    L += [" & Success & LLM & \\multicolumn{2}{c}{LLM tokens} & \\multicolumn{4}{c}{Escalations} & Not & "
          "Latency (s)\\\\", "\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}",
          "Arm & held-out & calls & total & compl. & all & (a) & (b) & (c) & served & median [IQR]\\\\", "\\midrule"]
    if not sub:
        L.append(f"\\multicolumn{{{ncol}}}{{c}}{{\\textbf{{[pending]}}}}\\\\")
    first = True
    for title, part in _sections(sub):
        if not first:
            L.append("\\addlinespace")
        first = False
        L.append(_section_row(title, ncol))
        f2 = lambda v: "--" if v is None or not np.isfinite(v) else f"{v:.2f}"  # noqa: E731
        for r in part:
            if not r["has_data"]:
                continue
            h = r["suites"].get("held")
            succ = "--" if h is None else ptab(f"{100 * h['rate']:.0f}\\,\\%", h["partial"])
            hyb = r["group"] == "hybrid"
            lat = ("--" if not np.isfinite(r.get("lat_med", float("nan")))
                   else f"{r['lat_med']:.0f} [{r['lat_q1']:.0f}, {r['lat_q3']:.0f}]")
            cells = [succ, f"{r['calls_ep']:.1f}", fmt_tok(r["tok_ep"]), f"{r['ctok_ep']:.0f}",
                     f2(r.get("esc_ep")) if hyb else "--", f2(r.get("esc_a_ep")) if hyb else "--",
                     f2(r.get("esc_b_ep")) if hyb else "--", f2(r.get("esc_c_ep")) if hyb else "--",
                     f2(r.get("supp_ep")) if hyb else f2(r.get("exhaust_ep")), lat]
            L.append(f"\\hspace{{0.6em}}{tex(short_name(r['arm']))} & " + " & ".join(cells) + "\\\\")
        pend = [short_name(r["arm"]) for r in part if not r["has_data"]]
        if pend:
            L.append(_pending_row(pend, ncol))
    L += _close("Means per held-out episode (anticipated + novel suites). Escalation triggers: (a) unexplained "
                "abnormality, (b) the playbook's proposal fails the nominal-twin forward check, (c) conformal novelty; an "
                "escalation with two triggers counts under both, so (a)+(b)+(c) can exceed the total. Not served = "
                "for hybrids, triggers that fired but were not escalated (cooldown, per-episode cap or empty token "
                "bucket); for online arms, invocations refused because the token bucket was empty. Latency = "
                "wall-clock time of one LLM decision over calls not replayed from the cache (all suites); the workstation "
                "was shared with unrelated jobs, so latencies are pessimistic. \\dag\\ run in progress.")
    _write_table(path, L)


def table_coverage(cov: list[dict], path: str):
    ncol = 7
    L = _open("l" + "r" * (ncol - 1), "3pt")
    L += [" & \\multicolumn{3}{c}{Anticipated families, fresh sample} & \\multicolumn{2}{c}{Independent adversary} & "
          "Family shift\\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-7}",
          "Playbook & failures & rate (\\%) & CP bound (\\%) & first round & local search & V/C/K\\\\", "\\midrule"]
    names = [c["name"] for c in cov]
    rate = lambda d: "--" if not d or not d.get("n") else f"{100 * d['fail'] / d['n']:.1f} ({d['fail']}/{d['n']})"  # noqa: E731
    for c in cov:
        i, a0, a1, fs = c["in_dist"], c["adv_first"] or {}, c["adv_last"] or {}, c.get("family_shift") or {}
        L.append(f"{tex(coverage_label(c['name']))} & {i['fail']}/{i['n']} & {100 * i['fail'] / max(i['n'], 1):.1f} & "
                 f"{100 * i['cp_upper95']:.1f} & {rate(a0)} & {rate(a1)} & {rate(fs)}\\\\")
    pend = [coverage_label(n) for n in EXPECTED_COVERAGE if n not in names]
    if pend:
        L.append(_pending_row(pend, ncol))
    L += _close("Failures of the playbook on fresh scenarios drawn from the anticipated-family sampler "
                "(oracle-unrescued scenarios excluded) and the one-sided 95\\,\\% Clopper-Pearson (CP) upper bound on the "
                "failure probability under that sampler. Adversary and family-shift cells: failure rate in \\% "
                "(failures/scenarios). The bound is conditional on the scenario distribution; it is not a "
                "safety-integrity claim. Independent adversary: first round = a different, harsher sampler (comparable "
                "with the bound); local search = failure rate among perturbed neighbours of the failures found earlier "
                "(last round), which is conditional on failure and measures how clustered the failures are, not a "
                "distribution-level rate. Family shift = novel families V, C and K.")
    _write_table(path, L)


def stream_label(name: str) -> str:
    return ("With compile-back" if name.startswith("cb_") else "Without compile-back" if name.startswith("nocb_")
            else name.replace("_", " "))


def table_stream(streams: list[dict], path: str):
    if not streams:
        open(path, "w").write("% GENERATED by code/experiments/analyze.py: no results/stream/*/block*.json yet\n"
                              "\\begin{tabular}{@{}l@{}}\\toprule Compile-back stream: \\textbf{[pending]}\\\\"
                              "\\bottomrule\\end{tabular}\n")
        return
    names = [s["name"] for s in streams]
    nb = max(max(b["block"] for b in s["blocks"]) + 1 for s in streams)
    L = _open("l" + "rrrrr" * len(streams), "2.5pt")
    L += [" & " + " & ".join(f"\\multicolumn{{5}}{{c}}{{{tex(stream_label(n))}}}" for n in names) + "\\\\",
          "".join(f"\\cmidrule(lr){{{2 + 5 * i}-{6 + 5 * i}}}" for i in range(len(names))),
          "Block & " + " & ".join("success & esc. & online tok. & compile tok. & version" for _ in names) + "\\\\",
          "\\midrule"]
    for b in range(nb):
        cells = []
        for s in streams:
            x = next((y for y in s["blocks"] if y["block"] == b), None)
            cells += ["", "", "", "", ""] if x is None else [f"{x['k']}/{x['n']}", str(x["esc"]), fmt_tok(x["tok"]),
                                                             fmt_tok(x["cb_tok"]), f"v{x['version']}"]
        L.append(f"{b + 1} & " + " & ".join(cells) + "\\\\")
    L.append("\\midrule")
    cells = []
    for s in streams:
        k, n = sum(x["k"] for x in s["blocks"]), sum(x["n"] for x in s["blocks"])
        cells += [ptab(f"{k}/{n}", n < NH()), str(sum(x["esc"] for x in s["blocks"])),
                  fmt_tok(sum(x["tok"] for x in s["blocks"])), fmt_tok(sum(x["cb_tok"] for x in s["blocks"])),
                  f"v{s['blocks'][-1]['version']}"]
    L.append("Total & " + " & ".join(cells) + "\\\\")
    L += _close("Held-out episodes (anticipated + novel) deployed as a stream in a fixed random order, in blocks of 18 "
                "episodes (maintenance intervals). With compile-back, verified online resolutions seen in at least two "
                "episodes are compiled by the coder into a new guarded rule; at most one rule is admitted per block, and "
                "only if training-batch success does not fall; version = playbook version in force after the block. "
                "\\dag\\ stream in progress.")
    _write_table(path, L)


def _budget_grid(vals: list[float]) -> list[float]:
    vals = [v for v in vals if np.isfinite(v) and v > 0]
    if not vals:
        return []
    lo, hi = np.floor(np.log10(min(vals))), np.ceil(np.log10(max(vals)))
    grid = [m * 10 ** e for e in range(int(lo), int(hi) + 1) for m in (1, 3)]
    return [g for g in grid if min(vals) * 0.999 <= g <= max(vals) * 3.0001] or [max(vals)]


def allocation_rows(rows: list[dict]) -> list[dict]:
    """Iso-budget view at N = held-out deployment: for each budget B, the best arm of each allocation class that fits."""
    llm = [r for r in rows if r["has_data"] and r["group"] in ("playbook", "online", "hybrid") and r["main"]
           and r["suites"].get("held") and np.isfinite(r["budget"]) and r["budget"] > 0]
    out = []
    for B in _budget_grid([r["budget"] for r in llm]):
        row = {"B": B}
        for cls, groups in (("offline", ("playbook",)), ("online", ("online",)), ("hybrid", ("hybrid",))):
            fit = [r for r in llm if r["group"] in groups and r["budget"] <= B * 1.0001]
            if fit:
                # ties: prefer the "final" label (the last synthesis checkpoint is the same file as the final playbook)
                b = max(fit, key=lambda r: (r["suites"]["held"]["rate"], -r["budget"], parse_arm(r["arm"])["ckpt"] == "final"))
                row[cls] = {"arm": b["arm"], "name": short_name(b["arm"]), "rate": b["suites"]["held"]["rate"],
                            "k": b["suites"]["held"]["k"], "n": b["suites"]["held"]["n"], "f": b["f"],
                            "budget": b["budget"], "partial": b["suites"]["held"]["partial"]}
        out.append(row)
    return out


def table_allocation(alloc: list[dict], rows: list[dict], path: str):
    ncol = 4
    L = _open("l" + "l" * (ncol - 1), "3pt")
    L += ["Budget $B$ & Offline only ($f=1$) & Online only ($f=0$) & Hybrid ($0<f<1$)\\\\",
          "(tokens) & success (arm) & success (arm) & success (arm, $f$)\\\\", "\\midrule"]
    if not alloc:
        L.append(f"\\multicolumn{{{ncol}}}{{c}}{{\\textbf{{[pending]}}}}\\\\")
    for a in alloc:
        cells = []
        for cls in ("offline", "online", "hybrid"):
            c = a.get(cls)
            if c is None:
                cells.append("--")
                continue
            extra = f", $f={c['f']:.2f}$" if cls == "hybrid" else ""
            cells.append(ptab(f"{100 * c['rate']:.0f}\\,\\%", c["partial"]) + f" ({tex(c['name'])}{extra})")
        bl = f"{a['B'] / 1e6:g}M" if a["B"] >= 1e6 else f"{a['B'] / 1e3:g}k"
        L.append(f"$\\le$ {bl} & " + " & ".join(cells) + "\\\\")
    free = [r for r in rows if r["has_data"] and r["group"] in ("rule", "fdd", "search", "gp") and r["suites"].get("held")]
    note = ""
    if free:
        npriv = [r for r in free if r["arm"] != "B2b"]
        parts = []
        if npriv:
            b = max(npriv, key=lambda r: r["suites"]["held"]["rate"])
            parts.append(f"the best non-privileged LLM-free supervisor ({tex(pretty(b['arm']))}) reached "
                         f"{100 * b['suites']['held']['rate']:.0f}\\,\\%")
        if any(r["arm"] == "B2b" for r in free):
            parts.append(f"the privileged expert table "
                         f"{100 * next(r for r in free if r['arm'] == 'B2b')['suites']['held']['rate']:.0f}\\,\\%")
        note = " For comparison, at zero LLM tokens " + " and ".join(parts) + "."
    L += _close(f"$B$ = all LLM tokens actually spent over a deployment of the {NH()} held-out episodes (offline synthesis "
                "tokens of the playbook checkpoint plus online tokens), not the size of the token bucket; $f$ = offline "
                "share of $B$. Each cell gives the highest pooled held-out success among the arms of that allocation class "
                "whose $B$ fits within the budget (the arm in parentheses)." + note +
                " The GP rule language is expert-curated. \\dag\\ run in progress.")
    _write_table(path, L)


_STAGES = [("First draft (v0)", "main", "v0"), ("Calibration only, one round", "calibonly", "R1"),
           ("Full procedure, one round", "main", "B25000"), ("Full procedure, final", "main", "final")]
ABLATIONS = ([("#Stages of synthesis by seed: first draft, calibration only (no repair), full procedure", "", "", False)]
             + [(lab, f"{tag}_qwen3-coder-30b_s{s}", ck, False) for s in (1, 2, 3) for lab, tag, ck in _STAGES]
             + [("#Seed-1 ablations at about 210k tokens, and early runs", "", "", False),
                ("Full procedure", "main_qwen3-coder-30b_s1", "B200000", False),  # matched-budget reference
                ("No counterfactual remedies", "norem_qwen3-coder-30b_s1", "final", False),
                ("No CPU calibration", "notune_qwen3-coder-30b_s1", "final", False),
                ("No calibration, archived run", "notune_qwen3-coder-30b_s1", "latest", True),
                ("Coder-7B, no calibration", "main_qwen2.5-coder-7b_s1", "B100000", False)])


def ablation_rows(runs: list[dict], summ: dict) -> list[dict]:
    out = []
    by = {(r["name"], r["archived"]): r for r in runs}
    for label, run, ck, arch in ABLATIONS:
        if label.startswith("#"):
            out.append({"section": label[1:]})
            continue
        rr = by.get((run, arch))
        if ck == "latest" and rr and rr["ckpts"]:
            ck = max(rr["ckpts"], key=lambda c: (rr["ckpts"][c]["x"] or 0, ckpt_rank(c)))
        meta = (rr or {}).get("ckpts", {}).get(ck)
        arm = f"playbook:{os.path.join(SYN_ARCH if arch else SYN, run, f'playbook_{ck}.py')}"
        a = summ.get(arm)
        out.append({"label": label, "run": run, "ckpt": ck, "archived": arch, "arm": arm, "tokens": meta and meta["x"],
                    "sr": meta and meta["sr"], "id": a and a["suites"].get("test_id"), "ood": a and a["suites"].get("test_ood"),
                    "evaluated": bool(a and a["has_data"]), "planned": arm in summ})
    return out


def table_ablations(abl: list[dict], path: str):
    ncol = 7
    L = _open("l" + "r" * (ncol - 1), "3pt")
    L += [" & & & & & \\multicolumn{2}{c}{Held-out success}\\\\", "\\cmidrule(l){6-7}",
          "Variant & Seed & Checkpoint & Coder tokens & Training success & Anticipated & Novel\\\\", "\\midrule"]
    for r in abl:
        if "section" in r:
            L.append(_section_row(r["section"], ncol))
            continue

        def kn(v):
            if v is None:
                return PEND_CELL if r["planned"] else "not run"
            return ptab(f"{v['k']}/{v['n']}", v["partial"])
        tok = fmt_tok(r["tokens"]) if r["tokens"] else PEND_CELL
        sr = f"{100 * r['sr']:.1f}\\,\\%" if r["sr"] is not None else PEND_CELL
        ck = "--" if r["ckpt"] == "latest" else ckpt_label(r["ckpt"]) if r["ckpt"] not in ("v0", "final") else r["ckpt"]
        seed = re.search(r"_s(\d+)$", r["run"]).group(1)
        L.append(f"{tex(r['label'])} & {seed} & {ck} & {tok} & {sr} & {kn(r['id'])} & {kn(r['ood'])}\\\\")
    L += _close("First draft = checkpoint v0, written once without repair, falsification or calibration. Calibration only = "
                "that draft after the first falsification round and CPU calibration pass of the full procedure, without any "
                "repair call (R1); full procedure, one round = the checkpoint saved after the first 25k tokens (two repairs, "
                "one falsification round, one calibration pass). The full procedure at 200k is the matched-budget reference for "
                "the two ablation runs (paired tests in the text). "
                "Coder tokens = tokens spent when the checkpoint was saved; training success = incumbent success "
                "on that run's adversarially grown training batch at the checkpoint (not comparable across runs). The "
                "Coder-7B run and the archived runs predate CPU calibration; their checkpoints were not evaluated on the "
                "held-out suites unless stated. pend. = planned, not yet available; \\dag\\ run in progress.")
    _write_table(path, L)


def table_synthesis(runs: list[dict], path: str):
    ncol = 11
    L = _open("ll" + "r" * (ncol - 2), "1.6pt")
    L += ["Run & Status & LLM & Coder & Iter. & Lint & Fals. & Tuning & CPU & Training & Rules\\\\",
          " & & tokens & calls & & fixes & rounds & gains & episodes & success & \\\\", "\\midrule"]
    blocks = [("LLM synthesis", [r for r in runs if not r["gp"] and not r["archived"]]),
              ("Genetic programming over the rule language (no LLM)", [r for r in runs if r["gp"]]),
              ("Archived runs before CPU tuning was added (no tuning)", [r for r in runs if r["archived"]])]
    first = True
    for title, sub in blocks:
        if not sub:
            continue
        if not first:
            L.append("\\addlinespace")
        first = False
        L.append(_section_row(title, ncol))
        for r in sub:
            if r["status"] == "pending":
                continue
            tune = r.get("tune") or []
            tg = "--" if r["gp"] else (f"{len(tune)}" + (f" ($+${100 * max(t['to'] - t['from'] for t in tune):.1f})" if tune else ""))
            g = lambda k: "--" if r.get(k) is None or (isinstance(r.get(k), float) and not np.isfinite(r[k])) else str(r[k])  # noqa: E731
            L.append(f"\\hspace{{0.6em}}{tex(r['label'])} & {r['status']} & {'0' if r['gp'] else fmt_tok(r['x_total'])} & "
                     f"{'0' if r['gp'] else g('calls')} & {g('iters')} & {'--' if r['gp'] else g('lint')} & {r['fals']} & "
                     f"{tg} & {fmt_tok(r['cpu_ep']) if r.get('cpu_ep') else '--'} & "
                     f"{fmt_pct(r['sr_now'], 1)}\\,\\% & {g('rules')}\\\\")
        pend = [r["label"] for r in sub if r["status"] == "pending"]
        if pend:
            L.append(_pending_row(pend, ncol))
    L += _close("LLM tokens = prompt + completion tokens of the coder so far; Iter. = repair iterations (GP: "
                "generations, including the initial population); Lint fixes = candidates sent back to the coder for sandbox or unknown-feature-key "
                "violations; Fals. rounds = falsification rounds; Tuning gains = CPU coordinate-search passes that improved "
                "the incumbent's synthesis score (largest gain in training success, pp; a pass can raise the score without "
                "raising success); CPU episodes = simulated playbook episodes for evaluation, falsification and tuning up to "
                "the last checkpoint (GP: episode evaluations), excluding the oracle simulations that label fresh scenarios "
                "and, for LLM runs, compute the counterfactual remedies; Training success = "
                "incumbent success on the current, adversarially grown training batch (not comparable across runs); "
                "Rules = rule identifiers in the latest checkpoint. Tokens, iterations, falsification rounds, tuning gains and "
                "training success come from the live log; coder calls, lint fixes, CPU episodes and rules from the latest "
                "saved checkpoint. The GP rule language was curated by the authors.")
    _write_table(path, L)


# ======================================================================================= macros (numbers.tex)
def build_macros(summ: dict, stats: list[dict], hgain: list[dict], runs: list[dict], cov: list[dict],
                 streams: list[dict], df: pd.DataFrame, case: dict | None, contrasts: list[dict] | None = None) -> Macros:
    M = Macros()
    M.sec("global")
    M.add("nId", str(SUITE_N["test_id"]), "held-out anticipated episodes")
    M.add("nOod", str(SUITE_N["test_ood"]), "held-out novel episodes")
    M.add("nUnres", str(SUITE_N["test_unrescuable"]), "oracle-unrescued episodes")
    M.add("nHeld", str(NH()), "held-out anticipated + novel episodes")
    M.add("nAnchor", str(SUITE_N["alhazmi"]), "Anchor-13 episodes (3 seeds)")
    M.add("nFrontier", str(SUITE_N["frontier"]), "Frontier-5 episodes (3 seeds)")
    M.add("amortN", str(NH()), "deployment horizon used for amortisation (held-out episodes)")
    M.add("amortNLarge", fmt_num(N_LARGE), "second deployment horizon")
    with_data = [r for r in summ.values() if r["has_data"]]
    M.add("nArmsData", str(len(with_data)), "arms with episodes")
    M.add("nArmsPending", str(sum(1 for r in summ.values() if not r["has_data"])), "expected arms without episodes")
    M.add("nEpisodesTotal", fmt_num(len(df)), "episodes analysed (all arms and suites)")
    M.add("analysisDate", _dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "time of this analysis run")
    for f, n in FAMILY_N.items():
        M.add("nFam" + f, str(n), f"held-out episodes of family {f}")

    for r in summ.values():
        k, nm = r["key"], r["name"]
        M.sec(f"arm {nm} [{anon(r['arm'])[:110]}]")
        M.add("status" + k, r["status"], "data status")
        for s in ("alhazmi", "frontier", "test_id", "test_ood", "test_unrescuable", "held"):
            sk = SUITE_KEY[s]
            planned = s in r["planned"] or (s == "held" and any(x in r["planned"] for x in HELD))
            v = r["suites"].get(s)
            if v is None:
                if not planned and r["has_data"]:
                    continue  # suite not part of this arm's protocol
                for m in ("succ", "rate", "ci", "nEp"):
                    M.add(m + sk + k, PENDING, f"{nm}, {SUITE_NAME[s]}")
                continue
            M.add("succ" + sk + k, partial(f"{v['k']}/{v['n']}", v["partial"]), f"{nm}, {SUITE_NAME[s]} successes")
            M.add("rate" + sk + k, partial(f"{100 * v['rate']:.1f}\\,\\%", v["partial"]), "success rate")
            M.add("ci" + sk + k, partial(f"[{100 * v['lo']:.1f}, {100 * v['hi']:.1f}]\\,\\%", v["partial"]), "Wilson 95% CI")
            M.add("nEp" + sk + k, str(v["n"]), "episodes")
        if not r["has_data"]:
            for m in ("tokEp", "callsEp", "actEp", "actBenign", "offTok", "amortTok", "allocF", "budgetB", "wallEp", "prod",
                      "prodStab", "opCost", "substEp", "repSubstEp", "substSN"):
                M.add(m + k, PENDING, nm)
            if r["group"] in ("playbook", "hybrid"):
                M.add("breakEven" + k, PENDING, nm)
            if r["group"] in ("online", "hybrid"):
                M.add("latEp" + k, PENDING, nm)
            if r["group"] == "hybrid":
                for m in ("escEp", "escAEp", "escBEp", "escCEp", "escTot", "escShare", "suppEp", "latMed", "latIQR"):
                    M.add(m + k, PENDING, nm)
            if r["group"] == "online":
                for m in ("latMed", "latIQR", "exhaustEp"):
                    M.add(m + k, PENDING, nm)
            continue
        part = r["suites"].get("held", {}).get("partial", False)
        M.add("tokEp" + k, partial(fmt_num(r["tok_ep"]), part), "mean LLM tokens per held-out episode (online)")
        M.add("ptokEp" + k, partial(fmt_num(r["ptok_ep"]), part), "prompt tokens per episode")
        M.add("ctokEp" + k, partial(fmt_num(r["ctok_ep"]), part), "completion tokens per episode")
        M.add("callsEp" + k, partial(f"{r['calls_ep']:.2f}", part), "LLM calls per episode")
        M.add("actEp" + k, partial(f"{r['act_ep']:.2f}", part), "accepted actions per episode")
        M.add("actBenign" + k, partial("--" if not np.isfinite(r["act_benign"]) else f"{r['act_benign']:.2f}", part),
              "accepted actions per benign episode")
        if np.isfinite(r["diag_any"]):
            M.add("diagAny" + k, partial(f"{100 * r['diag_any']:.1f}\\,\\%", part), "share of episodes with a correct diagnosis at some decision")
        if np.isfinite(r.get("wall_ep", float("nan"))):
            M.add("wallEp" + k, partial(f"{r['wall_ep']:.1f}\\,s", part), "mean wall-clock seconds per held-out episode (shared machine)")
        if np.isfinite(r.get("dec_overhead", float("nan"))):
            M.add("latDec" + k, partial(f"{max(r['dec_overhead'], 0.0):.2f}\\,s", part),
                  "median added wall-clock time per supervisor invocation vs the weak baseline (paired, episodes in which "
                  f"both ran the full horizon, n={r['overhead_n']}; features, decision, validator rollout; shared machine)")
            M.add("wallAdd" + k, partial(f"{max(r['ep_overhead'], 0.0):.1f}\\,s", part),
                  "median added wall-clock time per episode vs the weak baseline (same pairs as latDec; shared machine)")
        if np.isfinite(r.get("prod_mean", float("nan"))):
            M.add("prod" + k, partial(fmt_num(r["prod_mean"]), part), "mean production (m3) per held-out episode, integrated until the episode ends (trip or 48 h)")
            M.add("opCost" + k, partial(fmt_num(r["cost_mean"]), part), "mean operating cost per held-out episode")
            if np.isfinite(r.get("prod_stab", float("nan"))):
                M.add("prodStab" + k, partial(fmt_num(r["prod_stab"]), part),
                      f"mean production (m3) over the {r['n_stab']} held-out episodes this arm stabilised to 48 h")
        if "subst_ep" in r:
            M.add("substEp" + k, partial(f"{r['subst_ep']:.2f}", part), "accepted instrument substitutions per episode")
            M.add("repSubstEp" + k, partial(f"{r['rep_subst_ep']:.2f}", part),
                  "proposals to substitute an already substituted transmitter, per episode")
            if r["subst_sn"][1]:
                M.add("substSN" + k, partial(f"{r['subst_sn'][0]}/{r['subst_sn'][1]}", r["subst_sn"][1] < FAMILY_N.get("SN", 0)),
                      "SN-family episodes with at least one accepted substitution")
        if np.isfinite(r.get("lat_ep", float("nan"))):
            M.add("latEp" + k, f"{r['lat_ep']:.0f}\\,s", f"mean LLM wall-clock seconds per episode ({r['lat_ep_n']} uncached episodes)")
        if r["group"] in ("playbook", "hybrid", "online"):
            M.add("offTok" + k, fmt_num(r["off_tok"]) if np.isfinite(r["off_tok"]) else "--", f"offline tokens ({r['off_src']})")
            M.add("amortTok" + k, partial(fmt_num(r["amort"]), part), "amortised tokens/episode at N=amortN")
            M.add("budgetB" + k, partial(fmt_num(r["budget"]), part), "total tokens over N=amortN episodes")
            M.add("allocF" + k, partial(f"{r['f']:.2f}" if np.isfinite(r["f"]) else "--", part), "allocation f at N=amortN")
            M.add("allocFLarge" + k, partial(f"{r['f_large']:.2f}" if np.isfinite(r["f_large"]) else "--", part),
                  "allocation f at N=amortNLarge")
        onl = summ.get(f"onlineg:{MAIN_MODEL}")
        if r["group"] in ("playbook", "hybrid") and onl and onl["has_data"] and np.isfinite(r["off_tok"]) and r["off_tok"] > 0:
            gap = onl["tok_ep"] - (r["tok_ep"] if np.isfinite(r["tok_ep"]) else 0.0)
            M.add("breakEven" + k, fmt_num(math.ceil(r["off_tok"] / gap)) if gap > 0 else "never",
                  "deployment episodes after which this arm has used fewer LLM tokens than online gated deliberation")
        if r["group"] == "gp" and np.isfinite(r["off_cpu"]):
            M.add("offEvals" + k, fmt_num(r["off_cpu"]), "GP episode evaluations")
        if r["group"] == "hybrid" and "esc_ep" in r:
            M.add("escEp" + k, partial(f"{r['esc_ep']:.2f}", part), "escalations per episode")
            for code, _, lab in TRIGGERS:
                M.add(f"esc{code.upper()}Ep" + k, partial(f"{r[f'esc_{code}_ep']:.2f}", part), f"trigger ({code}) {lab} per episode")
                M.add(f"esc{code.upper()}Tot" + k, partial(str(r[f"esc_{code}_tot"]), part), f"trigger ({code}) total")
            M.add("escTot" + k, partial(str(r["esc_tot"]), part), "total escalations (held-out)")
            M.add("escShare" + k, partial(f"{100 * r['esc_share']:.1f}\\,\\%", part), "episodes with >=1 escalation")
            M.add("suppEp" + k, partial(f"{r['supp_ep']:.2f}", part), "suppressed triggers per episode")
        if r["group"] == "online" and "exhaust_ep" in r:
            M.add("exhaustEp" + k, partial(f"{r['exhaust_ep']:.2f}", part), "invocations refused (empty bucket) per episode")
        if r["group"] in ("online", "hybrid") and "lat_med" in r:
            ok = np.isfinite(r["lat_med"])
            M.add("latMed" + k, f"{r['lat_med']:.1f}\\,s" if ok else "--", "median uncached LLM decision latency")
            M.add("latIQR" + k, f"[{r['lat_q1']:.1f}, {r['lat_q3']:.1f}]\\,s" if ok else "--", "IQR")
            M.add("latN" + k, str(r["lat_n"]), "timed (uncached) LLM decisions")
        # per family
        held = df[(df["arm"] == r["arm"]) & df["suite"].isin(HELD)]
        for f in FAMILY_N:
            x = held[held["family"] == f]
            if len(x):
                pf = len(x) < FAMILY_N[f]
                M.add("fam" + f + k, partial(f"{int(x['success'].sum())}/{len(x)}", pf), f"family {f}")
                M.add("famRate" + f + k, partial(f"{100 * x['success'].mean():.0f}\\,\\%", pf), f"family {f} rate")

    # paired statistics
    M.sec("paired statistics vs references")
    have = {(s["ref_key"], s["suite"], s["arm"]) for s in stats}
    for s in stats:
        t = f"{SUITE_KEY[s['suite']]}{arm_key(s['arm'])}"
        R = s["ref_key"]
        p = s["partial"]
        M.add(f"dVs{R}{t}", partial(f"{fmt_signed(100 * s['diff'])}\\,pp", p), f"{pretty(s['arm'])} - {pretty(s['ref'])}")
        M.add(f"dciVs{R}{t}", partial(f"[{fmt_signed(100 * s['lo'])}, {fmt_signed(100 * s['hi'])}]\\,pp", p), "Newcombe CI")
        M.add(f"pVs{R}{t}", partial(fmt_p(s["p_holm"]), p), f"McNemar exact, Holm over {s['holm_m']} arms")
        M.add(f"prawVs{R}{t}", partial(fmt_p(s["p"]), p), "McNemar exact, unadjusted")
        M.add(f"bcVs{R}{t}", partial(f"{s['b']}/{s['c']}", p), "discordant pairs b/c")
        M.add(f"nVs{R}{t}", str(s["n"]), "paired episodes")
    present_refs = {s["ref_key"] for s in stats}
    for r in summ.values():
        if r["has_data"]:
            continue
        for _, R in REF_ARMS:
            if R not in present_refs:
                continue
            for suite in ("held", "test_id", "test_ood"):
                if not (suite in r["planned"] or (suite == "held" and any(x in r["planned"] for x in HELD))):
                    continue
                t = f"{SUITE_KEY[suite]}{r['key']}"
                if (R, suite, r["arm"]) in have:
                    continue
                for m in ("dVs", "dciVs", "pVs"):
                    M.add(f"{m}{R}{t}", PENDING, r["name"])
    M.sec("named contrasts (A minus B, paired; exact McNemar unadjusted)")
    for c in contrasts or []:
        t = c["name"] + SUITE_KEY[c["suite"]]
        if not c["available"]:
            for m in ("cDiff", "cCi", "cP", "cBc"):
                M.add(m + t, PENDING, c["desc"])
            continue
        p = c["partial"]
        M.add("cDiff" + t, partial(f"{fmt_signed(100 * c['diff'])}\\,pp", p), c["desc"])
        M.add("cCi" + t, partial(f"[{fmt_signed(100 * c['lo'])}, {fmt_signed(100 * c['hi'])}]\\,pp", p), "Newcombe CI")
        M.add("cP" + t, partial(fmt_p(c["p"]), p), "exact McNemar")
        M.add("cBc" + t, partial(f"{c['b']}/{c['c']}", p), "discordant b/c")
        M.add("cN" + t, str(c["n"]), "paired episodes")
    M.sec("hybrid vs its own playbook checkpoint")
    for h in hgain:
        t = f"{SUITE_KEY[h['suite']]}{arm_key(h['arm'])}"
        M.add("hybGain" + t, f"{fmt_signed(100 * h['diff'])}\\,pp", f"{pretty(h['arm'])} - {pretty(h['playbook'])}")
        M.add("hybGainCi" + t, f"[{fmt_signed(100 * h['lo'])}, {fmt_signed(100 * h['hi'])}]\\,pp", "Newcombe CI")
        M.add("hybGainP" + t, fmt_p(h["p"]), "exact McNemar, unadjusted")
        M.add("hybGainBc" + t, f"{h['b']}/{h['c']}", "discordant b/c")
    for r in summ.values():
        if r["group"] == "hybrid":
            for suite in ("held", "test_id", "test_ood"):
                M.add("hybGain" + SUITE_KEY[suite] + r["key"], PENDING, r["name"])

    # synthesis
    for run in runs:
        k = run["key"]
        M.sec(f"synthesis run {run['name']}")
        exp = EXPECTED_RUNS.get(run["name"], ())
        if run["status"] == "pending":
            for m in ("synTokTotal", "synCalls", "synIters", "synLint", "synFals", "synEp", "synTuneN", "synSrNow",
                      "synStatus", "synTokToNinety", "synTuneBest", "synTuneMaxGain"):
                M.add(m + k, PENDING, run["label"])
            for c in exp:
                M.add("synTok" + k + ckpt_key(c), PENDING, c)
                M.add("synSr" + k + ckpt_key(c), PENDING, c)
            continue
        M.add("synStatus" + k, run["status"], run["label"])
        unit = "evals" if run["gp"] else "tokens"
        M.add(("synEvalsTotal" if run["gp"] else "synTokTotal") + k, fmt_num(run["x_total"]), f"{unit} spent so far")
        M.add("synIters" + k, str(run["iters"] or 0), "iterations (GP: generations)")
        M.add("synFals" + k, str(run["fals"]), "falsification rounds")
        M.add("synFalsFailed" + k, str(run["fals_failed"]), "recoverable failures added by falsification")
        M.add("synSrNow" + k, f"{fmt_pct(run['sr_now'])}\\,\\%", "current incumbent training success")
        if run.get("wall_h"):
            M.add("synWall" + k, partial(f"{run['wall_h']:.1f}\\,h", run["status"] not in ("done", "archived")),
                  "wall-clock hours since the run (re)started (shared machine); [partial] while the run is still going")
        if run.get("cpu_ep"):
            M.add("synEp" + k, fmt_num(run["cpu_ep"]), "CPU episodes at last checkpoint")
        if run.get("reach90") is not None:
            M.add(("synEvalsToNinety" if run["gp"] else "synTokToNinety") + k, fmt_num(run["reach90"]),
                  f"{unit} until training success >= 90%")
        else:
            M.add(("synEvalsToNinety" if run["gp"] else "synTokToNinety") + k, "not reached", "")
        if not run["gp"]:
            M.add("synCalls" + k, str(run.get("calls") or 0), "coder calls")
            M.add("synLint" + k, str(run.get("lint") or 0), "lint fixes")
            M.add("synInvalid" + k, str(run.get("invalid") or 0), "candidates rejected after 2 fixes")
            tune = run.get("tune") or []
            M.add("synTuneN" + k, str(len(tune)), "tuning passes that improved the incumbent's synthesis score")
            M.add("synTuneGainN" + k, str(sum(1 for t in tune if t["to"] > t["from"] + 1e-9)),
                  "tuning passes that raised training success")
            if tune:
                best = max(tune, key=lambda t: t["to"] - t["from"])
                M.add("synTuneBest" + k, f"{best['from']:.3f} $\\to$ {best['to']:.3f}", "largest tuning improvement")
                M.add("synTuneMaxGain" + k, f"$+${100 * (best['to'] - best['from']):.1f}\\,pp", "largest tuning gain")
                M.add("synTuneFirst" + k, f"{tune[0]['from']:.3f} $\\to$ {tune[0]['to']:.3f}", "first tuning improvement")
                M.add("synTuneEvals" + k, str(sum(t["evals"] for t in tune)), "batch evaluations in improving passes")
                M.add("synTuneFirstEvals" + k, str(tune[0]["evals"]), "batch evaluations of the first improving pass")
            else:
                for m in ("synTuneBest", "synTuneMaxGain", "synTuneFirst"):
                    M.add(m + k, "none yet", "")
            if np.isfinite(run.get("lat_med", float("nan"))):
                M.add("synLatMed" + k, f"{run['lat_med']:.0f}\\,s", "median uncached coder-call latency")
        for c in sorted(set(exp) | set(run["ckpts"]), key=ckpt_rank):
            v = run["ckpts"].get(c)
            ck = ckpt_key(c)
            if v is None:
                M.add(("synEvals" if run["gp"] else "synTok") + k + ck, PENDING, c)
                M.add("synSr" + k + ck, PENDING, c)
                if not run["gp"]:
                    M.add("synEp" + k + ck, PENDING, c)
                continue
            M.add(("synEvals" if run["gp"] else "synTok") + k + ck, fmt_num(v["x"]), f"{unit} at checkpoint {c}")
            M.add("synSr" + k + ck, f"{fmt_pct(v['sr'])}\\,\\%", f"training success at {c}")
            if v.get("episodes"):
                M.add("synEp" + k + ck, fmt_num(v["episodes"]), f"CPU episodes at {c}")

    # coverage
    M.sec("coverage bounds")
    names = set()
    for c in cov:
        k = letters_key(c["name"])
        names.add(c["name"])
        i = c["in_dist"]
        M.add("covN" + k, str(i["n"]), "fresh in-distribution scenarios (rescuable/benign/quality gap)")
        M.add("covFail" + k, f"{i['fail']}/{i['n']}", "failures")
        M.add("covRate" + k, f"{100 * i['fail'] / max(i['n'], 1):.1f}\\,\\%", "failure rate")
        M.add("covUpper" + k, f"{100 * i['cp_upper95']:.1f}\\,\\%", "one-sided 95% Clopper-Pearson upper bound")
        for nm, d in (("covAdvRate", c["adv_first"]), ("covAdvFinal", c["adv_last"]), ("covShiftRate", c.get("family_shift"))):
            if d and d.get("n"):
                M.add(nm + k, f"{100 * d['fail'] / d['n']:.1f}\\,\\%", f"{d['fail']}/{d['n']}")
                M.add(nm.replace("Rate", "N").replace("Final", "FinalN") + k, str(d["n"]), "scenarios")
        if c["adv_first"] and c["adv_first"].get("n"):
            # compare the bound with the independent sampler's distribution-level failure rate (first round); later
            # local-search rounds evaluate only neighbours of failures and are conditional on failure by construction
            within = c["adv_first"]["fail"] / c["adv_first"]["n"] <= i["cp_upper95"]
            M.add("covAdvWithin" + k, "yes" if within else "no", "first-round adversary failure rate within the bound")
    for n in EXPECTED_COVERAGE:
        if n not in names:
            for m in ("covN", "covFail", "covRate", "covUpper", "covAdvRate", "covAdvFinal", "covShiftRate", "covAdvWithin"):
                M.add(m + letters_key(n), PENDING, n)

    # stream
    M.sec("compile-back stream")
    snames = set()
    for s in streams:
        k = letters_key(s["name"])
        snames.add(s["name"])
        B = s["blocks"]
        kk, nn = sum(b["k"] for b in B), sum(b["n"] for b in B)
        part = nn < NH()
        third = max(1, len(B) // 3)
        first, last = B[:third], B[-third:]
        epr = lambda bs: sum(b["esc"] for b in bs) / max(sum(b["n"] for b in bs), 1)  # noqa: E731
        M.add("strSucc" + k, partial(f"{kk}/{nn}", part), stream_label(s["name"]))
        M.add("strRate" + k, partial(f"{100 * kk / max(nn, 1):.1f}\\,\\%", part), "success rate")
        M.add("strBlocks" + k, str(len(B)), "blocks completed")
        M.add("strRules" + k, str(B[-1]["version"]), "admitted compile-back rules (final version)")
        M.add("strEscEp" + k, partial(f"{epr(B):.2f}", part), "escalations per episode")
        M.add("strEscFirst" + k, f"{epr(first):.2f}", "escalations/ep. in the first third of blocks")
        M.add("strEscLast" + k, partial(f"{epr(last):.2f}", part), "escalations/ep. in the last third of blocks")
        M.add("strTok" + k, partial(fmt_num(sum(b["tok"] for b in B)), part), "online tokens")
        M.add("strCbTok" + k, partial(fmt_num(sum(b["cb_tok"] for b in B)), part), "compile-back tokens")
        lg = s.get("log") or []
        M.add("strCbProposed" + k, partial(str(len(lg)), part), "compile-back rules proposed by the coder")
        M.add("strCbRejected" + k, partial(str(sum(1 for x in lg if not x.get("admitted"))), part),
              "proposed rules rejected (sandbox or regression falsification)")
    cb = next((s for s in streams if s["name"].startswith("cb_")), None)
    nocb = next((s for s in streams if s["name"].startswith("nocb_")), None)
    if cb and nocb:
        A = {k: v for b in cb["blocks"] for k, v in b["outcomes"].items()}
        B_ = {k: v for b in nocb["blocks"] for k, v in b["outcomes"].items()}
        idx = sorted(set(A) & set(B_))
        if len(idx) >= 10:
            x = np.array([A[i] for i in idx]); y = np.array([B_[i] for i in idx])
            b, c, p = mcnemar_exact(x, y)
            d, lo, hi = newcombe_paired_diff_ci(x, y)
            M.add("strDiff", f"{fmt_signed(100 * d)}\\,pp", "compile-back minus no compile-back (paired)")
            M.add("strDiffCi", f"[{fmt_signed(100 * lo)}, {fmt_signed(100 * hi)}]\\,pp", "Newcombe CI")
            M.add("strDiffP", fmt_p(p), "exact McNemar")
            M.add("strDiffBc", f"{b}/{c}", "discordant b/c")
    for n in EXPECTED_STREAMS:
        if n not in snames:
            for m in ("strSucc", "strRate", "strBlocks", "strRules", "strEscEp", "strEscFirst", "strEscLast", "strTok", "strCbTok"):
                M.add(m + letters_key(n), PENDING, n)
    M.add("strDiff", PENDING, "")
    M.add("strDiffCi", PENDING, "")
    M.add("strDiffP", PENDING, "")

    # aliases requested by the results text (kept stable; each points to a scheme macro)
    M.sec("aliases used by sections/results.tex (target macro in the comment)")
    gp_final = "GPFinalSOne" if f"playbook:{_gp(1, 'final')}" in summ and summ[f"playbook:{_gp(1, 'final')}"]["has_data"] \
        else "GPEvTwentyKSOne"
    aliases = [("prodExpert", "prodStabBTwob"), ("prodGp", "prodStab" + gp_final), ("latEpOnline", "latEpOnlGated"),
               ("latEpSearch", "wallAddSearch"), ("pGpEarlyVsGpOod", "cPGpEarlyVsGpOod"), ("subOnlineSN", "substSNOnlGated"),
               ("cbProposed", "strCbProposedCbSOneFinalSixteenK"), ("cbRejected", "strCbRejectedCbSOneFinalSixteenK"),
               ("wallSynthMain", "synWallMainSOne")]
    main_pb = next((summ[a]["key"] for a in [f"playbook:{_pb(1, c)}" for c in ("final", "B400000", "B200000", "B100000",
                                                                                "B25000")]
                    + sorted(x for x in summ if "/preview/" in x and x.startswith("playbook:"))
                    if a in summ and summ[a]["has_data"]), "PbFinalSOne")
    aliases += [("breakEven", "breakEven" + main_pb), ("latPlaybook", "latDec" + main_pb)]
    M.add("mainPbArm", main_pb, "arm key that the breakEven/latPlaybook aliases follow (latest evaluated seed-1 main checkpoint)")
    main_s1 = next((r for r in runs if r["name"] == "main_qwen3-coder-30b_s1" and not r["archived"]), None)
    pending_until_done = {"wallSynthMain": main_s1 is None or main_s1.get("status") != "done"}  # quoted as the FINAL run time
    for al, target in aliases:
        ok = target in M.names and not pending_until_done.get(al, False)
        M.add(al, f"\\{target}" if ok else PENDING, f"alias of \\{target}")

    # case study
    M.sec("case study (fig_case.pdf)")
    if case:
        M.add("caseScenario", tex(case["scenario"]), "held-out scenario id")
        M.add("caseFamily", case["family"], "fault family")
        M.add("caseDesc", tex(case["desc"]).replace("°", "\\textdegree{}"), "fault description")
        M.add("caseRule", case["rule"], "selection rule")
        for role, info in case["arms"].items():
            M.add("caseArm" + role, tex(info["name"]) if info else PENDING, f"{role} arm")
            M.add("caseOut" + role, tex(info["outcome"]) if info else PENDING, f"{role} outcome")
            if info and info.get("first_action"):
                M.add("caseAct" + role, tex(info["first_action"]), f"{role} first accepted action")
                M.add("caseDiag" + role, tex(info["first_diag"]), f"{role} diagnosis of the first accepted action")
    else:
        for m in ("caseScenario", "caseFamily", "caseDesc", "caseRule"):
            M.add(m, PENDING, "")
    return M


# ======================================================================================= case-study selection
def describe_scenario(sd: dict) -> str:
    parts = []
    for e in sd.get("idv_events", []):
        parts.append(f"{e['idv'].upper().replace('_0', '').replace('_', '')} at {e['start_h']:.1f} h"
                     + ("" if e.get("end_h") is None else f" until {e['end_h']:.1f} h"))
    for f in sd.get("sensor_faults", []):
        unit = "kPa" if "pressure" in f["measurement"] else "°C" if "temperature" in f["measurement"] else ""
        unit = (unit + "/h" if f["kind"] == "drift" else unit) if f["kind"] in ("drift", "bias") else ""
        mag = "" if f["kind"] == "freeze" else f" ({f['magnitude']:+.1f}{' ' + unit if unit else ''})"
        parts.append(f"{f['measurement'].replace('_', ' ')} transmitter {f['kind']}{mag} at {f['start_h']:.1f} h")
    for f in sd.get("actuator_faults", []):
        v = str(f.get("valve", f.get("mv", "valve"))).replace("_", " ")
        v = v if v.endswith("valve") else v + " valve"
        parts.append(f"{v} {f.get('kind', 'fault').replace('_', ' ')} at {f.get('start_h', 0):.1f} h")
    for t, ch in sd.get("setpoint_shifts", []):
        for k, v in ch:
            if k == "production_rate":
                parts.append(f"production {v:.1f} m3/h from {t:.1f} h")
    return "; ".join(parts) or sd.get("description", "")


def _first_present(cands: list[str], present: set) -> str | None:
    return next((c for c in cands if c in present), None)


def select_case(df: pd.DataFrame, force: str | None = None) -> dict | None:
    present = set(df["arm"])
    roles = {
        "Weak": ["weak"],
        "Onl": [f"onlineg:{MAIN_MODEL}"],
        "Pb": [f"playbook:{_pb(1, c)}" for c in ("final", "B400000", "B200000", "B100000", "B25000")] +
              [a for a in sorted(present) if "/preview/" in a and a.startswith("playbook:")],
        "Hyb": [f"hybrid:{_pb(1, c)}:{MAIN_MODEL}@16000" for c in ("final", "B100000", "B25000")] +
               sorted(a for a in present if a.startswith("hybrid:")),
    }
    chosen = {r: _first_present(c, present) for r, c in roles.items()}
    arms = [a for a in chosen.values() if a]
    if "weak" not in arms or len(arms) < 2:
        return None
    d = df[(df["suite"] == "test_id") & df["family"].isin(["S", "SN"]) & df["arm"].isin(arms)]
    w = d.pivot_table(index=["scenario", "seed"], columns="arm", values="success", aggfunc="first").dropna()
    if not len(w):
        return None
    ft = d[d["arm"] == "weak"].set_index(["scenario", "seed"])["fail_time"]
    others = [a for a in arms if a != "weak"]
    cand = []
    for idx, row in w.iterrows():
        weak_fail = not bool(row["weak"])
        disagree = len({bool(row[a]) for a in others}) > 1
        cand.append((not weak_fail, not disagree, float(ft.get(idx, 99)), idx[0], idx))
    if not cand:
        return None
    cand.sort()
    rule_pick = cand[0][4]
    if force:
        forced = [c for c in cand if c[3] == force]
        if not forced:
            return None
        idx = forced[0][4]
    else:
        idx = rule_pick
    paths = d[(d["scenario"] == idx[0]) & (d["seed"] == idx[1])].set_index("arm")["path"].to_dict()
    sd = json.load(open(paths["weak"]))["scenario_def"]
    info = {}
    for role, arm in chosen.items():
        if arm is None or arm not in paths:
            info[role] = None
            continue
        rec = json.load(open(paths[arm]))
        if rec["stabilized"]:
            outc = "stabilised to 48 h"
        elif rec.get("shutdown"):
            outc = f"orderly shutdown at {rec.get('shutdown_time') or 0:.1f} h"
        else:
            outc = f"trip at {rec.get('trip_time') or 0:.1f} h ({(rec.get('trip_msg') or '').strip().rstrip('!').split('!')[0].strip().lower()})"
        acts = [x for x in rec.get("decisions", []) if x.get("accepted")]
        fa = fdiag = None
        if acts:
            fa = f"{action_phrase(acts[0]['action'])} at {acts[0]['t']:.2f} h"
            fdiag = (acts[0].get("diagnosis") or "none").replace("_", " ")
        info[role] = {"arm": arm, "name": pretty(arm), "outcome": outc, "first_action": fa, "first_diag": fdiag,
                      "path": paths[arm], "success": bool(w.loc[idx, arm])}
    rule = ("the anticipated-family (S or SN) held-out scenario in which the weak baseline fails and the plotted "
            "supervisors disagree, choosing the earliest weak-baseline trip")
    if idx != rule_pick:  # pinned with --case to a scenario the rule no longer selects: say so instead of citing the rule
        print(f"WARNING: --case {force} is not the rule-based choice ({rule_pick[0]}); caseRule says so")
        rule = (f"scenario fixed in advance by the authors; on the current data the selection rule (an anticipated-family "
                f"held-out scenario in which the weak baseline fails and the plotted supervisors disagree, earliest "
                f"weak-baseline trip) would pick {tex(rule_pick[0])}")
    return {"scenario": idx[0], "seed": int(idx[1]), "family": sd["family"], "desc": describe_scenario(sd), "sd": sd,
            "arms": info, "rule": rule}


# ======================================================================================= preview document
def write_preview(path: str, figs: list[str]):
    tabs = [("results_main", "Main results (generated)"), ("results_family", "Per-family success (generated)"),
            ("stats_vs_ref", "Paired statistics vs references (generated)"),
            ("cost_online", "Online and hybrid cost (generated)"), ("allocation", "Iso-budget allocation (generated)"),
            ("ablations", "Ablations (generated)"), ("synthesis", "Synthesis runs (generated)"),
            ("coverage", "Coverage bounds (generated)"), ("stream", "Compile-back stream (generated)"),
            ("results_all", "All arms incl. previews (generated)")]
    L = [r"% GENERATED by code/experiments/analyze.py: compile check for every generated table and figure.",
         r"\documentclass[preprint,review,12pt,authoryear]{elsarticle}",
         r"\usepackage[T1]{fontenc}\usepackage{amsmath,amssymb}\usepackage{graphicx}\usepackage{booktabs,threeparttable}",
         r"\usepackage[table]{xcolor}\usepackage{siunitx}\usepackage{hyperref}",
         r"\sisetup{group-separator={,},group-minimum-digits=4}", r"\input{numbers}",
         r"\begin{document}", r"\section*{Generated results preview (\analysisDate)}",
         r"Held-out anticipated success of the expert table: \succIdBTwob{} (\rateIdBTwob, CI \ciIdBTwob)."]
    for name, cap in tabs:
        L += [r"\begin{table}[!htbp]\centering", f"\\caption{{{cap}}}", f"\\input{{{name}}}", r"\end{table}", r"\clearpage"]
    for fpath in figs:
        nm = os.path.basename(fpath)
        lab = "fig:pareto" if "pareto" in nm else "fig:" + nm.replace(".pdf", "")
        L += [r"\begin{figure}[!htbp]\centering", f"\\includegraphics[width=\\textwidth]{{../figures/{nm}}}",
              f"\\caption{{{tex(nm)}}}\\label{{{lab}}}", r"\end{figure}"]
    L += [r"\end{document}"]
    _write_table(path, L)


# ======================================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=4, help="loader processes (at most 4)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--no-pending", action="store_true", help="drop pending rows from the tables")
    ap.add_argument("--case", default=None, help="force the case-study scenario id (e.g. SN0003)")
    a = ap.parse_args(argv)
    os.makedirs(GEN, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    df, ext = load_all(a.workers)
    if df.empty:
        print("no episodes")
        return
    expected = expected_arms()
    PROMOTED.clear()
    PROMOTED.update(promote_previews(set(df["arm"])))
    summ = summarise(df, ext, expected)
    rows_all = [summ[k] for k in sorted(summ, key=sort_key)]
    present = set(df["arm"])
    rows_main = [r for r in rows_all if main_table_keep(r["arm"], present) and (r["has_data"] or not a.no_pending)]
    stats = paired_stats(df, [r["arm"] for r in rows_all])
    hgain = hybrid_gain(df, [r["arm"] for r in rows_all])
    runs = synthesis_runs()
    cov = coverage_results()
    streams = stream_results()
    case = select_case(df, a.case)

    note_prev = (" Preview = a code-identical copy of that main-run checkpoint, evaluated before the main checkpoints; it "
                 "is shown until the checkpoint itself has been evaluated." if PROMOTED else "")
    table_results(rows_main, os.path.join(GEN, "results_main.tex"), note_prev)
    table_results([r for r in rows_all if r["has_data"] and not same_as_final(r["arm"], present)
                   and parse_arm(r["arm"]).get("origin") != "preview"], os.path.join(GEN, "results_all.tex"),
                  " Checkpoints whose code is identical to the final playbook of their run are not repeated.", sep="1.5pt")
    table_family([r for r in rows_main if r["has_data"]], df, os.path.join(GEN, "results_family.tex"))
    table_stats([r for r in rows_main if r["has_data"] and stats_keep(r["arm"], present)], stats,
                os.path.join(GEN, "stats_vs_ref.tex"))
    table_cost(rows_main, os.path.join(GEN, "cost_online.tex"))
    table_coverage(cov, os.path.join(GEN, "coverage.tex"))
    table_stream(streams, os.path.join(GEN, "stream.tex"))
    table_synthesis(runs, os.path.join(GEN, "synthesis.tex"))
    alloc = allocation_rows(rows_all)
    table_allocation(alloc, rows_all, os.path.join(GEN, "allocation.tex"))
    abl = ablation_rows(runs, summ)
    table_ablations(abl, os.path.join(GEN, "ablations.tex"))
    contrasts = contrast_stats(df)
    M = build_macros(summ, stats, hgain, runs, cov, streams, df, case, contrasts)
    M.write(os.path.join(GEN, "numbers.tex"), NUMBERS_HEADER.format(date=_dt.datetime.now().strftime("%Y-%m-%d %H:%M")))

    figs = []
    if not a.no_figs:
        import figures_extra as FX
        figs = FX.make_all(df=df, ext=ext, summ=summ, rows_main=rows_main, rows_proto=[r for r in rows_all if r["main"]],
                           runs=runs, case=case, out_dir=FIG)
    write_preview(os.path.join(GEN, "preview.tex"), figs or [os.path.join(FIG, f) for f in
                                                              ("fig_family.pdf", "fig_pareto.pdf", "fig_synthesis.pdf",
                                                               "fig_case.pdf", "fig_escalation.pdf")])

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items() if k not in ("outcomes", "sd")}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, float) and not math.isfinite(o):
            return None
        if isinstance(o, (np.floating, np.integer)):
            return _clean(o.item())
        return o
    out = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "suite_sizes": SUITE_N, "family_sizes": FAMILY_N,
           "amortisation_N": [NH(), N_LARGE], "arms": rows_all, "paired": stats, "hybrid_gain": hgain,
           "contrasts": contrasts, "allocation": alloc, "ablations": abl, "synthesis": runs,
           "coverage": cov, "stream": streams, "case": case}
    # paths are anonymised (paper/generated may be bundled with the manuscript source for double-anonymised review)
    open(os.path.join(GEN, "analysis.json"), "w").write(anon(json.dumps(_clean(out), indent=1, default=str)))
    n_pend = sum(1 for r in rows_all if not r["has_data"])
    n_part = sum(1 for r in rows_all if r["status"].startswith("partial"))
    print(f"episodes {len(df)}; arms with data {len(rows_all) - n_pend} (partial {n_part}); pending {n_pend}; "
          f"macros {len(M.items)}; figures {len(figs)}")
    for r in rows_all:
        print(f"  [{r['status']:>24}] {r['key']:<34} {r['name']}")


if __name__ == "__main__":
    main()
