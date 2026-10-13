"""Regenerate the appendix material on prompts and reproducibility from code, environment metadata and results.

  cd <workspace> && PYTHONPATH=c10_compile_or_deliberate/code .venv/bin/python \
      c10_compile_or_deliberate/code/scripts/gen_repro.py [--offline] [--card-arm SUBSTRING]

Everything is read, nothing is simulated and no model is run: prompt text comes from cod.prompts,
cod.synth.llm_synth and cod.supervisors.llm_online; the counterexample card is rendered with
cod.synth.evaluate.counterexample_card from an episode JSON already on disk; the counterfactual
remedy line is taken from the stored oracle record of the held-out screen when that record is
provably the same computation (see `screen_remedy`); the Ollama server is asked only for metadata
(/api/version, /api/tags). If the server cannot be reached (or --offline), the last snapshot in
paper/generated/repro_env.json is used for the Ollama version and the model digests.

Outputs (paper/generated/):
  repro_table.tex            models table + reproducibility record (software, hardware, serving, prompt hash,
                             suite hashes, seeds, invocation policy, synthesis/GP/online/hybrid parameters)
  repro_env.json             environment snapshot (no host or user names)
  repro_macros.tex           \\repro... macros quoted in the appendix text
  repro_partb.tex            structured summary of Part B (sections, content, characters)
  repro_part_a.txt           Part A, verbatim
  repro_contract.txt         synthesis contract, verbatim
  repro_online_instruction.txt  instruction appended to the online process-state card, verbatim
  repro_card.txt             example counterexample card (real failed episode)
  repro_playbook.txt         excerpt (<= 30 lines) of the latest synthesised playbook checkpoint
"""
from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime as dt
import glob
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import re
import subprocess
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(CODE)
WS = os.path.dirname(PROJ)
sys.path.insert(0, CODE)
sys.path.insert(0, WS)

GEN = os.environ.get("COD_GEN", os.path.join(PROJ, "paper", "generated"))  # output folder (as in analyze.py)
RESULTS = os.path.join(PROJ, "results")
SYN = os.path.join(RESULTS, "synth")
SUITES = os.path.join(PROJ, "data", "suites")
MAIN_SYNTH = "main_qwen3-coder-30b_s1"
PACKAGES = ["tep-studio", "numpy", "scipy", "scikit-learn", "pandas", "statsmodels", "matplotlib", "requests", "psutil"]


# ----------------------------------------------------------------------------- helpers
def tex(s) -> str:
    s = str(s)
    rep = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&", "#": r"\#", "$": r"\$", "{": r"\{",
           "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(rep.get(c, c) for c in s)


def tt(s) -> str:
    return r"\texttt{" + tex(s) + "}"


ASCII = {"–": "-", "—": "-", "−": "-", "‘": "'", "’": "'", "“": '"', "”": '"',
         "°": " deg", "µ": "u", "≤": "<=", "≥": ">=", "×": "x", "…": "..."}


def ascii_only(s: str) -> str:
    """listings under pdflatex cannot typeset multi-byte UTF-8; map the few that occur, '?' otherwise."""
    return "".join(c if ord(c) < 128 else ASCII.get(c, "?") for c in s)


def state_card(scenario_id: str, suite: str) -> str:
    """The process-state card of the deliberator's first invocation in a held-out episode (weak configuration,
    the supervisor holds), rendered by the same function the online deliberator uses."""
    from cod.actions import NO_ACTION
    from cod.runner import Decision, run_episode
    from cod.summary import render
    from experiments.run_arms import load_suite
    sc, seeds = next((x, sd) for x, sd in load_suite(suite) if x.id == scenario_id)
    cards: list[str] = []

    class Recorder:
        def episode_start(self, plant, scenario, seed):
            pass

        def decide(self, plant, f, trigger, history):
            if not cards and (f["alarms"] or trigger != "heartbeat"):
                cards.append(f"[{trigger}]\n" + render(f, []))
            return Decision(NO_ACTION, "none", "", source="recorder")

        def revise(self, plant, f, rejected, reasons, history):
            return None

        def episode_end(self, plant, outcome):
            pass

    run_episode(sc, (seeds or [1])[0], Recorder(), baseline="weak")
    return cards[0] if cards else "(no invocation)"


def write(name: str, text: str):
    os.makedirs(GEN, exist_ok=True)
    path = os.path.join(GEN, name)
    with open(path + ".tmp", "w") as f:
        f.write(text)
    os.replace(path + ".tmp", path)
    print("wrote", os.path.relpath(path, PROJ))


def src_default(path: str, flag: str):
    """Literal default of an argparse flag, read from the script source (keeps the table in sync with code)."""
    s = open(os.path.join(CODE, path)).read()
    m = re.search(r'add_argument\("' + re.escape(flag) + r'"[^\n]*?default=([^,\)\n]+)', s)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1).strip())
    except (ValueError, SyntaxError):
        return m.group(1).strip()


def src_search(path: str, regex: str, default=None):
    s = open(os.path.join(CODE, path) if not os.path.isabs(path) else path).read()
    m = re.search(regex, s)
    return m.group(1) if m else default


def fmt_num(v) -> str:
    if isinstance(v, float) and v.is_integer() and abs(v) >= 1000:
        return f"{int(v):,}"
    if isinstance(v, int) and abs(v) >= 10000:
        return f"{v:,}"
    return f"{v:g}" if isinstance(v, float) else str(v)


# ----------------------------------------------------------------------------- environment
def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _gpu_cores() -> str | None:
    try:
        out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True, timeout=30).stdout
        m = re.search(r"Total Number of Cores:\s*(\d+)", out)
        return m.group(1) if m else None
    except Exception:  # noqa: BLE001
        return None


def ollama_meta(host: str) -> dict | None:
    """Metadata only (server version and installed model manifests); no inference request."""
    import requests
    try:
        ver = requests.get(host + "/api/version", timeout=5).json()["version"]
        tags = requests.get(host + "/api/tags", timeout=10).json()["models"]
    except Exception as e:  # noqa: BLE001
        print("ollama metadata unavailable:", e)
        return None
    models = {}
    for m in tags:
        d = m.get("details", {})
        models[m["name"]] = {"digest": m.get("digest", ""), "size_bytes": m.get("size"),
                             "parameter_size": d.get("parameter_size"), "quantization_level": d.get("quantization_level"),
                             "family": d.get("family"), "format": d.get("format")}
    return {"version": ver, "models": models}


def collect_env(offline: bool) -> dict:
    snap_path = os.path.join(GEN, "repro_env.json")
    if not os.path.exists(snap_path):  # the release keeps the snapshot with the result summaries
        snap_path = os.path.join(PROJ, "results_summary", "repro_env.json")
    old = json.load(open(snap_path)) if os.path.exists(snap_path) else {}
    pk = {}
    for p in PACKAGES:
        try:
            pk[p] = md.version(p)
        except md.PackageNotFoundError:
            pk[p] = None
    osname = platform.system()
    if osname == "Darwin":
        osname = "macOS " + platform.mac_ver()[0]
    mem = _sysctl("hw.memsize")
    env = {
        # offline: keep the date of the snapshot, so that a regenerated record is identical
        "collected": (old.get("collected") if offline and old.get("collected") else dt.date.today().isoformat()),
        "python": platform.python_version(),
        "packages": pk,
        "os": osname,
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "cpu_cores": os.cpu_count(),
        "gpu_cores": _gpu_cores(),
        "memory_gb": round(int(mem) / 2 ** 30) if mem else None,
    }
    from tools.llm import DEFAULT_HOST
    meta = None if offline else ollama_meta(os.environ.get("LLM_HOST", DEFAULT_HOST))
    if meta is not None:
        env["ollama"] = {"version": meta["version"], "source": "live /api/version, /api/tags on " + env["collected"]}
        env["models"] = meta["models"]
    else:
        prev = old.get("ollama", {"version": None})  # keep the provenance of the snapshot (when it was queried live)
        env["ollama"] = dict(prev, source=prev.get("source") or ("snapshot " + str(old.get("collected"))))
        env["models"] = old.get("models", {})
    env["roles"] = old.get("roles", {})  # {model: role in this study}; see models_used()
    return env


# ----------------------------------------------------------------------------- models actually used
def _arm_specs(base: str) -> dict[str, str]:
    out = {}
    for d in sorted(glob.glob(os.path.join(base, "*"))):
        if not os.path.isdir(d):
            continue
        fs = sorted(glob.glob(os.path.join(d, "*", "*.json")))
        if not fs:
            continue
        try:
            out[os.path.basename(d)] = json.load(open(fs[0]))["arm"]
        except Exception:  # noqa: BLE001
            continue
    return out


def models_used(recorded: dict | None = None) -> dict[str, str]:
    """{model tag: role description}, inferred from what is on disk (synthesis metadata, arm specs, pilots). Where
    the pilot runs are not on disk (the release omits them), roles recorded in the snapshot fill the gap."""
    main: dict[str, set] = {}
    pilot: dict[str, set] = {}
    for p in glob.glob(os.path.join(SYN, "*", "meta_*.json")):
        try:
            m = json.load(open(p)).get("model")
        except Exception:  # noqa: BLE001
            continue
        if m:
            main.setdefault(m, set()).add("offline coder")
    for base, dst in ((os.path.join(RESULTS, "arms"), main), (os.path.join(RESULTS, "pilots"), pilot)):
        for spec in _arm_specs(base).values():
            m = re.match(r"^(online|onlineg):([^@]+)", spec)
            h = re.match(r"^hybrid:.*?\.py:([^@]+)@", spec)
            if m:
                dst.setdefault(m.group(2), set()).add("online deliberator")
            elif h:
                dst.setdefault(h.group(1), set()).add("hybrid escalation")
    if glob.glob(os.path.join(RESULTS, "stream", "*.json")):
        main.setdefault(src_default("experiments/stream_hybrid.py", "--coder") or "qwen3-coder:30b", set()).add("compile-back coder")
    out = {}
    for m in sorted(set(main) | set(pilot)):
        txt = ", ".join(sorted(main.get(m, ())))
        extra = sorted(pilot.get(m, set()) - main.get(m, set()))
        if extra:
            txt = (txt + "; " if txt else "") + "pilot runs only: " + ", ".join(extra)
        out[m] = txt
    if recorded and not os.path.isdir(os.path.join(RESULTS, "pilots")):
        for m, txt in recorded.items():
            out.setdefault(m, txt)
    return out


# ----------------------------------------------------------------------------- Part B summary
def partb_sections():
    from cod.actions import BOUNDS, DIAGNOSES, PROCEDURES, SUBSTITUTABLE
    from cod.alarms import ALARMS
    from cod.prompts import PART_B
    from cod.runner import Policy
    pol = Policy()
    novel = [d for d in DIAGNOSES if d in ("reactor_cooling_fault", "condenser_cooling_fault", "kinetics_drift",
                                             "heat_transfer_degradation", "valve_stiction_or_stuck", "compound_fault")]
    spec = [
        ("PLANT:", "Plant and operating point",
         "Flowsheet in one paragraph (reactor, condenser, separator, recycle compressor with purge, stripper) and nominal "
         "values of the regulated variables."),
        ("HARD TRIP LIMITS", "Hard trip limits",
         "True-state limits of the independent safety-instrumented system (reactor pressure and temperature, three levels)."),
        ("REGULATORY LAYER", "Regulatory layer",
         "Pairings of the decentralised PI loops; the two pre-engineered components (composition trims, safety overrides), "
         "stated to be available but disarmed, and what each does."),
        ("ALARMS (on measured values)", "Alarm table",
         f"The {len(ALARMS)} alarms of \\cref{{tab:alarms}}, rendered from the same constants: limits for all, priorities "
         f"for the {sum(a.kind != 'DEV' for a in ALARMS)} high and low alarms and on-delays for the "
         f"{sum(a.kind == 'DEV' for a in ALARMS)} deviation alarms; alarms act on measured, possibly faulty, values."),
        ("SUPERVISORY ACTIONS", "Action grammar",
         f"JSON form of every action type: setpoint moves with the bounds, maximum moves and cooldowns of {len(BOUNDS)} "
         f"setpoints (\\cref{{tab:bounds}}); one-way arming of the two components; the {len(PROCEDURES)} procedures of "
         f"\\cref{{tab:procedures}}; soft-sensor substitution for {len(SUBSTITUTABLE)} measurements, with the inputs of the "
         "pressure and temperature soft sensors; no action; orderly shutdown (stated to count as failure)."),
        ("Actions land", "Validator and budget",
         f"Action latency ({pol.apply_delay_h:g}\\,h), the per-episode action budget ({pol.max_actions}), the static validator "
         f"and the {pol.shadow_horizon_h:g}-h forward check on a fault-blind nominal model."),
        ("ANTICIPATED ABNORMAL SITUATIONS", "HAZOP catalogue",
         "Causes of the anticipated situations only (reactant-composition upsets and C-header pressure loss, loss or "
         "restriction of the A feed, ordered grade changes, reactor pressure or temperature transmitter failures, "
         "production changes). No parameters, onset times, magnitudes or scenario labels."),
        ("DIAGNOSIS VOCABULARY", "Diagnosis vocabulary",
         f"The {len(DIAGNOSES)} labels a decision must use. {len(novel)} of them name causes outside the HAZOP catalogue "
         f"({', '.join(tt(x) for x in novel[:3])}, \\ldots) as bare labels, without description or parameters."),
    ]
    idx = []
    pos = 0
    for marker, _, _ in spec:
        i = PART_B.find(marker, pos)
        if i < 0:
            raise RuntimeError(f"Part B no longer contains the section marker {marker!r}; update gen_repro.partb_sections")
        idx.append(i)
        pos = i + 1
    idx.append(len(PART_B))
    rows = []
    for k, (_, title, desc) in enumerate(spec):
        rows.append((title, desc, idx[k + 1] - idx[k]))
    return rows


# ----------------------------------------------------------------------------- counterexample card
def _playbook_priority(spec: str):
    path = spec[len("playbook:"):].split("@")[0]
    if "/gp_" in path or ("/synth/" not in path and "/preview/" not in path):
        return None
    meta = path.replace("playbook_", "meta_").replace(".py", ".json")
    tokens = json.load(open(meta)).get("tokens", 0) if os.path.exists(meta) else 0
    m = re.search(r"/synth/(\w+?)_(.+?)_s(\d+)/", path)
    if m and m.group(1) == "main" and m.group(2) == "qwen3-coder-30b":
        return (0, int(m.group(3)), -tokens, path)
    if m and m.group(1) == "main":
        return (1, int(m.group(3)), -tokens, path)
    if m:
        return (2, int(m.group(3)), -tokens, path)
    return (3, 0, -tokens, path)


def screen_remedy(sc_d: dict, seed: int) -> dict | None:
    """The synthesiser's remedy is oracle_rescue(sc, seed, [on+0.5, on+1.5], two_step=False). The held-out screen
    ran oracle_rescue(sc, seed, [on+0.5, on+1.5, on+3], two_step=True), which tries the same singles in the same
    order at each decision time before any pair. Its stored plan therefore equals the remedy exactly when it is a
    single action found at on+0.5 or on+1.5; anything else returns None (no stored equivalent)."""
    from cod.faults import Scenario
    sc = Scenario.from_dict(sc_d)
    h, on = sc.content_hash(), sc.first_onset_h
    for split in ("screen_101", "screen"):
        for p in glob.glob(os.path.join(RESULTS, split, sc.family, "*.json")):
            r = json.load(open(p))
            if int(r["seed"]) != int(seed) or Scenario.from_dict(r["scenario"]).content_hash() != h:
                continue
            o = r.get("oracle") or {}
            plan, t = o.get("plan"), o.get("t")
            if o.get("rescued") and plan and plan != ["hold"] and len(plan) == 1 and t is not None and \
                    min(abs(t - (on + 0.5)), abs(t - (on + 1.5))) < 0.011:
                return {"action": plan[0], "t": t, "split": split}
            return None
    return None


def pick_card(card_arm: str | None):
    from cod.analysis.load import episode_label
    from cod.faults import Scenario
    from cod.synth.evaluate import counterexample_card, success
    from cod.synth.llm_synth import simplicity
    arms = _arm_specs(os.path.join(RESULTS, "arms"))
    cands = []
    for d, spec in arms.items():
        if not spec.startswith("playbook:"):
            continue
        if card_arm and card_arm not in spec:
            continue
        pr = _playbook_priority(spec)
        if pr is not None:
            cands.append((pr, d, spec))
    cands.sort()
    for pr, d, spec in cands:
        for suite in ("test_id", "frontier", "alhazmi"):
            eps = []
            for p in sorted(glob.glob(os.path.join(RESULTS, "arms", d, suite, "*.json"))):
                rec = json.load(open(p))
                sd = rec.get("scenario_def")
                if not sd or not any("fx" in x for x in rec["decisions"]):
                    continue
                label, _, _ = episode_label(rec, suite, Scenario.from_dict(sd).content_hash())
                if label != "rescuable" or success(rec, label):
                    continue
                e = dict(rec, id=rec["scenario"], scenario=sd, label=label)
                eps.append(e)
            eps.sort(key=lambda e: (simplicity(e), e["id"], e["seed"]))
            for e in eps:
                rem = screen_remedy(e["scenario"], e["seed"])
                if rem is not None:
                    card = counterexample_card(e, remedy=rem, use_remedy=True)
                    return card, {"arm": spec, "suite": suite, "episode": f"{e['id']} (seed {e['seed']})",
                                  "remedy": True, "family": e["family"]}
            if eps:
                e = eps[0]
                return counterexample_card(e), {"arm": spec, "suite": suite, "episode": f"{e['id']} (seed {e['seed']})",
                                                "remedy": False, "family": e["family"]}
    return None, None


def arm_label(spec: str) -> str:
    path = spec[len("playbook:"):].split("@")[0]
    m = re.search(r"/synth/(\w+?)_(.+?)_s(\d+)/playbook_(\w+)\.py", path)
    if m:
        return f"{m.group(2)} synthesis seed {m.group(3)}, checkpoint {m.group(4)}"
    m = re.search(r"/preview/tuned_s(\d+)_(\w+)\.py", path)
    if m:
        lab = f"preview checkpoint {m.group(0)[9:-3]} (calibrated {m.group(2)} checkpoint of a seed-{m.group(1)} synthesis"
        if os.path.exists(path):  # same code as the corresponding main checkpoint (ignoring its two header lines)?
            body = open(path).read()
            for p in sorted(glob.glob(os.path.join(SYN, f"main_*_s{m.group(1)}", f"playbook_{m.group(2)}.py"))):
                src = open(p).read().split("\n", 2)
                if len(src) == 3 and src[2] == body:
                    run = os.path.basename(os.path.dirname(p))[5:].rsplit("_s", 1)[0]
                    return lab + f"; code identical to checkpoint {m.group(2)} of the main {run} seed-{m.group(1)} run)"
        return lab + ")"
    return os.path.basename(path)


# ----------------------------------------------------------------------------- playbook excerpt
def latest_checkpoint(run_dir: str):
    best = None
    for p in glob.glob(os.path.join(run_dir, "meta_*.json")):
        m = json.load(open(p))
        name = os.path.basename(p)[5:-5]
        key = (m.get("tokens", 0), name == "final")
        if os.path.exists(os.path.join(run_dir, f"playbook_{name}.py")) and (best is None or key > best[0]):
            best = (key, name, m)
    return (best[1], best[2]) if best else (None, None)


def excerpt(src: str, max_lines: int = 30) -> tuple[str, int, int]:
    lines = [ln.rstrip() for ln in ascii_only(src).splitlines()]
    header = [ln for ln in lines[:3] if ln.startswith("#")]
    body = lines[len(header):]
    try:
        d = next(i for i, ln in enumerate(body) if ln.startswith("def playbook("))
    except StopIteration:
        d = 0
    out = list(header)
    if d > 0:
        n_helper = sum(1 for ln in body[:d] if ln.strip())
        out.append(f"# ... helper functions omitted ({n_helper} lines)")
    out.append(body[d])
    rest = [ln for ln in body[d + 1:] if ln.strip()]
    blocks, cur = [], []
    for ln in rest:  # a block starts at a comment line at function-body indentation
        if re.match(r"^ {4}#", ln) and cur:
            blocks.append(cur)
            cur = []
        cur.append(ln)
    if cur:
        blocks.append(cur)
    used, shown = 0, 0
    budget = max_lines - len(out) - 1
    for b in blocks:
        if used + len(b) > budget:
            break
        out += b
        used += len(b)
        shown += 1
    if shown < len(blocks):
        out.append(f"    # ... {len(blocks) - shown} further rule blocks omitted "
                   f"({sum(len(b) for b in blocks[shown:])} lines)")
    return "\n".join(out) + "\n", shown, len(blocks)


# ----------------------------------------------------------------------------- reproducibility table
def suite_rows():
    from cod.scenarios import ALHAZMI13, FRONTIER5
    rows = []
    for p in sorted(glob.glob(os.path.join(SUITES, "*.json"))):
        name = os.path.basename(p)[:-5]
        d = json.load(open(p))
        h = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]
        stored = open(p + ".sha").read().strip() if os.path.exists(p + ".sha") else None
        n_ep = sum(len(x.get("seeds") or [1]) for x in d)
        fams = sorted({x["scenario"]["family"] for x in d})
        rows.append((name, len(d), n_ep, "/".join(fams), h, stored == h))
    for name, scs in (("alhazmi", ALHAZMI13), ("frontier", FRONTIER5)):
        h = hashlib.sha256(json.dumps([s.as_dict() for s in scs], sort_keys=True).encode()).hexdigest()[:16]
        rows.append((name, len(scs), 3 * len(scs), "/".join(sorted({s.family for s in scs})), h, None))
    return rows


def build_table(env: dict, roles: dict[str, str]) -> str:
    from cod.prompts import PROMPT_HASH
    from cod.runner import Policy
    from cod.supervisors.hybrid import HybridSupervisor
    from cod.supervisors.llm_online import OnlineLLMSupervisor
    from cod.synth.evaluate import ACTION_PENALTY, QUALITY_IAE_MAX
    from cod.synth.llm_synth import Synthesiser
    from tools.llm import LLM

    def defaults(fn, keys):
        sig = inspect.signature(fn)
        return {k: sig.parameters[k].default for k in keys if k in sig.parameters}

    M, E, P = [], [], []
    # ---- models
    M.append(r"\begin{tabular}{@{}lp{0.29\linewidth}lll@{}}")
    M.append(r"\toprule Model tag & Role in this study & Size & Quant. & Digest\\\midrule")
    order = sorted(roles, key=lambda m: ("pilot runs only" in roles[m] and "; " not in roles[m], "coder" not in roles[m], m))
    for m in order:
        info = env.get("models", {}).get(m, {})
        dig = (info.get("digest") or "")[:12] or "n/a"
        M.append(f"{tt(m)} & {tex(roles[m])} & {tex(info.get('parameter_size') or 'n/a')} & "
                 f"{tex(info.get('quantization_level') or 'n/a')} & {tt(dig)}\\\\")
    M.append(r"\bottomrule\end{tabular}")
    # ---- records
    head = r"\begin{tabular}{@{}p{0.22\linewidth}p{\dimexpr0.78\linewidth-2\tabcolsep\relax}@{}}" + "\n" + \
        r"\toprule Item & Value\\\midrule"
    E.append(head)
    P.append(head)
    L = E

    def sec(t):
        L.append(r"\multicolumn{2}{@{}l}{\textit{" + t + r"}}\\")

    def row(k, v):
        L.append(f"{k} & {v}\\\\")

    pk = env["packages"]
    sec("Software")
    row("Python", tex(env["python"]))
    row(tt("tep-studio"), tex(pk.get("tep-studio")) + " (revised TEP kernel; pip dependency, BSD-3-Clause; not redistributed)")
    row("Scientific stack", ", ".join(f"{tex(k)} {tex(v)}" for k, v in pk.items() if k not in ("tep-studio",) and v))
    row("LLM server", f"Ollama {tex(env['ollama'].get('version') or 'n/a')} ({tex(env['ollama'].get('source', ''))})")
    row("Operating system", tex(env["os"]))
    sec("Hardware")
    hw = f"{tex(env['cpu'])}, {env['cpu_cores']} CPU cores"
    if env.get("gpu_cores"):
        hw += f", {env['gpu_cores']}-core integrated GPU"
    if env.get("memory_gb"):
        hw += f", {env['memory_gb']}\\,GB unified memory"
    row("Machine", hw + "; one shared workstation (other workloads ran concurrently, so wall-clock latencies are inflated)")
    mg = src_search(os.path.join(WS, "tools", "memguard.py"), r'--limit-gb"[^\n]*default=([\d.]+)', None)
    row("Resource guard", f"project memory cap {float(mg or 200):g}\\,GB enforced by a watchdog that restarts jobs with halved parallelism")
    sec("LLM serving and decoding")
    npar = src_search(os.path.join(WS, "tools", "start_ollama_private.sh"), r"OLLAMA_NUM_PARALLEL:-(\d+)", "n/a")
    ctx = inspect.signature(LLM.__init__).parameters["num_ctx"].default
    row("Server", f"private Ollama instance, {npar} parallel sequences, context {ctx:,} tokens for every request, flash attention")
    sd = defaults(Synthesiser.__init__, ["temperature", "max_out_tokens"])
    od = defaults(OnlineLLMSupervisor.__init__, ["temperature", "max_tokens", "est_call_tokens"])
    cb_t = src_search("experiments/stream_hybrid.py", r"temperature=([\d.]+), seed=version", "n/a")
    row("Decoding", f"synthesis: temperature {sd['temperature']:g}, at most {sd['max_out_tokens']:,} output tokens, "
                    f"sampling seed $1000\\,s+i$ (run seed $s$, iteration $i$; $+500\\,k$ for the $k$-th fix of a rejected "
                    f"candidate); online: temperature {od['temperature']:g}, "
                    f"at most {od['max_tokens']} output tokens, JSON-schema constrained; compile-back: temperature {cb_t}")
    row("LLM cache", "SQLite, keyed by SHA-256 of (model, messages, options, format); a resumed run never repeats a call, "
                     "and cached runs replay exactly")
    sec("Prompts")
    row("Prompt hash", f"{tt(PROMPT_HASH)} (SHA-256 of Part~A $+$ Part~B, first 16 hex)")
    L = P
    sec("Test suites")
    for name, n, n_ep, fams, h, ok in suite_rows():
        chk = " (matches stored hash)" if ok else (" (STORED HASH DIFFERS)" if ok is False else " (computed here)")
        row(tt(name), f"{n} scenarios, {n_ep} episodes, families {tex(fams)}; content hash {tt(h)}{chk}")
    sec("Seeds")
    fdd_seed = src_search("experiments/fdd_data.py", r"BASE_SEED = (\d+)", "n/a")
    cov_seed = src_search("experiments/coverage.py", r"base seed (\d+)", "n/a")
    order_seed = src_default("experiments/stream_hybrid.py", "--order-seed")
    synth_seeds = sorted({int(m.group(1)) for p in glob.glob(os.path.join(SYN, "main_*_s*"))
                          for m in [re.search(r"_s(\d+)$", p)] if m})
    gp_seeds = sorted({int(m.group(1)) for p in glob.glob(os.path.join(SYN, "gp_s*")) for m in [re.search(r"_s(\d+)$", p)] if m})
    row("Scenario sampling", f"headroom screen base seed 7 (training and synthesis) and 101 (held-out suites); FDD data {fdd_seed}; "
                             f"coverage study {cov_seed}; falsification round $r$ of synthesis run $s$ draws from the generator seeded with $(s, 9000+r)$")
    row("Episode noise", "screen and held-out suites: $1000+$scenario index (stored per suite entry); falsification "
                         "scenarios: $7000+1000\\,r+i$; anchors: 1, 2, 3; arms are paired on (scenario, seed)")
    row("Runs", f"LLM synthesis seeds {', '.join(map(str, synth_seeds)) or 'n/a'}; GP seeds "
                f"{', '.join(map(str, gp_seeds)) or 'n/a'}; deployment-stream order seed {order_seed}")
    sec("Invocation policy (all supervised arms)")
    pol = Policy()
    row(tt("Policy") + " defaults", "; ".join(f"{tt(f.name)}$=${tex(getattr(pol, f.name))}" for f in dataclasses.fields(pol)))
    sec("Method parameters")
    sy = defaults(Synthesiser.__init__, ["falsify_every", "falsify_n", "falsify_families", "n_cards", "pop_k", "batch_cap",
                                         "tune_budget"])
    budget = src_search(os.path.join(SYN, MAIN_SYNTH, "log.txt"), r"budget (\d+)", None) if os.path.exists(
        os.path.join(SYN, MAIN_SYNTH, "log.txt")) else None
    ck = src_search("scripts/launch_synth.sh", r"--checkpoints ([\d,]+)", "n/a")
    head = f"budget {int(budget):,} tokens per run; checkpoints {tex(ck.replace(',', ', '))}; " if budget else ""
    tb = os.path.join(SYN, MAIN_SYNTH, "train_batch.json")
    if os.path.exists(tb):
        head += f"initial training batch {len(json.load(open(tb)))} scenarios (stratified from the seed-7 screen); "
    row("LLM synthesis", head +
        f"population {sy['pop_k']}; at most {sy['n_cards']} counterexample cards per repair; falsification every "
        f"{sy['falsify_every']} iterations on {sy['falsify_n']} fresh scenarios of families "
        f"{tex('/'.join(sy['falsify_families']))}; training batch capped at {sy['batch_cap']}; "
        f"{sy['tune_budget']} batch evaluations of CPU tuning per round; quality threshold \\%G IAE $\\le$ {QUALITY_IAE_MAX:g}\\,mol\\,\\%\\,h; "
        f"restraint penalty {ACTION_PENALTY:g} per action on benign scenarios")
    gp = {k: src_default("experiments/gp_run.py", "--" + k) for k in ("evals", "pop", "elite", "falsify-every", "falsify-n")}
    row("GP (B5)", f"{gp['evals']:,} episode evaluations; population {gp['pop']}, elite {gp['elite']}; falsification every "
                   + (f"{gp['falsify-every']} generations" if gp['falsify-every'] != 1 else "generation") +
                   f" on {gp['falsify-n']} fresh scenarios")
    hy = defaults(HybridSupervisor.__init__, ["triggers", "cooldown_h", "max_escalations", "ttl_h", "resid_P", "resid_T"])
    g_thr = src_search("cod/supervisors/hybrid.py", r'abs\(f\["G_cmd_err"\]\) > ([\d.]+)', "n/a")
    on_src = inspect.getsource(OnlineLLMSupervisor.abnormal)  # the gated online arm must use the same test
    if (inspect.signature(OnlineLLMSupervisor.abnormal).parameters["ttl_h"].default != hy["ttl_h"]
            or f"> {hy['resid_P']}" not in on_src or f"> {hy['resid_T']}" not in on_src or f"> {g_thr}" not in on_src):
        print("WARNING: online gate and hybrid abnormality test differ; update the 'Online and hybrid' row")
    from cod.fdd import NoveltyDetector
    try:
        alpha = f"{NoveltyDetector.load().alpha:g}"
    except Exception:  # noqa: BLE001 - detector not trained yet
        alpha = f"{inspect.signature(NoveltyDetector.__init__).parameters['alpha'].default:g}"
    row("Online and hybrid", f"token bucket per episode; estimated cost per call {od['est_call_tokens']:,.0f} tokens; "
                             f"abnormality test (gate of the gated online arm, part of trigger (a)): an active alarm, "
                             f"time-to-limit $<$ {hy['ttl_h']:g}\\,h, pressure or temperature residual above "
                             f"{hy['resid_P']:g}\\,kPa or {hy['resid_T']:g}\\,\\si{{\\celsius}}, or grade error above "
                             f"{float(g_thr):g}\\,mol\\,\\%; hybrid escalation triggers "
                             f"{', '.join('(' + c + ')' for c in hy['triggers'])} of \\cref{{sec:hybrid}} (conformal level "
                             f"$\\alpha={alpha}$), cooldown {hy['cooldown_h']:g}\\,h, at most {hy['max_escalations']} "
                             f"escalations per episode")
    E.append(r"\bottomrule\end{tabular}")
    P.append(r"\bottomrule\end{tabular}")
    out = ["% generated by code/scripts/gen_repro.py - do not edit"]
    for name, body in (("reproTableModels", M), ("reproTableEnv", E), ("reproTableProtocol", P)):
        out.append(f"\\newcommand{{\\{name}}}{{%")
        out += body
        out.append("}")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true", help="do not query the Ollama server; reuse the snapshot")
    ap.add_argument("--card-arm", default=None, help="restrict the card to playbook arms whose spec contains this")
    a = ap.parse_args()

    from cod.prompts import FEATURE_GLOSSARY, PART_A, PART_B, PROMPT_HASH
    from cod.supervisors.llm_online import INSTRUCTION
    from cod.synth import llm_synth

    env = collect_env(a.offline)
    roles = models_used(env.get("roles"))
    env["roles"] = roles
    write("repro_env.json", json.dumps(env, indent=1, sort_keys=True) + "\n")

    # verbatim prompt material
    write("repro_part_a.txt", ascii_only(PART_A) + "\n")
    write("repro_contract.txt", ascii_only(llm_synth.CONTRACT) + "\n")
    write("repro_online_instruction.txt", ascii_only(INSTRUCTION) + "\n")
    write("repro_part_b_full.txt", ascii_only(PART_B) + "\n")
    write("repro_glossary.txt", ascii_only(FEATURE_GLOSSARY) + "\n")
    write("repro_state_card.txt", ascii_only(state_card("SN0003", "test_id")) + "\n")

    rows = partb_sections()
    t = [r"\begin{tabular}{@{}p{0.2\linewidth}p{\dimexpr0.68\linewidth-4\tabcolsep\relax}r@{}}",
         r"\toprule Section & Content & Characters\\\midrule"]
    t += [f"{tex(title)} & {desc} & {n:,}\\\\" for title, desc, n in rows]
    t += [r"\midrule Total & & " + f"{len(PART_B):,}" + r"\\\bottomrule\end{tabular}"]
    write("repro_partb.tex", "\n".join(t) + "\n")

    # counterexample card
    card, cinfo = pick_card(a.card_arm)
    if card is None:
        card = "(no failed playbook episode with feature snapshots is available yet)"
        cinfo = {"arm": "n/a", "suite": "n/a", "episode": "n/a", "remedy": False, "family": "n/a"}
    write("repro_card.txt", ascii_only(card) + "\n")

    # playbook excerpt
    run = os.path.join(SYN, MAIN_SYNTH)
    name, meta = latest_checkpoint(run)
    if name is not None:
        src = open(os.path.join(run, f"playbook_{name}.py")).read()
        ex, shown, nblocks = excerpt(src)
    else:
        ex, shown, nblocks, meta = "# no checkpoint yet\n", 0, 0, {}
    write("repro_playbook.txt", ex)
    # CPU calibration passes (tune.py) that replaced the incumbent before this checkpoint was saved
    n_tuned = None
    logp = os.path.join(run, "log.txt")
    if name is not None and os.path.exists(logp):
        n_tuned = 0
        for ln in open(logp):
            if ln.strip().startswith(f"checkpoint {name}:"):
                break
            n_tuned += ln.strip().startswith("tuned incumbent:")
    if n_tuned is None:
        tune_note = ""
    elif n_tuned == 0:
        tune_note = ("No CPU calibration pass had modified the incumbent before this checkpoint was saved, so the "
                     "rules are the coder's output.")
    else:
        tune_note = (f"Before this checkpoint was saved, the CPU calibration step (rule reordering and threshold "
                     f"rescaling, no language-model tokens) had replaced the incumbent "
                     f"{'once' if n_tuned == 1 else str(n_tuned) + ' times'}, so the rule order and some thresholds "
                     f"may differ from the coder's output.")

    # measured prompt sizes
    first_ptok = None
    if os.path.exists(os.path.join(run, "meta_v0.json")):
        h = json.load(open(os.path.join(run, "meta_v0.json"))).get("history") or []
        first_ptok = h[0]["ptok"] if h else None
    from cod.analysis.load import arm_dir_name
    pt = calls = 0
    for p in glob.glob(os.path.join(RESULTS, "arms", arm_dir_name("onlineg:qwen3-coder:30b"), "*", "*.json")):
        r = json.load(open(p))
        pt += int(r.get("prompt_tokens", 0))
        calls += int(r.get("llm_calls", 0))

    def mac(n, v):
        return f"\\newcommand{{\\{n}}}{{{v}}}"

    mlines = [
        "% generated by code/scripts/gen_repro.py - do not edit",
        mac("reproPromptHash", tt(PROMPT_HASH)),
        mac("reproPartAChars", f"{len(PART_A):,}"),
        mac("reproPartBChars", f"{len(PART_B):,}"),
        mac("reproGlossaryChars", f"{len(FEATURE_GLOSSARY):,}"),
        mac("reproContractChars", f"{len(llm_synth.CONTRACT):,}"),
        mac("reproSynthSystemChars", f"{len(llm_synth.SYSTEM):,}"),
        mac("reproNumValidKeys", f"{len(llm_synth.valid_feature_keys()):,}"),
        mac("reproSynthFirstPromptTokens", f"{first_ptok:,}" if first_ptok else r"\todo{n/a}"),
        mac("reproOnlinePromptTokens", f"{pt / calls:,.0f}" if calls else r"\todo{online arm pending}"),
        mac("reproOnlineCalls", f"{calls:,}"),
        mac("reproCardArm", tex(arm_label(cinfo["arm"])) if cinfo["arm"] != "n/a" else "n/a"),
        mac("reproCardSuite", tt(cinfo["suite"])),
        mac("reproCardEpisode", tex(cinfo["episode"])),
        mac("reproCardFamily", tex(cinfo["family"])),
        mac("reproCardChars", f"{len(card):,}"),
        mac("reproCardRemedyNote",
            "The counterfactual line is the stored oracle record of the held-out screen for the same scenario and seed, "
            "which is the computation the synthesiser performs for its own cards."
            if cinfo["remedy"] else "No stored counterfactual is available for this episode, so the card is shown without one."),
        mac("reproPlaybookRun", tex(MAIN_SYNTH.replace("main_", "").replace("_s", ", seed "))),
        mac("reproPlaybookCkpt", tt(name or "n/a")),
        mac("reproPlaybookTokens", f"{int(meta.get('tokens', 0)):,}" if meta else "n/a"),
        mac("reproPlaybookIter", str(meta.get("best_it", "n/a")) if meta else "n/a"),
        mac("reproPlaybookTrainSR", f"{meta['best_sr']:.3f}" if meta and "best_sr" in meta else "n/a"),
        mac("reproPlaybookBatch", str(meta.get("batch_size", "n/a")) if meta else "n/a"),
        mac("reproPlaybookLoc", str(meta.get("loc", "n/a")) if meta else "n/a"),
        mac("reproPlaybookRules", str(len(meta.get("rules", []))) if meta else "n/a"),
        mac("reproPlaybookTuneNote", tune_note),
        mac("reproPlaybookBlocksShown", str(shown)),
        mac("reproPlaybookBlocks", str(nblocks)),
        mac("reproEnvDate", tex(env["collected"])),
    ]
    write("repro_macros.tex", "\n".join(mlines) + "\n")

    write("repro_table.tex", build_table(env, roles))


if __name__ == "__main__":
    main()
