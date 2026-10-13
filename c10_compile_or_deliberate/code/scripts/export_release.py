"""Build the double-anonymised release copy of the study: code, suites, data descriptors, selected results, prompts.

  cd <workspace> && PYTHONPATH=<project>/code .venv/bin/python <project>/code/scripts/export_release.py --dry-run
  ... export_release.py [--out DIR] [--episodes] [--with-fdd-artifacts] [--with-llm-cache] [--with-paper] [--force]

The release mirrors the workspace, so the code runs unchanged from its root:
  <out>/README.md                         pointer to the package README
  <out>/tools/                            cached LLM client, memory guard, private-server launcher
  <out>/<project>/code/                   cod/, experiments/, scripts/, tests/, README.md, LICENSE, run_all.sh,
                                          requirements.txt (pinned from the current environment)
  <out>/<project>/data/                   *.json descriptors and suites/ (+ content hashes)
  <out>/<project>/results/                screen/ and screen_101/ (labels, oracle plans), synth/ and synth_archive/
                                          (playbook checkpoints, metadata, training batches, run logs), preview/,
                                          coverage/, stream/ (when present)
  <out>/<project>/results_summary/        episodes.csv (one row per episode, arm specs anonymised) and analysis.json
  <out>/<project>/prompts/                rendered prompt strings and PROMPT_HASH
  <out>/MANIFEST.txt                      sha256, size and path of every file, plus what was excluded and why
Options: --episodes adds the per-episode JSONs of results/arms (decision logs; the per-step plant trace is dropped);
--with-fdd-artifacts adds data/*.npz and data/*.pkl; --with-llm-cache adds a consistent snapshot of the LLM call cache;
--with-paper adds the anonymised manuscript and supplementary sources (LaTeX, generated tables and numbers, figures,
listings, bibliography) and their PDFs, but never the files that identify the authors (authors.tex, titlepage, cover
letter, acknowledgment, the named build) or the internal submission notes. Per-step plant traces are dropped from the
episode JSONs except for the case-study scenario (CASE_SCENARIO), which the case-study figure plots.

Anonymisation. Absolute paths are rewritten to relative ones: paths under the workspace become workspace-relative
(all commands run from the workspace root, so arm specs such as playbook:<project>/results/synth/... keep resolving),
a bare workspace path becomes `<workspace>` (a script-relative `cd` in shell scripts), the home directory `~`.
Identifying strings are derived at run time from this machine and never stored in this file: the login and full
user name, the host name, the git user name and e-mail, and the workspace path components below the home
directory (plus any --extra-pattern). After rewriting, every remaining case-insensitive match in a text file is
replaced by `anon`; a binary file that still matches is dropped. File and directory names are rewritten the same
way. The dry run prints what would be included, what the rewrite changes, and the residual matches; it writes
nothing.

Always excluded: refs/ (third-party code and papers, incl. safetep-core), cache/ (unless --with-llm-cache), logs/,
scratch/, configs/, paper/ (except generated/analysis.json and generated/repro_env.json), results/pilots/ (development runs), GP resume
pickles (state.pkl), __pycache__/, .pytest_cache/, *.pyc, *.tmp, *.log, .DS_Store.
"""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import glob
import hashlib
import importlib.metadata as md
import io
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJ = os.path.dirname(CODE)
WS = os.path.dirname(PROJ)
PNAME = os.path.basename(PROJ)
HOME = os.path.expanduser("~")
sys.path.insert(0, CODE)
sys.path.insert(0, WS)

MARKER = "# anonymised release manifest (export_release.py)"
TEXT_EXT = {".py", ".sh", ".md", ".txt", ".json", ".csv", ".tex", ".bib", ".yaml", ".yml", ".toml", ".cfg", ".sha", ""}
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".mypy_cache", ".ruff_cache"}
SKIP_FILES = {".DS_Store"}
SKIP_EXT = {".pyc", ".tmp", ".log", ".swp"}
GENERIC = {"documents", "desktop", "downloads", "users", "home", "mac", "macbook", "studio", "pro", "air", "local",
           "imac", "mini", "admin", "user", "root", "lan", "localdomain", "work", "code", "projects", "src"}
REQUIREMENTS = ["tep-studio", "numpy", "scipy", "scikit-learn", "pandas", "statsmodels", "matplotlib", "requests",
                "psutil", "tqdm", "pyyaml", "pytest"]
TOOLS = ["__init__.py", "llm.py", "memguard.py", "memstatus.py", "start_ollama_private.sh"]
CASE_SCENARIO = "SN0003"  # analyze.py --case: its episodes keep their per-step trace (case-study figure)
# paper/ files that identify the authors or are internal submission notes: never released
PAPER_PRIVATE = {"authors.tex", "titlepage.tex", "titlepage.pdf", "cover_letter.tex", "cover_letter.pdf",
                 "manuscript_named.pdf", "SUBMISSION_CHECKLIST.md", "mock_review_response.md",
                 "suggested_reviewers_TEMPLATE.md", "acknowledgment.tex"}
PAPER_BUILD_EXT = {".aux", ".log", ".fls", ".fdb_latexmk", ".out", ".blg", ".toc", ".synctex", ".gz", ".bbl", ".xdv"}


# ----------------------------------------------------------------------------- identifying strings
def _git(key: str) -> str | None:
    try:
        r = subprocess.run(["git", "config", "--global", "--get", key], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def identifiers(extra: list[str]) -> list[tuple[str, str]]:
    """[(category, pattern)] gathered from this machine; patterns are matched case-insensitively."""
    out: list[tuple[str, str]] = []

    def add(cat, s):
        if s and len(s) >= 3 and s.lower() not in GENERIC and (cat, s) not in out:
            out.append((cat, s))

    add("login name", getpass.getuser())
    try:
        import pwd
        pw = pwd.getpwuid(os.getuid())
        add("login name", pw.pw_name)
        for w in re.split(r"[\s,]+", pw.pw_gecos or ""):
            add("full name", w)
    except Exception:  # noqa: BLE001
        pass
    host = socket.gethostname()
    add("host name", host)
    stem = host.split(".")[0]
    add("host name", stem)
    for w in re.split(r"[-_.]", stem):
        add("host name", w)
    name, email = _git("user.name"), _git("user.email")
    for w in re.split(r"\s+", name or ""):
        add("git user name", w)
    if email:
        add("git e-mail", email)
        add("git e-mail", email.split("@")[0])
    rel = os.path.relpath(WS, HOME) if WS.startswith(HOME) else WS
    for comp in rel.split(os.sep):
        add("workspace path component", comp)
    for e in extra:
        add("extra pattern", e)
    # longest first so that a host name is reported before its fragments
    return sorted(out, key=lambda x: -len(x[1]))


def id_regex(pat: str) -> re.Pattern:
    """Case-insensitive; patterns shorter than 5 characters only match as whole alphanumeric tokens, so that a
    short name cannot corrupt an ordinary word that happens to contain it."""
    core = re.escape(pat)
    if len(pat) < 5:
        core = r"(?<![A-Za-z0-9])" + core + r"(?![A-Za-z0-9])"
    return re.compile(core, re.I)


def mask(s: str) -> str:
    return s[0] + "*" * max(len(s) - 2, 1) + s[-1] if len(s) > 2 else "**"


# ----------------------------------------------------------------------------- rewriting
def path_rules() -> list[tuple[str, str]]:
    """Absolute paths under the workspace become workspace-relative (every command runs from the workspace root,
    so e.g. a playbook arm spec keeps resolving); arm directory names carry the path with '/' -> '_'."""
    us = WS.replace(os.sep, "_")
    return [
        (us + "_", ""), (us, ""),
        (WS + "/", ""), (WS, "<workspace>"),
        (HOME + "/", "~/"), (HOME, "~"),
    ]


def rewrite_text(text: str, rel_dst: str, ids: list[tuple[str, str]], stats: dict) -> str:
    if rel_dst.endswith(".sh"):  # `cd <workspace>` -> cd relative to the script's own location
        depth = rel_dst.count("/")
        rep = 'cd "$(dirname "${BASH_SOURCE[0]}")/' + "../" * depth + '"'
        n = len(re.findall(r"cd\s+" + re.escape(WS) + r"(?=\s|$)", text))
        if n:
            text = re.sub(r"cd\s+" + re.escape(WS) + r"(?=\s|$)", lambda _: rep, text)
            stats["shell cd rewritten"] = stats.get("shell cd rewritten", 0) + n
    for a, b in path_rules():
        if a and a in text:
            stats["absolute path rewritten"] = stats.get("absolute path rewritten", 0) + text.count(a)
            text = text.replace(a, b)
    for cat, pat in ids:
        text, n = id_regex(pat).subn("anon", text)
        if n:
            stats[f"stripped: {cat}"] = stats.get(f"stripped: {cat}", 0) + n
    return text


def rewrite_name(rel: str, ids) -> str:
    us = WS.replace(os.sep, "_")
    rel = rel.replace(us + "_", "").replace(us, "")
    for _, pat in ids:
        rel = id_regex(pat).sub("anon", rel)
    return rel


def scan(data: bytes, ids) -> dict[str, int]:
    s = data.decode("latin-1")
    hits = {}
    for cat, pat in ids:
        n = len(id_regex(pat).findall(s))
        if n:
            hits[cat] = hits.get(cat, 0) + n
    n = s.count(HOME + "/") + s.count(HOME.replace(os.sep, "_") + "_")
    if n:
        hits["absolute home path"] = n
    return hits


# ----------------------------------------------------------------------------- items
@dataclass
class Item:
    dst: str                      # release-relative path
    group: str                    # manifest group
    produce: Callable[[], bytes]  # raw content (before anonymisation)
    text: bool


def _reader(p):
    return lambda: open(p, "rb").read()


def _pickle_reader(p):
    """Our own pickles (fitted FDD model, novelty gate) may store absolute paths in their metadata: rewrite string
    metadata under the workspace to workspace-relative paths and re-pickle; other files are copied unchanged."""
    if not p.endswith(".pkl"):
        return _reader(p)

    def produce():
        import pickle
        obj = pickle.load(open(p, "rb"))
        meta = getattr(obj, "meta", None)
        if isinstance(meta, dict):
            for k, v in list(meta.items()):
                if isinstance(v, str) and v.startswith(WS + os.sep):
                    meta[k] = os.path.relpath(v, WS)
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    return produce


def walk(src_root: str, dst_root: str, group: str, keep: Callable[[str], bool] = lambda p: True) -> list[Item]:
    items = []
    if os.path.isfile(src_root):
        files = [src_root]
        base = os.path.dirname(src_root)
    else:
        files = []
        for d, dirs, fs in os.walk(src_root):
            dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
            files += [os.path.join(d, f) for f in sorted(fs)]
        base = src_root
    for p in files:
        name = os.path.basename(p)
        ext = os.path.splitext(name)[1]
        if name in SKIP_FILES or ext in SKIP_EXT or not keep(p):
            continue
        rel = os.path.relpath(p, base)
        dst = os.path.normpath(os.path.join(dst_root, rel)) if not os.path.isfile(src_root) else os.path.join(dst_root, name)
        items.append(Item(dst, group, _reader(p), ext in TEXT_EXT))
    return items


def prompt_items() -> list[Item]:
    from cod.prompts import FEATURE_GLOSSARY, PART_A, PART_B, PROMPT_HASH
    from cod.supervisors.llm_online import INSTRUCTION
    from cod.synth import llm_synth
    d = {"part_a.txt": PART_A, "part_b.txt": PART_B, "feature_glossary.txt": FEATURE_GLOSSARY,
         "contract.txt": llm_synth.CONTRACT, "synthesis_system_message.txt": llm_synth.SYSTEM,
         "online_system_message.txt": PART_A + "\n\n" + PART_B, "online_instruction.txt": INSTRUCTION,
         "PROMPT_HASH.txt": PROMPT_HASH + "  sha256(PART_A + PART_B)[:16]"}
    return [Item(f"{PNAME}/prompts/{k}", "prompts", (lambda v=v: (v + "\n").encode()), True) for k, v in d.items()]


def requirements_item() -> Item:
    lines = ["# pinned from the environment that produced the paper's results (python "
             f"{sys.version.split()[0]})"]
    for p in REQUIREMENTS:
        try:
            lines.append(f"{p}=={md.version(p)}")
        except md.PackageNotFoundError:
            pass
    return Item(f"{PNAME}/code/requirements.txt", "code", lambda: ("\n".join(lines) + "\n").encode(), True)


def episodes_csv_item() -> Item:
    def produce():
        from cod.analysis.load import load_episodes
        df = load_episodes()
        df = df.drop(columns=["path"])
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return buf.getvalue().encode()
    return Item(f"{PNAME}/results_summary/episodes.csv", "results summary", produce, True)


def arm_episode_items(ids) -> list[Item]:
    from cod.analysis.load import arm_dir_name
    items = []
    base = os.path.join(PROJ, "results", "arms")
    for d in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        fs = sorted(glob.glob(os.path.join(base, d, "*", "*.json")))
        if not fs:
            continue
        try:
            spec = json.load(open(fs[0]))["arm"]
            new_d = arm_dir_name(rewrite_text(spec, "x.json", ids, {}))
        except Exception:  # noqa: BLE001
            new_d = rewrite_name(d, ids)
        for p in fs:
            def produce(p=p):
                r = json.load(open(p))
                if not os.path.basename(p).startswith(CASE_SCENARIO + "__"):
                    r.pop("trace", None)
                return json.dumps(r, separators=(",", ":")).encode()
            items.append(Item(f"{PNAME}/results/arms/{new_d}/{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}",
                              "results: episodes", produce, True))
    return items


def llm_cache_item(ids) -> Item | None:
    """Consistent snapshot of the LLM call cache with every text column rewritten like a text file. Identifying
    strings occur only in the bookkeeping column `tag` (arm specs with absolute paths); keys (request hashes),
    prompts and responses are left byte-identical when they contain none, so a replay from the cache still hits."""
    src = os.path.join(WS, "cache", "llm_cache.sqlite")
    if not os.path.exists(src):
        return None

    def produce():
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dst = os.path.join(td, "c.sqlite")
            s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            t = sqlite3.connect(dst)
            s.backup(t)  # consistent snapshot while writers are active
            s.close()
            stats: dict = {}
            for (table,) in t.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
                cols = [r[1] for r in t.execute(f"PRAGMA table_info({table})") if (r[2] or "").upper() in ("TEXT", "")]
                for c in cols:
                    rows = t.execute(f"SELECT rowid, {c} FROM {table} WHERE typeof({c})='text'").fetchall()
                    for rowid, v in rows:
                        w = rewrite_text(v, "cache.txt", ids, stats)
                        if w != v:
                            t.execute(f"UPDATE {table} SET {c}=? WHERE rowid=?", (w, rowid))
            t.commit()
            t.execute("PRAGMA journal_mode=DELETE")
            t.execute("VACUUM")  # drop free pages that may still hold the old text
            t.commit()
            t.close()
            return open(dst, "rb").read()
    return Item("cache/llm_cache.sqlite", "LLM cache", produce, False)


def collect(a, ids) -> tuple[list[Item], list[tuple[str, str]]]:
    P = PNAME
    items: list[Item] = []
    excluded = [
        ("refs/", "third-party code and papers (safetep-core has no licence; the paper PDF is copyrighted)"),
        ("cache/", "LLM call cache (opt in with --with-llm-cache)" if not a.with_llm_cache else ""),
        ("logs/, scratch/, configs/", "run logs and scratch files"),
        ("paper/", "manuscript sources are submitted separately (only generated/analysis.json and generated/repro_env.json "
                   "are included; opt in with --with-paper)" if not a.with_paper else
                   "author-identifying files (authors.tex, title page, cover letter, acknowledgment, named build), internal "
                   "submission notes and LaTeX build files"),
        ("results/pilots/", "development pilot runs, not used in the paper"),
        ("results/synth/*/state.pkl", "GP resume state (pickle)"),
        ("data/fdd_rows/", "per-episode FDD feature cache (regenerated by experiments/fdd_data.py)"),
    ]
    if not a.episodes:
        excluded.append(("results/arms/", "per-episode JSONs (opt in with --episodes; episodes.csv summarises them)"))
    if not a.with_fdd_artifacts:
        excluded.append(("data/*.npz, data/*.pkl", "FDD feature matrices and fitted models (opt in with --with-fdd-artifacts)"))
    # code (with --readme, the given file replaces the package README as the release's only README)
    keep_code = (lambda p: os.path.abspath(p) != os.path.join(CODE, "README.md")) if a.readme else (lambda p: True)  # noqa: E731
    items += walk(CODE, f"{P}/code", "code", keep_code)
    items.append(requirements_item())
    # tools
    for t in TOOLS:
        p = os.path.join(WS, "tools", t)
        if os.path.exists(p):
            items.append(Item(f"tools/{t}", "tools", _reader(p), True))
    # data
    for p in sorted(glob.glob(os.path.join(PROJ, "data", "*.json"))):
        items.append(Item(f"{P}/data/{os.path.basename(p)}", "data", _reader(p), True))
    items += walk(os.path.join(PROJ, "data", "suites"), f"{P}/data/suites", "data")
    if a.with_fdd_artifacts:
        for p in sorted(glob.glob(os.path.join(PROJ, "data", "*.npz")) + glob.glob(os.path.join(PROJ, "data", "*.pkl"))):
            items.append(Item(f"{P}/data/{os.path.basename(p)}", "data (FDD artefacts)", _pickle_reader(p), False))
    # results
    R = os.path.join(PROJ, "results")
    for sub in ("screen", "screen_101"):
        items += walk(os.path.join(R, sub), f"{P}/results/{sub}", "results: screens")
    keep_synth = lambda p: os.path.basename(p) != "state.pkl"  # noqa: E731
    for sub in ("synth", "synth_archive"):
        items += walk(os.path.join(R, sub), f"{P}/results/{sub}", "results: synthesis", keep_synth)
    for sub in ("preview", "coverage", "stream"):
        if os.path.isdir(os.path.join(R, sub)):
            items += walk(os.path.join(R, sub), f"{P}/results/{sub}", "results: " + sub)
    gen = os.path.join(PROJ, "paper", "generated", "analysis.json")
    if os.path.exists(gen):
        items.append(Item(f"{P}/results_summary/analysis.json", "results summary", _reader(gen), True))
    env = os.path.join(PROJ, "paper", "generated", "repro_env.json")  # model digests; lets gen_repro.py --offline run
    if os.path.exists(env) and not a.with_paper:  # with --with-paper it comes with paper/generated/
        def env_produce(p=env):  # keep only the models this study used (the snapshot lists every installed model)
            e = json.load(open(p))
            if e.get("roles"):
                e["models"] = {m: v for m, v in e.get("models", {}).items() if m in e["roles"]}
            return (json.dumps(e, indent=1, sort_keys=True) + "\n").encode()
        items.append(Item(f"{P}/results_summary/repro_env.json", "results summary", env_produce, True))
    items.append(episodes_csv_item())
    if a.episodes:
        items += arm_episode_items(ids)
    if a.with_paper:
        items += paper_items()
    # prompts, cache, readme
    items += prompt_items()
    if a.with_llm_cache:
        it = llm_cache_item(ids)
        if it:
            items.append(it)
    readme = (f"# Anonymised release\n\nSee `{P}/code/README.md` for the package description and how to reproduce each "
              f"result, `{P}/code/run_all.sh` for the end-to-end script and `MANIFEST.txt` for the file list.\n")
    if a.with_paper:
        readme += (f"\nThe anonymised manuscript and supplementary material (LaTeX sources, generated tables and numbers, "
                   f"figures, PDFs) are in `{P}/paper/`. Rebuild them with\n\n    cd {P}/paper && latexmk -pdf manuscript.tex "
                   f"&& latexmk -pdf supplementary.tex && latexmk -pdf -g manuscript.tex\n\nafter regenerating the tables, "
                   f"numbers and figures with the `analysis` stage of `{P}/code/run_all.sh`.\n")
    if a.readme:
        items.append(Item("README.md", "readme", _reader(os.path.abspath(a.readme)), True))
    else:
        items.append(Item("README.md", "readme", lambda: readme.encode(), True))
    return items, [e for e in excluded if e[1]]


def paper_items() -> list[Item]:
    """Anonymised paper sources and PDFs (identifying files and LaTeX build files excluded)."""
    root = os.path.join(PROJ, "paper")
    keep = lambda p: (os.path.basename(p) not in PAPER_PRIVATE and os.path.splitext(p)[1] not in PAPER_BUILD_EXT  # noqa: E731
                      and not os.path.basename(p).startswith("manuscript_named"))
    items = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isfile(p) and keep(p) and name.endswith((".tex", ".bib", ".txt", ".pdf", ".png")):
            items.append(Item(f"{PNAME}/paper/{name}", "paper", _reader(p), os.path.splitext(name)[1] in TEXT_EXT))
    for sub in ("sections", "generated", "figures", "listings"):
        if os.path.isdir(os.path.join(root, sub)):
            items += walk(os.path.join(root, sub), f"{PNAME}/paper/{sub}", "paper", keep)
    return items


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(WS, "release_anonymous"))
    ap.add_argument("--dry-run", action="store_true", help="report what would be written; write nothing")
    ap.add_argument("--episodes", action="store_true", help="include per-episode JSONs (trace dropped)")
    ap.add_argument("--with-fdd-artifacts", action="store_true")
    ap.add_argument("--with-llm-cache", action="store_true")
    ap.add_argument("--with-paper", action="store_true", help="include the anonymised paper sources and PDFs")
    ap.add_argument("--readme", help="markdown file to use as the release README.md (replaces the package README)")
    ap.add_argument("--extra-pattern", action="append", default=[], help="additional identifying string to strip")
    ap.add_argument("--force", action="store_true", help="replace an existing release directory written by this script")
    ap.add_argument("--skip-csv", action="store_true", help="do not build results_summary/episodes.csv")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    if out == PROJ or out.startswith(PROJ + os.sep):
        sys.exit("refusing to write the release inside the project directory")
    ids = identifiers(a.extra_pattern)
    items, excluded = collect(a, ids)
    if a.skip_csv:
        items = [it for it in items if not it.dst.endswith("results_summary/episodes.csv")]

    if not a.dry_run:
        if os.path.exists(out):
            mf = os.path.join(out, "MANIFEST.txt")
            if not a.force:
                sys.exit(f"{out} exists; pass --force to replace it")
            if not (os.path.exists(mf) and open(mf).readline().startswith(MARKER)):
                sys.exit(f"{out} exists but was not written by this script; refusing to delete it")
            shutil.rmtree(out)
        os.makedirs(out)

    stats: dict = {}
    pre_hits: dict = {}
    residual: list = []
    dropped: list = []
    groups: dict = {}
    manifest = []
    for it in items:
        dst = rewrite_name(it.dst, ids)
        if dst != it.dst:
            stats["file names rewritten"] = stats.get("file names rewritten", 0) + 1
        if a.dry_run and it.dst.endswith("results_summary/episodes.csv"):
            n = len(glob.glob(os.path.join(PROJ, "results", "arms", "*", "*", "*.json")))
            groups.setdefault(it.group, [0, 0])[0] += 1
            manifest.append((None, None, dst + f"  (would summarise {n} episode files)"))
            continue
        raw = it.produce()
        for k, v in scan(raw, ids).items():
            pre_hits.setdefault(k, [0, 0])
            pre_hits[k][0] += v
            pre_hits[k][1] += 1
        if it.text:
            data = rewrite_text(raw.decode("utf-8", errors="surrogateescape"), dst, ids, stats).encode(
                "utf-8", errors="surrogateescape")
        else:
            data = raw
        h = scan(data, ids) or scan(dst.encode(), ids)
        if h:
            if it.text:
                residual.append((dst, h))
            else:
                dropped.append((dst, h))
                continue
        g = groups.setdefault(it.group, [0, 0])
        g[0] += 1
        g[1] += len(data)
        manifest.append((hashlib.sha256(data).hexdigest(), len(data), dst))
        if not a.dry_run:
            p = os.path.join(out, dst)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
            if dst.endswith(".sh"):
                os.chmod(p, 0o755)

    total = sum(g[1] for g in groups.values())
    opts = " ".join(k for k in ("episodes", "with_fdd_artifacts", "with_llm_cache", "with_paper") if getattr(a, k)) or "default"
    lines = [MARKER, f"# generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}; options: {opts}",
             f"# {len(manifest)} files, {total / 2**20:.1f} MiB", "# excluded:"]
    lines += [f"#   {k}: {v}" for k, v in excluded]
    lines += ["# sha256  bytes  path"]
    lines += [f"{h}  {n}  {p}" for h, n, p in manifest if h]
    if not a.dry_run:
        with open(os.path.join(out, "MANIFEST.txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    # ---------------------------------------------------------------- report
    print(("DRY RUN - nothing written. " if a.dry_run else "") + f"release: {out}")
    print(f"identifying patterns derived from this machine ({len(ids)}):")
    for cat, pat in ids:
        print(f"  {cat:26s} {mask(pat)}")
    print("would include:" if a.dry_run else "included:")
    for g, (n, b) in sorted(groups.items()):
        print(f"  {g:28s} {n:6d} files {b / 2**20:8.2f} MiB")
    print(f"  {'TOTAL':28s} {len(manifest):6d} files {total / 2**20:8.2f} MiB (episodes.csv size not counted in a dry run)")
    print("excluded:")
    for k, v in excluded:
        print(f"  {k}: {v}")
    print("identifying matches in the source content (before rewriting):")
    for k, (n, f) in sorted(pre_hits.items()):
        print(f"  {k:26s} {n:7d} matches in {f} files")
    print("rewrites applied:")
    for k, v in sorted(stats.items()):
        print(f"  {k:34s} {v}")
    print(f"residual matches after rewriting: {len(residual)} text files")
    for dst, h in residual[:30]:
        print(f"  {dst}: {h}")
    print(f"binary files dropped because they still match: {len(dropped)}")
    for dst, h in dropped[:30]:
        print(f"  {dst}: {h}")
    if residual and not a.dry_run:
        sys.exit(2)


if __name__ == "__main__":
    main()
