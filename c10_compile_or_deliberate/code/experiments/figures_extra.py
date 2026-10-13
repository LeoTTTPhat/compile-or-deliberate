"""Figures of the results section (called by analyze.py; vector PDF into paper/figures/).

  fig_family.pdf      per-family success heatmap of the main arms
  fig_pareto.pdf      (a) held-out success vs LLM tokens per episode, offline cost amortised over the deployment;
                      (b) allocation view: for a deployment of N episodes, total budget B and allocation f of each arm
  fig_synthesis.pdf   training success vs cumulative synthesis tokens (seeds; ablations), tuning events marked;
                      CPU-episode efficiency vs genetic programming
  fig_case.pdf        case-study trajectories (true reactor pressure + one more variable) with action markers
  fig_escalation.pdf  hybrid escalations by trigger and by fault family (placeholder until hybrid results exist)

Style: serif 8 pt, thin marks, fixed categorical colour per method family plus a marker shape per family
(secondary encoding for print), no dual axes, legends for >= 2 series, selective direct labels.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analyze as A  # noqa: E402

W = 5.4  # \textwidth of the elsarticle preprint layout (390 pt)
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE_SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
TRIG_COL = {"a": "#0f7a55", "b": "#1baf7a", "c": "#8fdcbd"}
TRIG_HATCH = {"a": "", "b": "////", "c": "...."}


def style():
    plt.rcParams.update({
        "font.size": 8, "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "lines.linewidth": 1.2, "lines.markersize": 4.5, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150})


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
    return path


def placeholder(path: str, text: str, h: float = 1.4) -> str:
    fig, ax = plt.subplots(figsize=(W, h))
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", color=MUTED, fontsize=9, style="italic", transform=ax.transAxes)
    return _save(fig, path)


def _best_per_family(pts: list[dict]) -> list[dict]:
    out = []
    for g in ("online", "playbook", "hybrid"):
        sub = [p for p in pts if p["group"] == g]
        if sub:
            out.append(max(sub, key=lambda p: (p["sr"], -p["amort"], A.parse_arm(p["arm"])["ckpt"] == "final")))
    return out


def slot_labels(ax, items: list[tuple], x0: float, y0: float, dy: float, fontsize: float = 5.8):
    """Selective direct labels in a fixed column of slots (axes coordinates) with thin leader lines to the points;
    robust to crowded scatter. items: (x, y, text) in data coordinates, placed top-down by y."""
    items = sorted(items, key=lambda t: -t[1])
    for k, (x, y, text) in enumerate(items):
        ax.annotate(text, (x, y), xytext=(x0, y0 - k * dy), textcoords="axes fraction", fontsize=fontsize, color=INK2,
                    ha="left", va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1, shrinkB=3, relpos=(0, 0.5)))


def place_labels(ax, items: list[dict], fontsize: float = 6.0, min_gap_pt: float = 7.0, col_px: float = 70.0):
    """Direct labels with simple vertical repulsion. items: {x, y, text, dx (pt), ha}. Labels whose anchors lie
    within `col_px` horizontally are kept at least `min_gap_pt` apart; a moved label gets a thin leader line."""
    if not items:
        return
    px = ax.figure.dpi / 72.0
    T = ax.transData
    rows = []
    for it in items:
        X, Y = T.transform((it["x"], it["y"]))
        rows.append({"X": X + it.get("dx", 4) * px * (1 if it.get("ha", "left") == "left" else -1), "Y0": Y, "Y": Y, "it": it})
    rows.sort(key=lambda r: r["Y0"])
    placed: list[dict] = []
    gap = min_gap_pt * px
    for r in rows:
        for _ in range(len(placed) + 1):  # each pass moves the label strictly upwards; bounded
            clash = [q for q in placed if abs(q["X"] - r["X"]) < col_px * px and abs(q["Y"] - r["Y"]) < gap - 1e-3]
            if not clash:
                break
            r["Y"] = max(q["Y"] for q in clash) + gap
        placed.append(r)
    for r in rows:
        it = r["it"]
        dy = (r["Y"] - r["Y0"]) / px
        dx = it.get("dx", 4) * (1 if it.get("ha", "left") == "left" else -1)
        ax.annotate(it["text"], (it["x"], it["y"]), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    color=INK2, ha=it.get("ha", "left"), va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=0, shrinkB=2) if abs(dy) > 3 else None)


# ======================================================================================= (a) family heatmap
def heatmap_rows(rows: list[dict]) -> list[dict]:
    """Main arms for the heatmap: drop online budget variants, hybrid budgets other than 16k/unbounded, and
    non-final checkpoints of seeds 2-3 when a later checkpoint of that seed exists."""
    keep = []
    latest: dict = {}
    for r in rows:
        a = A.parse_arm(r["arm"])
        if a["kind"] == "playbook" and a["group"] == "playbook" and a["seed"]:
            k = (a["tag"], a["model"], a["seed"])
            latest[k] = max(latest.get(k, -1), A.ckpt_rank(a["ckpt"]))
    for r in rows:
        a = A.parse_arm(r["arm"])
        if a["group"] == "online" and a["budget"] is not None:
            continue
        if a["group"] == "hybrid" and (a["budget"] != 16000.0 or a["ckpt"] != "final"):
            continue  # final playbook + 16k bucket only; the other hybrids are in the supplementary tables
        if a["kind"] == "search" or r["arm"] == "B2a":
            continue  # weakest LLM-free supervisors; in Table 2
        if a["kind"] == "playbook" and a["group"] == "playbook" and a["seed"]:
            if a["tag"] != "main" or A.ckpt_rank(a["ckpt"]) < latest[(a["tag"], a["model"], a["seed"])]:
                continue  # final playbook of each main run only; checkpoints and ablations are in Table 4
        keep.append(r)
    return keep


def fig_family(df: pd.DataFrame, rows: list[dict], path: str) -> str:
    rows = [r for r in heatmap_rows(rows) if r["has_data"]]
    held = df[df["suite"].isin(A.HELD)]
    fams = [f for f in A.FAMILIES_ID + A.FAMILIES_OOD if f in A.FAMILY_N]
    if not rows or not fams:
        return placeholder(path, "pending: no held-out episodes")
    gap = len(A.FAMILIES_ID)  # column index where the novel families start (after a visual gap)
    xs = [i if i < gap else i + 0.35 for i in range(len(fams))]
    M = np.full((len(rows), len(fams)), np.nan)
    P = np.zeros_like(M, dtype=bool)
    for i, r in enumerate(rows):
        g = held[held["arm"] == r["arm"]]
        for j, f in enumerate(fams):
            x = g[g["family"] == f]
            if len(x):
                M[i, j] = x["success"].mean()
                P[i, j] = len(x) < A.FAMILY_N[f]
    h = 0.165 * len(rows) + 0.75
    fig, ax = plt.subplots(figsize=(W, h))
    for i in range(len(rows)):
        for j in range(len(fams)):
            v = M[i, j]
            col = "#f0efec" if np.isnan(v) else BLUE_SEQ(v)
            ax.add_patch(plt.Rectangle((xs[j] - 0.5, i - 0.5), 1, 1, facecolor=col, edgecolor="white", linewidth=1.2))
            if not np.isnan(v):
                ax.text(xs[j], i, f"{100 * v:.0f}" + ("*" if P[i, j] else ""), ha="center", va="center", fontsize=6.5,
                        color="white" if v > 0.62 else INK)
    # group separators and family markers
    prev = None
    for i, r in enumerate(rows):
        if prev is not None and r["group"] != prev:
            ax.axhline(i - 0.5, color="white", lw=3.0)
        prev = r["group"]
        ax.plot([-0.012], [i], marker=A.MARKER[r["group"]], color=A.COLOR[r["group"]], ms=4, clip_on=False,
                markeredgecolor="white", markeredgewidth=0.4, transform=ax.get_yaxis_transform())
    ax.set_xlim(-0.5, xs[-1] + 0.5)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xticks(xs, [f"{f}\n({A.FAMILY_N[f]})" for f in fams])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows])
    ax.tick_params(axis="y", length=0, pad=9)
    ax.tick_params(axis="x", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for x0, x1, label in ((xs[0], xs[gap - 1], "anticipated families"), (xs[gap], xs[-1], "novel families (not used in synthesis)")):
        ax.annotate(label, xy=((x0 + x1) / 2, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 21),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, color=INK2, annotation_clip=False)

    sm = plt.cm.ScalarMappable(cmap=BLUE_SEQ, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.015, aspect=30)
    cb.set_label("Success rate (cell text in %)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, width=0.4)
    cb.outline.set_linewidth(0.4)
    handles = [Line2D([], [], marker=A.MARKER[g], color=A.COLOR[g], ls="", ms=4, label=A.GROUP_LABEL[g])
               for g in A.GROUP_ORDER if any(r["group"] == g for r in rows)]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=min(4, len(handles)), fontsize=6.5,
              handletextpad=0.2, columnspacing=1.0)
    if P.any():
        fig.text(0.99, 0.0, "* run in progress (fewer episodes than planned)", ha="right", va="bottom", fontsize=6, color=INK2)
    return _save(fig, path)


# ======================================================================================= (b) Pareto / allocation
def _llm_points(rows: list[dict]) -> list[dict]:
    pts = []
    for r in rows:
        h = r["suites"].get("held")
        if not r["has_data"] or h is None or h["n"] < 0.5 * A.NH():
            continue
        pts.append({**r, "sr": h["rate"], "lo": h["lo"], "hi": h["hi"], "partial": h["partial"]})
    return pts


def _tok_fmt(v, _pos=None):
    if v <= 0:
        return "0"
    if v >= 1e6:
        return f"{v / 1e6:g}M"
    if v >= 1e3:
        return f"{v / 1e3:g}k"
    return f"{v:g}"


def fig_pareto(rows: list[dict], path: str) -> str:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    pts = _llm_points(rows)
    if not pts:
        return placeholder(path, "pending: no arm with held-out results")
    zero = [p for p in pts if p["group"] in ("ref", "rule", "fdd", "search", "gp")]
    llm = [p for p in pts if p["group"] in ("online", "playbook", "hybrid")]
    N, NL = A.NH(), A.N_LARGE
    fig = plt.figure(figsize=(W, 3.2))
    bot, top = 0.15, 0.76
    az = fig.add_axes([0.085, bot, 0.07, top - bot])
    al = fig.add_axes([0.165, bot, 0.335, top - bot], sharey=az)
    ab = fig.add_axes([0.64, bot, 0.35, top - bot])
    # -- zero-token strip (CPU-only supervisors), slight horizontal offset per family
    gx = {"ref": 0.0, "rule": 0.25, "fdd": 0.5, "search": 0.75, "gp": 1.0}
    for p in zero:
        x = gx[p["group"]]
        az.errorbar(x, p["sr"], yerr=[[p["sr"] - p["lo"]], [p["hi"] - p["sr"]]], fmt="none", ecolor=A.COLOR[p["group"]],
                    elinewidth=0.6, capsize=0, alpha=0.7)
        az.plot(x, p["sr"], marker=A.MARKER[p["group"]], color=A.COLOR[p["group"]], ls="", ms=4.5,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    az.set_xlim(-0.3, 1.3)
    az.set_xticks([0.5], ["0"])
    az.set_ylim(0, 1.02)
    az.set_ylabel(f"Success on held-out suites (n = {N})")
    az.grid(axis="y")
    az.spines["right"].set_visible(False)
    # -- token-using arms on a log axis
    for p in llm:
        c, mk = A.COLOR[p["group"]], A.MARKER[p["group"]]
        x = max(p["amort"], 1.0)
        al.errorbar(x, p["sr"], yerr=[[p["sr"] - p["lo"]], [p["hi"] - p["sr"]]], fmt="none", ecolor=c, elinewidth=0.6,
                    alpha=0.7, capsize=0)
        if p["off_tok"] and np.isfinite(p["off_tok"]) and p["off_tok"] > 0:
            xl = max(p["amort_large"], 1.0)
            al.plot([x, xl], [p["sr"], p["sr"]], color=c, lw=0.6, alpha=0.8)
            al.plot(xl, p["sr"], marker=mk, color="white", markeredgecolor=c, markeredgewidth=0.8, ls="", ms=4.2, zorder=3)
        al.plot(x, p["sr"], marker=mk, color=c, markeredgecolor="white", markeredgewidth=0.5, ls="",
                ms=5 if mk != "*" else 7, zorder=4)
    refl = {"strong": ("strong reference (level)", A.COLOR["ref"], (0, (1, 1.5))),
            "B2b": ("expert table B2b, privileged (level)", A.COLOR["rule"], (0, (4, 2)))}
    ref_handles = []
    for p in zero:
        if p["arm"] in refl:
            lab, c, ls = refl[p["arm"]]
            al.axhline(p["sr"], color=c, lw=0.7, ls=ls, zorder=0)
            ref_handles.append(Line2D([], [], color=c, lw=0.7, ls=ls, label=lab))
    xs = [max(p["amort"], 1.0) for p in llm] + [max(p["amort_large"], 1.0) for p in llm if p["off_tok"]]
    lo_x = 10 ** np.floor(np.log10(min(xs) / 1.3)) if xs else 100.0
    hi_x = 10 ** np.ceil(np.log10(max(xs) * 1.3)) if xs else 1e5
    al.set_xscale("log")
    al.set_xlim(lo_x, hi_x)
    al.xaxis.set_major_locator(LogLocator(base=10))
    al.xaxis.set_major_formatter(FuncFormatter(_tok_fmt))
    al.xaxis.set_minor_formatter(NullFormatter())
    al.tick_params(axis="y", labelleft=False, length=0)
    al.spines["left"].set_visible(False)
    al.grid(axis="both", which="major")
    al.set_xlabel("LLM tokens per episode, offline cost amortised")
    d = 0.015  # break marks between the zero strip and the log axis
    az.plot([1 - 4 * d, 1 + 4 * d], [-d * 2, d * 2], transform=az.transAxes, color=INK2, lw=0.6, clip_on=False)
    al.plot([-d * 0.6, d * 0.6], [-d * 2, d * 2], transform=al.transAxes, color=INK2, lw=0.6, clip_on=False)
    best = _best_per_family(llm)
    slot_labels(al, [(max(b["amort"], 1.0), b["sr"], b["name"] + ("*" if b["partial"] else "")) for b in best],
                x0=0.03, y0=0.30, dy=0.075)
    az.text(-1.2, 1.04, "(a)", transform=az.transAxes, fontsize=8, va="bottom")
    # -- (b) allocation view: for a deployment of N episodes, B = offline + N x online/ep., f = offline / B
    for p in llm:
        c, mk = A.COLOR[p["group"]], A.MARKER[p["group"]]
        f0, b0, f1, b1 = p["f"], p["budget"], p["f_large"], p["budget_large"]
        if not (np.isfinite(f0) and b0 > 0):
            continue
        if abs(b1 - b0) > 1 or abs(f1 - f0) > 1e-6:
            ab.plot([f0, f1], [b0, b1], color=c, lw=0.6, alpha=0.8)
            ab.plot(f1, b1, marker=mk, color="white", markeredgecolor=c, markeredgewidth=0.8, ls="", ms=4.2, zorder=3)
        ab.plot(f0, b0, marker=mk, color=c, markeredgecolor="white", markeredgewidth=0.5, ls="", ms=5 if mk != "*" else 7,
                zorder=4)
    bs = [v for p in llm for v in (p["budget"], p["budget_large"]) if np.isfinite(v) and v > 0]
    ab.set_yscale("log")
    if bs:
        ab.set_ylim(10 ** np.floor(np.log10(min(bs) / 1.3)), 10 ** np.ceil(np.log10(max(bs) * 1.3)))
    slot_labels(ab, [(b["f"], b["budget"], f"{b['name']}: {100 * b['sr']:.0f}%" + ("*" if b["partial"] else ""))
                     for b in best if np.isfinite(b["f"]) and b["budget"] > 0], x0=0.30, y0=0.30, dy=0.075)
    ab.yaxis.set_major_locator(LogLocator(base=10))
    ab.yaxis.set_major_formatter(FuncFormatter(_tok_fmt))
    ab.yaxis.set_minor_formatter(NullFormatter())
    ab.set_xlim(-0.06, 1.06)
    ab.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ab.text(0.0, 1.0, "online only", transform=ab.get_xaxis_transform(), fontsize=6, color=INK2, ha="left", va="bottom")
    ab.text(1.0, 1.0, "offline only", transform=ab.get_xaxis_transform(), fontsize=6, color=INK2, ha="right", va="bottom")
    ab.set_xlabel("Allocation f = offline / total LLM tokens")
    ab.set_ylabel("Total LLM tokens B over the deployment")
    ab.grid(axis="both", which="major")
    ab.text(-0.27, 1.04, "(b)", transform=ab.transAxes, fontsize=8, va="bottom")
    groups = [g for g in A.GROUP_ORDER if any(p["group"] == g for p in pts)]
    handles = [Line2D([], [], marker=A.MARKER[g], color=A.COLOR[g], ls="", ms=4.5, label=A.GROUP_LABEL[g]) for g in groups]
    handles += [Line2D([], [], marker="o", color=INK2, ls="", ms=4, label=f"deployment N = {N} episodes"),
                Line2D([], [], marker="o", color="white", markeredgecolor=INK2, ls="", ms=4, label=f"N = {NL:,} episodes")]
    handles += ref_handles
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.01), ncol=4, fontsize=6.3, handletextpad=0.2,
               columnspacing=0.9, handlelength=1.8)
    if any(p["partial"] for p in pts):
        fig.text(0.99, 0.0, "* run in progress (rate on the episodes available)", ha="right", va="bottom", fontsize=6,
                 color=INK2)
    fig.savefig(path, bbox_inches=None)
    plt.close(fig)
    return path


# ======================================================================================= (c) synthesis curves
def _curve_xy(run: dict):
    pts = [(x, y) for x, y, ev in run["curve"] if ev != "ckpt"]
    if not pts:
        pts = [(x, y) for x, y, _ in run["curve"]]
    return pts


def fig_synthesis(runs: list[dict], path: str) -> str:
    llm_main = [r for r in runs if not r["gp"] and not r["archived"] and r["name"].startswith("main_") and r["curve"]]
    # panel (b) is titled "seed 1": keep the seed-1 ablation runs and the seed-1 archived (pre-tuning) run only
    abl = [r for r in runs if not r["gp"] and (r["archived"] or not r["name"].startswith("main_")) and r["curve"]
           and r["seed"] == 1 and not r["name"].startswith("calibonly_")]  # calibration-only spends no tokens after v0
    gp = [r for r in runs if r["gp"] and r["curve"]]
    if not (llm_main or gp):
        return placeholder(path, "pending: no synthesis logs")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.15), sharey=True)
    ls_seed = {1: "-", 2: (0, (4, 1.6)), 3: (0, (1, 1.2))}
    blue = A.COLOR["playbook"]
    ends: dict = {}

    def draw(ax, run, color, ls, lw=1.1, label=None, alpha=1.0):
        pts = _curve_xy(run)
        if not pts:
            return
        x, y = zip(*pts)
        x = np.array(x) / 1000
        ax.step(x, y, where="post", color=color, ls=ls, lw=lw, alpha=alpha, label=label)
        for t in run.get("tune") or []:
            ax.plot([t["x"] / 1000] * 2, [t["from"], t["to"]], color=color, lw=0.6, alpha=alpha)
            ax.plot(t["x"] / 1000, t["to"], marker="^", color=color, markeredgecolor=INK, markeredgewidth=0.4, ms=4,
                    ls="", alpha=alpha, zorder=4)
        for f in run.get("fals_events") or []:
            ax.plot(f["x"] / 1000, 0.015, marker="|", color=color, ms=4, alpha=0.7 * alpha, clip_on=False)
        ends.setdefault(ax, []).append({"x": x[-1], "y": y[-1], "text": run["tag_label"], "dx": 2})

    for r in llm_main:
        coder7 = "7b" in r["name"]
        r["tag_label"] = f"s{r['seed']}" + (" 7B" if coder7 else "")
        draw(axs[0], r, blue if not coder7 else "#86b6ef", ls_seed.get(r["seed"], "-") if not coder7 else (0, (3, 1, 1, 1)),
             label=None)
    axs[0].set_title("(a) Main runs and Coder-7B", loc="left", fontsize=8)
    axs[0].set_ylabel("Incumbent training success")
    axs[0].set_xlabel("Cumulative synthesis tokens (k)")
    s1 = next((r for r in llm_main if r["seed"] == 1 and "30b" in r["name"]), None)
    abl_styles = {"norem": ((0, (4, 1.6)), "no remedies"), "notune": ((0, (1, 1.2)), "no tuning")}
    if s1 is not None:
        s1 = {**s1, "tag_label": "main"}
        draw(axs[1], s1, blue, "-")
    if abl:
        for r in abl:
            tag = r["name"].split("_")[0]
            if r["archived"]:
                r["tag_label"] = "archived"
                draw(axs[1], r, MUTED, "-", lw=0.7, alpha=0.9)
            else:
                ls, lab = abl_styles.get(tag, ("-.", tag))
                r["tag_label"] = lab
                draw(axs[1], r, blue, ls)
    else:
        axs[1].text(0.5, 0.5, "ablation runs pending", transform=axs[1].transAxes, ha="center", color=MUTED, style="italic")
    axs[1].set_title("(b) Seed 1 ablations", loc="left", fontsize=8)
    axs[1].set_xlabel("Cumulative synthesis tokens (k)")
    # (c) CPU efficiency: GP generations vs LLM checkpoints, in simulated episodes
    gcol = A.COLOR["gp"]
    for r in gp:
        pts = [(x, y) for x, y, ev in r["curve"] if ev == "gen"]
        if pts:
            x, y = zip(*pts)
            axs[2].step(np.array(x) / 1000, y, where="post", color=gcol, ls=ls_seed.get(r["seed"], "-"), lw=1.1)
            ends.setdefault(axs[2], []).append({"x": x[-1] / 1000, "y": y[-1], "text": f"GP s{r['seed']}", "dx": 2})
    for r in llm_main:
        if "30b" not in r["name"]:
            continue
        ck = sorted(((v["episodes"], v["sr"]) for v in r["ckpts"].values() if v.get("episodes")), key=lambda t: t[0])
        if ck:
            x, y = zip(*ck)
            axs[2].plot(np.array(x) / 1000, y, color=blue, ls=ls_seed.get(r["seed"], "-"), lw=0.9, marker="o", ms=3,
                        markeredgecolor="white", markeredgewidth=0.4)
            ends.setdefault(axs[2], []).append({"x": x[-1] / 1000, "y": y[-1], "text": f"LLM s{r['seed']}", "dx": 2})
    axs[2].set_title("(c) Simulation cost vs GP", loc="left", fontsize=8)
    axs[2].set_xlabel("Simulated playbook episodes (k)")
    for ax in axs:
        ax.set_ylim(0, 1.02)
        ax.grid(axis="y")
        x0, x1 = ax.get_xlim()
        ax.set_xlim(x0, x1 + 0.22 * (x1 - x0))  # room for the end-of-line labels
    fig.tight_layout(w_pad=0.6)
    for ax, items in ends.items():
        place_labels(ax, items, fontsize=6, min_gap_pt=6.5, col_px=40)
    handles = [Line2D([], [], color=blue, lw=1.1, label="LLM synthesis (Coder-30B)"),
               Line2D([], [], color=gcol, lw=1.1, label="genetic programming"),
               Line2D([], [], color=MUTED, lw=0.7, label="archived run without tuning"),
               Line2D([], [], marker="^", color=INK2, ls="", ms=4, label="CPU tuning gain"),
               Line2D([], [], marker="|", color=INK2, ls="", ms=5, label="falsification round")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=6.5, handletextpad=0.3,
               columnspacing=1.0)
    return _save(fig, path)


# ======================================================================================= (d) case study
ROLE_STYLE = {"Weak": ("ref", (0, (4, 1.6))), "Onl": ("online", "-"), "Pb": ("playbook", "-"), "Hyb": ("hybrid", "-")}
ROLE_LABEL = {"Weak": "weak baseline", "Onl": "online LLM", "Pb": "LLM playbook", "Hyb": "hybrid"}


def fig_case(case: dict | None, path: str) -> str:
    import json
    if not case:
        return placeholder(path, "pending: case-study arms not available")
    sd = case["sd"]
    sens = [f["measurement"] for f in sd.get("sensor_faults", [])]
    if "reactor_pressure" in sens:
        var2 = (8, "Reactor pressure reading (kPa)", None)
    elif "reactor_temperature" in sens:
        var2 = (3, "Reactor temperature (°C)", 9)
    else:
        var2 = (7, "Product G (mol%)", None)
    traces = {}
    for role, info in case["arms"].items():
        if info:
            rec = json.load(open(info["path"]))
            tr = np.array(rec.get("trace") or [])
            tfail = rec.get("trip_time") or rec.get("shutdown_time")
            if len(tr) and tfail is not None and not rec["stabilized"]:
                tr = tr[tr[:, 0] <= float(tfail) + 1e-9]
            traces[role] = (tr, [d for d in rec.get("decisions", []) if d.get("accepted")], rec)
    ends = [tr[-1, 0] for tr, _, rec in traces.values() if len(tr) and not rec["stabilized"]]
    xmax = 48.0 if not ends else float(min(48.0, max(12.0, np.ceil(max(ends) * 1.6 / 2) * 2)))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(W, 3.4), sharex=True, gridspec_kw={"height_ratios": [1.25, 1.0], "hspace": 0.12})
    onset = min([f["start_h"] for k in ("idv_events", "sensor_faults", "actuator_faults") for f in sd.get(k, [])] or [None]) \
        if any(sd.get(k) for k in ("idv_events", "sensor_faults", "actuator_faults")) else None
    for ax in (a1, a2):
        if onset is not None:
            ax.axvline(onset, color=MUTED, lw=0.6, ls=(0, (1, 1.5)))
        ax.grid(axis="y")
    a1.axhline(3000, color="#d03b3b", lw=0.7)
    a1.text(xmax * 0.995, 3000, "trip limit", color="#d03b3b", fontsize=6, va="bottom", ha="right")
    a1.axhline(2760, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    a1.text(xmax * 0.995, 2760, "high-pressure alarm", color=INK2, fontsize=6, va="bottom", ha="right")
    if onset is not None:
        a1.text(onset, 1.0, " fault onset", transform=a1.get_xaxis_transform(), fontsize=6, color=INK2, va="top")
    handles = []
    joined_dt = None  # trace sampling interval, set when a run is joined to its trip point
    for role in ("Weak", "Onl", "Pb", "Hyb"):
        if role not in traces:
            continue
        tr, acts, rec = traces[role]
        if not len(tr):
            continue
        g, ls = ROLE_STYLE[role]
        c, mk = A.COLOR[g], A.MARKER[g]
        lw = 1.0 if role != "Weak" else 0.9
        a1.plot(tr[:, 0], tr[:, 1], color=c, ls=ls, lw=lw)
        a2.plot(tr[:, 0], tr[:, var2[0]], color=c, ls=ls, lw=lw)
        if var2[2] is not None and role == "Weak":
            a2.plot(tr[:, 0], tr[:, var2[2]], color=INK, ls=(0, (1, 1.2)), lw=0.8)
        for d in acts:
            t = d["t"] + 0.05
            i = min(np.searchsorted(tr[:, 0], t), len(tr) - 1)
            a1.plot(tr[i, 0], tr[i, 1], marker=mk, color=c, markeredgecolor="white", markeredgewidth=0.4,
                    ms=5 if mk != "*" else 7, ls="", zorder=4)
        if not rec["stabilized"]:
            tt = rec.get("trip_time") or rec.get("shutdown_time") or tr[-1, 0]
            if "pressure" in (rec.get("trip_msg") or "").lower() and "high" in (rec.get("trip_msg") or "").lower():
                # high-pressure trip: the kernel trips when true pressure exceeds 3000 kPa; join the last sample to it
                a1.plot([tr[-1, 0], tt], [tr[-1, 1], 3000.0], color=c, ls=(0, (1, 1)), lw=0.8)
                joined_dt = float(tr[1, 0] - tr[0, 0]) if len(tr) > 1 else joined_dt
                a1.plot(tt, 3000.0, marker="x", color=c, ms=5, mew=1.0, ls="", zorder=5)
            else:
                a1.plot(tr[-1, 0], tr[-1, 1], marker="x", color=c, ms=5, mew=1.0, ls="", zorder=5)
        info = case["arms"][role]
        handles.append(Line2D([], [], color=c, ls=ls, lw=lw, marker=mk, ms=4 if mk != "*" else 6,
                              markeredgecolor="white", label=f"{ROLE_LABEL[role]}: {info['outcome'].split(' (')[0]}"))
    if var2[2] is not None:
        handles.append(Line2D([], [], color=INK, ls=(0, (1, 1.2)), lw=0.8, label="transmitter reading (weak run)"))
    handles.append(Line2D([], [], marker="x", color=INK2, ls="", ms=5, label="trip"))
    if joined_dt:
        handles.append(Line2D([], [], color=INK2, ls=(0, (1, 1)), lw=0.8,
                              label=f"last sample to trip (trace every {joined_dt:.2g} h)"))
    a1.set_ylabel("True reactor\npressure (kPa)")
    a2.set_ylabel(var2[1].replace(" (", "\n("))
    a2.set_xlabel("Time (h); markers = accepted supervisory actions")
    a1.set_xlim(0, xmax)
    lo = min(2600, np.nanmin([traces[r][0][:, 1].min() for r in traces if len(traces[r][0])]) - 20)
    a1.set_ylim(lo, 3060)
    a2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=6.5, handlelength=2.2,
              columnspacing=1.2)
    a1.set_title(f"{case['scenario']}: {case['desc']}", loc="left", fontsize=7, color=INK2)
    return _save(fig, path)


# ======================================================================================= (e) escalations
def fig_escalation(df: pd.DataFrame, ext: pd.DataFrame, rows: list[dict], path: str) -> str:
    hyb = [r for r in rows if r["group"] == "hybrid" and r["has_data"] and "esc_ep" in r]
    if not hyb or ext.empty:
        return placeholder(path, "pending: hybrid (playbook + escalation) results not available yet")
    e = ext.merge(df[["arm", "suite", "scenario", "seed", "family", "success"]], on=["arm", "suite", "scenario", "seed"])
    e = e[e["suite"].isin(A.HELD) & e["arm"].isin([r["arm"] for r in hyb])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 0.22 * len(hyb) + 1.75), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.55})
    y = np.arange(len(hyb))
    left = np.zeros(len(hyb))
    fr = e.groupby("arm")[[f"escf_{c}" for c, _, _ in A.TRIGGERS]].mean()
    for code, _, lab in A.TRIGGERS:
        v = np.array([fr.loc[r["arm"], f"escf_{code}"] if r["arm"] in fr.index else 0.0 for r in hyb])
        a1.barh(y, v, left=left, height=0.62, color=TRIG_COL[code], hatch=TRIG_HATCH[code], edgecolor="white", linewidth=0.8,
                label=f"({code}) {lab}")
        left += v
    for i, r in enumerate(hyb):
        a1.text(left[i], i, f" {r['esc_ep']:.2f}", va="center", fontsize=6, color=INK2)
    a1.set_yticks(y, [r["name"] for r in hyb])
    a1.invert_yaxis()
    a1.set_xlabel("Escalations per held-out episode")
    a1.set_title("(a) by hybrid arm and trigger", loc="left", fontsize=8)
    a1.grid(axis="x")
    fams = [f for f in A.FAMILIES_ID + A.FAMILIES_OOD if f in set(e["family"])]
    bottom = np.zeros(len(fams))
    for code, _, lab in A.TRIGGERS:
        v = np.array([e[e["family"] == f]["escf_" + code].mean() for f in fams])
        a2.bar(range(len(fams)), v, bottom=bottom, width=0.66, color=TRIG_COL[code], hatch=TRIG_HATCH[code],
               edgecolor="white", linewidth=0.8)
        bottom += v
    ep = e.assign(esc_any=e["n_esc"] > 0)
    for j, f in enumerate(fams):  # outcome of the escalated episodes: success rate of episodes with >= 1 escalation
        x = ep[(ep["family"] == f) & ep["esc_any"]]
        if len(x):
            a2.text(j, bottom[j], f"{100 * x['success'].mean():.0f}%", ha="center", va="bottom", fontsize=5.8, color=INK2)
    a2.set_ylim(0, max(bottom.max() * 1.15, 0.1))
    a2.set_xticks(range(len(fams)), [f + ("*" if f in A.FAMILIES_OOD else "") for f in fams])
    a2.set_ylabel("Escalations per episode\n(all hybrid arms pooled)")
    a2.set_xlabel("Fault family (* = novel)")
    a2.set_title("(b) by fault family", loc="left", fontsize=8)
    a2.grid(axis="y")
    h, l_ = a1.get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.30)
    fig.legend(h, [x + " trigger" for x in l_], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=6.5,
               handletextpad=0.3,
               title_fontsize=6.3)
    return _save(fig, path)


# ======================================================================================= driver
def make_all(df, ext, summ, rows_main, runs, case, out_dir, rows_proto=None) -> list[str]:
    style()
    out = []
    jobs = [("fig_family.pdf", lambda p: fig_family(df, rows_main, p)),
            ("fig_pareto.pdf", lambda p: fig_pareto(rows_proto or rows_main, p)),
            ("fig_synthesis.pdf", lambda p: fig_synthesis(runs, p)),
            ("fig_case.pdf", lambda p: fig_case(case, p)),
            ("fig_escalation.pdf", lambda p: fig_escalation(df, ext, rows_main, p))]
    for name, fn in jobs:
        p = os.path.join(out_dir, name)
        try:
            out.append(fn(p))
        except Exception as e:  # noqa: BLE001 - a failing figure must not block the tables and macros
            import traceback
            traceback.print_exc()
            print(f"{name} failed: {e}")
            out.append(placeholder(p, f"figure failed to render: {type(e).__name__}"))
    return out
