"""Figures added in the revision (vector PDF into paper/figures/; style of figures_extra.py).

  fig_stages.pdf   held-out success of each synthesis seed at four stages: first draft, calibration only (one round,
                   no repair), full procedure after one round, final playbook (anticipated and novel families)
  fig_confirm.pdf  success of the frozen supervisors on the confirmation suites (three noise seeds per scenario) with
                   scenario-level cluster-bootstrap 95 % intervals, next to their held-out success
  PYTHONPATH=code python code/experiments/figures_revision.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402
import figures_extra as FX  # noqa: E402
import final_stats as FS  # noqa: E402
from cod.analysis.load import load_episodes  # noqa: E402

plt = FX.plt
OUT = os.environ.get("COD_FIG", os.path.join(FS.PROJ, "paper", "figures"))  # output folder (as in analyze.py)
BLUE = A.COLOR["playbook"]
SEED_STYLE = {1: dict(color="#184f95", marker="o", ls="-"), 2: dict(color="#6da7ec", marker="s", ls="--"),
              3: dict(color="#3987e5", marker="^", ls="-.")}
STAGES = [("First\ndraft", {1: "PbVzero", 2: "PbS2Vzero", 3: "PbS3Vzero"}),
          ("Calibration\nonly", {1: "CalS1ROne", 2: "CalS2ROne", 3: "CalS3ROne"}),
          ("Full,\none round", {1: "PbS1TwentyFiveK", 2: "PbS2TwentyFiveK", 3: "PbS3TwentyFiveK"}),
          ("Full,\nfinal", {1: "PbS1", 2: "PbS2", 3: "PbS3"})]


def _rates(keys: list[str], suites: list[str]):
    inv = {FS.ARMS[k]: k for k in keys}
    df = load_episodes(list(inv), suites)
    df = df[df["arm"].isin(inv)].copy()
    df["key"] = df["arm"].map(inv)
    return df


def fig_stages(path: str) -> str:
    keys = sorted({k for _, m in STAGES for k in m.values()})
    df = _rates(keys, ["test_id", "test_ood"])
    FX.style()
    fig, axs = plt.subplots(1, 2, figsize=(FX.W, 2.05), sharey=False)
    xs = np.arange(len(STAGES))
    for ax, (suite, title, n) in zip(axs, (("test_id", "(a) Anticipated families", 98), ("test_ood", "(b) Novel families", 64))):
        pooled = np.zeros(len(STAGES))
        for s in (1, 2, 3):
            ys = []
            for _, m in STAGES:
                x = df[(df["key"] == m[s]) & (df["suite"] == suite)]["success"]
                ys.append(100 * x.mean())
            pooled += np.array(ys) / 3
            st = SEED_STYLE[s]
            ax.plot(xs, ys, color=st["color"], marker=st["marker"], ls=st["ls"], lw=1.1, ms=4, label=f"seed {s}", zorder=3)
            ax.annotate(f"{ys[-1]:.0f}", (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=6.5, color=st["color"])
        ax.plot(xs, pooled, color=FX.INK, lw=1.6, ls=":", marker="D", ms=3, label="mean of seeds", zorder=4)
        ax.axvspan(0.6, 1.4, color=FX.GRID, alpha=0.55, lw=0, zorder=0)
        ax.set_xticks(xs, [lab for lab, _ in STAGES])
        ax.set_xlim(-0.35, len(STAGES) - 0.55)
        ax.set_ylim(20 if suite == "test_id" else 30, 102)
        ax.set_ylabel(f"Held-out success (%, n = {n})")
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(axis="y")
        ax.tick_params(axis="x", length=0)
    axs[0].text(1.0, 23.5, "no repair:\nsimulation only", ha="center", va="bottom", fontsize=6.3, color=FX.INK2)
    axs[1].legend(loc="upper left", fontsize=6.3, handlelength=2.4, ncol=2, columnspacing=1.0)
    fig.tight_layout(w_pad=1.6)
    return FX._save(fig, path)


CONFIRM_ROWS = [("Strong reference", "Strong", "ref"), ("Expert table", "Expert", "rule"), ("FDD with lookup", "FDD", "fdd"),
                ("GP seed 1, final", "GPS1", "gp"), ("GP seed 1, 5k episodes", "GPS1Early", "gp"), ("GP seed 2, final", "GPS2", "gp"),
                ("GP seed 3, final", "GPS3", "gp"), ("LLM seed 1, final", "PbS1", "playbook"),
                ("LLM seed 2, final", "PbS2", "playbook"), ("LLM seed 3, final", "PbS3", "playbook"),
                ("LLM seed 1, 210k tokens", "PbS1TwoH", "playbook"), ("LLM seed 1, 210k, no remedies", "PbNorem", "playbook"),
                ("LLM seed 1, 210k, no calibration", "PbNotune", "playbook")]


def _cluster_ci(x, rng, reps=2000):
    per = x.groupby("scenario")["success"].mean().values
    boot = [rng.choice(per, len(per)).mean() for _ in range(reps)]
    return 100 * per.mean(), 100 * np.percentile(boot, 2.5), 100 * np.percentile(boot, 97.5)


def fig_confirm(path: str) -> str:
    keys = [k for _, k, _ in CONFIRM_ROWS]
    conf = _rates(keys, ["confirm_id", "confirm_ood"])
    held = _rates(keys, ["test_id", "test_ood"])
    rng = np.random.default_rng(2026)
    FX.style()
    fig, axs = plt.subplots(1, 2, figsize=(FX.W, 2.5), sharey=True)
    ys = np.arange(len(CONFIRM_ROWS))[::-1]
    for ax, (cs, hs, title) in zip(axs, (("confirm_id", "test_id", "(a) Anticipated families"),
                                          ("confirm_ood", "test_ood", "(b) Novel families"))):
        for y, (lab, k, grp) in zip(ys, CONFIRM_ROWS):
            col, mk = A.COLOR[grp], A.MARKER[grp]
            x = conf[(conf["key"] == k) & (conf["suite"] == cs)]
            if len(x):
                m, lo, hi = _cluster_ci(x, rng)
                ax.plot([lo, hi], [y, y], color=col, lw=1.6, alpha=0.75, solid_capstyle="round", zorder=2)
                ax.plot([m], [y], marker=mk, color=col, ms=4.2, ls="", zorder=3)
            h = held[(held["key"] == k) & (held["suite"] == hs)]["success"]
            if len(h):
                ax.plot([100 * h.mean()], [y], marker=mk, mfc="white", mec=col, mew=0.9, ms=4.2, ls="", zorder=4)
        strong = conf[(conf["key"] == "Strong") & (conf["suite"] == cs)]["success"].mean() * 100
        ax.axvline(strong, color=A.COLOR["ref"], lw=0.7, ls="--", zorder=1)
        ax.set_title(title, loc="left", fontsize=8)
        ax.set_xlabel("Success (%)")
        ax.grid(axis="x")
        ax.set_xlim(45 if cs == "confirm_id" else 45, 101)
    axs[0].set_yticks(ys, [lab for lab, _, _ in CONFIRM_ROWS])
    axs[0].tick_params(axis="y", length=0)
    for y in (ys[0] - 0.5, ys[2] - 0.5, ys[6] - 0.5, ys[9] - 0.5):
        for ax in axs:
            ax.axhline(y, color=FX.GRID, lw=0.6, zorder=0)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", color=FX.INK2, ls="-", lw=1.4, ms=4, label="confirmation suites, 95 % cluster-bootstrap interval"),
               Line2D([], [], marker="o", mfc="white", mec=FX.INK2, ls="", ms=4, label="held-out suites")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=6.5, bbox_to_anchor=(0.6, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.2)
    return FX._save(fig, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    print(fig_stages(os.path.join(OUT, "fig_stages.pdf")))
    print(fig_confirm(os.path.join(OUT, "fig_confirm.pdf")))


if __name__ == "__main__":
    main()
