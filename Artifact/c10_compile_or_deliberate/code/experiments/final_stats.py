"""Final paired statistics and escalation/compile-back figures quoted in the results text.

Writes paper/generated/numbers_final.tex (macros prefixed \\fs...) and paper/generated/seeds.tex
(per-seed robustness table). All numbers come from the per-episode records. TeX command names cannot
contain digits, so digits in macro names are spelled out on output (fsTrainS1 -> \\fsTrainSOne).
  python experiments/final_stats.py
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cod.analysis.load import load_episodes  # noqa: E402
from cod.analysis.stats import mcnemar_exact, newcombe_paired_diff_ci  # noqa: E402

PROJ = os.path.dirname(ROOT)
GEN = os.environ.get("COD_GEN", os.path.join(PROJ, "paper", "generated"))  # output folder (as in analyze.py)
SYN = os.path.join(PROJ, "results", "synth")
STREAM = os.path.join(PROJ, "results", "stream")


def pb(run: str, ck: str) -> str:
    return f"playbook:{os.path.join(SYN, run, f'playbook_{ck}.py')}"


def hy(run: str, ck: str, tok: str, model: str = "qwen3-coder:30b") -> str:
    return f"hybrid:{os.path.join(SYN, run, f'playbook_{ck}.py')}:{model}@{tok}"


M = "main_qwen3-coder-30b_s"
ARMS = {
    "PbS1": pb(M + "1", "final"), "PbS2": pb(M + "2", "final"), "PbS3": pb(M + "3", "final"),
    "PbS1TwoH": pb(M + "1", "B200000"), "PbNorem": pb("norem_qwen3-coder-30b_s1", "final"),
    "PbNotune": pb("notune_qwen3-coder-30b_s1", "final"), "PbVzero": pb(M + "1", "v0"),
    "HyS1": hy(M + "1", "final", "16000"), "HyS2": hy(M + "2", "final", "16000"), "HyS3": hy(M + "3", "final", "16000"),
    "HyS1Inf": hy(M + "1", "final", "inf"),
    "Onl": "onlineg:qwen3-coder:30b", "OnlFourteen": "onlineg:qwen2.5:14b-instruct-q4_K_M",
    "Expert": "B2b", "FDD": "fdd:default", "Strong": "strong", "Weak": "weak", "Oracle": "search:perfect@gate=0",
    "GPS1": pb("gp_s1", "E20000"), "GPS1Early": pb("gp_s1", "E5000"), "GPS2": pb("gp_s2", "final"),
    "GPS2Early": pb("gp_s2", "E5000"), "GPS3": pb("gp_s3", "final"), "GPS3Early": pb("gp_s3", "E5000"),
    "CalS1": pb("calibonly_qwen3-coder-30b_s1", "final"), "CalS2": pb("calibonly_qwen3-coder-30b_s2", "final"),
    "CalS3": pb("calibonly_qwen3-coder-30b_s3", "final"), "CalS1ROne": pb("calibonly_qwen3-coder-30b_s1", "R1"),
    "CalS2ROne": pb("calibonly_qwen3-coder-30b_s2", "R1"), "CalS3ROne": pb("calibonly_qwen3-coder-30b_s3", "R1"),
    "PbS1TwentyFiveK": pb(M + "1", "B25000"), "PbS2TwentyFiveK": pb(M + "2", "B25000"), "PbS3TwentyFiveK": pb(M + "3", "B25000"),
    "PbS2Vzero": pb(M + "2", "v0"), "PbS3Vzero": pb(M + "3", "v0"),
}
HELD = ("test_id", "test_ood")
DIGITS = dict(zip("0123456789", ("Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine")))


def texname(key: str) -> str:
    return re.sub(r"\d", lambda m: DIGITS[m.group()], key)


def fmt_p(p: float) -> str:
    """p-value with its relation sign, so that "$p$ \\fs...P" reads "p = 0.017" or "p < 0.0001"."""
    if p < 1e-4:
        return "$<$\\,0.0001"
    return "$=$\\," + (f"{p:.4f}" if p < 0.01 else f"{p:.3f}" if p < 0.1 else f"{p:.2f}")


def sgn(x: float) -> str:
    """Signed percentage-point value with a typographic minus: $+$12.3, $-$4.5, 0.0."""
    v = round(x, 1)
    return "0.0" if v == 0 else ("$+$" if v > 0 else "$-$") + f"{abs(v):.1f}"


def main():
    df = load_episodes(list(ARMS.values()), list(HELD) + ["alhazmi", "frontier"], workers=4) \
        if "workers" in load_episodes.__code__.co_varnames else load_episodes(list(ARMS.values()), list(HELD) + ["alhazmi", "frontier"])
    inv = {v: k for k, v in ARMS.items()}
    df = df[df["arm"].isin(inv)].copy()
    df["key"] = df["arm"].map(inv)
    out = {}
    raw_p = {}  # unadjusted exact McNemar p of every paired contrast, by tag

    def series(key, suites):
        x = df[(df["key"] == key) & (df["suite"].isin(suites))]
        return x.set_index(["suite", "scenario", "seed"])["success"].astype(int)

    def paired(a, b, suites, tag):
        A, B = series(a, suites), series(b, suites)
        idx = A.index.intersection(B.index)
        if len(idx) == 0:
            return
        x, y = A.loc[idx].values, B.loc[idx].values
        bb, cc, p = mcnemar_exact(x, y)
        raw_p[tag] = p
        d, lo, hi = newcombe_paired_diff_ci(x, y)
        out[f"fs{tag}N"] = str(len(idx))
        out[f"fs{tag}Diff"] = sgn(100 * d)
        out[f"fs{tag}CI"] = f"{sgn(100 * lo)} to {sgn(100 * hi)}"
        out[f"fs{tag}P"] = fmt_p(p)
        out[f"fs{tag}Disc"] = f"{bb}/{cc}"

    for sname, suites in (("Held", HELD), ("Id", ("test_id",)), ("Ood", ("test_ood",))):
        paired("PbS1", "Onl", suites, f"PbOneVsOnl{sname}")
        paired("HyS1", "PbS1", suites, f"HyOneVsPb{sname}")
        paired("HyS2", "PbS2", suites, f"HyTwoVsPb{sname}")
        paired("HyS3", "PbS3", suites, f"HyThreeVsPb{sname}")
        paired("PbS1", "Expert", suites, f"PbOneVsExpert{sname}")
        paired("PbS1", "GPS1", suites, f"PbOneVsGPOne{sname}")
        paired("PbS1", "Strong", suites, f"PbOneVsStrong{sname}")
        paired("PbS1TwoH", "PbNotune", suites, f"TuneAbl{sname}")
        paired("PbS1TwoH", "PbNorem", suites, f"RemAbl{sname}")
        paired("GPS1Early", "GPS1", suites, f"GPOneEarlyVsFinal{sname}")
        paired("Onl", "OnlFourteen", suites, f"OnlVsFourteen{sname}")
        # calibration only (first draft + falsification + calibration, no LLM repair) against the full procedure
        paired("PbS1", "CalS1", suites, f"PbOneVsCalOne{sname}")
        paired("PbS2", "CalS2", suites, f"PbTwoVsCalTwo{sname}")
        paired("PbS3", "CalS3", suites, f"PbThreeVsCalThree{sname}")
        paired("PbS1TwentyFiveK", "CalS1ROne", suites, f"PbTwentyFiveKVsCalROne{sname}")
        paired("PbS2TwentyFiveK", "CalS2ROne", suites, f"PbTwoTwentyFiveKVsCalTwoROne{sname}")
        paired("PbS3TwentyFiveK", "CalS3ROne", suites, f"PbThreeTwentyFiveKVsCalThreeROne{sname}")
    # primary contrasts, Holm-adjusted together (designated after the mock review; all other contrasts exploratory)
    primary = ["PbOneVsOnlId", "PbOneVsOnlOod", "PbOneVsStrongId", "PbOneVsExpertOod", "HyOneVsPbOod", "TuneAblId"]
    ps = [raw_p[t] for t in primary]
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    for t, v in zip(primary, adj):
        out[f"fsHolm{t}"] = fmt_p(v)
    # seed-pooled means
    for grp, keys in (("Pb", ["PbS1", "PbS2", "PbS3"]), ("Hy", ["HyS1", "HyS2", "HyS3"]), ("GP", ["GPS1", "GPS2", "GPS3"]),
                      ("Cal", ["CalS1", "CalS2", "CalS3"]), ("CalROne", ["CalS1ROne", "CalS2ROne", "CalS3ROne"]),
                      ("Vzero", ["PbVzero", "PbS2Vzero", "PbS3Vzero"]),
                      ("TwentyFiveK", ["PbS1TwentyFiveK", "PbS2TwentyFiveK", "PbS3TwentyFiveK"]),
                      ("GPEarly", ["GPS1Early", "GPS2Early", "GPS3Early"])):
        for sname, suites in (("Id", ("test_id",)), ("Ood", ("test_ood",)), ("Held", HELD)):
            vals = [series(k, suites).mean() for k in keys if len(series(k, suites))]
            if len(vals) == len(keys):  # only when every seed is available (no partial means)
                out[f"fsMean{grp}{sname}"] = f"{100 * np.mean(vals):.1f}\\,\\%"
                out[f"fsRange{grp}{sname}"] = f"{100 * min(vals):.0f}--{100 * max(vals):.0f}\\,\\%"
    # escalation behaviour of hybrids. Escalations are read from the hybrid's own log (outcome["escalations"], one entry
    # per escalation with its triggers). The decision logged at an escalation is the deliberator's FINAL proposal:
    # source "escalation:llm", or "escalation:llm_revision" when the validator rejected the first one. The playbook's
    # proposal is stored only on the former; it is None whenever trigger (a) fired (no rule fired), and unknown for a
    # revised escalation without trigger (a), which is therefore left out of the override share.
    for key in ("HyS1", "HyS2", "HyS3"):
        for sname, suites in (("Id", ("test_id",)), ("Ood", ("test_ood",))):
            x = df[(df["key"] == key) & (df["suite"].isin(suites))]
            esc = ep_esc = over = known = acted = 0
            trig = {"unexplained_abnormality": 0, "playbook_action_fails_forward_check": 0, "out_of_distribution": 0}
            for path in x["path"]:
                d = json.load(open(path))
                log = d.get("escalations") or []
                esc += len(log)
                ep_esc += int(bool(log))
                final = {round(dec["t"], 6): dec for dec in d["decisions"] if dec["source"].startswith("escalation:llm")}
                for e in log:
                    why = e.get("why") or []
                    for t in why:
                        trig[t] = trig.get(t, 0) + 1
                    dec = final.get(round(e["t"], 6))
                    if dec is None:
                        continue
                    act = dec["action"]
                    acted += int(act.get("type") != "no_action")
                    if dec["source"] == "escalation:llm":
                        prop = (dec.get("meta") or {}).get("playbook_proposal")
                    elif "unexplained_abnormality" in why:
                        prop = None
                    else:
                        continue
                    known += 1
                    prop = prop or {"type": "no_action"}
                    over += int(json.dumps(prop, sort_keys=True) != json.dumps(act, sort_keys=True))
            n = len(x)
            if n:
                out[f"fsEsc{key}{sname}"] = f"{esc / n:.2f}"
                out[f"fsEscEp{key}{sname}"] = f"{100 * ep_esc / n:.0f}\\,\\%"
                out[f"fsOverride{key}{sname}"] = f"{100 * over / max(known, 1):.0f}\\,\\%"
                out[f"fsActed{key}{sname}"] = f"{100 * acted / max(esc, 1):.0f}\\,\\%"
                tot = max(sum(trig.values()), 1)
                for t, short in (("unexplained_abnormality", "TrigA"), ("playbook_action_fails_forward_check", "TrigB"),
                                 ("out_of_distribution", "TrigC")):
                    out[f"fs{short}{key}{sname}"] = f"{100 * trig.get(t, 0) / tot:.0f}\\,\\%"
    # discordant anticipated-family episodes of a hybrid against its own playbook: what the deliberator did
    for key, pk in (("HyS2", "PbS2"), ("HyS3", "PbS3")):
        H, P = series(key, ("test_id",)), series(pk, ("test_id",))
        paths = df[(df["key"] == key) & (df["suite"] == "test_id")].set_index(["suite", "scenario", "seed"])["path"]
        win = loss = win_arm = loss_q = loss_q_grade = 0
        for idx in H.index.intersection(P.index):
            if H[idx] == P[idx]:
                continue
            acc = [x["action"] for x in json.load(open(paths[idx]))["decisions"]
                   if x["source"].startswith("escalation") and x["accepted"]]
            if H[idx]:
                win += 1
                win_arm += any(a["type"] == "reconfigure" and "composition_control" in a.get("components", []) for a in acc)
            else:
                loss += 1
                if idx[1].startswith("Q"):
                    loss_q += 1
                    loss_q_grade += any(a["type"] == "adjust_setpoints" and "pct_g" in a.get("changes", {}) for a in acc)
        out[f"fsDiscWin{key}Id"], out[f"fsDiscLoss{key}Id"] = str(win), str(loss)
        out[f"fsDiscWinArm{key}Id"], out[f"fsDiscLossQ{key}Id"] = str(win_arm), str(loss_q)
        out[f"fsDiscLossQGrade{key}Id"] = str(loss_q_grade)
    # novelty gate: per-invocation trigger rates by family, episode-level discrimination (anticipated vs novel) of the
    # number of conformal-trigger firings, and escalation against need (episodes the playbook alone fails)
    from sklearn.metrics import roc_auc_score
    for key, pk in (("HyS1", "PbS1"), ("HyS2", "PbS2"), ("HyS3", "PbS3")):
        x = df[(df["key"] == key) & (df["suite"].isin(HELD))]
        pb_ok = series(pk, HELD)
        n_inv = {"id": [0, 0], "ood": [0, 0]}
        score, novel, esc_ep, need = [], [], [], []
        for (suite, scen, seed), path in x.set_index(["suite", "scenario", "seed"])["path"].items():
            d = json.load(open(path))
            log = {round(e["t"], 6): e.get("why") or [] for e in d.get("escalations") or []}
            n_c = 0
            for dec in d["decisions"]:
                why = (dec.get("meta") or {}).get("triggers") or log.get(round(dec["t"], 6), [])
                c = "out_of_distribution" in why
                n_c += c
                k = "id" if suite == "test_id" else "ood"
                n_inv[k][0] += c
                n_inv[k][1] += 1
            score.append(n_c)
            novel.append(int(suite == "test_ood"))
            esc_ep.append(int(bool(log)))
            need.append(int(not pb_ok.get((suite, scen, seed), 1)))
        out[f"fsTrigCInv{key}Id"] = f"{100 * n_inv['id'][0] / max(n_inv['id'][1], 1):.0f}\\,\\%"
        out[f"fsTrigCInv{key}Ood"] = f"{100 * n_inv['ood'][0] / max(n_inv['ood'][1], 1):.0f}\\,\\%"
        out[f"fsGateAuc{key}"] = f"{roc_auc_score(novel, score):.2f}"
        tp = sum(e and n for e, n in zip(esc_ep, need))
        out[f"fsEscPrec{key}"] = f"{100 * tp / max(sum(esc_ep), 1):.0f}\\,\\%"
        out[f"fsEscRecall{key}"] = f"{100 * tp / max(sum(need), 1):.0f}\\,\\%"
        out[f"fsEscNeed{key}"] = str(sum(need))
        out[f"fsEscEpisodes{key}"] = str(sum(esc_ep))
    # diagnosis-required subset: held-out episodes that the strong reference (arm everything) fails but the
    # perfect-twin oracle rescues, i.e. where a fault-specific response is needed
    st, orc = series("Strong", HELD), series("Oracle", HELD)
    sub = [i for i in st.index if i in orc.index and not st[i] and orc[i]]
    out["fsDiagSubN"] = str(len(sub))
    out["fsDiagSubIdN"] = str(sum(i[0] == "test_id" for i in sub))
    out["fsDiagSubOodN"] = str(sum(i[0] == "test_ood" for i in sub))
    for key in ("PbS1", "PbS2", "PbS3", "GPS1", "GPS1Early", "GPS2", "GPS3", "Expert", "FDD", "Onl", "OnlFourteen",
                "HyS1", "HyS2", "HyS3", "Weak"):
        v = series(key, HELD)
        for tag, part in (("", sub), ("Id", [i for i in sub if i[0] == "test_id"]),
                          ("Ood", [i for i in sub if i[0] == "test_ood"])):
            k = [int(v[i]) for i in part if i in v.index]
            if k:
                out[f"fsDiagSub{tag}{key}"] = f"{sum(k)}/{len(k)}"
    # outcome severity, supervisor-induced failures, the invocation cap and validator vetoes (held-out suites)
    weak_ok = series("Weak", HELD)
    for key in ("PbS1", "PbS2", "PbS3", "GPS1", "GPS1Early", "Expert", "FDD", "Onl", "HyS1", "HyS3", "Strong"):
        x = df[(df["key"] == key) & (df["suite"].isin(HELD))].set_index(["suite", "scenario", "seed"])
        n = len(x)
        if not n:
            continue
        trips = shut = induced = capped = postcap = vetoes = shadow = 0
        for idx, path in x["path"].items():
            d = json.load(open(path))
            ok = bool(x.loc[idx, "success"])
            trips += bool(d.get("tripped"))
            shut += bool(d.get("shutdown")) and not d.get("tripped")
            induced += (not ok) and bool(weak_ok.get(idx, 0))
            dec = d.get("decisions") or []
            if len(dec) >= 12:
                capped += 1
                end = d.get("trip_time") or d.get("shutdown_time")
                if not ok and end is not None and end > dec[-1]["t"] + 0.05:
                    postcap += 1
            for y in dec:
                r = y.get("rejected_reasons") or []
                if r and (y.get("action") or {}).get("type") != "no_action":
                    vetoes += 1
                    shadow += any("shadow" in str(z) for z in r)
        out[f"fsTrip{key}"], out[f"fsShut{key}"] = str(trips), str(shut)
        out[f"fsInduced{key}"] = str(induced)
        out[f"fsCapped{key}"] = f"{100 * capped / n:.0f}\\,\\%"
        out[f"fsPostCap{key}"] = str(postcap)
        out[f"fsVetoEp{key}"] = f"{vetoes / n:.2f}"
        out[f"fsShadowVetoEp{key}"] = f"{shadow / n:.2f}"
    # economic cost hidden by the success endpoint: production (stripper underflow integrated over the episode)
    # relative to the strong reference, paired on held-out episodes that BOTH stabilised, and on the benign family A
    # relative to the weak baseline (no action is needed there); episodes with an accepted production-cut procedure
    def prod_of(key):
        x = df[(df["key"] == key) & (df["suite"].isin(HELD))].set_index(["suite", "scenario", "seed"])
        return x

    ref_s, ref_w = prod_of("Strong"), prod_of("Weak")
    for key in ("PbS1", "PbS3", "GPS1", "GPS1Early", "Expert", "FDD", "Onl", "HyS1"):
        x = prod_of(key)
        prod = {i: json.load(open(p_))["production"] for i, p_ in x["path"].items()}
        refp = {i: json.load(open(p_))["production"] for i, p_ in ref_s["path"].items()}
        both = [i for i in x.index if i in ref_s.index and x.loc[i, "success"] and ref_s.loc[i, "success"]]
        if both:
            out[f"fsProdLoss{key}"] = f"{int(round(100 * (1 - np.mean([prod[i] / refp[i] for i in both]))))}\\,\\%"
        famA = [i for i in x.index if i[1].startswith("A") and i in ref_w.index]
        if famA:
            wp = {i: json.load(open(ref_w.loc[i, "path"]))["production"] for i in famA}
            out[f"fsProdLossA{key}"] = f"{int(round(100 * (1 - np.mean([prod[i] / wp[i] for i in famA]))))}\\,\\%"
        cut = sum(any(a["accepted"] and a["action"]["type"] == "procedure" and "cut" in a["action"].get("name", "")
                      for a in json.load(open(p_))["decisions"]) for p_ in x["path"])
        out[f"fsCutEp{key}"] = str(cut)
        out[f"fsProdN{key}"] = str(len(both))
    # amortised offline cost
    meta = json.load(open(os.path.join(SYN, M + "1", "meta_final.json")))
    n_held = int(len(series("PbS1", HELD)))
    out["fsOffTokSOne"] = f"\\num{{{meta['tokens']}}}"
    out["fsOffTokEpSOne"] = f"\\num{{{meta['tokens'] / max(n_held, 1):.0f}}}"
    onl_tok = df[(df["key"] == "Onl") & (df["suite"].isin(HELD))]
    onl_mean = float((onl_tok["prompt_tokens"] + onl_tok["completion_tokens"]).mean())
    out["fsOnlTokEp"] = f"\\num{{{onl_mean:.0f}}}"
    out["fsCostRatio"] = f"{onl_mean / (meta['tokens'] / max(n_held, 1)):.0f}"
    out["fsBreakEven"] = f"{meta['tokens'] / onl_mean:.0f}"
    tot3 = sum(json.load(open(os.path.join(SYN, M + s, "meta_final.json")))["tokens"] for s in "123")
    out["fsOffTokThree"] = f"\\num{{{tot3}}}"
    out["fsBreakEvenThree"] = f"{tot3 / onl_mean:.0f}"
    # break-even under two further cost models: completion tokens only (decoding dominated the measured latency;
    # prefill of the cached system prompt is cheap) and measured LLM wall-clock time on the shared accelerator.
    # Offline: per-call history of the synthesis run (meta_final.json); online: held-out gated episodes, time over
    # episodes without cached decisions (as latEp in analyze.py).
    hist = {s: json.load(open(os.path.join(SYN, M + s, "meta_final.json")))["history"] for s in "123"}
    off_c = {s: sum(x["ctok"] for x in h) for s, h in hist.items()}
    off_t = {s: sum(x["latency"] for x in h) for s, h in hist.items()}
    onl_c = float(onl_tok["completion_tokens"].mean())
    times = []
    for path in onl_tok["path"]:
        d = json.load(open(path))
        llm = [x for x in d["decisions"] if int(x.get("llm_calls", 0)) > 0]
        if llm and not any((x.get("meta") or {}).get("cached") or float(x.get("gpu_s", 0.0)) <= 0 for x in llm):
            times.append(float(d.get("gpu_s", 0.0) or 0.0))
    onl_t = float(np.mean(times))
    out["fsOffComplSOne"] = f"\\num{{{off_c['1']}}}"
    out["fsOnlComplEp"] = f"\\num{{{onl_c:.0f}}}"
    out["fsBreakEvenCompl"] = f"{off_c['1'] / onl_c:.0f}"
    out["fsBreakEvenComplThree"] = f"{sum(off_c.values()) / onl_c:.0f}"
    out["fsOffTimeSOne"] = f"\\num{{{off_t['1'] / 3600:.1f}}}\\,h"
    out["fsOnlTimeEp"] = f"\\num{{{onl_t:.0f}}}\\,s"
    out["fsBreakEvenTime"] = f"{off_t['1'] / onl_t:.0f}"
    out["fsBreakEvenTimeThree"] = f"{sum(off_t.values()) / onl_t:.0f}"
    out["fsOnlTimeN"] = str(len(times))
    # the seed-1 checkpoint saved after the first 25k tokens (B25000): cost and break-even under the three models
    m25 = json.load(open(os.path.join(SYN, M + "1", "meta_B25000.json")))
    h25 = m25["history"]
    out["fsOffTokTwentyFiveK"] = f"\\num{{{m25['tokens']}}}"
    out["fsBreakEvenTwentyFiveK"] = f"{m25['tokens'] / onl_mean:.1f}"
    out["fsBreakEvenComplTwentyFiveK"] = f"{sum(x['ctok'] for x in h25) / onl_c:.0f}"
    out["fsBreakEvenTimeTwentyFiveK"] = f"{sum(x['latency'] for x in h25) / onl_t:.0f}"
    # tail of the wall-clock latency of single uncached online decisions (modelled decision latency: 0.05 h = 180 s)
    lats = []
    for path in onl_tok["path"]:
        for x in json.load(open(path))["decisions"]:
            if int(x.get("llm_calls", 0)) > 0 and not (x.get("meta") or {}).get("cached") and float(x.get("gpu_s", 0.0)) > 0:
                lats.append(float(x["gpu_s"]))
    out["fsLatPNinetyFive"] = f"{np.percentile(lats, 95):.0f}\\,s"
    out["fsLatMax"] = f"{max(lats):.0f}\\,s"
    out["fsLatOverModel"] = f"{100 * np.mean([v > 180 for v in lats]):.1f}\\,\\%"
    out["fsLatN"] = f"\\num{{{len(lats)}}}"
    # selection of a synthesis run on a common, fresh validation sample (suite select_id, sampler seed 707), and the
    # expected held-out success of "run k syntheses, deploy the one with the best validation success", averaged over
    # all k-subsets of the three runs (k = 1 is the seed mean)
    from itertools import combinations
    sel = load_episodes(list(ARMS.values()), ["select_id"])
    sel = sel[sel["arm"].isin(inv)].copy()
    sel["key"] = sel["arm"].map(inv)
    for fam_keys, tag in ((("PbS1", "PbS2", "PbS3"), "Pb"), (("GPS1", "GPS2", "GPS3"), "GP")):
        vsel = {k: sel[sel["key"] == k]["success"].astype(int) for k in fam_keys}
        if not all(len(v) for v in vsel.values()):
            continue
        for k in fam_keys:
            out[f"fsSel{k}"] = f"{int(vsel[k].sum())}/{len(vsel[k])}"
        if tag == "Pb":
            v25 = sel[sel["key"] == "PbS1TwentyFiveK"]["success"]
            if len(v25):
                out["fsSelPbSOneTwentyFiveK"] = f"{int(v25.sum())}/{len(v25)}"
        out["fsSelN"] = str(len(vsel[fam_keys[0]]))
        held = {k: series(k, ("test_id",)).mean() for k in fam_keys}
        for kk, name in ((1, "One"), (2, "Two"), (3, "Three")):
            picks = [max(c, key=lambda k: (vsel[k].mean(), -fam_keys.index(k))) for c in combinations(fam_keys, kk)]
            out[f"fsBestOf{name}{tag}Id"] = f"{100 * np.mean([held[k] for k in picks]):.1f}\\,\\%"
    # "onset" (first scheduled event, including an initial production change) versus the first fault: in how many
    # held-out scenarios do the oracle's decision times (onset + 0.5 h) precede the fault, i.e. may be pre-emptive
    from cod.faults import Scenario
    for suite, tag in (("test_id", "Id"), ("test_ood", "Ood")):
        items = json.load(open(os.path.join(PROJ, "data", "suites", f"{suite}.json")))
        pre = 0
        for x in items:
            sc = Scenario.from_dict(x["scenario"])
            f0 = [e.start_h for e in sc.idv_events] + [e.start_h for e in sc.sensor_faults] + [e.start_h for e in sc.actuator_faults]
            pre += bool(f0) and sc.first_onset_h + 0.5 < min(f0)
        out[f"fsPreFault{tag}"] = str(pre)
    # confirmation on fresh suites (confirm_id / confirm_ood: coverage scenarios, three noise seeds each). Episodes of
    # one scenario are not independent, so contrasts use scenario-level means: a cluster bootstrap over scenarios
    # (2,000 replicates) for the interval and a sign-flip permutation test (10,000) for p
    conf = load_episodes(list(ARMS.values()), ["confirm_id", "confirm_ood"])
    conf = conf[conf["arm"].isin(inv)].copy()
    if len(conf):
        conf["key"] = conf["arm"].map(inv)
        rng_c = np.random.default_rng(2026)
        for suite, tag in (("confirm_id", "Id"), ("confirm_ood", "Ood")):
            x = conf[conf["suite"] == suite]
            per = x.groupby(["key", "scenario"])["success"].mean().unstack("key")
            out[f"fsConfNEp{tag}"] = f"\\num{{{len(x[x['key'] == 'PbS1'])}}}"  # episodes per arm on this suite
            for key in per.columns:
                v = x[x["key"] == key]["success"]
                out[f"fsConf{tag}{key}"] = f"{int(v.sum())}/{len(v)}"
                out[f"fsConfRate{tag}{key}"] = f"{100 * v.mean():.1f}\\,\\%"
            for a_, b_ in (("PbS1", "Expert"), ("PbS1", "GPS1"), ("PbS1", "FDD"), ("PbS1", "Strong"), ("PbS1", "CalS1"),
                           ("GPS1", "Strong"), ("Expert", "Strong"), ("FDD", "Strong"), ("GPS1Early", "GPS1"),
                           ("PbS1TwoH", "PbNotune"), ("PbS1TwoH", "PbNorem")):
                if a_ not in per.columns or b_ not in per.columns:
                    continue
                d = (per[a_] - per[b_]).dropna().values
                if not len(d):
                    continue
                boot = [rng_c.choice(d, len(d)).mean() for _ in range(2000)]
                perm = [abs((d * rng_c.choice([-1, 1], len(d))).mean()) for _ in range(10000)]
                pval = (1 + sum(p_ >= abs(d.mean()) - 1e-12 for p_ in perm)) / 10001
                t = f"{a_}Vs{b_}{tag}"
                out[f"fsConfDiff{t}"] = sgn(100 * d.mean())
                out[f"fsConfDiffAbs{t}"] = f"{abs(100 * d.mean()):.0f}"  # rounded magnitude, for the abstract
                out[f"fsConfCI{t}"] = f"{sgn(100 * np.percentile(boot, 2.5))} to {sgn(100 * np.percentile(boot, 97.5))}"
                out[f"fsConfP{t}"] = fmt_p(pval)
    if "fsConfNEpId" in out and "fsConfNEpOod" in out:
        out["fsConfNEp"] = f"\\num{{{int(out['fsConfNEpId'][5:-1]) + int(out['fsConfNEpOod'][5:-1])}}}"
    # longest prompt in the language-model call cache (requests used num_ctx = 16384; Ollama truncates silently)
    import sqlite3
    cache = os.path.join(os.path.dirname(PROJ), "cache", "llm_cache.sqlite")
    if os.path.exists(cache):
        con = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
        out["fsMaxPromptTok"] = f"\\num{{{con.execute('SELECT MAX(ptok) FROM cache').fetchone()[0]}}}"
        con.close()
    # R3 robustness checks of the compiled supervisors: (a) on top of the strong configuration ("onstrong:" arms),
    # (b) dense invocation (suites test_*_dense: re-invocation every 0.5 h, at most 96 invocations)
    r3 = {"Expert": "B2b", "FDD": "fdd:default", "GPSOne": ARMS["GPS1"], "PbSOne": ARMS["PbS1"], "PbSThree": ARMS["PbS3"]}
    on = load_episodes(["onstrong:" + v for v in r3.values()], list(HELD))
    dense = load_episodes(list(r3.values()), ["test_id_dense", "test_ood_dense"])
    for k, v in r3.items():
        for suite, tag in (("test_id", "Id"), ("test_ood", "Ood")):
            x = on[(on["arm"] == "onstrong:" + v) & (on["suite"] == suite)] if len(on) else on
            if len(x):
                out[f"fsOnStrong{tag}{k}"] = f"{int(x['success'].sum())}/{len(x)}"
            y = dense[(dense["arm"] == v) & (dense["suite"] == suite + "_dense")] if len(dense) else dense
            if len(y):
                out[f"fsDense{tag}{k}"] = f"{int(y['success'].sum())}/{len(y)}"
    # run-time errors of the language-model playbooks on the held-out suites. A playbook that raises abstains at that
    # invocation. The seed-2 playbooks read alarm ages under the flag names (f["alarm_ages"]["A_GRADE_DEV"] instead of
    # f["alarm_ages"]["GRADE_DEV"]) from the first draft on; the coder was never told (the repair message's error summary
    # was overwritten whenever the batch contained a benign scenario, and the cards list only actions taken)
    def _err(dec):
        return bool((dec.get("meta") or {}).get("error")) or str(dec.get("rationale", "")).startswith("runtime error")
    for k in ("PbS1", "PbS2", "PbS3", "PbVzero", "PbS2Vzero", "PbS3Vzero"):
        x = df[(df["key"] == k) & (df["suite"].isin(HELD))]
        if not len(x):
            continue
        n_ep = n_dec = n_err = 0
        fam_err: dict = {}
        for path, fam in zip(x["path"], x["family"]):
            dec = json.load(open(path))["decisions"]
            e = sum(_err(d_) for d_ in dec)
            n_ep += e > 0
            n_dec += len(dec)
            n_err += e
            if e:
                fam_err[fam] = fam_err.get(fam, 0) + 1
        out[f"fsErrEp{k}"] = f"{n_ep}/{len(x)}"
        out[f"fsErrInv{k}"] = f"{100 * n_err / max(n_dec, 1):.0f}\\,\\%"
        for fam in ("Q", "P", "S"):
            out[f"fsErrEp{k}{fam}"] = f"{fam_err.get(fam, 0)}/{int((x['family'] == fam).sum())}"
    # training success per seed (selection rule)
    for s in "123":
        mm = json.load(open(os.path.join(SYN, M + s, "meta_final.json")))
        out[f"fsTrainS{s}"] = f"{100 * mm['best_sr']:.0f}\\,\\%"
    # coverage: exact (Clopper-Pearson) two-sided 95 % intervals for the in-distribution, adversary and
    # family-shift failure rates of each final playbook
    from scipy.stats import beta
    def cp(k, n):
        lo = 0.0 if k == 0 else beta.ppf(0.025, k, n - k + 1)
        hi = 1.0 if k == n else beta.ppf(0.975, k + 1, n - k)
        return f"{100 * lo:.1f}--{100 * hi:.1f}\\,\\%"
    for name, key in (("llm_final_s1", "LlmSOne"), ("llm_final_s2", "LlmSTwo"), ("llm_final_s3", "LlmSThree"),
                      ("gp_final_s1", "GpSOne")):
        fp = os.path.join(PROJ, "results", "coverage", f"{name}.json")
        if not os.path.exists(fp):
            continue
        c = json.load(open(fp))
        for part, tag in (("in_dist", "In"), ("adversary_round0", "Adv"), ("family_shift", "Shift")):
            if part in c:
                out[f"fsCovCI{tag}{key}"] = cp(c[part]["fail"], c[part]["n"])
    # compile-back: admitted rules and how often they fired
    cb = os.path.join(STREAM, "cb_s1_final_16k")
    if os.path.isdir(cb):
        v0 = open(os.path.join(cb, "playbook_v0.py")).read()
        vers = sorted(glob.glob(os.path.join(cb, "playbook_v*.py")), key=lambda p: int(re.search(r"_v(\d+)", p).group(1)))
        last = open(vers[-1]).read()
        rules = lambda s: set(re.findall(r'"rule":\s*"([^"]+)"', s))
        new = rules(last) - rules(v0)
        fired = 0
        for p in glob.glob(os.path.join(cb, "block*.json")):
            for e in json.load(open(p))["episodes"]:
                fired += sum(1 for d in e["decisions"] if d["source"].startswith("rule:") and d["source"][5:] in new)
        log = json.load(open(os.path.join(cb, "summary.json")))["log"]
        out["fsCbAdmitted"] = str(sum(1 for e in log if e.get("admitted")))
        out["fsCbSyntax"] = str(sum(1 for e in log if not e.get("admitted") and "syntax" in str(e.get("reason", ""))))
        per_block = {}
        for e in log:
            if not e.get("admitted") and "syntax" in str(e.get("reason", "")):
                per_block[e["block"]] = per_block.get(e["block"], 0) + 1
        if per_block:
            b_max = max(per_block, key=per_block.get)
            out["fsCbSyntaxMax"], out["fsCbSyntaxMaxBlock"] = str(per_block[b_max]), str(b_max + 1)
        out["fsCbRegress"] = str(sum(1 for e in log if not e.get("admitted") and "syntax" not in str(e.get("reason", ""))))
        out["fsCbNewRules"] = str(len(new))
        out["fsCbFired"] = str(fired)
    os.makedirs(GEN, exist_ok=True)
    lines = ["% GENERATED by code/experiments/final_stats.py -- do not edit"]
    lines += [f"\\providecommand{{\\{texname(k)}}}{{{v}}}" for k, v in sorted(out.items())]
    open(os.path.join(GEN, "numbers_final.tex"), "w").write("\n".join(lines) + "\n")
    # per-seed robustness table
    rows = []
    for s, pk, hk, gk, ge in (("1", "PbS1", "HyS1", "GPS1", "GPS1Early"), ("2", "PbS2", "HyS2", "GPS2", "GPS2Early"),
                              ("3", "PbS3", "HyS3", "GPS3", "GPS3Early")):
        def c(k, suites):
            v = series(k, suites)
            return f"{int(v.sum())}/{len(v)}" if len(v) else "--"
        rows.append(f"{s} & {out.get(f'fsTrainS{s}', '--')} & {out.get(f'fsSel{pk}', '--')} & {c(pk, ('test_id',))} & {c(pk, ('test_ood',))} & "
                    f"{c(hk, ('test_id',))} & {c(hk, ('test_ood',))} & {c(ge, ('test_id',))} & {c(ge, ('test_ood',))} & "
                    f"{c(gk, ('test_id',))} & {c(gk, ('test_ood',))}\\\\")
    tab = ["\\begin{tabular}{@{}lrrrrrrrrrr@{}}", "\\toprule",
           " & & & \\multicolumn{2}{c}{LLM playbook (final)} & \\multicolumn{2}{c}{Hybrid (16k/ep.)} & \\multicolumn{2}{c}{GP @5k evals} & \\multicolumn{2}{c}{GP final}\\\\",
           "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(l){10-11}",
           "Seed & Training & Validation & Antic. & Novel & Antic. & Novel & Antic. & Novel & Antic. & Novel\\\\", "\\midrule"] + rows + \
          ["\\midrule", f"Mean & & & {out.get('fsMeanPbId', '')} & {out.get('fsMeanPbOod', '')} & {out.get('fsMeanHyId', '')} & "
           f"{out.get('fsMeanHyOod', '')} & {out.get('fsMeanGPEarlyId', '')} & {out.get('fsMeanGPEarlyOod', '')} & "
           f"{out.get('fsMeanGPId', '')} & {out.get('fsMeanGPOod', '')}\\\\", "\\bottomrule", "\\end{tabular}"]
    open(os.path.join(GEN, "seeds.tex"), "w").write("\n".join(tab) + "\n")
    # supplementary tables: confirmation suites (frozen arms) and the R3 robustness checks
    def held(k, suite):
        v = series(k, (suite,))
        return f"{int(v.sum())}/{len(v)}" if len(v) else "--"
    conf_rows = (("Weak configuration", "Weak"), ("Strong reference", "Strong"), ("Expert table", "Expert"),
                 ("FDD with lookup", "FDD"), ("GP seed 1, 5k episodes", "GPS1Early"), ("GP seed 1, final", "GPS1"),
                 ("GP seed 2, final", "GPS2"), ("GP seed 3, final", "GPS3"), ("LLM seed 1, final", "PbS1"),
                 ("LLM seed 2, final", "PbS2"), ("LLM seed 3, final", "PbS3"), ("LLM seed 1, 210k tokens", "PbS1TwoH"),
                 ("\\quad no counterfactual remedies", "PbNorem"), ("\\quad no CPU calibration", "PbNotune"))
    if any(f"fsConfId{k}" in out for _, k in conf_rows):
        rows = [f"{lab} & {held(k, 'test_id')} & {out.get(f'fsConfId{k}', '--')} & {out.get(f'fsConfRateId{k}', '--')} & "
                f"{held(k, 'test_ood')} & {out.get(f'fsConfOod{k}', '--')} & {out.get(f'fsConfRateOod{k}', '--')}\\\\"
                for lab, k in conf_rows if f"fsConfId{k}" in out or f"fsConfOod{k}" in out]
        tab = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
               " & \\multicolumn{3}{c}{Anticipated families} & \\multicolumn{3}{c}{Novel families}\\\\",
               "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
               "Arm & Held-out & Confirmation & Rate & Held-out & Confirmation & Rate\\\\", "\\midrule"] + rows + \
              ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(GEN, "confirm.tex"), "w").write("\n".join(tab) + "\n")
        pairs = (("LLM s1 $-$ expert table", "PbS1VsExpert"), ("LLM s1 $-$ GP s1", "PbS1VsGPS1"),
                 ("LLM s1 $-$ FDD with lookup", "PbS1VsFDD"), ("LLM s1 $-$ strong reference", "PbS1VsStrong"),
                 ("GP s1 $-$ strong reference", "GPS1VsStrong"), ("Expert table $-$ strong reference", "ExpertVsStrong"),
                 ("FDD with lookup $-$ strong reference", "FDDVsStrong"), ("GP s1, 5k $-$ GP s1, final", "GPS1EarlyVsGPS1"),
                 ("LLM s1 210k $-$ no CPU calibration", "PbS1TwoHVsPbNotune"),
                 ("LLM s1 210k $-$ no remedies", "PbS1TwoHVsPbNorem"))
        pv = lambda x: x.replace("$=$\\,", "")  # p column: drop the '= ' prefix of fmt_p
        rows = [f"{lab} & {out.get(f'fsConfDiff{t}Id', '--')} & {out.get(f'fsConfCI{t}Id', '--')} & {pv(out.get(f'fsConfP{t}Id', '--'))} & "
                f"{out.get(f'fsConfDiff{t}Ood', '--')} & {out.get(f'fsConfCI{t}Ood', '--')} & {pv(out.get(f'fsConfP{t}Ood', '--'))}\\\\"
                for lab, t in pairs if f"fsConfDiff{t}Id" in out or f"fsConfDiff{t}Ood" in out]
        tab = ["\\begin{tabular}{@{}lrcrrcr@{}}", "\\toprule",
               " & \\multicolumn{3}{c}{Anticipated families} & \\multicolumn{3}{c}{Novel families}\\\\",
               "\\cmidrule(lr){2-4}\\cmidrule(l){5-7}",
               "Contrast & pp & 95\\,\\% interval & $p$ & pp & 95\\,\\% interval & $p$\\\\", "\\midrule"] + rows + \
              ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(GEN, "confirm_contrasts.tex"), "w").write("\n".join(tab) + "\n")
    if any(k.startswith("fsOnStrong") or k.startswith("fsDense") for k in out):
        rows = [f"Strong reference (no supervisor) & {held('Strong', 'test_id')} & {held('Strong', 'test_ood')} & -- & -- & -- & --\\\\"]
        for lab, k, ak in (("Expert table", "Expert", "Expert"), ("FDD with lookup", "FDD", "FDD"),
                           ("GP seed 1, final", "GPSOne", "GPS1"), ("LLM seed 1, final", "PbSOne", "PbS1"),
                           ("LLM seed 3, final", "PbSThree", "PbS3")):
            rows.append(f"{lab} & {held(ak, 'test_id')} & {held(ak, 'test_ood')} & {out.get(f'fsOnStrongId{k}', '--')} & "
                        f"{out.get(f'fsOnStrongOod{k}', '--')} & {out.get(f'fsDenseId{k}', '--')} & {out.get(f'fsDenseOod{k}', '--')}\\\\")
        tab = ["\\begin{tabular}{@{}lrrrrrr@{}}", "\\toprule",
               " & \\multicolumn{2}{c}{As in the article} & \\multicolumn{2}{c}{On the strong configuration} & \\multicolumn{2}{c}{Dense invocation}\\\\",
               "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}",
               "Supervisor & Antic. & Novel & Antic. & Novel & Antic. & Novel\\\\", "\\midrule"] + rows + \
              ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(GEN, "r3checks.tex"), "w").write("\n".join(tab) + "\n")
    print(f"{len(out)} macros")
    for k in sorted(out):
        print(texname(k), "=", out[k])


if __name__ == "__main__":
    main()
