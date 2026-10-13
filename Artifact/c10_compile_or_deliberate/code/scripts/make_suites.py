"""Build the frozen held-out test suites from the seed-101 screen (never used for synthesis/tuning).

test_id  : anticipated families (A, S, P, Q, SN), stratified by headroom label.
test_ood : novel families (V actuator faults, C compound faults, K kinetics/heat-transfer at high rate).
Unrescuable scenarios (no oracle plan) are kept in a separate suite `test_unrescuable` (reported apart).
A content hash of each suite is written next to it.
"""
import glob, hashlib, json, os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = sorted((json.load(open(p)) for p in glob.glob(os.path.join(ROOT, "results", "screen_101", "*", "*.json"))),
           key=lambda r: r["scenario"]["id"])
STRATA = {
    "test_id": {"A": {"benign": 16}, "S": {"rescuable": 18, "benign": 4}, "P": {"rescuable": 16, "benign": 4},
                "Q": {"quality_gap": 16}, "SN": {"rescuable": 18, "benign": 6}},
    "test_ood": {"V": {"rescuable": 20, "benign": 6}, "C": {"rescuable": 20, "benign": 6}, "K": {"benign": 12}},
    "test_unrescuable": {"V": {"unrescuable": 99}, "C": {"unrescuable": 99}},
}
for name, strata in STRATA.items():
    out = []
    for fam, want in strata.items():
        for lab, n in want.items():
            sel = [r for r in R if r["scenario"]["family"] == fam and r["label"] == lab][:n]
            out += [{"scenario": r["scenario"], "seeds": [r["seed"]], "label": r["label"]} for r in sel]
    path = os.path.join(ROOT, "data", "suites", f"{name}.json")
    json.dump(out, open(path, "w"), indent=0)
    h = hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()[:16]
    open(path + ".sha", "w").write(h + "\n")
    print(name, len(out), h)
