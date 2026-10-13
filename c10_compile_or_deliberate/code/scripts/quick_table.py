"""Quick success table across arms and suites (uses screen labels for quality criterion)."""
import json, glob, os, sys, collections
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from cod.synth.evaluate import success
labels = {}
for p in glob.glob(os.path.join(ROOT, "data", "suites", "*.json")):
    for x in json.load(open(p)):
        labels[x["scenario"]["id"]] = x["label"]
for sid in ["Q1_grade_to_40", "Q2_grade_to_65", "Q3_grade_40_idv08"]:
    labels[sid] = "quality_gap"
arms = sys.argv[1].split(",") if len(sys.argv) > 1 else None
rows = collections.defaultdict(lambda: collections.defaultdict(list))
for p in glob.glob(os.path.join(ROOT, "results", "arms", "*", "*", "*.json")):
    d = json.load(open(p))
    arm, suite = d["arm"], d["suite"]
    if arms and arm not in arms: continue
    lab = labels.get(d["scenario"])
    rows[arm][suite].append((success(d, lab), d.get("n_accepted", 0), d.get("prompt_tokens", 0) + d.get("completion_tokens", 0), d.get("llm_calls", 0), d["family"]))
suites = ["alhazmi", "frontier", "test_id", "test_ood", "test_unrescuable"]
print(f"{'arm':42s}" + "".join(f"{s:>20s}" for s in suites))
for arm in sorted(rows):
    line = f"{arm[:42]:42s}"
    for s in suites:
        v = rows[arm].get(s, [])
        if v:
            k = sum(x[0] for x in v); n = len(v); tok = sum(x[2] for x in v) / n
            line += f"{k:>4d}/{n:<4d}{'%.0fk' % (tok / 1000) if tok else '':>6s} a{sum(x[1] for x in v)/n:4.1f}"
        else:
            line += f"{'-':>20s}"
    print(line)
if len(sys.argv) > 2:
    fam = collections.defaultdict(lambda: collections.defaultdict(list))
    for arm in rows:
        for s in ("test_id", "test_ood"):
            for x in rows[arm].get(s, []):
                fam[arm][x[4]].append(x[0])
    fams = sorted({f for a in fam for f in fam[a]})
    print(f"{'arm':42s}" + "".join(f"{f:>8s}" for f in fams))
    for arm in sorted(fam):
        print(f"{arm[:42]:42s}" + "".join(f"{sum(fam[arm][f])}/{len(fam[arm][f])}".rjust(8) for f in fams))
