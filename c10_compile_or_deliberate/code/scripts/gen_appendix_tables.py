"""Generate LaTeX appendix tables from the code constants (single source of truth)."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
from cod.alarms import ALARMS
from cod.actions import BOUNDS, PROCEDURES, DIAGNOSES, SUBSTITUTABLE
OUT = os.environ.get("COD_GEN", os.path.join(ROOT, "paper", "generated"))  # output folder (as in analyze.py)
os.makedirs(OUT, exist_ok=True)
tt = lambda s: "\\texttt{" + s.replace("_", "\\_") + "}"
rows = []
for a in ALARMS:
    cond = (f"$|{tt(a.measurement)} - {tt(a.setpoint)}| > {a.limit:g}$" if a.kind == "DEV"
            else f"{tt(a.measurement)} {'$>$' if a.kind == 'HI' else '$<$'} {a.limit:g}")
    rows.append(f"{tt(a.name)} & {cond} & {a.deadband:g} & {a.on_delay_h:g} & {a.priority} \\\\")
open(os.path.join(OUT, "alarms.tex"), "w").write(
    "\\begin{tabular}{@{}llrrr@{}}\\toprule Alarm & Condition (observed) & Deadband & On-delay (h) & Priority\\\\\\midrule\n"
    + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
rows = [f"{tt(k)} & {b.lo:g} & {b.hi:g} & {b.max_delta:g} & {b.cooldown_h:g} \\\\" for k, b in BOUNDS.items()]
open(os.path.join(OUT, "bounds.tex"), "w").write(
    "\\begin{tabular}{@{}lrrrr@{}}\\toprule Setpoint & Low & High & Max move & Cooldown (h)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
desc = {"rate_cut_25": "production target $\\times 0.75$", "rate_cut_50": "production target $\\times 0.5$",
        "depressurize_step": "reactor-pressure setpoint $-75$\\,kPa",
        "min_stable_park": "production $\\times 0.5$; after 1\\,h reactor pressure 2600\\,kPa"}
open(os.path.join(OUT, "procedures.tex"), "w").write(
    "\\begin{tabular}{@{}ll@{}}\\toprule Procedure & Steps (clamped to bounds)\\\\\\midrule\n"
    + "\n".join(f"{tt(k)} & {desc[k]} \\\\" for k in sorted(PROCEDURES)) + "\n\\bottomrule\\end{tabular}\n")
ss = json.load(open(os.path.join(ROOT, "data", "soft_sensors.json")))
rows = []
for k in ("reactor_pressure", "reactor_temperature", "separator_temperature", "stripper_pressure"):
    m = ss[k]
    ins = ", ".join(tt(i.replace("mvf:", "filt.\\ ").replace("mv:", "")) for i in m["inputs"])
    rows.append(f"{tt(k)} & {ins} & {m['test_rmse']:.2f} & {m['n_train']} / {m['n_test']} \\\\")
open(os.path.join(OUT, "softsensors.tex"), "w").write(
    "\\begin{tabular}{@{}lp{7.5cm}rr@{}}\\toprule Target & Inputs & Test RMSE & Samples (train/test)\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print("written", os.listdir(OUT))
