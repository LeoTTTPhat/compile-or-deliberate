"""B5: genetic programming over the playbook DSL (the non-LLM synthesis baseline).

Everything except the proposal operator is shared with the LLM synthesiser (llm_synth.py): the
playbook contract and sandbox (PlaybookSupervisor), the training batch, the episode evaluation and
score (evaluate.py), and the falsification round (same fresh-scenario stream, same labelling, same
hardening rule, same batch cap). Where the LLM repairs a parent from counterexample cards, GP varies
rule lists blindly and selects on the score.

Genome: ordered list of 1-8 rules; a rule is a conjunction of 1-2 atoms -> one action.
  * atoms: `f[x] < q` / `f[x] > q` on a curated numeric feature x, with q drawn from x's empirical
    quantiles at supervisor invocations on training-family episodes (DEFAULT_QUANTILES); alarm atoms
    `f["A_<NAME>"]` (optionally negated); armed-state atoms.
  * actions: a fixed catalogue = the action grammar instantiated relative to the CURRENT setpoints
    (as in oracle.candidate_actions): reconfigurations, procedures, soft-sensor substitutions and
    setpoint moves whose absolute targets the playbook computes from f[K+"_sp"], clamped to BOUNDS.
Every genome compiles to playbook source run by the identical sandbox. Each rule's action is
proposed at most once per episode (mem guard); a proposal vetoed by the validator is released at the
revision call and may be retried later. Actions carry admissibility guards (not already armed /
substituted, setpoint not at its bound) so rules do not waste invocations on certain rejections.

GA: generational, population 24, tournament selection (k=3), one-point crossover on rule lists,
mutation (threshold / feature / operator / atom / action / add-drop atom / add-remove-swap rule),
elitism 2. Fitness = evaluate(...).score on the training batch; episode results are cached per
(source, scenario, seed, label), which is exact because episodes use common random numbers.
Budget = playbook EPISODE EVALUATIONS (cache hits are free; scenario labelling jobs in falsification
rounds are counted separately); checkpoint E<n> holds the incumbent found with at most n evaluations.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import pickle
import random
import time
from dataclasses import dataclass, field

import numpy as np

from ..actions import BOUNDS, NO_ACTION, PROCEDURES
from ..alarms import ALARM_NAMES
from ..faults import Scenario
from ..playbook import check_source, loc, rule_names
from ..runner import Decision, run_episode
from ..scenarios import sample
from .evaluate import ACTION_PENALTY, EvalResult, _job, _pool, evaluate, label_scenarios, success

MAX_RULES = 8
MAX_ATOMS = 2
NUM_FEATURES = ("P_avg", "P_slope", "P_ttl", "P_resid", "dP_rs", "T_avg", "T_slope", "T_resid", "RL_avg", "SL_avg",
                "XL_avg", "XL_slope", "G_cmd_err", "PROD_cmd_err", "yA_dev_nom", "mv_r_cooling_water",
                "mv_r_cooling_water_std", "mv_a_feed", "n_saturated", "n_alarms")
INT_THRESHOLDS = {"n_saturated": (0.5, 1.5, 2.5, 3.5), "n_alarms": (0.5, 1.5, 2.5, 3.5)}
STATE_KEYS = ("armed_composition", "armed_overrides")
QUANT_LEVELS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)

# Empirical quantiles (QUANT_LEVELS, finite values only) of the features at supervisor invocations of
# no-action episodes on the 78-scenario training batch (synthesis families, seed-7 screen), produced by
# feature_quantiles(training_batch()) (T_resid re-run after summary.features was fixed to pass the filtered
# valve input to the temperature soft sensor). Never computed on the held-out test split.
DEFAULT_QUANTILES: dict[str, tuple[float, ...]] = {
    "P_avg": (2595.2, 2623.7, 2685.8, 2701.9, 2704.0, 2704.6, 2705.0, 2705.6, 2706.4, 2711.8, 2752.1, 2850.6,
              2890.1, 2936.3, 2960.9),
    "P_slope": (-63.378, -43.244, -13.864, -4.8551, -2.0072, -0.73607, 0.0, 0.67697, 1.8638, 7.3088, 28.289, 40.454,
                104.5, 124.24, 135.25),
    "P_ttl": (0.19295, 0.30741, 0.72103, 1.1994, 4.6584, 5.8432, 7.8661, 14.282, 41.116, 100.27, 163.33, 275.98,
              354.1, 424.23, 444.52),
    "P_resid": (-265.93, -214.1, -163.97, -14.251, -0.40151, -0.31292, -0.24601, -0.16415, -0.043017, 0.23425,
                0.51798, 1.3113, 1.791, 30.778, 102.11),
    "dP_rs": (-262.56, -211.18, -161.51, -12.811, -0.276, 0.030331, 0.11161, 0.18738, 0.31624, 0.9335, 2.1767,
              4.2977, 5.5438, 33.842, 101.23),
    "T_avg": (120.3, 120.36, 120.38, 120.39, 120.4, 120.41, 120.43, 120.45),
    "T_slope": (-0.15111, -0.044512, -0.017028, -0.010072, -0.0065283, -0.0042413, -0.0016519, 6.9228e-05,
                0.0023752, 0.0047965, 0.0074247, 0.013107, 0.028099, 0.061239, 0.14588),
    "T_resid": (-7.6559, -3.8103, -0.42994, -0.16155, 0.011841, 0.045723, 0.059771, 0.06695, 0.074601, 0.083059,
                0.094342, 0.12132, 0.1425, 0.16094, 0.2096),
    "RL_avg": (70.552, 72.782, 74.297, 74.656, 74.878, 74.966, 75.025, 75.085, 75.241, 75.657, 76.963, 78.802,
               79.491, 80.038, 80.569),
    "SL_avg": (42.339, 43.126, 44.164, 46.325, 48.865, 49.449, 49.767, 49.951, 50.152, 50.375, 51.174, 54.429,
               58.288, 59.952, 60.721),
    "XL_avg": (10.334, 12.28, 14.639, 17.664, 33.762, 44.318, 48.371, 49.419, 49.861, 50.311, 51.091, 55.401, 60.27,
               67.543, 75.322),
    "XL_slope": (-19.013, -17.856, -15.88, -10.365, -5.3784, -1.6576, -0.74787, -0.25352, 0.077541, 0.5327, 1.3255,
                 4.56, 7.8172, 12.326, 13.014),
    "G_cmd_err": (-10.395, -10.113, -8.432, -1.311, -0.57379, -0.21342, 0.047172, 0.28857, 0.58679, 1.0175, 1.6949,
                  2.31, 9.2208, 11.235, 13.263),
    "PROD_cmd_err": (-6.2284, -6.1123, -5.5411, -4.6377, -2.1339, -0.45456, -0.049239, -0.0084656, 0.0038726,
                     0.016386, 0.065953, 0.32845, 1.4851, 2.7974, 3.1286),
    "yA_dev_nom": (-17.311, -16.507, -14.8, -13.127, -9.6374, -4.9329, -0.48009, -0.17001, 0.047297, 0.25531,
                   0.47547, 1.0335, 1.9251, 2.624, 2.9144),
    "mv_r_cooling_water": (33.948, 34.731, 35.81, 37.899, 40.039, 40.596, 40.854, 41.04, 41.179, 41.375, 41.686,
                           42.407, 43.075, 44.694, 45.397),
    "mv_r_cooling_water_std": (0.16492, 0.17441, 0.18573, 0.19569, 0.21705, 0.22713, 0.23701, 0.24792, 0.25723,
                               0.27317, 0.29482, 0.37084, 0.54749, 2.409, 3.1166),
    "mv_a_feed": (6.9922, 15.731, 19.532, 23.281, 24.181, 24.569, 24.672, 24.694, 24.727, 25.096, 26.265, 100.0),
}

# --------------------------------------------------------------------------- action catalogue
SP_KEY = {"reactor_pressure": "P", "reactor_temperature": "T", "reactor_level": "RL", "separator_level": "SL",
          "stripper_level": "XL", "production_rate": "PROD"}
SP_MOVES = (("reactor_pressure", (-100.0, -50.0, 50.0)), ("reactor_temperature", (-3.0, 3.0)),
            ("reactor_level", (-10.0, 10.0)), ("separator_level", (-10.0, 10.0)),
            ("stripper_level", (-10.0, 10.0)), ("production_rate", (-4.0, -2.0, 2.0)))


def _catalogue() -> dict[str, tuple[str, dict | tuple[str, float]]]:
    """action key -> (group, fixed action dict | (setpoint field, relative move))."""
    C: dict[str, tuple[str, dict | tuple[str, float]]] = {
        "arm_composition": ("reconfigure", {"type": "reconfigure", "components": ["composition_control"]}),
        "arm_overrides": ("reconfigure", {"type": "reconfigure", "components": ["safety_overrides"]}),
        "arm_both": ("reconfigure", {"type": "reconfigure", "components": ["composition_control", "safety_overrides"]}),
    }
    for name in sorted(PROCEDURES):
        C[name] = ("procedure", {"type": "procedure", "name": name})
    for m, k in (("reactor_pressure", "subst_P"), ("reactor_temperature", "subst_T")):
        C[k] = ("substitute", {"type": "substitute_instrument", "measurement": m, "method": "soft"})
    for fld, deltas in SP_MOVES:
        for d in deltas:
            C[f"{fld}{d:+g}"] = ("setpoint", (fld, d))
    return C


ACTIONS = _catalogue()
GROUPS = ("reconfigure", "procedure", "substitute", "setpoint")
GROUP_P = (0.3, 0.25, 0.15, 0.3)
DIAGNOSIS = {"subst_P": "sensor_fault", "subst_T": "sensor_fault"}


def _num(v: float) -> str:
    return f"{v:.5g}"


def action_source(key: str) -> tuple[list[str], str]:
    """(admissibility guards, action expression) for a catalogue entry."""
    group, spec = ACTIONS[key]
    if group == "setpoint":
        fld, d = spec
        b, k = BOUNDS[fld], SP_KEY[fld] + "_sp"
        if d < 0:
            return [f'f["{k}"] > {b.lo!r}'], f'{{"type": "adjust_setpoints", "changes": {{"{fld}": max({b.lo!r}, f["{k}"] - {-d!r})}}}}'
        return [f'f["{k}"] < {b.hi!r}'], f'{{"type": "adjust_setpoints", "changes": {{"{fld}": min({b.hi!r}, f["{k}"] + {d!r})}}}}'
    guards = []
    if group == "reconfigure":
        comps = spec["components"]
        if len(comps) == 1:
            guards.append(f'not f["armed_{"composition" if comps[0] == "composition_control" else "overrides"}"]')
        else:
            guards.append('not (f["armed_composition"] and f["armed_overrides"])')
    elif group == "substitute":
        guards.append(f'"{spec["measurement"]}" not in f["substituted"]')
    return guards, json.dumps(spec)


# --------------------------------------------------------------------------- genome
@dataclass(frozen=True, order=True)
class Atom:
    kind: str  # "num" | "alarm" | "state"
    key: str  # feature / alarm name / state flag
    op: str  # "<" | ">" for num; "" | "not" for alarm/state
    thr: float = 0.0

    def src(self) -> str:
        if self.kind == "num":
            return f'f["{self.key}"] {self.op} {_num(self.thr)}'
        name = f"A_{self.key}" if self.kind == "alarm" else self.key
        return ("not " if self.op == "not" else "") + f'f["{name}"]'

    def text(self) -> str:
        if self.kind == "num":
            return f"{self.key} {self.op} {_num(self.thr)}"
        return ("not " if self.op == "not" else "") + self.key


@dataclass(frozen=True)
class Rule:
    atoms: tuple[Atom, ...]
    action: str


Genome = tuple[Rule, ...]


def normalize(g: Genome) -> Genome:
    """Canonical form: atoms sorted and de-duplicated, duplicate rules dropped, at most MAX_RULES."""
    out, seen = [], set()
    for r in g:
        r2 = Rule(tuple(sorted(set(r.atoms)))[:MAX_ATOMS], r.action)
        if r2 not in seen:
            seen.add(r2)
            out.append(r2)
    return tuple(out[:MAX_RULES])


def compile_genome(g: Genome, header: str = "") -> str:
    """Playbook source for a genome (first matching rule wins). At the revision call the vetoed rule, and any
    rule with the same action (it would re-propose the identical action), is skipped."""
    L = [f"# {header}"] if header else []
    L += ["def playbook(f, mem):",
          "    # each rule proposes its action at most once per episode; a vetoed proposal is released",
          '    vetoed = ""',
          '    if f.get("trigger") == "revision":',
          '        vetoed = mem.get("last", "")',
          "        mem[vetoed] = 0"]
    for i, r in enumerate(g, 1):
        rid = f"R{i}"
        guards, act = action_source(r.action)
        why = " and ".join(a.text() for a in r.atoms)
        same = [f'"R{j}"' for j, r2 in enumerate(g, 1) if r2.action == r.action]
        skip = f"vetoed != {same[0]}" if len(same) == 1 else f"vetoed not in ({', '.join(same)})"
        conds = [f'not mem.get("{rid}")', skip] + guards + [a.src() for a in r.atoms]
        L += [f"    # {rid}: if {why} -> {r.action}",
              f"    if {' and '.join(conds)}:",
              f'        mem["{rid}"] = 1',
              f'        mem["last"] = "{rid}"',
              f'        return {{"rule": "{rid}", "diagnosis": "{DIAGNOSIS.get(r.action, "unknown")}", '
              f'"why": "{why}", "action": {act}}}']
    L.append("    return None")
    return "\n".join(L) + "\n"


def genome_json(g: Genome) -> list[dict]:
    return [{"atoms": [[a.kind, a.key, a.op, a.thr] for a in r.atoms], "action": r.action} for r in g]


def genome_from_json(d: list[dict]) -> Genome:
    return tuple(Rule(tuple(Atom(*a) for a in r["atoms"]), r["action"]) for r in d)


class Variation:
    """Random genome construction, mutation and crossover over the catalogue."""

    def __init__(self, rng: random.Random, quantiles: dict[str, tuple[float, ...]] | None = None,
                 max_rules: int = MAX_RULES):
        self.rng = rng
        q = dict(quantiles or DEFAULT_QUANTILES)
        q.update(INT_THRESHOLDS)
        self.q = {k: tuple(v) for k, v in q.items() if k in NUM_FEATURES and len(v)}
        self.features = tuple(k for k in NUM_FEATURES if k in self.q)
        self.max_rules = max_rules
        self.by_group = {gr: [k for k, (g2, _) in ACTIONS.items() if g2 == gr] for gr in GROUPS}

    # ---------------------------------------------------------------- random parts
    def atom(self) -> Atom:
        u = self.rng.random()
        if u < 0.6:
            k = self.rng.choice(self.features)
            return Atom("num", k, self.rng.choice("<>"), self.rng.choice(self.q[k]))
        if u < 0.9:
            return Atom("alarm", self.rng.choice(ALARM_NAMES), "not" if self.rng.random() < 0.2 else "")
        return Atom("state", self.rng.choice(STATE_KEYS), self.rng.choice(("", "not")))

    def action(self) -> str:
        return self.rng.choice(self.by_group[self.rng.choices(GROUPS, GROUP_P)[0]])

    def rule(self) -> Rule:
        return Rule(tuple(self.atom() for _ in range(self.rng.choice((1, 1, 2)))), self.action())

    def genome(self) -> Genome:
        return normalize(tuple(self.rule() for _ in range(self.rng.randint(1, 4))))

    # ---------------------------------------------------------------- operators
    def _shift(self, a: Atom, steps: int) -> Atom:
        tab = self.q[a.key]
        i = min(range(len(tab)), key=lambda j: abs(tab[j] - a.thr))
        return Atom(a.kind, a.key, a.op, tab[min(max(i + steps, 0), len(tab) - 1)])

    def mutate(self, g: Genome) -> Genome:
        rng = self.rng
        rules = list(g)
        i = rng.randrange(len(rules))
        r = rules[i]
        atoms = list(r.atoms)
        j = rng.randrange(len(atoms))
        a = atoms[j]
        ops = ["threshold", "feature", "operator", "atom", "action", "add_atom", "drop_atom", "add_rule",
               "remove_rule", "swap"]
        op = rng.choices(ops, (3, 1, 1, 2, 2, 1, 1, 1.5, 1, 1))[0]
        if op == "threshold" and a.kind == "num":
            atoms[j] = self._shift(a, rng.choice((-2, -1, 1, 2)))
        elif op == "feature" and a.kind == "num":
            k = rng.choice(self.features)
            lvl = self.q[a.key].index(a.thr) / max(len(self.q[a.key]) - 1, 1) if a.thr in self.q[a.key] else rng.random()
            tab = self.q[k]
            atoms[j] = Atom("num", k, a.op, tab[round(lvl * (len(tab) - 1))])
        elif op == "operator":
            if a.kind == "num":
                atoms[j] = Atom(a.kind, a.key, "<" if a.op == ">" else ">", a.thr)
            else:
                atoms[j] = Atom(a.kind, a.key, "" if a.op == "not" else "not", a.thr)
        elif op == "action":
            rules[i] = Rule(r.atoms, self.action())
            return normalize(tuple(rules))
        elif op == "add_atom" and len(atoms) < MAX_ATOMS:
            atoms.append(self.atom())
        elif op == "drop_atom" and len(atoms) > 1:
            atoms.pop(j)
        elif op == "add_rule" and len(rules) < self.max_rules:
            rules.insert(rng.randint(0, len(rules)), self.rule())
            return normalize(tuple(rules))
        elif op == "remove_rule" and len(rules) > 1:
            rules.pop(i)
            return normalize(tuple(rules))
        elif op == "swap" and len(rules) > 1:
            k = rng.randrange(len(rules))
            rules[i], rules[k] = rules[k], rules[i]
            return normalize(tuple(rules))
        else:  # operator not applicable: replace the atom
            atoms[j] = self.atom()
        rules[i] = Rule(tuple(atoms), r.action)
        return normalize(tuple(rules))

    def crossover(self, a: Genome, b: Genome) -> Genome:
        """One-point crossover on rule lists: head of `a` + tail of `b`."""
        i = self.rng.randint(0, len(a))
        j = self.rng.randint(0, len(b))
        child = a[:i] + b[j:]
        return normalize(child or a[:1])


# --------------------------------------------------------------------------- feature quantiles
class _Recorder:
    name = "recorder"

    def episode_start(self, plant, scenario, seed):
        self.rows: list[dict] = []

    def decide(self, plant, f, trigger, history):
        self.rows.append({k: float(f[k]) for k in NUM_FEATURES})
        return Decision(NO_ACTION, "none", source="recorder")

    def revise(self, plant, f, rejected, reasons, history):
        return None

    def episode_end(self, plant, outcome):
        pass


def _record_job(args):
    sc_d, seed = args
    rec = _Recorder()
    run_episode(Scenario.from_dict(sc_d), seed, rec)
    return rec.rows


def feature_quantiles(batch: list[dict], workers: int = 6) -> dict[str, tuple[float, ...]]:
    """Empirical quantiles of NUM_FEATURES at supervisor invocations of no-action episodes."""
    rows = [r for rs in _pool(workers).map(_record_job, [(b["scenario"], b["seed"]) for b in batch], chunksize=1)
            for r in rs]
    out = {}
    for k in NUM_FEATURES:
        if k in INT_THRESHOLDS:
            continue
        v = np.array([r[k] for r in rows], dtype=float)
        v = v[np.isfinite(v)]
        if len(v):
            out[k] = tuple(sorted({float(f"{x:.5g}") for x in np.quantile(v, QUANT_LEVELS)}))
    return out


# --------------------------------------------------------------------------- GA
@dataclass
class Individual:
    genome: Genome
    src: str
    gen: int
    evals_at_birth: int
    score: float = -math.inf
    sr: float = 0.0
    table: str = ""
    eval: EvalResult | None = field(default=None, repr=False)

    def rank(self) -> tuple:
        return (-self.score, len(self.genome), sum(len(r.atoms) for r in self.genome), self.src)


def aggregate(eps: list[dict]) -> EvalResult:
    """evaluate()'s aggregation applied to cached episodes (gives the identical score)."""
    fam: dict = {}
    for e in eps:
        d = fam.setdefault(e["family"], {"ok": 0, "n": 0})
        d["n"] += 1
        d["ok"] += int(e["success"])
    sr = sum(e["success"] for e in eps) / max(len(eps), 1)
    benign_acts = [e["n_accepted"] for e in eps if e["label"] == "benign"]
    pen = ACTION_PENALTY * (sum(benign_acts) / len(benign_acts)) if benign_acts else 0.0
    return EvalResult(sr - pen, sr, fam, eps, sum(e["errors"] for e in eps))


SLIM_KEYS = ("id", "family", "seed", "label", "success", "stabilized", "trip_time", "shutdown", "g_iae", "cost",
             "n_accepted", "errors")


def _slim(e: dict) -> dict:
    out = {k: e[k] for k in SLIM_KEYS}
    out["fired"] = [d["source"] for d in e.get("decisions", []) if d.get("accepted")]
    return out


class GPSynthesiser:
    def __init__(self, out_dir: str, train_batch: list[dict], budget_evals: int = 20_000,
                 checkpoints: tuple[int, ...] = (2_500, 5_000, 10_000, 20_000), pop_size: int = 24, elite: int = 2,
                 tournament: int = 3, p_crossover: float = 0.7, p_mutation: float = 0.9, max_rules: int = MAX_RULES,
                 max_generations: int | None = None, falsify: bool = True, falsify_every: int = 1, falsify_n: int = 48,
                 falsify_families=("A", "S", "P", "Q", "SN"), batch_cap: int = 140, seed: int = 0, workers: int = 6,
                 quantiles: dict | None = None, resume: bool = True, log=print):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.batch = list(train_batch)
        self.budget = budget_evals
        self.checkpoints = sorted(c for c in checkpoints if c <= budget_evals)
        self.pop_size, self.elite, self.tournament = pop_size, elite, tournament
        self.p_crossover, self.p_mutation = p_crossover, p_mutation
        self.max_generations = max_generations
        self.falsify, self.falsify_every, self.falsify_n, self.falsify_families = falsify, falsify_every, falsify_n, falsify_families
        self.batch_cap, self.seed, self.workers = batch_cap, seed, workers
        self.rng = random.Random(seed)
        self.var = Variation(self.rng, quantiles, max_rules)
        self.log = log
        self.cache: dict[tuple, dict] = {}
        self.evals = 0
        self.label_evals = 0
        self.gen = 0
        self.pop: list[Individual] = []
        self.offspring: list[Individual] = []
        self.history: list[dict] = []
        self.falsify_log: list[dict] = []
        self.next_ckpt = 0
        self.falsify_round = 0
        self.exhausted = False
        self.wall = 0.0
        self._t0 = time.time()
        if resume:
            self._load_state()

    # ------------------------------------------------------------- evaluation
    @staticmethod
    def _key(src: str, b: dict) -> tuple:
        return (hashlib.sha1(src.encode()).hexdigest(), b["scenario"]["id"], b["seed"], b.get("label"))

    def _cost(self, src: str) -> int:
        return sum(self._key(src, b) not in self.cache for b in self.batch)

    def _run(self, srcs: list[str]) -> None:
        """Simulate every missing (source, batch item) episode in one parallel map (evaluate()'s job)."""
        jobs, keys, seen = [], [], set()
        for s in srcs:
            check_source(s)
            for b in self.batch:
                k = self._key(s, b)
                if k not in self.cache and k not in seen:
                    seen.add(k)
                    keys.append(k)
                    jobs.append((s, b["scenario"], b["seed"], b.get("label"), {}))
        if not jobs:
            return
        eps = _pool(self.workers).map(_job, jobs, chunksize=1)
        self.evals += len(jobs)
        for k, e in zip(keys, eps):
            self.cache[k] = _slim(e)

    def _score(self, ind: Individual) -> None:
        ev = aggregate([self.cache[self._key(ind.src, b)] for b in self.batch])
        ind.score, ind.sr, ind.table, ind.eval = ev.score, ev.success_rate, ev.table(), ev

    def _room(self, need: int) -> int:
        """Evaluations available before the next checkpoint mark or the budget end. A mark that `need` would
        cross is saved first, so checkpoint E<n> never uses more than n evaluations."""
        while self.next_ckpt < len(self.checkpoints) and self.evals + need > self.checkpoints[self.next_ckpt]:
            if self.incumbent() is not None:
                self._save(f"E{self.checkpoints[self.next_ckpt]}")
            self.next_ckpt += 1
        limit = self.checkpoints[self.next_ckpt] if self.next_ckpt < len(self.checkpoints) else self.budget
        return min(limit, self.budget) - self.evals

    def _evaluate_new(self, inds: list[Individual]) -> list[Individual]:
        """Evaluate individuals in parallel groups within the budget; individuals that do not fit are dropped."""
        pending, done = list(inds), []
        while pending:
            c0 = self._cost(pending[0].src)
            if self.evals + c0 > self.budget:
                self.exhausted = True
                break
            room = self._room(c0)
            take, cost = [], 0
            while pending and cost + self._cost(pending[0].src) <= room:
                cost += self._cost(pending[0].src)
                take.append(pending.pop(0))
            self._run([i.src for i in take])
            for i in take:
                self._score(i)
            done += take
            self.offspring += take
        return done

    def incumbent(self) -> Individual | None:
        cands = [i for i in self.pop + self.offspring if i.eval is not None]
        return min(cands, key=Individual.rank) if cands else None

    # ------------------------------------------------------------- variation
    def _new(self, g: Genome) -> Individual:
        return Individual(g, compile_genome(g), self.gen, self.evals)

    def _select(self) -> Individual:
        return min(self.rng.sample(self.pop, min(self.tournament, len(self.pop))), key=Individual.rank)

    def _breed(self, n: int, taken: set[str]) -> list[Individual]:
        kids = []
        for _ in range(n):
            for _attempt in range(20):
                a = self._select()
                g = self.var.crossover(a.genome, self._select().genome) if self.rng.random() < self.p_crossover else a.genome
                if self.rng.random() < self.p_mutation or g == a.genome:
                    g = self.var.mutate(g)
                ind = self._new(g)
                if ind.src not in taken:
                    break
            else:
                ind = self._new(self.var.genome())
                if ind.src in taken:
                    continue
            taken.add(ind.src)
            kids.append(ind)
        return kids

    # ------------------------------------------------------------- falsification (as Synthesiser)
    def _falsification_round(self) -> tuple[int, int] | None:
        if self._room(self.falsify_n) < self.falsify_n:
            self.exhausted = self.evals + self.falsify_n > self.budget
            return None
        self.falsify_round += 1
        inc = self.incumbent()
        rng = np.random.default_rng([self.seed, 9000 + self.falsify_round])
        fresh = []
        for i in range(self.falsify_n):
            fam = self.falsify_families[i % len(self.falsify_families)]
            sc = sample(fam, rng, idx=50_000 + 1000 * self.falsify_round + i)
            fresh.append({"scenario": sc.as_dict(), "seed": 7000 + 1000 * self.falsify_round + i, "label": None})
        ev = evaluate(inc.src, fresh, workers=self.workers)
        self.evals += len(fresh)
        failing = [(b, e) for b, e in zip(fresh, ev.episodes) if not e["success"]]
        labels = label_scenarios([b for b, _ in failing], workers=self.workers) if failing else []
        self.label_evals += len(failing)
        new = []
        for (b, e), lab in zip(failing, labels):
            b["label"] = lab
            if lab in ("benign", "quality_gap", "rescuable"):  # unrescuable/benign_offspec cannot be fixed: not added
                e = dict(e, label=lab, success=success(e, lab))
                self.cache[self._key(inc.src, b)] = _slim(e)  # the incumbent's episode is already known
                new.append(b)
        # the population must be re-scored on the new cases: keep only as many as the budget allows
        others = [i for i in self.pop if i.src != inc.src]
        self._room(len(others) * len(new))  # save a checkpoint mark the re-scoring would cross
        room = self.budget - self.evals
        if others and len(others) * len(new) > room:
            new = new[: room // len(others)]
        self.batch.extend(new)
        if len(self.batch) > self.batch_cap:  # keep the newest hard cases
            self.batch = self.batch[-self.batch_cap:]
        self.log(f"  falsification round {self.falsify_round}: incumbent failed {len(new)}/{len(fresh)} fresh scenarios; "
                 f"batch -> {len(self.batch)}")
        self._rescore()
        self.falsify_log.append({"round": self.falsify_round, "gen": self.gen, "n_new": len(new), "n_fresh": len(fresh),
                                 "n_failing": len(failing), "labels": labels, "evals": self.evals, "batch": len(self.batch)})
        return len(new), len(fresh)

    def _rescore(self):
        self._run([i.src for i in self.pop])
        for i in self.pop:
            self._score(i)
        self.pop.sort(key=Individual.rank)

    # ------------------------------------------------------------- persistence
    def _save(self, name: str):
        best = self.incumbent()
        hdr = (f"synthesised by GP (B5); episode evaluations {self.evals}; generation {best.gen}; seed {self.seed}\n"
               f"# training-batch success {best.sr:.3f} ({best.table})")
        with open(os.path.join(self.out, f"playbook_{name}.py"), "w") as f:
            f.write(compile_genome(best.genome, hdr))
        meta = {"name": name, "method": "gp", "evals": self.evals, "label_evals": self.label_evals, "generation": self.gen,
                "best_gen": best.gen, "best_evals_at_birth": best.evals_at_birth, "best_sr": best.sr,
                "best_score": best.score, "table": best.table, "loc": loc(best.src), "rules": rule_names(best.src),
                "n_rules": len(best.genome), "genome": genome_json(best.genome), "batch_size": len(self.batch),
                "seed": self.seed, "budget_evals": self.budget, "wall_s": self.wall + time.time() - self._t0,
                "ga": {"pop_size": self.pop_size, "elite": self.elite, "tournament": self.tournament,
                       "p_crossover": self.p_crossover, "p_mutation": self.p_mutation, "max_rules": self.var.max_rules,
                       "falsify": self.falsify, "falsify_every": self.falsify_every, "falsify_n": self.falsify_n,
                       "batch_cap": self.batch_cap},
                "history": self.history, "falsification": self.falsify_log}
        json.dump(meta, open(os.path.join(self.out, f"meta_{name}.json"), "w"), indent=1)
        self.log(f"  checkpoint {name}: evals={self.evals} best gen={best.gen} sr={best.sr:.3f} {best.table}")

    _STATE = ("batch", "cache", "evals", "label_evals", "gen", "pop", "history", "falsify_log", "next_ckpt",
              "falsify_round")

    def _save_state(self):
        st = {k: getattr(self, k) for k in self._STATE}
        st.update(pop=[dataclasses.replace(i, eval=None) for i in self.pop], rng=self.rng.getstate(),
                  wall=self.wall + time.time() - self._t0)
        p = os.path.join(self.out, "state.pkl")
        with open(p + ".tmp", "wb") as f:
            pickle.dump(st, f)
        os.replace(p + ".tmp", p)

    def _load_state(self):
        p = os.path.join(self.out, "state.pkl")
        if not os.path.exists(p):
            return
        with open(p, "rb") as f:
            st = pickle.load(f)
        for k in self._STATE:
            setattr(self, k, st[k])
        self.rng.setstate(st["rng"])
        self.wall = st["wall"]
        for i in self.pop:
            self._score(i)
        self.log(f"resumed at generation {self.gen}: evals={self.evals}, batch={len(self.batch)}")

    # ------------------------------------------------------------- run
    def _log_gen(self):
        best = self.pop[0]
        scores = [i.score for i in self.pop]
        rec = {"gen": self.gen, "evals": self.evals, "label_evals": self.label_evals, "best_score": best.score,
               "best_sr": best.sr, "mean_score": float(np.mean(scores)), "best_rules": len(best.genome),
               "batch": len(self.batch), "wall_s": round(self.wall + time.time() - self._t0, 1)}
        self.history.append(rec)
        self.log(f"[gen {self.gen}] best sr={best.sr:.3f} score={best.score:.3f} rules={len(best.genome)} {best.table} | "
                 f"mean score {rec['mean_score']:.3f} | evals={self.evals} batch={len(self.batch)} ({rec['wall_s']:.0f}s)")

    def _done(self) -> bool:
        return self.exhausted or self.evals >= self.budget or \
            (self.max_generations is not None and self.gen >= self.max_generations)

    def run(self) -> Individual:
        if not self.pop:
            taken: set[str] = set()
            init = []
            for _ in range(50 * self.pop_size):
                if len(init) >= self.pop_size:
                    break
                ind = self._new(self.var.genome())
                if ind.src not in taken:
                    taken.add(ind.src)
                    init.append(ind)
            self.pop = self._evaluate_new(init)
            self.offspring = []
            if not self.pop:
                raise RuntimeError("budget too small to evaluate one individual")
            self.pop.sort(key=Individual.rank)
            self._log_gen()
            self._save("g0")
            self._save_state()
        while not self._done():
            self.gen += 1
            elites = self.pop[: self.elite]
            kids = self._breed(self.pop_size - len(elites), {i.src for i in elites})
            kids = self._evaluate_new(kids)
            self.offspring = []
            if not kids:  # budget exhausted before any offspring could be evaluated
                self.gen -= 1
                break
            # generational replacement with elitism; a generation cut short by the budget is filled from the old one
            rest = self.pop[self.elite:] if len(kids) < self.pop_size - len(elites) else []
            self.pop = sorted(elites + kids + rest, key=Individual.rank)[: self.pop_size]
            self._log_gen()
            if self.falsify and (self.gen % self.falsify_every == 0 or not self.pop[0].eval.failures()):
                self._falsification_round()
            elif not self.falsify and not self.pop[0].eval.failures():
                break
            self._save_state()
        if self.exhausted or self.evals >= self.budget:  # the incumbent is final for every remaining mark
            self._room(self.budget + 1)
        self._save("final")
        self._save_state()
        return self.pop[0]
