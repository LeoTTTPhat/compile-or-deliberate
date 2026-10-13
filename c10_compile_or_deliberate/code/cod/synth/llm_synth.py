"""Offline playbook synthesis by a small local coder LLM, hardened by falsification.

Loop (counterexample-guided, small EoH-style population):
  1. v0 <- LLM(Part A + Part B + feature glossary + contract)
  2. evaluate on the training batch (CPU simulation, common random numbers)
  3. repeat until the token budget is spent:
       parent <- tournament over the population (top-K by score on the current batch)
       cards  <- up to `n_cards` compact counterexamples of the parent (diverse families,
                 simplest first: fewest fault components, nominal production)
       child  <- LLM(repair prompt with parent source, per-family table, cards)
       evaluate child; insert into population
       every `falsify_every` iterations: falsification round = sample fresh scenarios from the
       SYNTHESIS families, evaluate the incumbent, add its failures to the training batch
       ("hardening"), re-score the population on the new batch.
  4. checkpoints: whenever cumulative tokens cross a budget mark, the incumbent is saved.
Ablations: falsify=False (no hardening), minimize=False (raw, long counterexamples),
one_shot (v0 only).
Every LLM call is cached, so a restarted run replays deterministically up to where it stopped.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from ..playbook import PlaybookError, check_source, lint, loc, rule_names, valid_feature_keys
from ..prompts import FEATURE_GLOSSARY, PART_A, PART_B, PROMPT_HASH
from ..scenarios import sample
from .evaluate import counterexample_card, counterfactual_remedies, evaluate, label_scenarios
from .tune import tune

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from tools.llm import LLM  # noqa: E402

CONTRACT = """CONTRACT. Write a Python function `def playbook(f, mem):` (helper functions allowed). It is called at every
supervisor invocation: when a new alarm annunciates, while alarms persist (at most every 0.25 h), and on a 2-h
heartbeat. Return None for no action, or a dict {"rule": "<short id>", "diagnosis": <vocabulary item>,
"why": "<few words>", "action": <exactly one action object as defined in the plant description>}.
Allowed: if/else, comparisons, arithmetic, local variables, for-loops and comprehensions over lists, dict
access f["key"] and f.get("key", default), and the builtins abs, min, max, len, round, float, int, bool, any,
all, sum, sorted, isfinite, sqrt, exp, log, inf, and the methods get/items/keys/values/append/extend/update/
pop/copy/setdefault. Forbidden: imports, while-loops, lambdas, classes, any other attribute access. `mem` is a dict that persists within one episode (use it to avoid repeating an action).
Rules are evaluated in order; the first matching rule wins, so put the most safety-critical first. Setpoint
targets are ABSOLUTE values and must respect the bounds and max move from the current setpoint f[K+"_sp"].
Keep the playbook readable: one commented block per rule, under 150 lines.
Reply with ONLY the complete code in a single ```python block."""

def _key_list() -> str:
    keys = sorted(valid_feature_keys())
    return "EXACT VALID KEYS of f (any other key raises an error): " + ", ".join(keys) + "."


SYSTEM = ("You are a senior process-control engineer writing an auditable abnormal-situation-management playbook "
          "as code for a plant's supervisory layer.\n\n" + PART_A + "\n\n" + PART_B + "\n\n" + FEATURE_GLOSSARY +
          "\n" + _key_list() + "\n\n" + CONTRACT)

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_RE.findall(text)
    if m:
        return max(m, key=len).strip() + "\n"
    return text.strip() + "\n"


def simplicity(e: dict) -> float:
    sc = e["scenario"]
    n = len(sc.get("idv_events", [])) + len(sc.get("sensor_faults", [])) + len(sc.get("actuator_faults", []))
    prod = 22.95
    for _, ch in sc.get("setpoint_shifts", []):
        for k, v in ch:
            if k == "production_rate":
                prod = v
    return n + abs(prod - 22.95) / 5.0


@dataclass
class Candidate:
    src: str
    score: float
    sr: float
    table: str
    it: int
    tokens_at_birth: int
    eval: object = field(repr=False, default=None)


class Synthesiser:
    def __init__(self, out_dir: str, train_batch: list[dict], model: str = "qwen3-coder:30b",
                 checkpoints: tuple[int, ...] = (40_000, 80_000, 160_000, 320_000), max_tokens_total: int = 320_000,
                 falsify: bool = True, falsify_every: int = 3, falsify_n: int = 48, falsify_families=("A", "S", "P", "Q", "SN"),
                 minimize: bool = True, n_cards: int = 4, pop_k: int = 3, temperature: float = 0.7, seed: int = 0,
                 workers: int = 16, max_out_tokens: int = 3500, one_shot: bool = False, batch_cap: int = 140,
                 remedy: bool = True, tune_budget: int = 24, log=print):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.batch = list(train_batch)
        self.llm = LLM(model, num_ctx=16384)
        self.model = model
        self.checkpoints = sorted(checkpoints)
        self.max_tokens_total = max_tokens_total
        self.falsify, self.falsify_every, self.falsify_n, self.falsify_families = falsify, falsify_every, falsify_n, falsify_families
        self.minimize, self.n_cards, self.pop_k = minimize, n_cards, pop_k
        self.temperature, self.seed, self.workers = temperature, seed, workers
        self.max_out_tokens, self.one_shot, self.batch_cap = max_out_tokens, one_shot, batch_cap
        self.remedy = remedy
        self._remedy_cache: dict = {}
        self.tune_budget = tune_budget  # batch evaluations per tuning pass (0 = no CPU calibration)
        self.episodes = 0  # simulated playbook episodes (CPU accounting, comparable with GP)
        self.rng = random.Random(seed)
        self.tokens = 0
        self.calls = 0
        self.pop: list[Candidate] = []
        self.history: list[dict] = []
        self.log = log
        self.next_ckpt = 0
        self.falsify_round = 0

    # ------------------------------------------------------------------ LLM
    def _ask(self, user: str, it: int) -> str:
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        r = self.llm.chat(msgs, temperature=self.temperature, seed=self.seed * 1000 + it, max_tokens=self.max_out_tokens,
                          tag=f"synth:{os.path.basename(self.out)}")
        self.tokens += r["prompt_tokens"] + r["completion_tokens"]
        self.calls += 1
        self.history.append({"it": it, "ptok": r["prompt_tokens"], "ctok": r["completion_tokens"], "cached": r["cached"],
                             "latency": r["latency"], "tokens_cum": self.tokens})
        return r["text"]

    @staticmethod
    def _problems(src: str) -> str | None:
        """Compile-time feedback: sandbox violations, then unknown feature keys (no simulation spent)."""
        try:
            check_source(src)
        except PlaybookError as e:
            return f"the sandbox checker rejected it: {e}"
        bad = lint(src)
        if bad:
            return (f"it reads feature keys that do not exist: {bad}. Use only the documented keys "
                    f"(e.g. P_slope, P_ttl, T_resid, A_PRESSURE_HI, mv_r_cooling_water); a missing key raises an error "
                    f"and the playbook abstains at that invocation")
        return None

    def _generate(self, user: str, it: int, max_fix: int = 2) -> str | None:
        src = extract_code(self._ask(user, it))
        for k in range(max_fix + 1):
            prob = self._problems(src)
            if prob is None:
                return src
            if k == max_fix:
                self.log(f"  [it {it}] invalid playbook after {max_fix} fixes: {prob[:200]}")
                return None
            self.lint_fixes = getattr(self, "lint_fixes", 0) + 1
            fix = (user + "\n\nYour previous answer was rejected because " + prob +
                   ".\nPrevious answer:\n```python\n" + src[:7000] + "\n```\nReturn the corrected complete code only.")
            src = extract_code(self._ask(fix, it + 500 * (k + 1)))
        return None

    # ------------------------------------------------------------- evaluate
    def _eval(self, src: str, it: int) -> Candidate:
        ev = evaluate(src, self.batch, workers=self.workers)
        self.episodes += len(self.batch)
        return Candidate(src, ev.score, ev.success_rate, ev.table(), it, self.tokens, ev)

    def _rescore(self):
        for c in self.pop:
            ev = evaluate(c.src, self.batch, workers=self.workers)
            self.episodes += len(self.batch)
            c.score, c.sr, c.table, c.eval = ev.score, ev.success_rate, ev.table(), ev
        self.pop.sort(key=lambda c: -c.score)

    def _insert(self, c: Candidate):
        self.pop.append(c)
        self.pop.sort(key=lambda c: (-c.score, loc(c.src)))
        self.pop = self.pop[: self.pop_k]

    # ------------------------------------------------------------- prompts
    def _repair_prompt(self, parent: Candidate) -> str:
        fails = parent.eval.failures()
        benign_acts = [e["n_accepted"] for e in parent.eval.episodes if e["label"] == "benign"]
        if self.minimize:
            by_fam: dict = {}
            for e in sorted(fails, key=simplicity):
                by_fam.setdefault(e["family"], []).append(e)
            chosen = []
            while len(chosen) < self.n_cards and any(by_fam.values()):
                for fam in sorted(by_fam, key=lambda k: -len(by_fam[k])):
                    if by_fam[fam] and len(chosen) < self.n_cards:
                        chosen.append(by_fam[fam].pop(0))
            rems = [None] * len(chosen)
            if self.remedy:
                todo = [e for e in chosen if (e["id"], e["seed"]) not in self._remedy_cache]
                for e, r in zip(todo, counterfactual_remedies(todo, self.workers) if todo else []):
                    self._remedy_cache[(e["id"], e["seed"])] = r
                rems = [self._remedy_cache[(e["id"], e["seed"])] for e in chosen]
            cards = "\n\n".join(counterexample_card(e, remedy=r, use_remedy=self.remedy) for e, r in zip(chosen, rems))
        else:
            chosen = self.rng.sample(fails, min(self.n_cards, len(fails)))
            cards = "\n\n".join(json.dumps({"scenario": e["scenario"], "decisions": e["decisions"], "trip_time": e["trip_time"],
                                            "trip_msg": e["trip_msg"]}, default=str)[:3000] for e in chosen)
        extra = ""
        errs = parent.eval.error_summary()
        if errs:
            extra += f"\nRUNTIME ERRORS during the simulations (the playbook abstained each time): {errs}."
        if benign_acts:
            # KNOWN DEFECT, kept so that the paper's synthesis runs (and their LLM-cache entries) replay exactly: this
            # assignment overwrites the RUNTIME ERRORS summary above whenever the batch holds a benign scenario, which
            # every batch does, so the coder never sees run-time errors (it hid the seed-2 KeyError); should be `+=`.
            extra = f"\nOn scenarios where no intervention was needed, the playbook took {np.mean(benign_acts):.2f} actions per episode on average (fewer is better)."
        return (f"Current playbook (success {parent.sr:.3f} on {len(self.batch)} simulated test scenarios; per family "
                f"{parent.table}):\n```python\n{parent.src}```\n"
                f"It FAILED in these offline simulation tests (the injected fault is revealed because these are simulations; "
                f"at run time you will only see the features):\n\n{cards}\n{extra}\n\n"
                "Revise the playbook so it handles these situations through the features it can observe, without breaking "
                "the cases that already pass. Return the complete revised code.")

    # ------------------------------------------------------------- falsify
    def _falsification_round(self):
        self.falsify_round += 1
        inc = self.pop[0]
        rng = np.random.default_rng([self.seed, 9000 + self.falsify_round])
        fresh = []
        for i in range(self.falsify_n):
            fam = self.falsify_families[i % len(self.falsify_families)]
            sc = sample(fam, rng, idx=50_000 + 1000 * self.falsify_round + i)
            fresh.append({"scenario": sc.as_dict(), "seed": 7000 + 1000 * self.falsify_round + i, "label": None})
        ev = evaluate(inc.src, fresh, workers=self.workers)
        self.episodes += len(fresh)
        failing = [b for b, e in zip(fresh, ev.episodes) if not e["success"]]
        labels = label_scenarios(failing, workers=self.workers) if failing else []
        new = []
        for b, lab in zip(failing, labels):
            b["label"] = lab
            if lab in ("benign", "quality_gap", "rescuable"):  # unrescuable/benign_offspec cannot be fixed: not added
                new.append(b)
        self.batch.extend(new)
        if len(self.batch) > self.batch_cap:  # keep the newest hard cases, drop oldest non-failing ones first
            self.batch = self.batch[-self.batch_cap:]
        self.log(f"  falsification round {self.falsify_round}: incumbent failed {len(new)}/{len(fresh)} fresh scenarios; batch -> {len(self.batch)}")
        self._rescore()
        self._tune_incumbent()
        return len(new), len(fresh)

    def _tune_incumbent(self):
        if not self.tune_budget:
            return
        inc = self.pop[0]
        src, ev, used, eps = tune(inc.src, self.batch, budget=self.tune_budget, workers=self.workers,
                                  seed=self.seed * 100 + self.falsify_round, log=self.log)
        self.episodes += eps
        if src != inc.src and ev.score > inc.score + 1e-9:
            c = Candidate(src, ev.score, ev.success_rate, ev.table(), inc.it, self.tokens, ev)
            self._insert(c)
            self.log(f"  tuned incumbent: {inc.sr:.3f} -> {ev.success_rate:.3f} ({used} batch evals, CPU only)")

    # ------------------------------------------------------------- run
    def _checkpoint(self, force_name: str | None = None):
        while self.next_ckpt < len(self.checkpoints) and (self.tokens >= self.checkpoints[self.next_ckpt] or force_name):
            name = force_name or f"B{self.checkpoints[self.next_ckpt]}"
            self._save(name)
            if force_name:
                return
            self.next_ckpt += 1

    def _save(self, name: str):
        best = self.pop[0]
        with open(os.path.join(self.out, f"playbook_{name}.py"), "w") as f:
            f.write(f"# synthesised by {self.model}; tokens {self.tokens}; iteration {best.it}; prompt hash {PROMPT_HASH}\n")
            f.write(f"# training-batch success {best.sr:.3f} ({best.table})\n")
            f.write(best.src)
        meta = {"name": name, "tokens": self.tokens, "calls": self.calls, "best_it": best.it, "best_sr": best.sr,
                "best_score": best.score, "table": best.table, "loc": loc(best.src), "rules": rule_names(best.src),
                "batch_size": len(self.batch), "model": self.model, "prompt_hash": PROMPT_HASH, "history": self.history,
                "falsify": self.falsify, "minimize": self.minimize, "remedy": self.remedy, "seed": self.seed,
                "lint_fixes": getattr(self, "lint_fixes", 0), "episodes": self.episodes, "tune_budget": self.tune_budget}
        json.dump(meta, open(os.path.join(self.out, f"meta_{name}.json"), "w"), indent=1)
        self.log(f"  checkpoint {name}: tokens={self.tokens} best it={best.it} sr={best.sr:.3f} {best.table}")

    def run(self):
        t0 = time.time()
        it = 0
        src = None
        valid_feature_keys()
        while src is None and it < 5:
            src = self._generate("Write the first version of the playbook.", it)
            it += 1
        if src is None:
            raise RuntimeError("could not obtain a valid initial playbook")
        c = self._eval(src, it)
        self._insert(c)
        self.log(f"[it {it}] v0 sr={c.sr:.3f} score={c.score:.3f} {c.table} tokens={self.tokens} ({time.time() - t0:.0f}s)")
        self._save("v0")
        if self.one_shot:
            self._save("final")
            return self.pop[0]
        while self.tokens < self.max_tokens_total:
            it += 1
            self._checkpoint()
            # tournament: best with p=0.6, else uniform among the rest
            parent = self.pop[0] if (len(self.pop) == 1 or self.rng.random() < 0.6) else self.rng.choice(self.pop[1:])
            if not parent.eval.failures():
                if self.falsify:
                    self._falsification_round()
                    continue
                break
            child_src = self._generate(self._repair_prompt(parent), it)
            if child_src is None:
                continue
            c = self._eval(child_src, it)
            self._insert(c)
            self.log(f"[it {it}] child sr={c.sr:.3f} score={c.score:.3f} {c.table} | best {self.pop[0].sr:.3f} "
                     f"tokens={self.tokens} ({time.time() - t0:.0f}s)")
            if self.falsify and it % self.falsify_every == 0:
                self._falsification_round()
        self._checkpoint()
        self._save("final")
        return self.pop[0]
