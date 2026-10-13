"""Online deliberation: a small local LLM decides at each invocation (Alhazmi-style, f = 0 arm).

The model receives Part A + Part B (system) and the process-state card (user) and must return
JSON {diagnosis, rationale, action} constrained by a JSON schema. A per-episode token bucket
caps online spending; when the bucket cannot cover a call the `fallback` supervisor decides
(hold by default). Every call is cached (tools/llm.py), so runs are exactly replayable.
"""
from __future__ import annotations

import json
import os
import sys

from ..actions import DECISION_SCHEMA, NO_ACTION, parse_action
from ..prompts import PART_A, PART_B
from ..runner import Decision
from ..summary import render

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
from tools.llm import LLM  # noqa: E402

INSTRUCTION = ("Decide the single best supervisory action now. Reply with JSON only: "
               '{"diagnosis": <one of the vocabulary>, "rationale": <at most 15 words>, "action": <one action object>}. '
               'If the regulatory layer is coping, use {"type": "no_action"}.')


class TokenBucket:
    def __init__(self, per_episode: float | None):
        self.per_episode = per_episode
        self.left = float("inf") if per_episode is None else float(per_episode)

    def refill(self):
        self.left = float("inf") if self.per_episode is None else float(self.per_episode)

    def can(self, est: float) -> bool:
        return self.left >= est

    def spend(self, n: float):
        self.left -= n


class OnlineLLMSupervisor:
    def __init__(self, model: str = "qwen2.5:7b", per_episode_tokens: float | None = None, fallback=None,
                 name: str | None = None, max_tokens: int = 220, est_call_tokens: float = 3600.0,
                 temperature: float = 0.0, system_extra: str = "", gate: bool = False):
        self.llm = LLM(model)
        self.model = model
        self.bucket = TokenBucket(per_episode_tokens)
        self.fallback = fallback
        self.name = name or f"online:{model}"
        self.max_tokens = max_tokens
        self.est = est_call_tokens
        self.temperature = temperature
        self.system = PART_A + "\n\n" + PART_B + (("\n\n" + system_extra) if system_extra else "")
        self.recent: list[str] = []
        self.gate = gate  # B6g: call the LLM only when a cheap abnormality test fires

    @staticmethod
    def abnormal(f, ttl_h: float = 4.0) -> bool:
        ttl = min(f[k + "_ttl"] for k in ("P", "T", "RL", "SL", "XL"))
        return bool(f["alarms"]) or ttl < ttl_h or abs(f["P_resid"]) > 40.0 or abs(f["T_resid"]) > 4.0 \
            or abs(f["G_cmd_err"]) > 3.0

    def episode_start(self, plant, scenario, seed):
        self.bucket.refill()
        self.recent = []
        self.ep_key = f"{scenario.id}|{seed}"
        self.k = 0
        if self.fallback is not None:
            self.fallback.episode_start(plant, scenario, seed)

    def _call(self, card: str, feedback: str | None = None) -> Decision:
        msgs = [{"role": "system", "content": self.system}, {"role": "user", "content": card + "\n\n" + INSTRUCTION}]
        if feedback:
            msgs.append({"role": "user", "content": feedback})
        self.k += 1
        r = self.llm.chat(msgs, temperature=self.temperature, seed=self.k, max_tokens=self.max_tokens,
                          fmt=DECISION_SCHEMA, tag=f"online:{self.name}")
        used = r["prompt_tokens"] + r["completion_tokens"]
        self.bucket.spend(used)
        try:
            obj = json.loads(r["text"])
            a, err = parse_action(obj.get("action"))
        except json.JSONDecodeError as e:
            obj, a, err = {}, None, f"invalid JSON: {e}"
        if a is None:
            a = NO_ACTION
        return Decision(a, str(obj.get("diagnosis", "unknown")), str(obj.get("rationale", ""))[:300],
                        prompt_tokens=r["prompt_tokens"], completion_tokens=r["completion_tokens"], llm_calls=1,
                        gpu_s=0.0 if r["cached"] else r["latency"], source="llm",
                        meta={"parse_error": err, "cached": r["cached"]})

    def decide(self, plant, f, trigger, history):
        if self.gate and not self.abnormal(f):
            return Decision(NO_ACTION, "none", "gate: no abnormality", source="gate")
        if not self.bucket.can(self.est):
            if self.fallback is not None:
                d = self.fallback.decide(plant, f, trigger, history)
                d.source = "fallback:" + d.source
                return d
            return Decision(NO_ACTION, "unknown", "online budget exhausted", source="budget_exhausted")
        budget = f"Supervisor budget: {12 - sum(1 for h in history if h['accepted'])} actions left this episode."
        card = render(f, self.recent, budget)
        d = self._call(card)
        if d.action.type != "no_action":
            self.recent.append(f"t={f['t']:.2f} {json.dumps(d.action.as_dict())}")
        return d

    def revise(self, plant, f, rejected, reasons, history):
        if rejected.source != "llm":
            return self.fallback.revise(plant, f, rejected, reasons, history) if self.fallback else None
        if not self.bucket.can(self.est):
            return None
        card = render(f, self.recent)
        fb = ("Your proposed action " + json.dumps(rejected.action.as_dict()) + " was REJECTED by the validator: "
              + "; ".join(reasons) + ". Propose a corrected action (or no_action). JSON only.")
        d = self._call(card, fb)
        return d

    def episode_end(self, plant, outcome):
        if self.fallback is not None:
            self.fallback.episode_end(plant, outcome)
