"""Hybrid compile-then-deliberate supervisor (the H arms).

At each invocation the compiled playbook proposes an action at zero tokens. The online
LLM deliberator is consulted only when a novelty trigger fires:
  (a) unexplained abnormality: alarms active, or a constraint time-to-limit < `ttl_h`, or a
      large consistency residual, while NO playbook rule fires;
  (b) the playbook's proposal fails the forward check on the nominal twin (it would be
      vetoed), i.e. the anticipated response does not fit the situation;
  (c) conformal novelty: the feature vector is out of the synthesis distribution
      (NoveltyDetector, split-conformal threshold at level alpha).
Escalations are rate-limited (`cooldown_h`, `max_escalations`) and paid from a per-episode
token bucket of size (1-f)B/N. The LLM sees the playbook's proposal and the trigger in
its card. Without escalation the playbook decision stands.

Spec: hybrid:<playbook.py>:<model>@<per_episode_tokens>[:trig=abc,cool=1.0,maxesc=3,ttl=4,alpha=0.05]
"""
from __future__ import annotations

import json

from ..actions import NO_ACTION
from ..playbook import PlaybookSupervisor
from ..runner import Decision
from ..summary import render
from ..twin import shadow_check
from .llm_online import OnlineLLMSupervisor

_NOV = None


def novelty_detector():
    global _NOV
    if _NOV is None:
        try:
            from ..fdd import NoveltyDetector
            _NOV = NoveltyDetector.load()
        except Exception:  # noqa: BLE001 - detector optional (trigger c disabled)
            _NOV = False
    return _NOV or None


class HybridSupervisor:
    def __init__(self, playbook_src: str, model: str, per_episode_tokens: float | None, *, triggers: str = "abc",
                 cooldown_h: float = 1.0, max_escalations: int = 3, ttl_h: float = 4.0, resid_P: float = 40.0,
                 resid_T: float = 4.0, name: str = "hybrid"):
        self.pb = PlaybookSupervisor(playbook_src, name="pb")
        self.llm = OnlineLLMSupervisor(model, per_episode_tokens=per_episode_tokens, name=name + ":llm")
        self.triggers, self.cooldown_h, self.max_esc = triggers, cooldown_h, max_escalations
        self.ttl_h, self.resid_P, self.resid_T = ttl_h, resid_P, resid_T
        self.name = name

    def episode_start(self, plant, scenario, seed):
        self.pb.episode_start(plant, scenario, seed)
        self.llm.episode_start(plant, scenario, seed)
        self.n_esc = 0
        self.last_esc = -1e9
        self.esc_log: list[dict] = []

    def _abnormal(self, f) -> bool:
        ttl = min(f[k + "_ttl"] for k in ("P", "T", "RL", "SL", "XL"))
        return bool(f["alarms"]) or ttl < self.ttl_h or abs(f["P_resid"]) > self.resid_P or abs(f["T_resid"]) > self.resid_T \
            or abs(f["G_cmd_err"]) > 3.0

    def _triggers(self, plant, f, d_pb):
        why = []
        if "a" in self.triggers and d_pb is None and self._abnormal(f):
            why.append("unexplained_abnormality")
        if "b" in self.triggers and d_pb is not None and d_pb.action.type not in ("no_action", "shutdown"):
            ok, reason, _ = shadow_check(plant, d_pb.action, 6.0, "nominal")
            if not ok:
                why.append("playbook_action_fails_forward_check")
        if "c" in self.triggers:
            nov = novelty_detector()
            if nov is not None and self._abnormal(f) and nov.is_novel(f):
                why.append("out_of_distribution")
        return why

    def decide(self, plant, f, trigger, history):
        d_pb, err = self.pb.propose(f, {"trigger": trigger})
        why = self._triggers(plant, f, d_pb)
        can = (why and self.n_esc < self.max_esc and f["t"] - self.last_esc >= self.cooldown_h - 1e-9
               and self.llm.bucket.can(self.llm.est))
        if not can:
            if d_pb is None:
                return Decision(NO_ACTION, "none", err or "no rule fired", source="playbook:none",
                                meta={"triggers": why} if why else {})
            d_pb.meta = {"triggers": why} if why else {}
            return d_pb
        self.n_esc += 1
        self.last_esc = f["t"]
        pb_txt = (f"The anticipated-situation playbook proposes {json.dumps(d_pb.action.as_dict())} (rule {d_pb.source})."
                  if d_pb is not None else "The anticipated-situation playbook has NO rule for this situation.")
        card = render(f, self.llm.recent, f"Escalated to you because: {', '.join(why)}. {pb_txt} "
                                           f"You may endorse the playbook's proposal, choose another action, or no_action.")
        d = self.llm._call(card)
        d.source = "escalation:llm"
        d.meta.update({"triggers": why, "playbook_proposal": d_pb.action.as_dict() if d_pb else None})
        if d.action.type != "no_action":
            self.llm.recent.append(f"t={f['t']:.2f} {json.dumps(d.action.as_dict())}")
        self.esc_log.append({"t": f["t"], "why": why, "action": d.action.as_dict(), "diagnosis": d.diagnosis})
        return d

    def revise(self, plant, f, rejected, reasons, history):
        if rejected.source.startswith("escalation") and self.llm.bucket.can(self.llm.est):
            card = render(f, self.llm.recent)
            fb = ("Your proposed action " + json.dumps(rejected.action.as_dict()) + " was REJECTED: " + "; ".join(reasons)
                  + ". Propose a corrected action (or no_action). JSON only.")
            d = self.llm._call(card, fb)
            d.source = "escalation:llm_revision"
            return d
        return self.pb.revise(plant, f, rejected, reasons, history)

    def episode_end(self, plant, outcome):
        outcome["escalations"] = self.esc_log


def build_hybrid(spec: str) -> HybridSupervisor:
    # hybrid:<path>:<model>@<tokens>[:k=v,...]   (model names contain ':' -> split carefully)
    body = spec[len("hybrid:"):]
    path, rest = body.split(".py:", 1)
    path += ".py"
    opts = {}
    if "@" in rest:
        model, tail = rest.split("@", 1)
        if ":" in tail:
            tok, optstr = tail.split(":", 1)
            for kv in optstr.split(","):
                k, v = kv.split("=")
                opts[k] = v
        else:
            tok = tail
        tokens = None if tok in ("inf", "") else float(tok)
    else:
        model, tokens = rest, None
    kw = {}
    if "trig" in opts:
        kw["triggers"] = opts["trig"]
    if "cool" in opts:
        kw["cooldown_h"] = float(opts["cool"])
    if "maxesc" in opts:
        kw["max_escalations"] = int(opts["maxesc"])
    if "ttl" in opts:
        kw["ttl_h"] = float(opts["ttl"])
    return HybridSupervisor(open(path).read(), model, tokens, name=spec, **kw)
