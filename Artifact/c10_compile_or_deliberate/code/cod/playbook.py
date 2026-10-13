"""Playbooks: auditable supervisory rule programs, written by an LLM (or GP, or a human).

A playbook is Python source defining

    def playbook(f, mem):
        ...
        return {"rule": "R3", "diagnosis": "...", "action": {...}}   # or None (no action)

where `f` is the feature dict from `summary.features` and `mem` a per-episode scratch dict
(e.g. to remember that a rule already fired). The source is checked against an AST
whitelist (no imports, no while-loops, loops/comprehensions only over finite lists, no attribute
access except dict .get/.items/.keys/.values and str.startswith/endswith, no dunder names) and executed in a namespace exposing
only safe builtins. Runtime errors make the playbook return None (and are counted).
"""
from __future__ import annotations

import ast
import math
from typing import Any

from .actions import NO_ACTION, parse_action
from .runner import Decision

ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.If, ast.IfExp, ast.Assign, ast.AugAssign,
    ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Store, ast.Constant,
    ast.Subscript, ast.Dict, ast.List, ast.Tuple, ast.Set, ast.Expr, ast.Pass, ast.For, ast.Break, ast.Continue,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.FloorDiv, ast.Attribute,
    ast.keyword, ast.Slice, ast.JoinedStr, ast.FormattedValue,
    ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp, ast.comprehension,
)
SAFE_FUNCS = {"abs": abs, "min": min, "max": max, "len": len, "round": round, "float": float, "int": int,
              "bool": bool, "str": str, "any": any, "all": all, "sum": sum, "sorted": sorted, "list": list,
              "isfinite": math.isfinite, "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "inf": math.inf}
ALLOWED_ATTRS = {"get", "items", "keys", "values", "startswith", "endswith", "append", "extend", "update", "pop",
                 "copy", "setdefault", "lower", "upper", "count", "index"}


class PlaybookError(Exception):
    pass


def check_source(src: str) -> ast.Module:
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PlaybookError(f"syntax error: {e}") from e
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not any(fn.name == "playbook" for fn in fns):
        raise PlaybookError("source must define `def playbook(f, mem):`")
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise PlaybookError(f"disallowed syntax: {type(node).__name__} (line {getattr(node, 'lineno', '?')})")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise PlaybookError("dunder names are not allowed")
        if isinstance(node, ast.Attribute) and node.attr not in ALLOWED_ATTRS:
            raise PlaybookError(f"attribute .{node.attr} not allowed (only {sorted(ALLOWED_ATTRS)})")
        if isinstance(node, ast.For) and not isinstance(node.iter, (ast.List, ast.Tuple, ast.Subscript, ast.Name, ast.Call)):
            raise PlaybookError("for-loops may only iterate over literals or feature lists")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id not in SAFE_FUNCS \
                and node.func.id not in {fn.name for fn in fns}:
            raise PlaybookError(f"call to {node.func.id}() not allowed")
    return tree


_VALID_KEYS: set[str] | None = None


def valid_feature_keys() -> set[str]:
    """All keys a playbook may read from `f` (computed once from a short nominal run)."""
    global _VALID_KEYS
    if _VALID_KEYS is None:
        from .faults import Scenario
        from .plant import Plant
        from .summary import features
        p = Plant(Scenario("lint", "A", ""), 1)
        p.run_until(1.0)
        _VALID_KEYS = set(features(p)) | {"trigger", "rejected", "reasons"}
    return _VALID_KEYS


def lint(src: str) -> list[str]:
    """Static check of feature keys used as string constants in f[...] / f.get(...): returns the unknown keys
    (a 'compile error' fed back to the synthesiser before any simulation is spent)."""
    keys = valid_feature_keys()
    bad = []
    for node in ast.walk(ast.parse(src)):
        k = None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "f" \
                and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
            k = node.slice.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "f" and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            k = node.args[0].value
        if k is not None and k not in keys and k not in bad:
            bad.append(k)
    return bad


def compile_playbook(src: str):
    tree = check_source(src)
    ns: dict[str, Any] = {"__builtins__": {}}
    ns.update(SAFE_FUNCS)
    exec(compile(tree, "<playbook>", "exec"), ns)  # noqa: S102 - whitelisted AST, no builtins
    return ns["playbook"]


def rule_names(src: str) -> list[str]:
    """Rule identifiers returned by the playbook (string constants under key 'rule')."""
    names = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and k.value == "rule" and isinstance(v, ast.Constant):
                    names.append(str(v.value))
    return sorted(set(names))


def loc(src: str) -> int:
    return sum(1 for ln in src.splitlines() if ln.strip() and not ln.strip().startswith("#"))


class PlaybookSupervisor:
    """Zero-token supervisor that executes a compiled playbook at each invocation."""

    def __init__(self, src: str, name: str = "playbook"):
        self.src = src
        self.fn = compile_playbook(src)
        self.name = name
        self.errors = 0
        self.error_msgs: dict[str, int] = {}
        self.fired: list[str] = []

    def episode_start(self, plant, scenario, seed):
        self.mem: dict = {}

    def _run(self, f: dict, extra: dict | None = None):
        ff = dict(f)
        if extra:
            ff.update(extra)
        try:
            out = self.fn(ff, self.mem)
        except Exception as e:  # noqa: BLE001 - playbook bugs must not kill the plant
            self.errors += 1
            msg = f"{type(e).__name__}: {e}"[:160]
            self.error_msgs[msg] = self.error_msgs.get(msg, 0) + 1
            return None, f"runtime error: {msg}"
        return out, None

    def propose(self, f: dict, extra: dict | None = None) -> tuple[Decision | None, str | None]:
        out, err = self._run(f, extra)
        if err is not None:
            return None, err
        if out is None:
            return None, None
        if not isinstance(out, dict) or "action" not in out:
            return None, "playbook returned a non-dict or no 'action'"
        a, perr = parse_action(out["action"])
        if a is None:
            return None, perr
        rule = str(out.get("rule", "?"))
        return Decision(a, str(out.get("diagnosis", "unknown")), str(out.get("why", ""))[:200], source=f"rule:{rule}"), None

    def decide(self, plant, f, trigger, history):
        d, err = self.propose(f, {"trigger": trigger})
        if d is None:
            return Decision(NO_ACTION, "none", err or "no rule fired", source="playbook:none",
                            meta={"error": err} if err else {})
        self.fired.append(d.source)
        return d

    def revise(self, plant, f, rejected, reasons, history):
        # One deterministic revision: re-run with the rejection visible to the rules.
        d, _ = self.propose(f, {"trigger": "revision", "rejected": rejected.action.as_dict(), "reasons": reasons})
        if d is None or d.action.key() == rejected.action.key():
            return None
        return d

    def episode_end(self, plant, outcome):
        pass
