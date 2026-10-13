"""CPU-side calibration of an LLM-written playbook: numeric thresholds and rule priority.

The language model is good at proposing *which* evidence should trigger *which* response,
and poor at choosing numbers and at ordering first-match rules. `tune` keeps the playbook's
structure and text (comments included) and only
  (i)  rescales numeric thresholds that are compared against a feature
       (f["X"] < c, abs(f["X"]) > c, c < f["X"] ...), by coordinate search over multiplicative
       candidates, and
  (ii) moves top-level rule blocks of `playbook` to the front (priority search).
Every candidate is scored with the same simulation evaluation as the synthesiser (common
random numbers on the current training batch); a change is kept only if it improves the
score. The budget is counted in batch evaluations.
"""
from __future__ import annotations

import ast
import random
from dataclasses import dataclass

from ..playbook import PlaybookError, check_source
from .evaluate import evaluate

MULTS = (0.25, 0.5, 0.75, 1.5, 2.0, 4.0)


@dataclass
class Param:
    start: tuple[int, int]  # (lineno, col) of the constant (1-based line, 0-based col)
    end: tuple[int, int]
    value: float
    feature: str


def _feature_of(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs" and node.args:
        return _feature_of(node.args[0])
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "f" \
            and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        return node.slice.value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get" \
            and isinstance(node.func.value, ast.Name) and node.func.value.id == "f" and node.args \
            and isinstance(node.args[0], ast.Constant):
        return str(node.args[0].value)
    return None


def _num(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node, float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant) \
            and isinstance(node.operand.value, (int, float)):
        return node, -float(node.operand.value)
    return None, None


def extract_params(src: str) -> list[Param]:
    out = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Compare) or len(node.comparators) != 1:
            continue
        left, right = node.left, node.comparators[0]
        for a, b in ((left, right), (right, left)):
            feat = _feature_of(a)
            cnode, val = _num(b)
            if feat and cnode is not None and val != 0.0 and not feat.startswith("A_"):
                out.append(Param((cnode.lineno, cnode.col_offset), (cnode.end_lineno, cnode.end_col_offset), val, feat))
                break
    return out


def _fmt(v: float) -> str:
    return f"{v:.4g}" if abs(v) < 1e4 else f"{v:.0f}"


def set_params(src: str, params: list[Param], values: list[float]) -> str:
    lines = src.splitlines(keepends=True)
    for p, v in sorted(zip(params, values), key=lambda z: z[0].start, reverse=True):
        (l0, c0), (l1, c1) = p.start, p.end
        if l0 != l1:
            continue
        ln = lines[l0 - 1]
        lines[l0 - 1] = ln[:c0] + _fmt(v) + ln[c1:]
    return "".join(lines)


def rule_blocks(src: str) -> tuple[list[str], list[tuple[int, int]]]:
    """Top-level statements of playbook() as text blocks (each including its leading comment lines)."""
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "playbook")
    lines = src.splitlines(keepends=True)
    spans = []
    prev_end = fn.body[0].lineno - 1
    # include comment lines that directly precede the first statement
    while prev_end - 1 >= fn.lineno and lines[prev_end - 1].strip().startswith("#"):
        prev_end -= 1
    for st in fn.body:
        start = prev_end + 1
        spans.append((start, st.end_lineno))
        prev_end = st.end_lineno
    return lines, spans


def move_to_front(src: str, k: int) -> str | None:
    lines, spans = rule_blocks(src)
    if k <= 0 or k >= len(spans):
        return None
    tree = ast.parse(src)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "playbook")
    if not isinstance(fn.body[k], ast.If):
        return None
    s0 = spans[0][0]
    blocks = ["".join(lines[a - 1:b]) for a, b in spans]
    tail = "".join(lines[spans[-1][1]:])
    head = "".join(lines[: s0 - 1])
    new = head + blocks[k] + "".join(b for i, b in enumerate(blocks) if i != k) + tail
    try:
        check_source(new)
    except PlaybookError:
        return None
    return new


def _block_rule_names(block: str) -> set[str]:
    names = set()
    try:
        for node in ast.walk(ast.parse("def _f():\n" + "".join("    " + ln for ln in block.splitlines(True)))):
            if isinstance(node, ast.Dict):
                for k, v in zip(node.keys, node.values):
                    if isinstance(k, ast.Constant) and k.value == "rule" and isinstance(v, ast.Constant):
                        names.add(str(v.value))
    except SyntaxError:
        pass
    return names


def tune(src: str, batch: list[dict], budget: int = 24, workers: int = 16, seed: int = 0, log=print):
    """Greedy coordinate search; returns (best_src, best_eval, n_batch_evals, n_episodes).
    Priority phase (at most half the budget): rule blocks whose rules never fired on the batch
    (possibly shadowed by an earlier first-match rule) are tried at the front first.
    Threshold phase: multiplicative rescaling of each compared constant, first improvement kept."""
    rng = random.Random(seed)
    best_src = src
    best = evaluate(src, batch, workers=workers)
    used, eps = 1, len(batch)
    improved = True
    while used < budget and improved:
        improved = False
        fired = {d["source"][5:] for e in best.episodes for d in e["decisions"] if d["source"].startswith("rule:")}
        lines, spans = rule_blocks(best_src)
        blocks = ["".join(lines[a - 1:b]) for a, b in spans]
        cand_k = [k for k in range(1, len(blocks)) if _block_rule_names(blocks[k]) and not (_block_rule_names(blocks[k]) & fired)]
        cand_k += [k for k in range(1, len(blocks)) if k not in cand_k]
        for k in cand_k:
            if used >= max(2, budget // 2):
                break
            cand = move_to_front(best_src, k)
            if cand is None:
                continue
            ev = evaluate(cand, batch, workers=workers)
            used += 1
            eps += len(batch)
            if ev.score > best.score + 1e-9:
                best_src, best, improved = cand, ev, True
                log(f"    tune: moved rule block {k} {sorted(_block_rule_names(blocks[k]))} to front -> {ev.success_rate:.3f}")
                break
        params = extract_params(best_src)
        order = list(range(len(params)))
        rng.shuffle(order)
        for i in order:
            if used >= budget:
                break
            params = extract_params(best_src)
            if i >= len(params):
                continue
            p = params[i]
            for v in [p.value * m for m in MULTS]:
                if used >= budget:
                    break
                cand = set_params(best_src, [p], [v])
                try:
                    check_source(cand)
                except PlaybookError:
                    continue
                ev = evaluate(cand, batch, workers=workers)
                used += 1
                eps += len(batch)
                if ev.score > best.score + 1e-9:
                    best_src, best, improved = cand, ev, True
                    log(f"    tune: {p.feature} {p.value:g} -> {v:.4g} -> {ev.success_rate:.3f}")
                    break
    return best_src, best, used, eps
