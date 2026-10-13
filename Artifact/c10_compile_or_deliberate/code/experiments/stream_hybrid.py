"""RQ3: hybrid deployment as a STREAM with compile-back of verified online resolutions.

The test episodes (test_id + test_ood) are processed in a fixed random deployment order, in
blocks of `block` episodes run in parallel ("one block = one maintenance interval"). After each
block, successful escalations (episode succeeded, LLM action accepted) are pooled by
(diagnosis, action). A pool with >= `min_cases` cases from different episodes that is not yet
covered is compiled by the coder LLM into ONE new guarded rule appended to the playbook; the
new playbook is admitted only if its success on the synthesis training batch does not drop
(regression falsification). Admitted rules take effect from the next block on; every
version is saved. Token use (online escalations + compile-back) is recorded per block.

  python experiments/stream_hybrid.py --playbook results/synth/main_qwen3-coder-30b_s1/playbook_B100000.py \
      --model qwen2.5:7b --tokens 16000 --name cb_s1_100k_16k
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.dirname(ROOT)))
from cod.faults import Scenario  # noqa: E402
from cod.playbook import PlaybookError, check_source, loc  # noqa: E402
from cod.runner import Policy, run_episode  # noqa: E402
from cod.supervisors.hybrid import HybridSupervisor  # noqa: E402
from cod.synth.evaluate import evaluate, success  # noqa: E402
from cod.synth.llm_synth import SYSTEM, extract_code  # noqa: E402
from tools.llm import LLM  # noqa: E402

OUT = os.path.join(os.path.dirname(ROOT), "results", "stream")


def _episode(args):
    src, model, tokens, sc_d, seed, label, opts = args
    sc = Scenario.from_dict(sc_d)
    sup = HybridSupervisor(src, model, tokens, name="stream", **opts)
    out = run_episode(sc, seed, sup, policy=Policy())
    out["success"] = success(out, label)
    out["label"] = label
    out.pop("decisions_full", None)
    return out


def compile_prompt(src: str, key: tuple, cases: list[dict]) -> str:
    diag, action = key
    lines = "\n".join(f"- t={c['t']:.2f} h, alarms={c['alarms']}, features={json.dumps(c['fx'])}" for c in cases[:6])
    return (f"Current playbook:\n```python\n{src}```\n"
            f"During operation, situations NOT covered by the playbook were escalated to an online deliberator, which "
            f"diagnosed '{diag}' and took the action {action}; the plant was stabilised every time. Observed features "
            f"at the moments of those decisions:\n{lines}\n\n"
            "Add exactly ONE new rule that takes this action in situations like these (choose robust, physically "
            "meaningful conditions with margins; guard with mem so it fires once), placed before any catch-all rule, "
            "without changing the behaviour of the existing rules. Return the complete revised code.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--playbook", required=True)
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--coder", default="qwen3-coder:30b")
    ap.add_argument("--tokens", type=float, default=16000)
    ap.add_argument("--name", required=True)
    ap.add_argument("--block", type=int, default=18)
    ap.add_argument("--min-cases", type=int, default=2)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--order-seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("MEMGUARD_PARALLEL", "12")))
    a = ap.parse_args()
    out_dir = os.path.join(OUT, a.name)
    os.makedirs(out_dir, exist_ok=True)
    stream = []
    for suite in ("test_id", "test_ood"):
        for x in json.load(open(os.path.join(os.path.dirname(ROOT), "data", "suites", f"{suite}.json"))):
            stream.append((x["scenario"], x["seeds"][0], x["label"], suite))
    order = np.random.default_rng(a.order_seed).permutation(len(stream))
    stream = [stream[i] for i in order]
    train = json.load(open(os.path.join(os.path.dirname(a.playbook), "train_batch.json")))
    src = open(a.playbook).read()
    base_train = evaluate(src, train, workers=a.workers).success_rate
    coder = LLM(a.coder, num_ctx=16384)
    pools: dict = {}
    covered: set = set()
    log = []
    version = 0
    open(os.path.join(out_dir, "playbook_v0.py"), "w").write(src)
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=20) as pool:
        for bi in range(0, len(stream), a.block):
            blk = stream[bi: bi + a.block]
            res_path = os.path.join(out_dir, f"block{bi // a.block:02d}.json")
            if os.path.exists(res_path):  # resume: replay the block's effect on the case pools
                saved = json.load(open(res_path))
                outs = saved["episodes"]
                src = open(os.path.join(out_dir, saved["playbook_after"])).read()
                version = saved["version_after"]
                log = saved["log"]
                covered = {tuple(e["key"]) for e in log}
                for o in outs:
                    if o["success"]:
                        for d in o["decisions"]:
                            if d["source"].startswith("escalation") and d["accepted"]:
                                key = (d["diagnosis"], json.dumps(d["action"], sort_keys=True))
                                pools.setdefault(key, []).append({"t": d["t"], "alarms": d["alarms"], "fx": d["fx"], "ep": o["scenario"]})
                if version > 0:
                    base_train = evaluate(src, train, workers=a.workers).success_rate
            else:
                outs = pool.map(_episode, [(src, a.model, a.tokens, s, seed, lab, {}) for s, seed, lab, _ in blk], chunksize=1)
                for o, (_, _, _, suite) in zip(outs, blk):
                    o["suite"] = suite
                    o["playbook_version"] = version
                cb_tokens = 0
                admitted = None
                if not a.no_compile:
                    for o in outs:
                        if not o["success"]:
                            continue
                        for d in o["decisions"]:
                            if d["source"].startswith("escalation") and d["accepted"]:
                                key = (d["diagnosis"], json.dumps(d["action"], sort_keys=True))
                                pools.setdefault(key, []).append({"t": d["t"], "alarms": d["alarms"], "fx": d["fx"], "ep": o["scenario"]})
                    for key, cases in sorted(pools.items(), key=lambda kv: -len(kv[1])):
                        if key in covered or len({c["ep"] for c in cases}) < a.min_cases:
                            continue
                        r = coder.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": compile_prompt(src, key, cases)}],
                                       temperature=0.2, seed=version + 1, max_tokens=3500, tag=f"compile:{a.name}")
                        cb_tokens += r["prompt_tokens"] + r["completion_tokens"]
                        new = extract_code(r["text"])
                        covered.add(key)
                        try:
                            check_source(new)
                        except PlaybookError as e:
                            log.append({"block": bi // a.block, "key": key, "admitted": False, "reason": str(e)})
                            continue
                        sr = evaluate(new, train, workers=a.workers).success_rate
                        ok = sr >= base_train - 1e-9
                        log.append({"block": bi // a.block, "key": key, "admitted": ok, "train_sr": sr, "base_train": base_train,
                                    "loc": loc(new), "cases": len(cases)})
                        if ok:
                            version += 1
                            src, base_train, admitted = new, sr, key
                            open(os.path.join(out_dir, f"playbook_v{version}.py"), "w").write(src)
                            break  # at most one admitted rule per maintenance interval
                json.dump({"episodes": outs, "compile_tokens": cb_tokens, "admitted": admitted, "version_after": version,
                           "playbook_after": f"playbook_v{version}.py", "log": log}, open(res_path, "w"), default=str)
            ok = sum(o["success"] for o in outs)
            esc = sum(len(o.get("escalations", [])) for o in outs)
            tok = sum(o["prompt_tokens"] + o["completion_tokens"] for o in outs)
            print(f"block {bi // a.block}: {ok}/{len(outs)} success, escalations {esc}, online tokens {tok}, "
                  f"playbook v{version} ({time.time() - t0:.0f}s)", flush=True)
    json.dump({"log": log, "final_version": version}, open(os.path.join(out_dir, "summary.json"), "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
