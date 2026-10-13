#!/usr/bin/env bash
# Mock-review (R3) robustness checks of the compiled supervisors, run after the confirmation suites finish:
#  (a) the same supervisors on top of the strong configuration m = (1,1)  ("onstrong:" arms)
#  (b) dense invocation: re-invocation every 0.5 h, at most 96 invocations (actions still capped at 12)
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
while pgrep -f "tag confirm --" > /dev/null; do sleep 60; done
B="B2b,fdd:default,playbook:c10_compile_or_deliberate/results/synth/gp_s1/playbook_E20000.py,playbook:c10_compile_or_deliberate/results/synth/main_qwen3-coder-30b_s1/playbook_final.py,playbook:c10_compile_or_deliberate/results/synth/main_qwen3-coder-30b_s3/playbook_final.py"
ON=$(echo "$B" | tr ',' '\n' | sed 's/^/onstrong:/' | paste -sd, -)
.venv/bin/python tools/memguard.py --limit-gb 200 --parallel 8 --tag r3_onstrong -- env PYTHONPATH=c10_compile_or_deliberate/code \
  .venv/bin/python c10_compile_or_deliberate/code/experiments/run_arms.py --arms "$ON" --suite test_id,test_ood --workers 8
.venv/bin/python tools/memguard.py --limit-gb 200 --parallel 8 --tag r3_dense -- env PYTHONPATH=c10_compile_or_deliberate/code \
  .venv/bin/python c10_compile_or_deliberate/code/experiments/run_arms.py --arms "$B" --suite test_id_dense,test_ood_dense --workers 8 \
  --policy '{"periodic_h": 0.5, "hold_interval_h": 0.5, "max_calls": 96}'
