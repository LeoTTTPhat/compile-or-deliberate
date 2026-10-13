#!/bin/bash
# usage: launch_synth.sh <model> <seed> <tag> <budget> [extra args...]
cd "$(dirname "${BASH_SOURCE[0]}")/../../../"
model=$1; seed=$2; tag=$3; budget=$4; shift 4
safe=${model//:/-}
PYTHONPATH=c10_compile_or_deliberate/code nohup .venv/bin/python tools/memguard.py --limit-gb 200 --parallel 8 \
  --tag "synth_${tag}_${safe}_s${seed}" -- .venv/bin/python c10_compile_or_deliberate/code/experiments/synth_run.py \
  --model "$model" --seed "$seed" --tag "$tag" --budget "$budget" --checkpoints 25000,50000,100000,200000,400000 "$@" \
  > "c10_compile_or_deliberate/logs/synth_${tag}_${safe}_s${seed}.log" 2>&1 &
echo "launched $model seed=$seed tag=$tag pid=$!"
