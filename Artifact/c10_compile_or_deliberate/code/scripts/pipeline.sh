#!/bin/bash
# Experiment conductor: dependency-aware launch of the remaining experiments (resumable; every
# step is idempotent because run_arms/synth/stream/coverage skip finished work).
# Run:  nohup bash c10_compile_or_deliberate/code/scripts/pipeline.sh > c10_compile_or_deliberate/logs/pipeline.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../../"
P=c10_compile_or_deliberate
PY=.venv/bin/python
export PYTHONPATH=$P/code
SYN=$P/results/synth
MG="$PY tools/memguard.py --limit-gb 200"
log() { echo "[$(date '+%F %T')] $*"; echo "- [$(date '+%F %T')] pipeline: $*" >> $P/logs/pipeline_events.md; }
wait_for() {  # wait_for <description> <shell test>
  log "waiting: $1"
  until eval "$2"; do sleep 60; done
  log "ready: $1"
}
synth_done() { grep -q "^DONE" $SYN/main_qwen3-coder-30b_s$1/log.txt 2>/dev/null; }
ckpt() { ls $SYN/main_qwen3-coder-30b_s$1/playbook_$2.py >/dev/null 2>&1; }
pb() { echo "$(pwd)/$SYN/main_qwen3-coder-30b_s$1/playbook_$2.py"; }
M=qwen3-coder:30b

./tools/start_ollama_private.sh

# ---------------------------------------------------------------- stage A: 100k checkpoints
wait_for "B100000 checkpoints of seeds 1-3" "ckpt 1 B100000 && ckpt 2 B100000 && ckpt 3 B100000"
ARMS="playbook:$(pb 1 v0),playbook:$(pb 1 B25000),playbook:$(pb 1 B100000),playbook:$(pb 2 B100000),playbook:$(pb 3 B100000)"
log "stage A1: zero-token playbook checkpoints (CPU)"
$MG --parallel 10 --tag A1_playbooks -- $PY $P/code/experiments/run_arms.py --arms "$ARMS" \
    --suite test_id,test_ood,alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 10
log "stage A2: hybrid grid at 100k (seed 1 x tokens; seeds 2,3 at 16k)"
H="hybrid:$(pb 1 B25000):$M@16000,hybrid:$(pb 1 B100000):$M@8000,hybrid:$(pb 1 B100000):$M@16000,hybrid:$(pb 1 B100000):$M@inf,hybrid:$(pb 2 B100000):$M@16000,hybrid:$(pb 3 B100000):$M@16000"
$MG --parallel 10 --tag A2_hybrid -- $PY $P/code/experiments/run_arms.py --arms "$H" --suite test_id,test_ood --seeds 1 --workers 10
log "stage A2 done"

# ---------------------------------------------------------------- stage B: budgeted online (cache replay)
wait_for "main online arm finished" "grep -q '^done' $P/logs/online_main.log"
log "stage B: budgeted online variants (mostly cache replay)"
$MG --parallel 10 --tag B_online_budget -- $PY $P/code/experiments/run_arms.py \
    --arms "onlineg:$M@4000,onlineg:$M@8000,onlineg:$M@16000,onlineg:$M@32000" --suite test_id,test_ood --seeds 1 --workers 10

log "stage B2: launch ablation syntheses in background (no remedies; no tuning), 200k tokens each"
nohup $MG --parallel 8 --tag D1_norem -- $PY $P/code/experiments/synth_run.py --model $M --seed 1 --tag norem --budget 200000 \
    --checkpoints 25000,50000,100000,200000 --no-remedy > $P/logs/synth_norem.log 2>&1 &
nohup $MG --parallel 8 --tag D1b_notune -- $PY $P/code/experiments/synth_run.py --model $M --seed 1 --tag notune --budget 200000 \
    --checkpoints 25000,50000,100000,200000 --tune-budget 0 > $P/logs/synth_notune.log 2>&1 &

# ---------------------------------------------------------------- stage C: synthesis finished
wait_for "synthesis seeds 1-3 finished" "synth_done 1 && synth_done 2 && synth_done 3"
ARMS=""
for s in 1 2 3; do
  for c in B200000 B400000 final; do
    if ckpt $s $c; then ARMS="$ARMS,playbook:$(pb $s $c)"; fi
  done
done
log "stage C1: final playbook checkpoints (CPU)"
$MG --parallel 10 --tag C1_playbooks -- $PY $P/code/experiments/run_arms.py --arms "${ARMS#,}" \
    --suite test_id,test_ood,alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 10
log "stage C2: coverage bounds of final playbooks (CPU)"
for s in 1 2 3; do
  $MG --parallel 10 --tag C2_cov_s$s -- $PY $P/code/experiments/coverage.py "$(pb $s final)" --name llm_final_s$s --workers 10
done
for g in 1; do
  if [ -f $SYN/gp_s$g/playbook_final.py ]; then
    $MG --parallel 10 --tag C2_cov_gp$g -- $PY $P/code/experiments/coverage.py "$(pwd)/$SYN/gp_s$g/playbook_final.py" --name gp_final_s$g --workers 10
  fi
done
log "stage C3: hybrid on final playbooks + anchors"
H="hybrid:$(pb 1 final):$M@8000,hybrid:$(pb 1 final):$M@16000,hybrid:$(pb 1 final):$M@inf,hybrid:$(pb 2 final):$M@16000,hybrid:$(pb 3 final):$M@16000"
$MG --parallel 10 --tag C3_hybrid -- $PY $P/code/experiments/run_arms.py --arms "$H" --suite test_id,test_ood --seeds 1 --workers 10
$MG --parallel 10 --tag C3_hybrid_anchor -- $PY $P/code/experiments/run_arms.py --arms "hybrid:$(pb 1 final):$M@16000" \
    --suite alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 10
log "stage C4: compile-back stream (seed 1 final, 16k) and its no-compile control"
$MG --parallel 10 --tag C4_stream -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb 1 final)" --model $M --tokens 16000 --name cb_s1_final_16k --workers 10
$MG --parallel 10 --tag C4_stream_nocb -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb 1 final)" --model $M --tokens 16000 --name nocb_s1_final_16k --no-compile --workers 10

# ---------------------------------------------------------------- stage D: ablations + GP evals + secondary model
wait_for "ablation syntheses finished" "grep -q '^DONE' $SYN/norem_qwen3-coder-30b_s1/log.txt 2>/dev/null && grep -q '^DONE' $SYN/notune_qwen3-coder-30b_s1/log.txt 2>/dev/null"
ARMS="playbook:$(pwd)/$SYN/norem_qwen3-coder-30b_s1/playbook_final.py,playbook:$(pwd)/$SYN/notune_qwen3-coder-30b_s1/playbook_final.py"
for g in 2 3; do
  wait_for "GP seed $g finished" "[ -f $SYN/gp_s$g/playbook_final.py ]"
  ARMS="$ARMS,playbook:$(pwd)/$SYN/gp_s$g/playbook_final.py,playbook:$(pwd)/$SYN/gp_s$g/playbook_E5000.py"
done
ARMS="$ARMS,playbook:$(pwd)/$SYN/gp_s1/playbook_final.py"
log "stage D2: ablation + GP playbook evaluation (CPU)"
$MG --parallel 10 --tag D2_eval -- $PY $P/code/experiments/run_arms.py --arms "$ARMS" \
    --suite test_id,test_ood,alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 10
log "stage D3: secondary online deliberator qwen2.5:14b (gated) on held-out suites"
$MG --parallel 8 --tag D3_online14 -- $PY $P/code/experiments/run_arms.py --arms "onlineg:qwen2.5:14b-instruct-q4_K_M" \
    --suite test_id,test_ood --seeds 1 --workers 8
log "ALL STAGES DONE"
