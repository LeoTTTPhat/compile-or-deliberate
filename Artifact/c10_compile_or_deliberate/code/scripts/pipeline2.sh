#!/bin/bash
# Experiment conductor v2 (replaces pipeline.sh from stage B on): same experiments, but CPU-bound
# steps (coverage) run concurrently with GPU-bound steps (hybrid, compile-back, secondary online
# model). Idempotent: every step skips finished work, so it can be re-launched at any time.
# Run:  nohup bash c10_compile_or_deliberate/code/scripts/pipeline2.sh > c10_compile_or_deliberate/logs/pipeline2.log 2>&1 &
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../../../"
P=c10_compile_or_deliberate
PY=.venv/bin/python
export PYTHONPATH=$P/code
SYN=$P/results/synth
MG="$PY tools/memguard.py --limit-gb 200"
log() { echo "[$(date '+%F %T')] $*"; echo "- [$(date '+%F %T')] pipeline2: $*" >> $P/logs/pipeline_events.md; }
wait_for() {
  log "waiting: $1"
  until eval "$2"; do sleep 60; done
  log "ready: $1"
}
synth_done() { grep -q "^DONE" $SYN/$1/log.txt 2>/dev/null; }
ckpt() { ls $SYN/main_qwen3-coder-30b_s$1/playbook_$2.py >/dev/null 2>&1; }
pb() { echo "$(pwd)/$SYN/main_qwen3-coder-30b_s$1/playbook_$2.py"; }
M=qwen3-coder:30b
./tools/start_ollama_private.sh

# ---------------------------------------------------------------- stage A (from pipeline.sh) must be finished
wait_for "stage A2 (hybrid grid at 100k) finished" "! pgrep -f 'tag A2_hybrid' >/dev/null && ! pgrep -f 'tag A1_playbooks' >/dev/null"

# ---------------------------------------------------------------- stage B: budgeted online (cache replay) + ablations
log "stage B: budgeted online variants (cache replay)"
$MG --parallel 10 --tag B_online_budget -- $PY $P/code/experiments/run_arms.py \
    --arms "onlineg:$M@4000,onlineg:$M@8000,onlineg:$M@16000,onlineg:$M@32000" --suite test_id,test_ood --seeds 1 --workers 10
log "stage B2: ablation syntheses in background (no remedies; no tuning), 200k tokens each"
if ! synth_done norem_qwen3-coder-30b_s1; then
  nohup $MG --parallel 8 --tag D1_norem -- $PY $P/code/experiments/synth_run.py --model $M --seed 1 --tag norem --budget 200000 \
      --checkpoints 25000,50000,100000,200000 --no-remedy >> $P/logs/synth_norem.log 2>&1 &
fi
if ! synth_done notune_qwen3-coder-30b_s1; then
  nohup $MG --parallel 8 --tag D1b_notune -- $PY $P/code/experiments/synth_run.py --model $M --seed 1 --tag notune --budget 200000 \
      --checkpoints 25000,50000,100000,200000 --tune-budget 0 >> $P/logs/synth_notune.log 2>&1 &
fi

# ---------------------------------------------------------------- stage C: main synthesis finished
wait_for "main synthesis seeds 1-3 finished" "synth_done main_qwen3-coder-30b_s1 && synth_done main_qwen3-coder-30b_s2 && synth_done main_qwen3-coder-30b_s3"
ARMS=""
for s in 1 2 3; do for c in B200000 B400000 final; do if ckpt $s $c; then ARMS="$ARMS,playbook:$(pb $s $c)"; fi; done; done
log "stage C1: final playbook checkpoints (CPU)"
$MG --parallel 10 --tag C1_playbooks -- $PY $P/code/experiments/run_arms.py --arms "${ARMS#,}" \
    --suite test_id,test_ood,alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 10

# three concurrent branches: CPU coverage | GPU hybrid + compile-back | GPU secondary online model
(
  log "branch C2 (CPU): coverage bounds"
  for s in 1 2 3; do
    $MG --parallel 8 --tag C2_cov_s$s -- $PY $P/code/experiments/coverage.py "$(pb $s final)" --name llm_final_s$s --workers 8
  done
  $MG --parallel 8 --tag C2_cov_gp1 -- $PY $P/code/experiments/coverage.py "$(pwd)/$SYN/gp_s1/playbook_final.py" --name gp_final_s1 --workers 8
  log "branch C2 done"
) &
C2=$!
(
  log "branch C3/C4 (GPU): hybrid on final playbooks, anchors, compile-back streams"
  H="hybrid:$(pb 1 final):$M@8000,hybrid:$(pb 1 final):$M@16000,hybrid:$(pb 1 final):$M@inf,hybrid:$(pb 2 final):$M@16000,hybrid:$(pb 3 final):$M@16000"
  $MG --parallel 8 --tag C3_hybrid -- $PY $P/code/experiments/run_arms.py --arms "$H" --suite test_id,test_ood --seeds 1 --workers 8
  $MG --parallel 8 --tag C3_hybrid_anchor -- $PY $P/code/experiments/run_arms.py --arms "hybrid:$(pb 1 final):$M@16000" \
      --suite alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 8
  $MG --parallel 8 --tag C4_stream -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb 1 final)" --model $M --tokens 16000 --name cb_s1_final_16k --workers 8
  $MG --parallel 8 --tag C4_stream_nocb -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb 1 final)" --model $M --tokens 16000 --name nocb_s1_final_16k --no-compile --workers 8
  log "branch C3/C4 done"
) &
C34=$!
(
  log "branch D3 (GPU): secondary online deliberator qwen2.5:14b (gated)"
  $MG --parallel 6 --tag D3_online14 -- $PY $P/code/experiments/run_arms.py --arms "onlineg:qwen2.5:14b-instruct-q4_K_M" \
      --suite test_id,test_ood --seeds 1 --workers 6
  log "branch D3 done"
) &
D3=$!

# ---------------------------------------------------------------- stage D2: ablation + GP evaluations (CPU)
wait_for "ablation syntheses finished" "synth_done norem_qwen3-coder-30b_s1 && synth_done notune_qwen3-coder-30b_s1"
ARMS="playbook:$(pwd)/$SYN/norem_qwen3-coder-30b_s1/playbook_final.py,playbook:$(pwd)/$SYN/notune_qwen3-coder-30b_s1/playbook_final.py"
for g in 2 3; do  # GP seed 1 (E5000, E20000 = final) was evaluated earlier
  if [ -f $SYN/gp_s$g/playbook_final.py ]; then ARMS="$ARMS,playbook:$(pwd)/$SYN/gp_s$g/playbook_final.py,playbook:$(pwd)/$SYN/gp_s$g/playbook_E5000.py"; fi
done
log "stage D2: ablation + GP playbook evaluation (CPU)"
$MG --parallel 8 --tag D2_eval -- $PY $P/code/experiments/run_arms.py --arms "$ARMS" \
    --suite test_id,test_ood,alhazmi,frontier,test_unrescuable --seeds 1,2,3 --workers 8
wait $C2 $C34 $D3
log "ALL STAGES DONE"
