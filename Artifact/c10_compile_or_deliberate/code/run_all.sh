#!/bin/bash
# End-to-end reproduction of "Compile or deliberate?" (all experiments, then all paper artefacts).
#
#   bash <project>/code/run_all.sh                  # every stage, in order
#   STAGES="analysis" bash <project>/code/run_all.sh  # selected stages only (space-separated names below)
#
# Stages: setup softsensors screens suites fdd baselines gp synth online playbooks hybrid stream coverage revision analysis
#
# Every experiment step is resumable: results are one JSON per (arm, scenario, seed) or per checkpoint, and every
# LLM call is cached in <workspace>/cache/llm_cache.sqlite, so re-running this script skips finished work and never
# repeats an LLM call. With a complete cache, the LLM stages replay without a GPU (and without the Ollama server).
# Heavy steps run under tools/memguard.py (project memory cap, restart with halved parallelism on breach).
#
# Layout assumed (as in the released archive):
#   <workspace>/.venv/                       python 3.12 virtual environment (created by the setup stage)
#   <workspace>/tools/                       llm.py (cached client), memguard.py, start_ollama_private.sh
#   <workspace>/<project>/code/              this package (cod/, experiments/, scripts/, tests/)
#   <workspace>/<project>/{data,results,paper}/
#
# Wall-clock: the CPU stages take hours on a 32-core machine; the LLM stages take days of decoding on one
# workstation-class GPU (see the paper's cost accounting). Nothing here is executed by the paper build itself.
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WS="$(dirname "$PROJ_DIR")"
P="$(basename "$PROJ_DIR")"
cd "$WS"

PY=.venv/bin/python
export PYTHONPATH="$P/code"
MG="$PY tools/memguard.py --limit-gb ${MEMGUARD_LIMIT_GB:-200}"
W=${WORKERS:-10}                       # worker processes per job
SEEDS=1,2,3                            # noise seeds for the anchor suites (held-out suites carry their own)
ALL="alhazmi,frontier,test_id,test_ood,test_unrescuable"
MAIN="test_id,test_ood"
CODER=qwen3-coder:30b                  # offline coder and main online deliberator
SYN="$P/results/synth"
STAGES=${STAGES:-"setup softsensors screens suites fdd baselines gp synth online playbooks hybrid stream coverage revision analysis"}

want() { [[ " $STAGES " == *" $1 "* ]]; }
log() { echo "[$(date '+%F %T')] $*"; }
pb() { echo "$WS/$SYN/$1/playbook_$2.py"; }            # absolute path, as in the paper's arm specs
have() { [ -f "$(pb "$1" "$2")" ]; }

# ------------------------------------------------------------------ setup
if want setup; then
  log "setup: python 3.12 venv + pinned packages; private Ollama server; models"
  if [ ! -x "$PY" ]; then uv venv --python 3.12 .venv; fi
  if [ -f "$P/code/requirements.txt" ]; then
    uv pip install --python "$PY" -r "$P/code/requirements.txt"
  else
    uv pip install --python "$PY" "tep-studio==0.2.0" numpy scipy scikit-learn pandas statsmodels matplotlib \
      requests psutil tqdm pyyaml pytest
  fi
  if [ "${SKIP_OLLAMA:-0}" != "1" ]; then
    bash tools/start_ollama_private.sh   # 127.0.0.1:11435, 8 parallel sequences, context 16384
    for m in qwen3-coder:30b qwen2.5-coder:7b qwen2.5:7b qwen2.5:14b-instruct-q4_K_M; do
      OLLAMA_HOST=127.0.0.1:11435 ollama pull "$m"
    done
  fi
fi

# ------------------------------------------------------------------ plant-side preparation (CPU)
if want softsensors; then
  log "soft sensors (data/soft_sensors.json, data/nominal.json)"
  $PY $P/code/scripts/fit_soft_sensors.py
fi
if want screens; then
  log "headroom screens: seed 7 (training/synthesis) and seed 101 (held-out suites)"
  $MG --parallel 16 --tag screen -- $PY $P/code/experiments/screen.py 50
  $MG --parallel 12 --tag screen101 -- $PY $P/code/experiments/screen.py 40 A,S,P,Q,SN,V,C,K 101
fi
if want suites; then
  log "frozen held-out suites + content hashes (data/suites/)"
  $PY $P/code/scripts/make_suites.py
fi
if want fdd; then
  log "FDD classifier (B3) and conformal novelty detector (hybrid trigger c)"
  $MG --parallel 6 --tag fdd_data -- $PY $P/code/experiments/fdd_data.py synth
  $PY $P/code/experiments/fdd_data.py heldout --n 15
  $PY -m cod.fdd train
fi

# ------------------------------------------------------------------ zero-token baselines (CPU)
if want baselines; then
  log "references and LLM-free baselines: weak, strong, B2a, B2b, FDD (B3), verifier search (B4)"
  $MG --parallel $W --tag baselines -- $PY $P/code/experiments/run_arms.py \
      --arms weak,strong,B2a,B2b,fdd:default,search:nominal --suite $ALL --seeds $SEEDS --workers $W
  log "oracle upper bound: perfect-twin search, perfect-fidelity shadow check"
  $MG --parallel $W --tag oracle -- $PY $P/code/experiments/run_arms.py --arms "search:perfect@gate=0" \
      --suite $ALL --seeds $SEEDS --workers $W --policy '{"shadow_fidelity": "perfect"}'
fi
if want gp; then
  log "genetic programming over the playbook DSL (B5), seeds 1-3 (resumable from state.pkl)"
  for s in 1 2 3; do
    $MG --parallel $W --tag gp_s$s -- $PY $P/code/experiments/gp_run.py --seed $s --evals 20000 --workers $W
  done
fi

# ------------------------------------------------------------------ offline LLM synthesis
if want synth; then
  CK=25000,50000,100000,200000,400000
  log "main synthesis: $CODER, seeds 1-3, 400k tokens each"
  for s in 1 2 3; do
    $MG --parallel 8 --tag synth_main_s$s -- $PY $P/code/experiments/synth_run.py --model $CODER --seed $s --tag main \
        --budget 400000 --checkpoints $CK
  done
  log "coder-size point: qwen2.5-coder:7b, seed 1 (the paper uses its checkpoints up to 100k tokens)"
  $MG --parallel 8 --tag synth_7b -- $PY $P/code/experiments/synth_run.py --model qwen2.5-coder:7b --seed 1 --tag main \
      --budget 200000 --checkpoints $CK
  log "ablations: one-shot, no counterfactual remedies, no CPU tuning (seed 1)"
  $MG --parallel 8 --tag synth_oneshot -- $PY $P/code/experiments/synth_run.py --model $CODER --seed 1 --tag oneshot --one-shot
  $MG --parallel 8 --tag synth_norem -- $PY $P/code/experiments/synth_run.py --model $CODER --seed 1 --tag norem \
      --budget 200000 --checkpoints 25000,50000,100000,200000 --no-remedy
  $MG --parallel 8 --tag synth_notune -- $PY $P/code/experiments/synth_run.py --model $CODER --seed 1 --tag notune \
      --budget 200000 --checkpoints 25000,50000,100000,200000 --tune-budget 0
fi

# ------------------------------------------------------------------ online deliberation (f = 0)
if want online; then
  log "gated online deliberator, unconstrained bucket, all suites"
  $MG --parallel 8 --tag online_main -- $PY $P/code/experiments/run_arms.py --arms "onlineg:$CODER" \
      --suite test_id,test_ood,alhazmi,frontier --seeds $SEEDS --workers 8
  log "token-bucket variants (mostly cache replay of the run above)"
  $MG --parallel $W --tag online_budget -- $PY $P/code/experiments/run_arms.py \
      --arms "onlineg:$CODER@4000,onlineg:$CODER@8000,onlineg:$CODER@16000,onlineg:$CODER@32000" \
      --suite $MAIN --seeds 1 --workers $W
  log "secondary deliberator"
  $MG --parallel 8 --tag online_14b -- $PY $P/code/experiments/run_arms.py --arms "onlineg:qwen2.5:14b-instruct-q4_K_M" \
      --suite $MAIN --seeds 1 --workers 8
fi

# ------------------------------------------------------------------ playbook checkpoints (f = 1, zero tokens, CPU)
if want playbooks; then
  ARMS=""
  for run in main_qwen3-coder-30b_s1 main_qwen3-coder-30b_s2 main_qwen3-coder-30b_s3 main_qwen2.5-coder-7b_s1 \
             oneshot_qwen3-coder-30b_s1 norem_qwen3-coder-30b_s1 notune_qwen3-coder-30b_s1 gp_s1 gp_s2 gp_s3; do
    for c in v0 B25000 B50000 B100000 B200000 B400000 E5000 E20000 final; do
      if have $run $c; then ARMS="$ARMS,playbook:$(pb $run $c)"; fi
    done
  done
  log "playbook checkpoints on all suites"
  $MG --parallel $W --tag playbooks -- $PY $P/code/experiments/run_arms.py --arms "${ARMS#,}" --suite $ALL \
      --seeds $SEEDS --workers $W
fi

# ------------------------------------------------------------------ hybrid (0 < f < 1)
if want hybrid; then
  S1=main_qwen3-coder-30b_s1
  H="hybrid:$(pb $S1 B25000):$CODER@16000,hybrid:$(pb $S1 B100000):$CODER@8000,hybrid:$(pb $S1 B100000):$CODER@16000"
  H="$H,hybrid:$(pb $S1 B100000):$CODER@inf,hybrid:$(pb $S1 final):$CODER@8000,hybrid:$(pb $S1 final):$CODER@16000"
  H="$H,hybrid:$(pb $S1 final):$CODER@inf"
  for s in 2 3; do
    for c in B100000 final; do H="$H,hybrid:$(pb main_qwen3-coder-30b_s$s $c):$CODER@16000"; done
  done
  log "hybrid grid (playbook checkpoint x per-episode token bucket) on the held-out suites"
  $MG --parallel $W --tag hybrid -- $PY $P/code/experiments/run_arms.py --arms "$H" --suite $MAIN --seeds 1 --workers $W
  $MG --parallel $W --tag hybrid_anchor -- $PY $P/code/experiments/run_arms.py --arms "hybrid:$(pb $S1 final):$CODER@16000" \
      --suite alhazmi,frontier,test_unrescuable --seeds $SEEDS --workers $W
fi

# ------------------------------------------------------------------ compile-back stream (RQ3)
if want stream; then
  log "deployment stream with compile-back, and its no-compile control"
  $MG --parallel $W --tag stream -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb main_qwen3-coder-30b_s1 final)" \
      --model $CODER --tokens 16000 --name cb_s1_final_16k --workers $W
  $MG --parallel $W --tag stream_nocb -- $PY $P/code/experiments/stream_hybrid.py --playbook "$(pb main_qwen3-coder-30b_s1 final)" \
      --model $CODER --tokens 16000 --name nocb_s1_final_16k --no-compile --workers $W
fi

# ------------------------------------------------------------------ coverage bound (RQ4)
if want coverage; then
  for s in 1 2 3; do
    log "coverage bound: LLM playbook, seed $s"
    $MG --parallel $W --tag cov_s$s -- $PY $P/code/experiments/coverage.py "$(pb main_qwen3-coder-30b_s$s final)" \
        --name llm_final_s$s --workers $W
  done
  log "coverage bound: GP playbook, seed 1"
  $MG --parallel $W --tag cov_gp1 -- $PY $P/code/experiments/coverage.py "$(pb gp_s1 final)" --name gp_final_s1 --workers $W
fi

# ------------------------------------------------------------------ revision runs (CPU only, no language-model call)
if want revision; then
  log "calibration-only ablation: each main run's first draft, one falsification round + calibration, no repair (R1)"
  for s in 1 2 3; do
    $MG --parallel $W --tag calibonly_s$s -- $PY $P/code/experiments/calib_only.py --seed $s --rounds 1 --workers $W
  done
  CAL="playbook:$(pb calibonly_qwen3-coder-30b_s1 R1),playbook:$(pb calibonly_qwen3-coder-30b_s2 R1)"
  CAL="$CAL,playbook:$(pb calibonly_qwen3-coder-30b_s3 R1)"
  $MG --parallel $W --tag calibonly_eval -- $PY $P/code/experiments/run_arms.py --arms "$CAL" --suite $MAIN --seeds 1 --workers $W
  log "fresh suites (drawn after the method was frozen): select_id, confirm_id, confirm_ood; dense-invocation copies"
  $PY $P/code/experiments/make_confirm_suites.py   # needs the coverage stage (labels of the coverage samples)
  for s in test_id test_ood; do
    cp "$P/data/suites/$s.json" "$P/data/suites/${s}_dense.json"
    cp "$P/data/suites/$s.json.sha" "$P/data/suites/${s}_dense.json.sha"
  done
  FROZEN="weak,strong,B2b,fdd:default,playbook:$(pb gp_s1 E20000),playbook:$(pb gp_s1 E5000)"
  FROZEN="$FROZEN,playbook:$(pb gp_s2 final),playbook:$(pb gp_s3 final)"
  for s in 1 2 3; do FROZEN="$FROZEN,playbook:$(pb main_qwen3-coder-30b_s$s final)"; done
  FROZEN="$FROZEN,playbook:$(pb norem_qwen3-coder-30b_s1 final),playbook:$(pb notune_qwen3-coder-30b_s1 final)"
  log "selection sample and confirmation suites (frozen arms; the suites carry their own noise seeds)"
  $MG --parallel $W --tag confirm -- $PY $P/code/experiments/run_arms.py --arms "$FROZEN" \
      --suite select_id,confirm_id,confirm_ood --workers $W
  $MG --parallel $W --tag sel_ckpts -- $PY $P/code/experiments/run_arms.py \
      --arms "playbook:$(pb main_qwen3-coder-30b_s1 B25000),playbook:$(pb main_qwen3-coder-30b_s1 B100000)" \
      --suite select_id --workers $W
  $MG --parallel $W --tag confirm_200k -- $PY $P/code/experiments/run_arms.py \
      --arms "playbook:$(pb main_qwen3-coder-30b_s1 B200000)" --suite confirm_id,confirm_ood --workers $W
  log "robustness checks: compiled supervisors on top of the strong configuration, and dense invocation"
  R3="B2b fdd:default playbook:$(pb gp_s1 E20000) playbook:$(pb main_qwen3-coder-30b_s1 final) playbook:$(pb main_qwen3-coder-30b_s3 final)"
  ON=""; DN=""
  for a in $R3; do ON="$ON,onstrong:$a"; DN="$DN,$a"; done
  $MG --parallel $W --tag r3_onstrong -- $PY $P/code/experiments/run_arms.py --arms "${ON#,}" --suite $MAIN --workers $W
  $MG --parallel $W --tag r3_dense -- $PY $P/code/experiments/run_arms.py --arms "${DN#,}" \
      --suite test_id_dense,test_ood_dense --policy '{"periodic_h": 0.5, "hold_interval_h": 0.5, "max_calls": 96}' --workers $W
fi

# ------------------------------------------------------------------ tables, figures, macros, manuscript
if want analysis; then
  log "analysis: tables, statistics, figures, appendix material"
  $PY $P/code/experiments/analyze.py --workers 4 --case SN0003 --no-pending
  $PY $P/code/experiments/final_stats.py   # paired contrasts, escalation, production -> generated/numbers_final.tex, seeds.tex
  $PY $P/code/experiments/figures_revision.py   # synthesis stages (fig_stages) and confirmation suites (fig_confirm)
  $PY $P/code/scripts/make_playbook_figure.py   # annotated playbook excerpt (paper/figures/playbook_excerpt.tex)
  $PY $P/code/scripts/gen_appendix_tables.py
  $PY $P/code/scripts/gen_repro.py ${GEN_REPRO_ARGS:-}
  log "unit tests (CPU only)"
  $PY -m pytest -q $P/code/tests
  if command -v latexmk >/dev/null && [ -f "$P/paper/manuscript.tex" ]; then
    # the manuscript and the supplementary material cross-reference each other (xr-hyper): build, build, rebuild
    (cd $P/paper && latexmk -pdf -interaction=nonstopmode -silent manuscript.tex \
      && latexmk -pdf -interaction=nonstopmode -silent supplementary.tex \
      && latexmk -pdf -g -interaction=nonstopmode -silent manuscript.tex)
  fi
fi
log "run_all: done ($STAGES)"
